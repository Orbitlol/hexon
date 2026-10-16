#!/usr/bin/env python3
"""Train hexon-1.8: hexon-1.2 depth-expanded to ~350M parameters.

Why staged training
-------------------
Expanding 1.2 (30 layers, 134.5M) to ~350M means adding 61 blocks, because
using 1.2 as the base fixes the hidden and FFN widths - changing them would
break weight inheritance. Training all 61 at once needs ~2.6 GB of AdamW
state on top of a 1.4 GB fp32 model; the sandbox has ~3.7 GB usable but
measured RSS of 3.8 GB on a 91-layer run, so it thrashes.

The new blocks are therefore trained in stages of `--layers-per-stage`,
bottom-up. Each stage caches the frozen trunk's hidden states once and
backpropagates only through its own blocks. After a stage, its blocks join the
frozen trunk and the next stage caches against the new boundary. Every stage
checkpoints, so a kill costs one stage rather than the whole run.

Why this is not free intelligence
---------------------------------
Blocks are identity-initialised (zeroed o_proj / down_proj), so the expanded
model's logits match 1.2's exactly before training. Those parameters are
mathematically no-ops until trained - this script is what gives them content.

    python3 scripts/train_18.py --total-layers 91 --layers-per-stage 6 \
        --epochs-per-stage 2 --examples-per-stage 700 --max-seq-len 384
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hexon.config import HexonConfig  # noqa: E402
from hexon.export_gguf import export_to_gguf  # noqa: E402
from hexon.model import HexonForCausalLM  # noqa: E402
import train_15 as t15  # noqa: E402

DEFAULT_SFT = ROOT / "data" / "sft" / "hexon_18_sft.jsonl"


def log(msg: str) -> None:
    print(msg, flush=True)


def save_checkpoint(model: HexonForCausalLM, path: Path,
                    meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": {k: v.detach().half()
                           for k, v in model.state_dict().items()},
            **meta,
        },
        str(path),
    )


def train_stage(
    model: HexonForCausalLM,
    examples: Sequence[Tuple[List[int], List[int]]],
    stage_start: int,
    stage_end: int,
    epochs: int,
    lr: float,
    min_lr: float,
    grad_clip: float,
    seed: int,
    tag: str,
    cache_chunk: int = 400,
    anchor_weight: float = 0.0,
    stats_base_layers: int = 0,
) -> float:
    """Train layers [stage_start, stage_end) with the lower trunk frozen."""
    trainable: List[torch.nn.Parameter] = []
    for idx in range(stage_start, stage_end):
        trainable.extend(model.layers[idx].parameters())
    trainable.extend(model.norm.parameters())
    trainable_ids = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.02,
                            betas=(0.9, 0.95))

    # Anchor the inherited layers to their pre-trained values. Without this the
    # model drifts off hexon-1.2's distribution and generation degenerates into
    # repetition loops; hexon-1.5 used anchor_weight 0.01 for the same reason.
    anchor_ref: Dict[str, torch.Tensor] = {}
    for idx in range(stage_start, min(stage_end, stats_base_layers)):
        for name, p in model.layers[idx].named_parameters():
            if p.requires_grad:
                anchor_ref[f"{idx}.{name}"] = p.detach().clone()
    if anchor_ref:
        log(f"  anchoring {len(anchor_ref)} tensors "
            f"(weight {anchor_weight})")

    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    chunks = [order[i:i + cache_chunk]
              for i in range(0, len(order), cache_chunk)]

    total_steps = epochs * len(examples)
    step = 0
    loss_sum = 0.0
    loss_n = 0

    log(f"  training layers {stage_start}..{stage_end - 1} on all "
        f"{len(examples)} examples ({total_steps} steps, "
        f"{len(chunks)} cached chunks of <={cache_chunk})")
    for ci, idxs in enumerate(chunks):
        sub = [examples[i] for i in idxs]
        t0 = time.time()
        cached = t15.precompute_trunk(model, sub, stage_start)
        lengths = [len(x[0]) for x in sub]
        all_batches = [
            t15.make_index_batches(len(sub), lengths, 1, seed + ep)
            for ep in range(epochs)
        ]
        log(f"  chunk {ci + 1}/{len(chunks)} ({len(sub)} ex, "
            f"cached in {time.time() - t0:.0f}s)")
        for epoch, batches in enumerate(all_batches):
            model.train()
            epoch_loss, n = 0.0, 0
            for batch_idx in batches:
                i = batch_idx[0]
                cache = cached[i]
                hidden = cache["hidden"].float()
                cos, sin, mask = cache["cos"], cache["sin"], cache["mask"]
                labels = torch.tensor([sub[i][1]], dtype=torch.long)

                progress = step / max(1, total_steps - 1)
                cur_lr = min_lr + 0.5 * (lr - min_lr) * (
                    1 + math.cos(math.pi * progress))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr

                for layer in model.layers[stage_start:]:
                    hidden, _ = layer(hidden, cos=cos, sin=sin,
                                      attention_mask=mask)
                hidden = model.norm(hidden)
                loss = t15.sparse_ce_loss(model, hidden, labels)

                if anchor_ref:
                    anchor = None
                    for idx in range(stage_start,
                                    min(stage_end, stats_base_layers)):
                        for name, p in model.layers[idx].named_parameters():
                            ref = anchor_ref.get(f"{idx}.{name}")
                            if ref is None:
                                continue
                            d = (p - ref).pow(2).sum()
                            anchor = d if anchor is None else anchor + d
                    if anchor is not None:
                        loss = loss + anchor_weight * anchor

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                opt.step()

                epoch_loss += float(loss.detach())
                n += 1
                step += 1
                loss_sum += float(loss.detach())
                loss_n += 1
                if step % 500 == 0 or step == 1:
                    log(f"    {tag} step {step}/{total_steps} "
                        f"loss={float(loss.detach()):.4f} lr={cur_lr:.2e}")
            log(f"    {tag} chunk {ci + 1} epoch {epoch + 1}/{epochs} "
                f"mean_loss={epoch_loss / max(1, n):.4f}")
        del cached

    del opt
    return loss_sum / max(1, loss_n)


def _export(model: HexonForCausalLM, tokenizer: Any, args: Any) -> None:
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    info = export_to_gguf(
        model, tokenizer, str(out), quant_type=args.quant,
        description=("Hexon 1.8 - hexon-1.2 depth-expanded to "
                     f"{model.config.num_hidden_layers} layers, staged "
                     "fine-tune on grounded ChromaDB SFT."),
    )
    size = out.stat().st_size
    log(f"   wrote {out}  {size/(1024*1024):.2f} MiB "
        f"({size/1e6:.2f} MB decimal)")
    log(f"   GitHub 100 MiB limit: "
        f"{'OK' if size <= 100*1024*1024 else 'EXCEEDED'}")

    report = ROOT / "exports" / "hexon-1.8-training-report.json"
    report.write_text(json.dumps({
        "model": "hexon-1.8",
        "base": args.base,
        "total_layers": model.config.num_hidden_layers,
        "sft": args.sft,
        "layers_per_stage": args.layers_per_stage,
        "epochs_per_stage": args.epochs_per_stage,
        "quant": args.quant,
        "export": info,
    }, indent=2))
    log(f"   wrote {report.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exports/hexon-1.2.gguf")
    ap.add_argument("--sft", default=str(DEFAULT_SFT))
    ap.add_argument("--total-layers", type=int, default=91)
    ap.add_argument("--layers-per-stage", type=int, default=6)
    ap.add_argument("--epochs-per-stage", type=int, default=2)
    ap.add_argument("--examples-per-stage", type=int, default=0,
                    help="0 = use every record each stage (recommended)")
    ap.add_argument("--cache-chunk", type=int, default=400,
                    help="examples cached at once; controls peak memory")
    ap.add_argument("--train-from-layer", type=int, default=28,
                    help="first layer to train; below this the trunk is frozen")
    ap.add_argument("--anchor-weight", type=float, default=0.01,
                    help="L2 pull of inherited layers to their base values")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--checkpoint",
                    default="/home/user/hexon/.deps/hexon18_ckpt.pt")
    ap.add_argument("--out", default="exports/hexon-1.8.gguf")
    ap.add_argument("--quant", default="q4_deep")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)

    ckpt_path = Path(args.checkpoint)

    log("== building hexon-1.8 ==")
    model, tokenizer, stats = t15.build_expanded_model(
        args.base, args.total_layers)
    log(f"   layers={stats['total_layers']} "
        f"base_layers={stats['base_layers']}")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"   parameters: {n_params/1e6:.1f}M")

    if args.export_only:
        if not ckpt_path.exists():
            log(f"no checkpoint at {ckpt_path}")
            return 1
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["state_dict"], strict=False)
        log("   loaded checkpoint for export")
        _export(model, tokenizer, args)
        return 0

    log("== building examples ==")
    t15.SFT_PATH = Path(args.sft)
    examples = t15.build_examples(tokenizer, args.max_seq_len)
    log(f"   {len(examples)} training examples from {Path(args.sft).name}")

    first = min(args.train_from_layer, stats["base_layers"])
    bounds = [first] + [b for b in range(stats["base_layers"],
                                        args.total_layers,
                                        args.layers_per_stage)
                        if b > first]
    if bounds[-1] < args.total_layers:
        bounds.append(args.total_layers)

    done: List[int] = []
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["state_dict"], strict=False)
        done = ck.get("done_stages", [])
        log(f"   resumed: completed stages = {done}")

    t_start = time.time()
    for s, start in enumerate(bounds[:-1]):
        end = bounds[s + 1]
        if s in done:
            log(f"== stage {s+1}/{len(bounds)-1} "
                f"(layers {start}..{end-1}) already done ==")
            continue
        log(f"== stage {s+1}/{len(bounds)-1}: layers {start}..{end-1} ==")
        if args.examples_per_stage and len(examples) > args.examples_per_stage:
            rng = random.Random(args.seed + s)
            subset = rng.sample(examples, args.examples_per_stage)
        else:
            subset = list(examples)

        loss = train_stage(model, subset, start, end,
                           epochs=args.epochs_per_stage, lr=args.lr,
                           min_lr=args.min_lr, grad_clip=args.grad_clip,
                           seed=args.seed + s, tag=f"S{s+1}",
                           cache_chunk=args.cache_chunk,
                           anchor_weight=args.anchor_weight,
                           stats_base_layers=stats["base_layers"])
        done = sorted(set(done) | {s})
        save_checkpoint(model, ckpt_path,
                        {"done_stages": done, "stage_loss": loss,
                         "total_layers": args.total_layers})
        log(f"   stage {s+1} done (loss {loss:.4f}, elapsed "
            f"{time.time()-t_start:.0f}s) -> checkpoint saved")

    log(f"== all stages complete in {time.time()-t_start:.0f}s ==")

    model.eval()
    model.mobile_inference_locked = False
    log("== sample generations ==")
    from hexon.inference import HexonMobileEngine
    engine = HexonMobileEngine(model=model, tokenizer=tokenizer)
    for q in ["What is the capital of Australia?",
              "What does RAM stand for?",
              "Write a Python function that checks whether a string is a "
              "palindrome."]:
        r = engine.reply(q, max_new_tokens=80)
        log(f"   Q: {q}")
        log(f"   A: {r['response'][:220]}")

    _export(model, tokenizer, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
