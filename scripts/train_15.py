#!/usr/bin/env python3
"""
Train hexon-1.5: a deeper Hexon built on top of hexon-1.2.

Approach — block expansion with identity initialisation
------------------------------------------------------
hexon-1.5 keeps hexon-1.2's width and vocabulary (576 hidden, 1536 FFN,
49152 vocab) and adds 18 decoder layers on top of its 30. Every new layer is
initialised so that it initially computes the identity function: `o_proj` and
`down_proj` weights are set to zero, and because Hexon is pre-norm with no
projection biases, a block whose output projection is zero adds nothing to the
residual stream.

That means hexon-1.5 starts out *exactly* equal to hexon-1.2 and can only
diverge by learning something useful. It is the same trick used for depth
upscaling in larger models, and it is why this script can add capacity without
re-training or destroying what hexon-1.2 already knows.

Training runs the frozen trunk under `torch.no_grad()` and only tracks
gradients through the trainable tail, which keeps activation memory low enough
to run on 2 CPU cores.

Usage
-----
    PYTHONPATH=/tmp/pydeps:/home/user/hexon python3 scripts/train_15.py \
        --base exports/hexon-1.2.gguf --layers 48 --epochs 12
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hexon.config import HexonConfig  # noqa: E402
from hexon.export_gguf import export_to_gguf  # noqa: E402
from hexon.gguf_runtime import GGUFReader  # noqa: E402
from hexon.model import HexonForCausalLM  # noqa: E402

SFT_PATH = ROOT / "data" / "sft" / "hexon_15_sft.jsonl"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_expanded_model(
    base_path: str | Path,
    total_layers: int,
    device: str = "cpu",
) -> Tuple[HexonForCausalLM, Any, Dict[str, Any]]:
    """Load a base model and return a deeper model that behaves identically."""
    reader = GGUFReader(base_path)
    base_model, tokenizer = reader.load_model_and_tokenizer(device=device)
    base_layers = base_model.config.num_hidden_layers
    if total_layers <= base_layers:
        raise ValueError(
            f"total_layers ({total_layers}) must exceed the base model's "
            f"{base_layers} layers"
        )

    cfg_dict = base_model.config.to_dict()
    cfg_dict["num_hidden_layers"] = total_layers
    cfg_dict["model_name"] = "hexon-1.5"
    new_config = HexonConfig.from_dict(cfg_dict)

    model = HexonForCausalLM(new_config)
    model = model.to(device)

    # Carry over everything the base model has.
    base_sd = base_model.state_dict()
    new_sd = model.state_dict()

    carried, fresh = 0, 0
    with torch.no_grad():
        for name, tensor in new_sd.items():
            # Layers are named `layers.<idx>.<rest>`; only indices below
            # base_layers exist in the base checkpoint.
            if name.startswith("layers."):
                idx = int(name.split(".")[1])
                if idx < base_layers:
                    src = base_sd[name]
                    if src.shape == tensor.shape:
                        tensor.copy_(src)
                        carried += 1
                        continue
            elif name in base_sd and base_sd[name].shape == tensor.shape:
                tensor.copy_(base_sd[name])
                carried += 1
                continue
            fresh += 1

        # Identity-initialise the new blocks: zero their output projections.
        for idx in range(base_layers, total_layers):
            model.layers[idx].self_attn.o_proj.weight.zero_()
            model.layers[idx].mlp.down_proj.weight.zero_()

    stats = {
        "base_layers": base_layers,
        "total_layers": total_layers,
        "tensors_carried_from_base": carried,
        "tensors_freshly_initialised": fresh,
        "identity_init_new_layers": list(range(base_layers, total_layers)),
    }
    return model, tokenizer, stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_examples(
    tokenizer: Any,
    max_seq_len: int,
) -> List[Tuple[List[int], List[int]]]:
    """Tokenise the SFT set with prompt masking (labels are -100 on the prompt)."""
    rows: List[Dict[str, Any]] = []
    with open(SFT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    examples: List[Tuple[List[int], List[int]]] = []
    for row in rows:
        if row.get("category") == "multiturn":
            # Conversation records carry turn-by-turn content; mask each user
            # turn and supervise each assistant turn.
            ids: List[int] = []
            labels: List[int] = []
            for turn in row["turns_text"]:
                u_ids = tokenizer.encode(
                    f"<|im_start|>user\n{turn['q']}<|im_end|>\n<|im_start|>assistant\n",
                    add_bos=not ids,
                    add_eos=False,
                )
                a_ids = tokenizer.encode(
                    f"{turn['a']}<|im_end|>\n", add_bos=False, add_eos=False
                )
                ids.extend(u_ids)
                labels.extend([-100] * len(u_ids))
                ids.extend(a_ids)
                labels.extend(a_ids)
        else:
            p_ids = tokenizer.encode(
                f"<|im_start|>user\n{row['user']}<|im_end|>\n<|im_start|>assistant\n",
                add_bos=True,
                add_eos=False,
            )
            a_ids = tokenizer.encode(
                f"{row['assistant']}<|im_end|>\n", add_bos=False, add_eos=False
            )
            ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids

        if len(ids) > max_seq_len:
            # Keep the tail: the answer is at the end, and the context head is
            # the most expendable part of a grounded prompt.
            overflow = len(ids) - max_seq_len
            ids = ids[overflow:]
            labels = labels[overflow:]
            # A prompt-only remainder teaches nothing; drop those.
            if not any(t != -100 for t in labels):
                continue
        if not ids or not any(t != -100 for t in labels):
            continue
        examples.append((ids, labels))
    return examples


def make_batches(
    examples: Sequence[Tuple[List[int], List[int]]],
    batch_size: int,
    seed: int,
    pad_id: int,
):
    """Length-bucketed batches to cut padding waste."""
    rng = random.Random(seed)
    idx = list(examples)
    rng.shuffle(idx)
    idx.sort(key=lambda x: len(x[0]))
    batches = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        max_len = max(len(x[0]) for x in chunk)
        inputs = torch.tensor(
            [x[0] + [pad_id] * (max_len - len(x[0])) for x in chunk], dtype=torch.long
        )
        labels = torch.tensor(
            [x[1] + [-100] * (max_len - len(x[1])) for x in chunk], dtype=torch.long
        )
        batches.append((inputs, labels))
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# Forward / loss
# ---------------------------------------------------------------------------


def forward_tail(
    model: HexonForCausalLM,
    input_ids: torch.Tensor,
    grad_from_layer: int,
) -> torch.Tensor:
    """Run the full stack, tracking gradients only from `grad_from_layer` on."""
    bsz, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    hidden = model.embed_tokens(input_ids)
    cos, sin = model.rotary_emb(hidden, position_ids)
    mask = model._build_causal_mask(
        q_len=seq_len, kv_len=seq_len, device=hidden.device, dtype=hidden.dtype
    )

    with torch.no_grad():
        for layer in model.layers[:grad_from_layer]:
            hidden, _ = layer(hidden, cos=cos, sin=sin, attention_mask=mask)

    for layer in model.layers[grad_from_layer:]:
        hidden, _ = layer(hidden, cos=cos, sin=sin, attention_mask=mask)

    hidden = model.norm(hidden)
    return model.lm_head(hidden)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="exports/hexon-1.2.gguf")
    ap.add_argument("--total-layers", type=int, default=48)
    ap.add_argument("--train-from-layer", type=int, default=28,
                    help="First layer that receives gradients (old layers below this stay frozen).")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--anchor-weight", type=float, default=0.01,
                    help="L2 pull back toward hexon-1.2 for the pre-existing trainable layers.")
    ap.add_argument("--max-seq-len", type=int, default=640)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="exports/hexon-1.5.gguf")
    ap.add_argument("--save-pt", default="", help="Optional fp16 checkpoint path.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.time()
    print(f"== building hexon-1.5 from {args.base} ==")
    model, tokenizer, stats = build_expanded_model(args.base, args.total_layers, args.device)
    print(f"   {stats}")
    total_params = model.num_parameters()
    print(f"   parameters: {total_params/1e6:.1f}M")

    # Unlock for training (the GGUF loader hands back a mobile-locked model).
    model.mobile_inference_locked = False
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for idx in range(args.train_from_layer, args.total_layers):
        trainable.extend(model.layers[idx].parameters())
    trainable.extend(model.norm.parameters())
    for p in trainable:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in trainable if p.requires_grad)
    print(f"   trainable params: {n_train/1e6:.1f}M "
          f"(layers {args.train_from_layer}..{args.total_layers-1} + final norm)")

    # Snapshot the pre-existing trainable weights for the anchor term.
    anchor_ref = {}
    for idx in range(args.train_from_layer, stats["base_layers"]):
        for name, p in model.layers[idx].named_parameters():
            if p.requires_grad:
                anchor_ref[f"{idx}.{name}"] = p.detach().clone()

    print("== tokenising SFT set ==")
    examples = build_examples(tokenizer, args.max_seq_len)
    lens = [len(x[0]) for x in examples]
    print(f"   {len(examples)} examples, len min={min(lens)} max={max(lens)} "
          f"avg={sum(lens)/len(lens):.0f}")

    pad_id = tokenizer.pad_token_id
    opt = torch.optim.AdamW(
        [p for p in trainable if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    all_batches = [make_batches(examples, args.batch_size, args.seed + ep, pad_id)
                   for ep in range(args.epochs)]
    total_steps = sum(len(b) for b in all_batches)
    step = 0

    print(f"== training {args.epochs} epochs / {total_steps} steps ==")
    for epoch, batches in enumerate(all_batches):
        model.train()
        epoch_loss, n = 0.0, 0
        for input_ids, labels in batches:
            progress = step / max(1, total_steps - 1)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
            for pg in opt.param_groups:
                pg["lr"] = lr

            logits = forward_tail(model, input_ids, args.train_from_layer)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

            if anchor_ref:
                anchor = None
                for idx in range(args.train_from_layer, stats["base_layers"]):
                    for name, p in model.layers[idx].named_parameters():
                        ref = anchor_ref.get(f"{idx}.{name}")
                        if ref is None:
                            continue
                        d = (p - ref).pow(2).sum()
                        anchor = d if anchor is None else anchor + d
                if anchor is not None:
                    loss = loss + args.anchor_weight * anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable if p.requires_grad], args.grad_clip
            )
            opt.step()

            epoch_loss += float(loss.detach())
            n += 1
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"   step {step}/{total_steps} loss={float(loss.detach()):.4f} lr={lr:.2e}")
        print(f"   epoch {epoch+1}/{args.epochs} mean_loss={epoch_loss/max(1,n):.4f}")

    print(f"   training took {time.time()-t0:.0f}s")

    # --- quick qualitative check before exporting ---------------------------
    model.eval()
    model.mobile_inference_locked = False
    print("== sample generations ==")
    for q in [
        "What is the capital of Australia?",
        "What does RAM stand for?",
        "Write a Python function that checks whether a string is a palindrome.",
    ]:
        ids = tokenizer.encode(
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
            add_bos=True, add_eos=False,
        )
        with torch.inference_mode():
            out = model.generate(
                input_ids=torch.tensor([ids], dtype=torch.long),
                max_new_tokens=90,
                temperature=0.0,
                top_k=20,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=5,
            )
        new_ids = out[0, len(ids):].tolist()
        if tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
        print(f"   Q: {q}")
        print(f"   A: {tokenizer.decode(new_ids, skip_special_tokens=True).strip()[:220]}")

    # --- export ------------------------------------------------------------
    print("== exporting ==")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = export_to_gguf(
        model,
        tokenizer,
        str(out_path),
        quant_type="q4_code",
        description=(
            "Hexon 1.5 - 48-layer depth-expanded Hexon (from hexon-1.2), "
            "identity-initialised new blocks, grounded ChromaDB RAG SFT."
        ),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"   wrote {out_path.relative_to(ROOT)}  {size_mb:.2f} MiB "
          f"({out_path.stat().st_size/1e6:.2f} MB decimal)")

    if args.save_pt:
        pt_path = ROOT / args.save_pt
        torch.save(
            {"config": model.config.to_dict(), "state_dict": model.state_dict()},
            str(pt_path),
        )
        print(f"   wrote {pt_path.relative_to(ROOT)}")

    meta = {
        "base_model": str(args.base),
        "architecture": {
            "layers": args.total_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "parameters": total_params,
            "trainable_parameters": n_train,
        },
        "expansion": stats,
        "training": {
            "sft_records": len(examples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "anchor_weight": args.anchor_weight,
            "train_from_layer": args.train_from_layer,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "total_steps": total_steps,
            "wall_clock_s": round(time.time() - t0, 1),
        },
        "export": {"path": str(args.out), "size_mb": round(size_mb, 2)},
    }
    meta_path = ROOT / "exports" / "hexon-1.5-training-report.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"   wrote {meta_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
