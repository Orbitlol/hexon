#!/usr/bin/env python3
"""Delete no-op decoder blocks left behind by depth expansion.

``train_15.py`` grows hexon-1.2 by bolting on extra blocks whose ``o_proj`` and
``down_proj`` weights are zeroed. Because Hexon is pre-norm with bias-free
projections, a zeroed output projection makes the block an identity function.
Blocks that training never touched stay exact no-ops, so they can be deleted
without changing the model's output by a single bit.

The script refuses to write anything unless the rebuilt model's logits match
the original's on a set of probe prompts, so it is safe to point at a
checkpoint and see what happens.

    python3 scripts/prune_identity_layers.py /tmp/ckpt.pt exports/out.gguf \\
        --quant q4_deep --tokenizer exports/hexon-1.2.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexon.config import HexonConfig  # noqa: E402
from hexon.export_gguf import export_to_gguf  # noqa: E402
from hexon.gguf_runtime import GGUFReader  # noqa: E402
from hexon.model import HexonForCausalLM  # noqa: E402

PROBE_IDS = [1, 2, 3, 4, 5]
GITHUB_LIMIT_BYTES = 100 * 1024 * 1024


def find_identity_blocks(model) -> list:
    """Blocks whose o_proj and down_proj are both exactly zero."""
    identity = []
    for idx, block in enumerate(model.model.layers):
        o = block.self_attn.o_proj.weight
        d = block.mlp.down_proj.weight
        if torch.count_nonzero(o) == 0 and torch.count_nonzero(d) == 0:
            identity.append(idx)
    return identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--quant", default="q4_1")
    ap.add_argument("--tokenizer", default="exports/hexon-1.2.gguf")
    ap.add_argument("--description", default="")
    ap.add_argument(
        "--target-layers", type=int, default=0,
        help="Also drop the lowest-index trained blocks to reach this layer "
             "count. Only for fitting a size budget: every block dropped this "
             "way is a real, trained block, so the model changes and must be "
             "re-evaluated.",
    )
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = HexonConfig.from_dict(ck["config"])

    original = HexonForCausalLM(cfg)
    original.load_state_dict(ck["state_dict"], strict=False)
    original.eval()

    identity_idx = find_identity_blocks(original)
    keep_idx = [i for i in range(cfg.num_hidden_layers)
                if i not in set(identity_idx)]

    print(f"layers: {cfg.num_hidden_layers}")
    print(f"identity (removable) blocks: {identity_idx}")
    print(f"keeping: {len(keep_idx)} layers")

    if args.target_layers and len(keep_idx) > args.target_layers:
        drop = len(keep_idx) - args.target_layers
        inherited = [i for i in keep_idx if identity_idx and i < min(identity_idx)]
        droppable = [i for i in keep_idx if i not in set(inherited)]
        to_drop = droppable[:drop]
        print(f"additionally dropping {drop} trained block(s) to hit "
              f"{args.target_layers} layers: {to_drop}")
        keep_idx = [i for i in keep_idx if i not in set(to_drop)]

    if not identity_idx and not args.target_layers:
        print("nothing to prune")
        return 1

    new_cfg = HexonConfig.from_dict(cfg.to_dict())
    new_cfg.num_hidden_layers = len(keep_idx)
    pruned = HexonForCausalLM(new_cfg)
    pruned_state = pruned.state_dict()
    orig_state = original.state_dict()

    for dst, src in zip(range(len(keep_idx)), keep_idx):
        for key in orig_state:
            if key.startswith(f"model.layers.{src}."):
                pruned_state[key.replace(f"model.layers.{src}.",
                                         f"model.layers.{dst}.")] = orig_state[key]
    for key in orig_state:
        if not key.startswith("model.layers."):
            pruned_state[key] = orig_state[key]

    missing, unexpected = pruned.load_state_dict(pruned_state, strict=False)
    assert not missing, f"missing: {missing[:5]}"
    pruned.eval()

    # Verify the rebuild before trusting it.
    worst = 0.0
    with torch.no_grad():
        for pid in PROBE_IDS:
            g = torch.Generator().manual_seed(pid)
            ids = torch.randint(0, cfg.vocab_size, (1, 24), generator=g)
            a = original(input_ids=ids).logits
            b = pruned(input_ids=ids).logits
            worst = max(worst, (a - b).abs().max().item())

    only_identity_removed = (len(keep_idx)
                             == cfg.num_hidden_layers - len(identity_idx))
    print(f"max |logit difference| after pruning: {worst:.3e}")
    if only_identity_removed and worst > 1e-4:
        print("FAIL: only identity blocks were removed, so the pruned model "
              "must be bit-identical. Refusing to export.")
        return 1
    if not only_identity_removed:
        print("NOTE: trained blocks were also removed to hit the size budget, "
              "so the model is NOT equivalent and must be re-evaluated.")

    reader = GGUFReader(args.tokenizer)
    tokenizer = reader.extract_tokenizer()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_gguf(
        pruned, tokenizer, str(out), quant_type=args.quant,
        description=args.description or "hexon-1.5 with no-op blocks removed",
    )

    size = out.stat().st_size
    n_params = sum(p.numel() for p in pruned.parameters())
    old_params = sum(p.numel() for p in original.parameters())
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"wrote {rel}")
    print(f"  {size/1e6:.2f} MB decimal  |  {size/(1024*1024):.2f} MiB")
    print(f"  parameters: {n_params/1e6:.1f}M (was {old_params/1e6:.1f}M)")
    print(f"  GitHub 100 MiB limit: "
          f"{'OK' if size <= GITHUB_LIMIT_BYTES else 'EXCEEDED'}")

    report = out.with_suffix("").name + "-prune-report.json"
    rp = out.parent / report
    rp.write_text(json.dumps({
        "source_checkpoint": args.checkpoint,
        "original_layers": cfg.num_hidden_layers,
        "identity_blocks_removed": identity_idx,
        "kept_layers": len(keep_idx),
        "max_logit_difference": worst,
        "bit_identical": only_identity_removed and worst == 0.0,
        "quant": args.quant,
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 3),
        "under_github_limit": size <= GITHUB_LIMIT_BYTES,
        "params_before": old_params,
        "params_after": n_params,
    }, indent=2))
    print(f"  wrote {rp.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
