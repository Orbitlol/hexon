#!/usr/bin/env python3
"""Re-export a train_15.py checkpoint under a different quantisation scheme.

Used to separate "the architecture is wrong" from "the quantisation is too
lossy": same weights, different file, rescore, compare.

    python3 scripts/reexport_ckpt.py /tmp/hexon15_ckpt.pt /tmp/out.gguf \\
        --quant q4_deep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexon.config import HexonConfig  # noqa: E402
from hexon.export_gguf import export_to_gguf  # noqa: E402
from hexon.gguf_runtime import GGUFReader  # noqa: E402
from hexon.model import HexonForCausalLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--quant", default="q4_deep")
    ap.add_argument("--tokenizer", default="exports/hexon-1.2.gguf")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = HexonConfig.from_dict(ck["config"])
    model = HexonForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    tokenizer = GGUFReader(args.tokenizer).extract_tokenizer()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_gguf(
        model, tokenizer, str(out), quant_type=args.quant,
        description=f"hexon-1.5 ({cfg.num_hidden_layers} layers, {args.quant})",
    )
    size = out.stat().st_size
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"wrote {rel}  {size/(1024*1024):.2f} MiB ({size/1e6:.2f} MB decimal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
