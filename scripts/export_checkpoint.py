#!/usr/bin/env python3
"""
Export a hexon-1.5 training checkpoint to a GGUF file.

`train_15.py` writes a fp16 checkpoint after every epoch. This turns any one of
them into a shippable GGUF, which matters because the final epoch of a small
SFT run is often not the best one: on 126 examples the training loss falls to
about 0.04 by epoch 8, which is memorisation, not learning. Being able to
export epoch 4 or 5 and score it is how you tell the difference.

Usage
-----
    PYTHONPATH=/tmp/pydeps:/home/user/hexon python3 scripts/export_checkpoint.py \
        /tmp/hexon15_epoch5.pt exports/hexon-1.5-ep5.gguf \
        --tokenizer exports/hexon-1.2.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from hexon.config import HexonConfig  # noqa: E402
from hexon.export_gguf import export_to_gguf  # noqa: E402
from hexon.gguf_runtime import GGUFReader  # noqa: E402
from hexon.model import HexonForCausalLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--tokenizer", default="exports/hexon-1.2.gguf",
                    help="GGUF file to take the tokenizer from.")
    ap.add_argument("--quant", default="q4_code")
    ap.add_argument("--description", default="")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    config = HexonConfig.from_dict(ck["config"])
    model = HexonForCausalLM(config)
    model.mobile_inference_locked = False

    sd = {k: v.float() for k, v in ck["state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  ! missing keys: {len(missing)}")
    if unexpected:
        print(f"  ! unexpected keys: {len(unexpected)}")

    reader = GGUFReader(args.tokenizer)
    _, tokenizer = reader.load_model_and_tokenizer(device="cpu")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    desc = args.description or (
        f"Hexon 1.5 (epoch {ck.get('epoch', '?')}, mean_loss "
        f"{ck.get('mean_loss', 0):.4f}) - depth-expanded from hexon-1.2."
    )
    export_to_gguf(model, tokenizer, str(out), quant_type=args.quant, description=desc)

    mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out.relative_to(ROOT)}  {mb:.2f} MiB "
          f"({out.stat().st_size/1e6:.2f} MB decimal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
