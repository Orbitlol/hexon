"""
Hexon PyTorch Mobile & TorchScript Exporter.

Exports a trained Hexon model into:
1. TorchScript (`torch.jit.trace` + `optimize_for_inference`) `.pt` artifact for
   native C++ / Android JNI / iOS Objective-C++ PyTorch Mobile execution.
2. INT8 Dynamically Quantized PyTorch inference bundle (`.pt`) with training
   permanently disabled.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer


class HexonTorchScriptWrapper(nn.Module):
    """
    Pure tensor-in / tensor-out wrapper around `HexonForCausalLM` for TorchScript
    tracing (`input_ids: LongTensor[B, S] -> logits: FloatTensor[B, S, V]`).
    """

    def __init__(self, model: HexonForCausalLM) -> None:
        super().__init__()
        self.model = model.freeze_for_mobile_inference()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        position_ids = (
            torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        hidden_states = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)

        # Build causal mask without data-dependent branching so TorchScript trace is general
        idx = torch.arange(seq_len, device=input_ids.device)
        cond = idx.unsqueeze(0) <= idx.unsqueeze(1)
        zero = torch.zeros((seq_len, seq_len), device=input_ids.device, dtype=hidden_states.dtype)
        neg_inf = torch.full(
            (seq_len, seq_len), -1e9, device=input_ids.device, dtype=hidden_states.dtype
        )
        causal_mask = torch.where(cond, zero, neg_inf).unsqueeze(0).unsqueeze(0)

        for layer in self.model.layers:
            hidden_states, _ = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=causal_mask,
                past_key_value=None,
                use_cache=False,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits


def export_to_torchscript(
    model: HexonForCausalLM,
    output_path: str | Path,
    example_seq_len: int = 16,
) -> Dict[str, Any]:
    """
    Trace and optimize `HexonForCausalLM` as a standalone TorchScript `.pt` file
    for mobile deployment (Android / iOS).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_copy = copy.deepcopy(model).cpu().freeze_for_mobile_inference()
    wrapper = HexonTorchScriptWrapper(model_copy).eval()

    example_input = torch.randint(
        0, model.config.vocab_size, (1, example_seq_len), dtype=torch.long
    )

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_input, strict=True)
        frozen_script = torch.jit.freeze(traced)
        optimized = torch.jit.optimize_for_inference(frozen_script)
        optimized.save(str(output_path))

    size_bytes = output_path.stat().st_size
    return {
        "output_path": str(output_path),
        "format": "torchscript-mobile",
        "mobile_inference_only": True,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 4),
    }


def export_to_mobile_int8_pt(
    model: HexonForCausalLM,
    tokenizer: HexonTokenizer,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Apply PyTorch dynamic INT8 quantization (`qint8`) to all linear projections
    and save a self-contained mobile inference bundle (no optimizer state, no gradients).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_copy = copy.deepcopy(model).cpu().freeze_for_mobile_inference()
    # Select supported CPU quantization engine (qnnpack on ARM mobile, fbgemm on x86)
    supported_engines = torch.backends.quantized.supported_engines
    if "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},
        dtype=torch.qint8,
    )

    bundle = {
        "format": "hexon-mobile-int8-pt",
        "mobile_inference_only": True,
        "trainable_on_device": False,
        "quant_engine": torch.backends.quantized.engine,
        "config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "quantized_state_dict": quantized_model.state_dict(),
    }
    torch.save(bundle, str(output_path))

    size_bytes = output_path.stat().st_size
    return {
        "output_path": str(output_path),
        "format": "pytorch-dynamic-int8",
        "quant_engine": torch.backends.quantized.engine,
        "mobile_inference_only": True,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 4),
    }
