"""
Hexon GGUF v3 Reader, Dequantizer, Inspector & Standalone Mobile Inference Loader.

Reads any Hexon `.gguf` file (F32, F16, Q8_0, Q4_0), extracts the embedded
architecture config and tokenizer, dequantizes GGML block tensors, and executes
inference-only generation without requiring any external config or tokenizer files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

import numpy as np
import torch

from .config import HexonConfig
from .export_gguf import (
    BLOCK_Q4_0_DTYPE,
    BLOCK_Q4_1_DTYPE,
    BLOCK_Q8_0_DTYPE,
    GGMLQuantizationType,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFValueType,
    align_offset,
)
from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer


@dataclass
class GGUFTensorInfo:
    name: str
    ggml_dims: Tuple[int, ...]
    torch_shape: Tuple[int, ...]
    ggml_type: GGMLQuantizationType
    offset: int
    num_elements: int
    byte_size: int


def compute_ggml_tensor_bytes(num_elements: int, ggml_type: GGMLQuantizationType) -> int:
    if ggml_type == GGMLQuantizationType.F32:
        return num_elements * 4
    if ggml_type == GGMLQuantizationType.F16:
        return num_elements * 2
    if ggml_type == GGMLQuantizationType.Q8_0:
        return (num_elements // 32) * 34
    if ggml_type == GGMLQuantizationType.Q4_0:
        return (num_elements // 32) * 18
    if ggml_type == GGMLQuantizationType.Q4_1:
        return (num_elements // 32) * 20
    raise ValueError(f"Unsupported GGMLQuantizationType: {ggml_type}")


def dequantize_tensor_q8_0(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize GGML Q8_0 blocks (34 bytes per 32 elements) to float32 numpy array."""
    blocks = np.frombuffer(raw_bytes, dtype=BLOCK_Q8_0_DTYPE)
    d = blocks["d"].astype(np.float32)[:, None]
    qs = blocks["qs"].astype(np.float32)
    out = (d * qs).reshape(shape)
    return out


def dequantize_tensor_q4_0(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize GGML Q4_0 blocks (18 bytes per 32 elements) to float32 numpy array."""
    blocks = np.frombuffer(raw_bytes, dtype=BLOCK_Q4_0_DTYPE)
    d = blocks["d"].astype(np.float32)[:, None]
    qs = blocks["qs"]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = ((qs >> 4) & 0x0F).astype(np.int8) - 8
    nibbles = np.concatenate([lo, hi], axis=1).astype(np.float32)
    out = (d * nibbles).reshape(shape)
    return out


def dequantize_tensor_q4_1(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize GGML Q4_1 blocks (20 bytes per 32 elements) to float32 numpy array."""
    blocks = np.frombuffer(raw_bytes, dtype=BLOCK_Q4_1_DTYPE)
    d = blocks["d"].astype(np.float32)[:, None]
    m = blocks["m"].astype(np.float32)[:, None]
    qs = blocks["qs"]
    lo = (qs & 0x0F).astype(np.float32)
    hi = ((qs >> 4) & 0x0F).astype(np.float32)
    nibbles = np.concatenate([lo, hi], axis=1)
    out = (d * nibbles + m).reshape(shape)
    return out


class GGUFReader:
    """Parser and loader for GGUF v3 model files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self.version: int = 0
        self.tensor_count: int = 0
        self.kv_count: int = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        self.tensor_data_offset: int = 0
        self.file_size_bytes: int = self.path.stat().st_size

        self._parse_file()

    @staticmethod
    def _read_string(f: BinaryIO) -> str:
        (length,) = struct.unpack("<Q", f.read(8))
        return f.read(length).decode("utf-8")

    def _read_value(self, f: BinaryIO, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        if vtype == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        if vtype == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        if vtype == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        if vtype == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        if vtype == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        if vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        if vtype == GGUFValueType.BOOL:
            return bool(struct.unpack("<B", f.read(1))[0])
        if vtype == GGUFValueType.STRING:
            return self._read_string(f)
        if vtype == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        if vtype == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        if vtype == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        if vtype == GGUFValueType.ARRAY:
            elem_type_raw, count = struct.unpack("<IQ", f.read(12))
            elem_type = GGUFValueType(elem_type_raw)
            if elem_type == GGUFValueType.STRING:
                return [self._read_string(f) for _ in range(count)]
            if elem_type == GGUFValueType.FLOAT32:
                return list(struct.unpack(f"<{count}f", f.read(4 * count)))
            if elem_type == GGUFValueType.INT32:
                return list(struct.unpack(f"<{count}i", f.read(4 * count)))
            if elem_type == GGUFValueType.UINT32:
                return list(struct.unpack(f"<{count}I", f.read(4 * count)))
            return [self._read_value(f, elem_type) for _ in range(count)]
        raise ValueError(f"Unsupported GGUFValueType: {vtype}")

    def _parse_file(self) -> None:
        with open(self.path, "rb") as f:
            magic, version, tensor_count, kv_count = struct.unpack("<IIQQ", f.read(24))
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Invalid GGUF magic header: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})"
                )
            self.version = version
            self.tensor_count = tensor_count
            self.kv_count = kv_count

            for _ in range(kv_count):
                key = self._read_string(f)
                (vtype_raw,) = struct.unpack("<I", f.read(4))
                vtype = GGUFValueType(vtype_raw)
                self.metadata[key] = self._read_value(f, vtype)

            for _ in range(tensor_count):
                name = self._read_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                ggml_dims = tuple(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                torch_shape = tuple(reversed(ggml_dims))
                ggml_type_raw, offset = struct.unpack("<IQ", f.read(12))
                ggml_type = GGMLQuantizationType(ggml_type_raw)
                num_elements = int(np.prod(torch_shape))
                byte_size = compute_ggml_tensor_bytes(num_elements, ggml_type)

                self.tensors[name] = GGUFTensorInfo(
                    name=name,
                    ggml_dims=ggml_dims,
                    torch_shape=torch_shape,
                    ggml_type=ggml_type,
                    offset=offset,
                    num_elements=num_elements,
                    byte_size=byte_size,
                )

            alignment = int(self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
            self.tensor_data_offset = align_offset(f.tell(), alignment)

    def load_tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Read and dequantize a single tensor from the GGUF file."""
        info = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.tensor_data_offset + info.offset)
            raw = f.read(info.byte_size)

        if info.ggml_type == GGMLQuantizationType.F32:
            arr = np.frombuffer(raw, dtype="<f4").reshape(info.torch_shape).copy()
        elif info.ggml_type == GGMLQuantizationType.F16:
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info.torch_shape)
        elif info.ggml_type == GGMLQuantizationType.Q8_0:
            arr = dequantize_tensor_q8_0(raw, info.torch_shape)
        elif info.ggml_type == GGMLQuantizationType.Q4_0:
            arr = dequantize_tensor_q4_0(raw, info.torch_shape)
        elif info.ggml_type == GGMLQuantizationType.Q4_1:
            arr = dequantize_tensor_q4_1(raw, info.torch_shape)
        else:
            raise ValueError(f"Unsupported tensor type {info.ggml_type}")

        return torch.from_numpy(arr).to(dtype=dtype)

    def extract_config(self) -> HexonConfig:
        """Reconstruct `HexonConfig` from GGUF metadata."""
        m = self.metadata
        ctx_len = min(int(m["llama.context_length"]), 2048)
        return HexonConfig(
            model_name=str(m.get("general.name", "hexon-gguf")),
            vocab_size=int(m["llama.vocab_size"]),
            hidden_size=int(m["llama.embedding_length"]),
            intermediate_size=int(m["llama.feed_forward_length"]),
            num_hidden_layers=int(m["llama.block_count"]),
            num_attention_heads=int(m["llama.attention.head_count"]),
            num_key_value_heads=int(m["llama.attention.head_count_kv"]),
            max_position_embeddings=ctx_len,
            rms_norm_eps=float(m.get("llama.attention.layer_norm_rms_epsilon", 1e-5)),
            rope_theta=float(m.get("llama.rope.freq_base", 10000.0)),
            tie_word_embeddings=bool(m.get("hexon.tie_word_embeddings", True)),
            pad_token_id=int(m.get("tokenizer.ggml.padding_token_id", 0)),
            bos_token_id=int(m.get("tokenizer.ggml.bos_token_id", 1)),
            eos_token_id=int(m.get("tokenizer.ggml.eos_token_id", 2)),
        )

    def extract_tokenizer(self) -> HexonTokenizer:
        """Reconstruct `HexonTokenizer` directly from embedded GGUF metadata."""
        tokens = list(self.metadata["tokenizer.ggml.tokens"])
        scores = list(self.metadata.get("tokenizer.ggml.scores", [0.0] * len(tokens)))
        merges = self.metadata.get("tokenizer.ggml.merges")
        tok_model = str(self.metadata.get("tokenizer.ggml.model", "llama"))
        pre_tok = str(self.metadata.get("tokenizer.ggml.pre", "default"))
        add_bos = self.metadata.get("tokenizer.ggml.add_bos_token")
        return HexonTokenizer(
            tokens=tokens,
            scores=scores,
            vocab_size=len(tokens),
            merges=list(merges) if merges is not None else None,
            tokenizer_model=tok_model,
            pre_tokenizer=pre_tok,
            add_bos_token=add_bos,
        )

    def load_model_and_tokenizer(
        self, device: str | torch.device = "cpu"
    ) -> Tuple[HexonForCausalLM, HexonTokenizer]:
        """
        Reconstruct a mobile-locked `HexonForCausalLM` and `HexonTokenizer`
        from this GGUF file alone.
        """
        config = self.extract_config()
        tokenizer = self.extract_tokenizer()
        model = HexonForCausalLM(config)

        state_dict: Dict[str, torch.Tensor] = {}
        state_dict["embed_tokens.weight"] = self.load_tensor("token_embd.weight")

        for i in range(config.num_hidden_layers):
            prefix = f"layers.{i}"
            state_dict[f"{prefix}.input_layernorm.weight"] = self.load_tensor(
                f"blk.{i}.attn_norm.weight"
            )
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = self.load_tensor(
                f"blk.{i}.attn_q.weight"
            )
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = self.load_tensor(
                f"blk.{i}.attn_k.weight"
            )
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = self.load_tensor(
                f"blk.{i}.attn_v.weight"
            )
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = self.load_tensor(
                f"blk.{i}.attn_output.weight"
            )
            state_dict[f"{prefix}.post_attention_layernorm.weight"] = self.load_tensor(
                f"blk.{i}.ffn_norm.weight"
            )
            state_dict[f"{prefix}.mlp.gate_proj.weight"] = self.load_tensor(
                f"blk.{i}.ffn_gate.weight"
            )
            state_dict[f"{prefix}.mlp.up_proj.weight"] = self.load_tensor(
                f"blk.{i}.ffn_up.weight"
            )
            state_dict[f"{prefix}.mlp.down_proj.weight"] = self.load_tensor(
                f"blk.{i}.ffn_down.weight"
            )

        state_dict["norm.weight"] = self.load_tensor("output_norm.weight")
        if "output.weight" in self.tensors:
            state_dict["lm_head.weight"] = self.load_tensor("output.weight")
        else:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        # Always lock GGUF-loaded models into inference-only mode (no on-device training)
        model.freeze_for_mobile_inference()
        return model, tokenizer

    def summary(self) -> Dict[str, Any]:
        """Return a structured inspection summary of the GGUF file."""
        type_counts: Dict[str, int] = {}
        total_elements = 0
        total_tensor_bytes = 0
        for info in self.tensors.values():
            tname = info.ggml_type.name
            type_counts[tname] = type_counts.get(tname, 0) + 1
            total_elements += info.num_elements
            total_tensor_bytes += info.byte_size

        return {
            "path": str(self.path),
            "gguf_version": self.version,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 4),
            "file_size_decimal_mb": round(self.file_size_bytes / 1_000_000, 4),
            "architecture": self.metadata.get("general.architecture"),
            "model_name": self.metadata.get("general.name"),
            "quantization": self.metadata.get("hexon.quantization"),
            "mobile_inference_only": self.metadata.get("hexon.mobile_inference_only"),
            "trainable_on_device": self.metadata.get("hexon.trainable_on_device"),
            "vocab_size": self.metadata.get("llama.vocab_size"),
            "context_length": self.metadata.get("llama.context_length"),
            "hidden_size": self.metadata.get("llama.embedding_length"),
            "num_layers": self.metadata.get("llama.block_count"),
            "num_heads": self.metadata.get("llama.attention.head_count"),
            "num_kv_heads": self.metadata.get("llama.attention.head_count_kv"),
            "tensor_count": self.tensor_count,
            "tensor_type_distribution": type_counts,
            "total_tensor_elements": total_elements,
            "total_tensor_bytes": total_tensor_bytes,
        }
