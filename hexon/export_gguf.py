"""
Hexon GGUF v3 Binary Exporter & Quantizer (F32, F16, Q8_0, Q4_0).

Produces spec-compliant GGUF v3 files (`0x46554747`) with full model architecture
metadata, embedded tokenizer tables, and 32-byte aligned tensor blocks for
on-device mobile inference runtimes.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config import HexonConfig
from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer


GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8


GGUF_FILE_TYPE_MAP: Dict[str, int] = {
    "f32": 0,   # ALL_F32
    "f16": 1,   # MOSTLY_F16
    "q4_0": 2,  # MOSTLY_Q4_0
    "q4_1": 3,  # MOSTLY_Q4_1
    "q8_0": 7,  # MOSTLY_Q8_0
}

BLOCK_Q8_0_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
BLOCK_Q4_0_DTYPE = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
BLOCK_Q4_1_DTYPE = np.dtype([("d", "<f2"), ("m", "<f2"), ("qs", "u1", (16,))])


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    """Round offset up to nearest multiple of `alignment`."""
    return (offset + alignment - 1) // alignment * alignment


def quantize_tensor_q8_0(arr_f32: np.ndarray) -> bytes:
    """
    Quantize a float32 array into GGML Q8_0 blocks (block_size=32, 34 bytes/block).
    Each block stores: [fp16 scale `d`, 32 x int8 values].
    """
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError(f"Q8_0 requires element count divisible by 32, got {flat.size}")
    num_blocks = flat.size // 32
    blocks_f32 = flat.reshape(num_blocks, 32)

    amax = np.max(np.abs(blocks_f32), axis=1)
    d = (amax / 127.0).astype("<f2")
    d_f32 = d.astype(np.float32)[:, None]
    inv_d = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)
    qs = np.clip(np.round(blocks_f32 * inv_d), -128, 127).astype(np.int8)

    out = np.empty(num_blocks, dtype=BLOCK_Q8_0_DTYPE)
    out["d"] = d
    out["qs"] = qs
    return out.tobytes()


def quantize_tensor_q4_0(arr_f32: np.ndarray) -> bytes:
    """
    Quantize a float32 array into GGML Q4_0 blocks (block_size=32, 18 bytes/block).
    Each block stores: [fp16 scale `d`, 16 x uint8 packed nibbles (offset +8)].
    """
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError(f"Q4_0 requires element count divisible by 32, got {flat.size}")
    num_blocks = flat.size // 32
    blocks_f32 = flat.reshape(num_blocks, 32)

    # Official GGML quantize_row_q4_0_ref formula:
    # Map the element with largest absolute value in each 32-element block to -8
    max_idx = np.argmax(np.abs(blocks_f32), axis=1)
    max_signed = blocks_f32[np.arange(num_blocks), max_idx]
    d = (max_signed / -8.0).astype("<f2")
    d_f32 = d.astype(np.float32)[:, None]
    inv_d = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)

    xi = np.clip(np.trunc(blocks_f32 * inv_d + 8.5), 0, 15).astype(np.uint8)
    lo = xi[:, :16] & 0x0F
    hi = (xi[:, 16:] & 0x0F) << 4

    out = np.empty(num_blocks, dtype=BLOCK_Q4_0_DTYPE)
    out["d"] = d
    out["qs"] = lo | hi
    return out.tobytes()


def quantize_tensor_q4_1(arr_f32: np.ndarray) -> bytes:
    """
    Quantize a float32 array into GGML Q4_1 blocks (block_size=32, 20 bytes/block).
    Each block stores: [fp16 scale `d`, fp16 min `m`, 16 x uint8 packed nibbles].
    """
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError(f"Q4_1 requires element count divisible by 32, got {flat.size}")
    num_blocks = flat.size // 32
    blocks_f32 = flat.reshape(num_blocks, 32)

    min_v = np.min(blocks_f32, axis=1)
    max_v = np.max(blocks_f32, axis=1)
    d = ((max_v - min_v) / 15.0).astype("<f2")
    m = min_v.astype("<f2")
    d_f32 = d.astype(np.float32)[:, None]
    m_f32 = m.astype(np.float32)[:, None]
    inv_d = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)

    xi = np.clip(np.round((blocks_f32 - m_f32) * inv_d), 0, 15).astype(np.uint8)
    lo = xi[:, :16] & 0x0F
    hi = (xi[:, 16:] & 0x0F) << 4

    out = np.empty(num_blocks, dtype=BLOCK_Q4_1_DTYPE)
    out["d"] = d
    out["m"] = m
    out["qs"] = lo | hi
    return out.tobytes()


def encode_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_gguf_kv(key: str, val_type: GGUFValueType, value: Any) -> bytes:
    buf = bytearray(encode_gguf_string(key))
    buf.extend(struct.pack("<I", int(val_type)))

    if val_type == GGUFValueType.UINT8:
        buf.extend(struct.pack("<B", int(value)))
    elif val_type == GGUFValueType.INT8:
        buf.extend(struct.pack("<b", int(value)))
    elif val_type == GGUFValueType.UINT16:
        buf.extend(struct.pack("<H", int(value)))
    elif val_type == GGUFValueType.INT16:
        buf.extend(struct.pack("<h", int(value)))
    elif val_type == GGUFValueType.UINT32:
        buf.extend(struct.pack("<I", int(value)))
    elif val_type == GGUFValueType.INT32:
        buf.extend(struct.pack("<i", int(value)))
    elif val_type == GGUFValueType.FLOAT32:
        buf.extend(struct.pack("<f", float(value)))
    elif val_type == GGUFValueType.BOOL:
        buf.extend(struct.pack("<B", 1 if value else 0))
    elif val_type == GGUFValueType.STRING:
        buf.extend(encode_gguf_string(str(value)))
    elif val_type == GGUFValueType.UINT64:
        buf.extend(struct.pack("<Q", int(value)))
    elif val_type == GGUFValueType.INT64:
        buf.extend(struct.pack("<q", int(value)))
    elif val_type == GGUFValueType.FLOAT64:
        buf.extend(struct.pack("<d", float(value)))
    elif val_type == GGUFValueType.ARRAY:
        elem_type, items = value
        buf.extend(struct.pack("<IQ", int(elem_type), len(items)))
        if elem_type == GGUFValueType.STRING:
            for item in items:
                buf.extend(encode_gguf_string(str(item)))
        elif elem_type == GGUFValueType.FLOAT32:
            buf.extend(struct.pack(f"<{len(items)}f", *[float(x) for x in items]))
        elif elem_type == GGUFValueType.INT32:
            buf.extend(struct.pack(f"<{len(items)}i", *[int(x) for x in items]))
        elif elem_type == GGUFValueType.UINT32:
            buf.extend(struct.pack(f"<{len(items)}I", *[int(x) for x in items]))
        else:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
    else:
        raise ValueError(f"Unsupported GGUF value type: {val_type}")

    return bytes(buf)


def map_hexon_tensors_for_gguf(
    model: HexonForCausalLM,
) -> List[Tuple[str, torch.Tensor]]:
    """
    Map Hexon PyTorch state_dict tensors to standard GGUF / llama.cpp tensor names.
    Ensures the model is frozen for inference first.
    """
    model.freeze_for_mobile_inference()
    sd = model.state_dict()
    tensors: List[Tuple[str, torch.Tensor]] = []

    tensors.append(("token_embd.weight", sd["embed_tokens.weight"].detach().cpu()))

    for i in range(model.config.num_hidden_layers):
        prefix = f"layers.{i}"
        tensors.extend(
            [
                (f"blk.{i}.attn_norm.weight", sd[f"{prefix}.input_layernorm.weight"].detach().cpu()),
                (f"blk.{i}.attn_q.weight", sd[f"{prefix}.self_attn.q_proj.weight"].detach().cpu()),
                (f"blk.{i}.attn_k.weight", sd[f"{prefix}.self_attn.k_proj.weight"].detach().cpu()),
                (f"blk.{i}.attn_v.weight", sd[f"{prefix}.self_attn.v_proj.weight"].detach().cpu()),
                (f"blk.{i}.attn_output.weight", sd[f"{prefix}.self_attn.o_proj.weight"].detach().cpu()),
                (
                    f"blk.{i}.ffn_norm.weight",
                    sd[f"{prefix}.post_attention_layernorm.weight"].detach().cpu(),
                ),
                (f"blk.{i}.ffn_gate.weight", sd[f"{prefix}.mlp.gate_proj.weight"].detach().cpu()),
                (f"blk.{i}.ffn_up.weight", sd[f"{prefix}.mlp.up_proj.weight"].detach().cpu()),
                (f"blk.{i}.ffn_down.weight", sd[f"{prefix}.mlp.down_proj.weight"].detach().cpu()),
            ]
        )

    tensors.append(("output_norm.weight", sd["norm.weight"].detach().cpu()))
    # Include output.weight explicitly for compact vocabularies; for large tied vocabs
    # (>16k tokens), omitting output.weight lets llama.cpp share token_embd.weight directly.
    if not (model.config.tie_word_embeddings and model.config.vocab_size > 16000):
        tensors.append(("output.weight", sd["lm_head.weight"].detach().cpu()))
    return tensors


def serialize_gguf_tensor(
    tensor: torch.Tensor,
    quant_type: str,
) -> Tuple[GGMLQuantizationType, Tuple[int, ...], bytes]:
    """
    Convert a PyTorch tensor to GGUF GGML dimensions and quantized raw bytes.
    Per GGML convention:
    - 1D norm tensors are always stored in F32 for numerical stability.
    - 2D weight tensors use the target quantization type (`f32`, `f16`, `q8_0`, `q4_0`, `q4_1`).
    - GGML dimension ordering reverses PyTorch shape: `(rows, cols)` -> `(cols, rows)`.
    """
    arr_f32 = tensor.to(torch.float32).numpy()
    ggml_dims = tuple(reversed(arr_f32.shape))

    # 1D tensors (e.g. RMSNorm weights) always stay F32 in GGUF
    if arr_f32.ndim == 1 or quant_type == "f32":
        return (
            GGMLQuantizationType.F32,
            ggml_dims,
            np.ascontiguousarray(arr_f32, dtype="<f4").tobytes(),
        )
    if quant_type == "f16":
        return (
            GGMLQuantizationType.F16,
            ggml_dims,
            np.ascontiguousarray(arr_f32, dtype="<f2").tobytes(),
        )
    if quant_type == "q8_0":
        return (
            GGMLQuantizationType.Q8_0,
            ggml_dims,
            quantize_tensor_q8_0(arr_f32),
        )
    if quant_type == "q4_0":
        return (
            GGMLQuantizationType.Q4_0,
            ggml_dims,
            quantize_tensor_q4_0(arr_f32),
        )
    if quant_type == "q4_1":
        return (
            GGMLQuantizationType.Q4_1,
            ggml_dims,
            quantize_tensor_q4_1(arr_f32),
        )

    raise ValueError(
        f"Unsupported quantization type '{quant_type}'. Choose from: f32, f16, q8_0, q4_0, q4_1"
    )


def export_to_gguf(
    model: HexonForCausalLM,
    tokenizer: HexonTokenizer,
    output_path: str | Path,
    quant_type: str = "q8_0",
    description: str = "Hexon compact mobile language model (inference-only GGUF export)",
) -> Dict[str, Any]:
    """
    Export a trained Hexon PyTorch model and tokenizer into a single self-contained
    GGUF v3 binary file ready for mobile inference.
    """
    quant_type = quant_type.lower().strip()
    if quant_type not in GGUF_FILE_TYPE_MAP:
        raise ValueError(
            f"Invalid quant_type '{quant_type}'. Supported: {list(GGUF_FILE_TYPE_MAP.keys())}"
        )

    config: HexonConfig = model.config
    model.freeze_for_mobile_inference()

    # Build GGUF metadata KV pairs
    kv_entries: List[Tuple[str, GGUFValueType, Any]] = [
        ("general.architecture", GGUFValueType.STRING, "llama"),
        ("general.name", GGUFValueType.STRING, config.model_name),
        ("general.description", GGUFValueType.STRING, description),
        ("general.file_type", GGUFValueType.UINT32, GGUF_FILE_TYPE_MAP[quant_type]),
        ("general.quantization_version", GGUFValueType.UINT32, 2),
        ("general.alignment", GGUFValueType.UINT32, GGUF_DEFAULT_ALIGNMENT),
        # Hexon mobile-specific metadata
        ("hexon.mobile_inference_only", GGUFValueType.BOOL, True),
        ("hexon.trainable_on_device", GGUFValueType.BOOL, False),
        ("hexon.quantization", GGUFValueType.STRING, quant_type),
        ("hexon.tie_word_embeddings", GGUFValueType.BOOL, config.tie_word_embeddings),
        # Architecture hyperparameters (llama.* namespace for ecosystem compatibility)
        ("llama.vocab_size", GGUFValueType.UINT32, config.vocab_size),
        ("llama.context_length", GGUFValueType.UINT32, config.max_position_embeddings),
        ("llama.embedding_length", GGUFValueType.UINT32, config.hidden_size),
        ("llama.block_count", GGUFValueType.UINT32, config.num_hidden_layers),
        ("llama.feed_forward_length", GGUFValueType.UINT32, config.intermediate_size),
        ("llama.attention.head_count", GGUFValueType.UINT32, config.num_attention_heads),
        ("llama.attention.head_count_kv", GGUFValueType.UINT32, config.num_key_value_heads),
        ("llama.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, config.rms_norm_eps),
        ("llama.rope.dimension_count", GGUFValueType.UINT32, config.head_dim),
        ("llama.rope.freq_base", GGUFValueType.FLOAT32, config.rope_theta),
        # Embedded Tokenizer metadata
        ("tokenizer.ggml.model", GGUFValueType.STRING, tokenizer.tokenizer_model),
        (
            "tokenizer.ggml.tokens",
            GGUFValueType.ARRAY,
            (GGUFValueType.STRING, tokenizer.tokens),
        ),
        (
            "tokenizer.ggml.scores",
            GGUFValueType.ARRAY,
            (GGUFValueType.FLOAT32, tokenizer.scores),
        ),
        (
            "tokenizer.ggml.token_type",
            GGUFValueType.ARRAY,
            (GGUFValueType.INT32, tokenizer.get_gguf_token_types()),
        ),
        ("tokenizer.ggml.bos_token_id", GGUFValueType.UINT32, tokenizer.bos_token_id),
        ("tokenizer.ggml.eos_token_id", GGUFValueType.UINT32, tokenizer.eos_token_id),
        ("tokenizer.ggml.unknown_token_id", GGUFValueType.UINT32, tokenizer.unk_token_id),
        ("tokenizer.ggml.padding_token_id", GGUFValueType.UINT32, tokenizer.pad_token_id),
        ("tokenizer.ggml.add_bos_token", GGUFValueType.BOOL, tokenizer.add_bos_token),
        ("tokenizer.ggml.add_space_prefix", GGUFValueType.BOOL, False),
        (
            "tokenizer.chat_template",
            GGUFValueType.STRING,
            tokenizer.to_dict()["chat_template"],
        ),
    ]
    if tokenizer.tokenizer_model == "gpt2":
        kv_entries.append(("tokenizer.ggml.pre", GGUFValueType.STRING, tokenizer.pre_tokenizer or "smollm"))
        if tokenizer.merges:
            kv_entries.append(
                (
                    "tokenizer.ggml.merges",
                    GGUFValueType.ARRAY,
                    (GGUFValueType.STRING, tokenizer.merges),
                )
            )

    mapped_tensors = map_hexon_tensors_for_gguf(model)

    # Encode all tensors and compute relative data offsets (32-byte aligned)
    tensor_infos_buf = bytearray()
    tensor_data_chunks: List[bytes] = []
    current_data_offset = 0

    for name, tensor in mapped_tensors:
        ggml_type, ggml_dims, raw_bytes = serialize_gguf_tensor(tensor, quant_type=quant_type)

        # Tensor info entry: name + n_dims + dims + ggml_type + offset
        tensor_infos_buf.extend(encode_gguf_string(name))
        tensor_infos_buf.extend(struct.pack("<I", len(ggml_dims)))
        for d in ggml_dims:
            tensor_infos_buf.extend(struct.pack("<Q", int(d)))
        tensor_infos_buf.extend(struct.pack("<IQ", int(ggml_type), current_data_offset))

        # Pad tensor payload to GGUF_DEFAULT_ALIGNMENT
        padded_len = align_offset(len(raw_bytes), GGUF_DEFAULT_ALIGNMENT)
        padding = b"\x00" * (padded_len - len(raw_bytes))
        tensor_data_chunks.append(raw_bytes + padding)
        current_data_offset += padded_len

    # Assemble header + KV metadata + tensor info table
    header_and_meta = bytearray()
    header_and_meta.extend(
        struct.pack(
            "<IIQQ",
            GGUF_MAGIC,
            GGUF_VERSION,
            len(mapped_tensors),
            len(kv_entries),
        )
    )
    for k, vtype, val in kv_entries:
        header_and_meta.extend(encode_gguf_kv(k, vtype, val))

    header_and_meta.extend(tensor_infos_buf)

    # Pad end of tensor info table to 32-byte alignment before tensor data section
    aligned_header_size = align_offset(len(header_and_meta), GGUF_DEFAULT_ALIGNMENT)
    header_and_meta.extend(b"\x00" * (aligned_header_size - len(header_and_meta)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header_and_meta)
        for chunk in tensor_data_chunks:
            f.write(chunk)

    total_bytes = output_path.stat().st_size
    return {
        "output_path": str(output_path),
        "quant_type": quant_type,
        "gguf_version": GGUF_VERSION,
        "tensor_count": len(mapped_tensors),
        "metadata_kv_count": len(kv_entries),
        "file_size_bytes": total_bytes,
        "file_size_mb": round(total_bytes / (1024 * 1024), 4),
    }
