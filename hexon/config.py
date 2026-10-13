"""
Hexon Model Configuration & Mobile Presets.

Hexon is an inference-first compact Transformer architecture designed specifically
for mobile and edge devices (smartphones, tablets, wearables). Training is performed
strictly offline; mobile deployments receive frozen, quantized GGUF or TorchScript
export files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class HexonConfig:
    """Configuration for the Hexon causal language model."""

    model_name: str = "hexon-nano"
    vocab_size: int = 1024
    hidden_size: int = 192
    intermediate_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 2  # Grouped-Query Attention (3:1 GQA ratio)
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for Grouped-Query Attention"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def estimate_parameter_count(self) -> int:
        """Calculate exact parameter count based on architecture hyperparameters."""
        embed_params = self.vocab_size * self.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim

        # Per layer:
        # Attention: q_proj (H*H) + k_proj (H*KV) + v_proj (H*KV) + o_proj (H*H)
        attn_params = 2 * (self.hidden_size * self.hidden_size) + 2 * (self.hidden_size * kv_dim)
        # SwiGLU MLP: gate_proj (H*I) + up_proj (H*I) + down_proj (I*H)
        mlp_params = 3 * (self.hidden_size * self.intermediate_size)
        # Norms: input_layernorm (H) + post_attention_layernorm (H)
        norm_params = 2 * self.hidden_size

        layer_params = self.num_hidden_layers * (attn_params + mlp_params + norm_params)
        final_norm_params = self.hidden_size
        lm_head_params = 0 if self.tie_word_embeddings else (self.vocab_size * self.hidden_size)

        return embed_params + layer_params + final_norm_params + lm_head_params

    def estimate_mobile_memory_mb(self, seq_len: int = 256) -> Dict[str, float]:
        """
        Estimate on-device mobile memory footprint (in MB) across quantization levels,
        including static KV cache for `seq_len` tokens in FP16.
        """
        params = self.estimate_parameter_count()
        kv_dim = self.num_key_value_heads * self.head_dim
        # KV cache in FP16: 2 (K and V) * layers * seq_len * kv_dim * 2 bytes
        kv_cache_bytes = 2 * self.num_hidden_layers * seq_len * kv_dim * 2
        kv_mb = kv_cache_bytes / (1024 * 1024)

        return {
            "f32_weights_mb": round((params * 4) / (1024 * 1024), 3),
            "f16_weights_mb": round((params * 2) / (1024 * 1024), 3),
            "q8_0_weights_mb": round((params * 34 / 32) / (1024 * 1024), 3),
            "q4_0_weights_mb": round((params * 18 / 32) / (1024 * 1024), 3),
            "kv_cache_fp16_mb": round(kv_mb, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HexonConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_preset(cls, preset: str, vocab_size: int | None = None) -> "HexonConfig":
        """Load one of the standard Hexon mobile architecture presets."""
        presets: Dict[str, Dict[str, Any]] = {
            "hexon-nano": {
                "model_name": "hexon-nano",
                "vocab_size": vocab_size or 1024,
                "hidden_size": 192,
                "intermediate_size": 512,
                "num_hidden_layers": 6,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "tie_word_embeddings": True,
            },
            "hexon-micro": {
                "model_name": "hexon-micro",
                "vocab_size": vocab_size or 4096,
                "hidden_size": 384,
                "intermediate_size": 1024,
                "num_hidden_layers": 8,
                "num_attention_heads": 6,
                "num_key_value_heads": 2,
                "max_position_embeddings": 1024,
                "tie_word_embeddings": True,
            },
            "hexon-mobile-35m": {
                "model_name": "hexon-mobile-35m",
                "vocab_size": vocab_size or 8192,
                "hidden_size": 512,
                "intermediate_size": 1408,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "max_position_embeddings": 2048,
                "tie_word_embeddings": True,
            },
            "hexon-mobile-125m": {
                "model_name": "hexon-mobile-125m",
                "vocab_size": vocab_size or 16384,
                "hidden_size": 768,
                "intermediate_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "max_position_embeddings": 2048,
                "tie_word_embeddings": True,
            },
            "hexon-135m": {
                "model_name": "hexon-135m",
                "vocab_size": vocab_size or 49152,
                "hidden_size": 576,
                "intermediate_size": 1536,
                "num_hidden_layers": 30,
                "num_attention_heads": 9,
                "num_key_value_heads": 3,
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-5,
                "rope_theta": 100000.0,
                "tie_word_embeddings": True,
                "pad_token_id": 2,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
        }
        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {list(presets.keys())}"
            )
        return cls(**presets[preset])
