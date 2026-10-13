"""
Hexon PyTorch Model Architecture.

Mobile-first Decoder-only Transformer featuring:
- Grouped-Query Attention (GQA) for reduced mobile KV-cache memory
- Pre-normalization with RMSNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU Feed-Forward Networks
- Shared Input/Output Token Embeddings (MobileLLM-style embedding tying)
- Static & Dynamic KV-Cache for fast autoregressive mobile inference

Note: Training happens exclusively offline on servers/workstations. When exported
for mobile deployment (`freeze_for_mobile_inference`), gradients and training hooks
are permanently disabled.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HexonConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (bias-free, fast on mobile NPUs/CPUs)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normed.to(input_dtype))


class RotaryEmbedding(nn.Module):
    """
    Precomputed Interleaved Rotary Position Embeddings (RoPE).
    Matches `llama.cpp` GGML_ROPE_TYPE_NORMAL (consecutive pairs 2i, 2i+1).
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim // 2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch_size, seq_len]
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)  # [B, S, head_dim // 2]
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)  # [B, S, head_dim // 2]
        return cos.unsqueeze(1), sin.unsqueeze(1)  # [B, 1, S, head_dim // 2]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply interleaved Rotary Position Embeddings to query and key tensors,
    matching `llama.cpp` C++ `GGML_ROPE_TYPE_NORMAL` bit-for-bit.
    """
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    q_embed = torch.stack((q_rot_even, q_rot_odd), dim=-1).flatten(-2)

    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos
    k_embed = torch.stack((k_rot_even, k_rot_odd), dim=-1).flatten(-2)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match query heads for Grouped-Query Attention.
    Input shape:  [batch, num_key_value_heads, seq_len, head_dim]
    Output shape: [batch, num_attention_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class HexonAttention(nn.Module):
    """Grouped-Query Attention (GQA) optimized for mobile KV-cache efficiency."""

    def __init__(self, config: HexonConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), present_kv


class HexonMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: HexonConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HexonDecoderLayer(nn.Module):
    """Single Transformer decoder block in Hexon."""

    def __init__(self, config: HexonConfig) -> None:
        super().__init__()
        self.self_attn = HexonAttention(config)
        self.mlp = HexonMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(
            hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_kv


class HexonForCausalLM(nn.Module):
    """
    Hexon Causal Language Model.

    Supports offline server-side training and locked inference-only execution
    for mobile devices.
    """

    def __init__(self, config: HexonConfig) -> None:
        super().__init__()
        self.config = config
        self.mobile_inference_locked = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HexonDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def freeze_for_mobile_inference(self) -> "HexonForCausalLM":
        """
        Permanently lock the model into mobile inference-only mode:
        - Disables gradients on all parameters
        - Switches all layers to eval mode
        - Blocks accidental `.train(True)` calls on mobile devices
        """
        super().train(False)
        for param in self.parameters():
            param.requires_grad_(False)
        self.mobile_inference_locked = True
        return self

    def train(self, mode: bool = True) -> "HexonForCausalLM":
        if mode and self.mobile_inference_locked:
            raise RuntimeError(
                "Hexon model is locked for mobile inference-only mode. "
                "Training on mobile devices is disabled by design."
            )
        return super().train(mode)

    @staticmethod
    def _build_causal_mask(
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if q_len == 1:
            return None
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        cond = torch.arange(kv_len, device=device) <= (
            torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)
        )
        mask = mask.masked_fill(cond, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if labels is not None and self.mobile_inference_locked:
            raise RuntimeError(
                "Loss computation / training is disabled on mobile-locked Hexon instances."
            )

        bsz, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._build_causal_mask(
            q_len=seq_len,
            kv_len=past_len + seq_len,
            device=input_ids.device,
            dtype=hidden_states.dtype,
        )

        next_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present_kv is not None:
                next_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return {
            "logits": logits,
            "past_key_values": next_cache if use_cache else None,
            "loss": loss,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        no_repeat_ngram_size: int = 0,
        ban_copy_sequences: Optional[List[List[int]]] = None,
        rag_token_ids: Optional[List[int]] = None,
        rag_boost: float = 0.0,
        generated_only_repetition_penalty: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV-cache, ChromaDB grounding logit
        boost, generated-only repetition penalty, and cross-turn n-gram blocking
        to prevent hallucination and multi-turn response copying.
        """
        eos_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        prompt_len = input_ids.shape[1]
        generated = input_ids.clone()
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        cur_input = generated

        # Precompute banned n-gram lookup table from previous assistant turns
        prior_banned_ngrams: Dict[Tuple[int, ...], set[int]] = {}
        prior_turn_tokens: List[int] = []
        if ban_copy_sequences:
            for seq in ban_copy_sequences:
                prior_turn_tokens.extend(seq)
                if no_repeat_ngram_size >= 2 and len(seq) >= no_repeat_ngram_size:
                    n = no_repeat_ngram_size
                    for idx in range(len(seq) - n + 1):
                        prefix = tuple(seq[idx : idx + n - 1])
                        prior_banned_ngrams.setdefault(prefix, set()).add(int(seq[idx + n - 1]))

        rag_set = [int(t) for t in rag_token_ids if 0 <= int(t) < self.config.vocab_size] if rag_token_ids else []

        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_position_embeddings:
                break

            out = self.forward(
                cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out["logits"][:, -1, :].clone()  # [B, V]
            past_key_values = out["past_key_values"]

            # 1. ChromaDB Grounding Logit Boost on unvisited RAG tokens
            if rag_set and rag_boost > 0.0:
                for b in range(generated.shape[0]):
                    gen_so_far = set(generated[b, prompt_len:].tolist())
                    active_rag = [t for t in rag_set if t not in gen_so_far]
                    if active_rag:
                        idx_t = torch.tensor(active_rag, dtype=torch.long, device=logits.device)
                        logits[b, idx_t] = logits[b, idx_t] + rag_boost

            # 2. Repetition penalty (either generated+prior-assistant only, or full sequence)
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    if generated_only_repetition_penalty:
                        gen_slice = generated[b, prompt_len:].tolist()
                        combined_seen = gen_slice + prior_turn_tokens
                        if not combined_seen:
                            continue
                        seen_tokens = torch.tensor(
                            list(set(combined_seen)), dtype=torch.long, device=logits.device
                        )
                    else:
                        seen_tokens = torch.unique(generated[b])
                    score = logits[b, seen_tokens]
                    logits[b, seen_tokens] = torch.where(
                        score < 0, score * repetition_penalty, score / repetition_penalty
                    )

            # 3. N-Gram Blocking across current response (8-gram) AND previous assistant turns (5-gram)
            if no_repeat_ngram_size >= 2:
                n = no_repeat_ngram_size
                for b in range(generated.shape[0]):
                    gen_list = generated[b, prompt_len:].tolist()
                    if len(gen_list) >= n - 1:
                        cur_prefix = tuple(gen_list[-(n - 1) :])
                        # Only block from prior assistant turns on 5-gram match,
                        # and only block within current response on >=8-gram match (skipping whitespace token 220/198)
                        banned_next: set[int] = set(prior_banned_ngrams.get(cur_prefix, set()))
                        intra_n = max(8, n)
                        if len(gen_list) >= intra_n and not all(t in (198, 220) for t in cur_prefix):
                            intra_prefix = tuple(gen_list[-(intra_n - 1) :])
                            for idx in range(len(gen_list) - intra_n + 1):
                                if tuple(gen_list[idx : idx + intra_n - 1]) == intra_prefix:
                                    banned_next.add(int(gen_list[idx + intra_n - 1]))
                        for bt in banned_next:
                            if 0 <= bt < logits.size(-1):
                                logits[b, bt] = float("-inf")

            if temperature <= 1e-5:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k_val = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k_val)
                    logits[logits < v[:, [-1]]] = float("-inf")
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            cur_input = next_token

            if (next_token == eos_id).all():
                break

        return generated

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
