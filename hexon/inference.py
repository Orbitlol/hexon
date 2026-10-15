"""
Hexon On-Device Mobile Inference Engine.

Loads either a quantized `.gguf` v3 file or a `.pt` checkpoint in strict
inference-only mode (no gradients, no training hooks) and performs fast
KV-cached generation with heavy ChromaDB RAG grounding and cross-turn
anti-copying n-gram protection.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .gguf_runtime import GGUFReader
from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer
from .train import load_hexon_checkpoint


class HexonMobileEngine:
    """
    Inference-only runtime for Hexon & Hexon-Code models on mobile/edge devices.
    Accepts `.gguf` files (F32, F16, Q8_0, Q4_0, Q4_1, Q4_CODE) or `.pt` checkpoints.
    """

    def __init__(
        self,
        model: HexonForCausalLM,
        tokenizer: HexonTokenizer,
        source_path: str = "",
        source_format: str = "pytorch",
        rag: Optional[Any] = None,
    ) -> None:
        self.model = model.freeze_for_mobile_inference()
        self.tokenizer = tokenizer
        self.source_path = source_path
        self.source_format = source_format
        self.rag = rag
        self.history: List[Dict[str, str]] = []

    def attach_rag(self, rag: Any) -> "HexonMobileEngine":
        """Attach a `HexonChromaRAG` instance to this inference engine."""
        self.rag = rag
        return self

    def reset_history(self) -> None:
        """Clear multi-turn conversation history."""
        self.history.clear()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: str = "cpu",
        rag: Optional[Any] = None,
    ) -> "HexonMobileEngine":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        if p.suffix.lower() == ".gguf":
            reader = GGUFReader(p)
            model, tokenizer = reader.load_model_and_tokenizer(device=device)
            qtype = str(reader.metadata.get("hexon.quantization", "gguf"))
            return cls(
                model=model,
                tokenizer=tokenizer,
                source_path=str(p),
                source_format=f"gguf-{qtype}",
                rag=rag,
            )

        model, tokenizer = load_hexon_checkpoint(p, device=device, lock_for_mobile=True)
        return cls(
            model=model,
            tokenizer=tokenizer,
            source_path=str(p),
            source_format="pytorch-pt",
            rag=rag,
        )

    def _extract_rag_grounding_ids(self, chunks: Sequence[Dict[str, Any]]) -> List[int]:
        """
        Extract substantive content and code token IDs from retrieved ChromaDB chunks
        for logit grounding during autoregressive decoding.
        """
        if not chunks or self.model.config.vocab_size < 16000:
            return []
        grounded_ids: set[int] = set()
        for ch in chunks[:2]:
            txt = str(ch.get("text", ""))
            # Extract technical terms, numbers, identifiers, and 4+ char words
            words = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b|\b\d+\b", txt))
            for w in words:
                for tid in self.tokenizer.encode(" " + w, add_bos=False, add_eos=False):
                    if tid > 256:
                        grounded_ids.add(int(tid))
        return list(grounded_ids)

    @torch.inference_mode()
    def reply(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 110,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: Optional[float] = None,
        use_rag: Optional[bool] = None,
        rag_top_k: int = 3,
        history: Optional[Sequence[Dict[str, str]]] = None,
        remember_turn: bool = False,
        no_repeat_ngram_size: Optional[int] = None,
        rag_boost: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate an assistant response to a user message with ChromaDB RAG grounding
        and cross-turn anti-copying protection.
        """
        is_large_model = self.model.config.vocab_size >= 16000
        if use_rag is None:
            # Every shipped Hexon at this scale was trained to answer from
            # retrieved ChromaDB context, so grounding is the default for all
            # of them. Only the small vocab models (hexon-nano and friends)
            # run bare, because their tokenizer cannot use the grounding ids.
            use_rag = is_large_model

        retrieved_chunks: List[Dict[str, Any]] = []
        effective_prompt = user_prompt

        if use_rag:
            if self.rag is None:
                from .rag import HexonChromaRAG
                self.rag = HexonChromaRAG()
            aug_prompt, chunks = self.rag.build_rag_prompt(user_prompt, top_k=rag_top_k)
            retrieved_chunks = [c.to_dict() for c in chunks]
            if is_large_model and chunks:
                effective_prompt = aug_prompt

        if repetition_penalty is None:
            repetition_penalty = 1.0 if is_large_model else 1.05
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = 5 if is_large_model else 0

        active_history = list(history) if history is not None else (list(self.history) if remember_turn else [])

        # Collect previous assistant turn token sequences so `generate()` blocks cross-turn copying
        ban_copy_sequences: List[List[int]] = []
        for msg in active_history:
            if msg.get("role") == "assistant":
                prev_text = msg.get("content", "").strip()
                if prev_text:
                    ban_copy_sequences.append(
                        self.tokenizer.encode(prev_text, add_bos=False, add_eos=False)
                    )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in active_history[-4:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": effective_prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(formatted, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        rag_token_ids = self._extract_rag_grounding_ids(retrieved_chunks) if (use_rag and is_large_model) else []

        t0 = time.perf_counter()
        output_tensor = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ban_copy_sequences=ban_copy_sequences,
            rag_token_ids=rag_token_ids,
            rag_boost=rag_boost if rag_token_ids else 0.0,
            generated_only_repetition_penalty=is_large_model,
        )
        elapsed = time.perf_counter() - t0

        new_ids = output_tensor[0, len(prompt_ids) :].tolist()
        if self.tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(self.tokenizer.eos_token_id)]

        response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        # If the model completed a fenced code block (```...```) and then got cut off
        # starting a second unclosed code block or trailing partial line, trim cleanly
        if is_large_model and response_text.count("```") >= 2:
            first_open = response_text.find("```")
            first_close = response_text.find("```", first_open + 3)
            if first_close != -1:
                after_code = response_text[first_close + 3 :]
                if "```" in after_code:
                    # Cut before a second unclosed code block
                    after_code = after_code[: after_code.find("```")].strip()
                elif "\ndef " in after_code:
                    after_code = after_code[: after_code.find("\ndef ")].strip()
                # If trailing sentence after code was truncated mid-sentence by max_new_tokens,
                # trim to the last complete sentence boundary
                if after_code and not after_code.endswith((".", "!", "?", "`")):
                    last_dot = max(after_code.rfind("."), after_code.rfind("!"))
                    if last_dot != -1:
                        after_code = after_code[: last_dot + 1].strip()
                    else:
                        after_code = ""
                response_text = (
                    response_text[: first_close + 3]
                    + ("\n\n" + after_code if after_code else "")
                ).strip()
        elif is_large_model and not response_text.startswith("```") and len(new_ids) >= max_new_tokens - 2:
            # Trim truncated trailing sentence on prose answers if cut off by max_new_tokens
            if not response_text.endswith((".", "!", "?", "`", ")")):
                last_dot = max(response_text.rfind("."), response_text.rfind("!"))
                if last_dot > len(response_text) // 3:
                    response_text = response_text[: last_dot + 1].strip()

        num_generated = max(1, len(new_ids))

        if remember_turn:
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": response_text})

        return {
            "prompt": user_prompt,
            "response": response_text,
            "tokens_generated": len(new_ids),
            "latency_ms": round(elapsed * 1000.0, 2),
            "ms_per_token": round((elapsed * 1000.0) / num_generated, 2),
            "tokens_per_sec": round(num_generated / max(elapsed, 1e-6), 2),
            "source_format": self.source_format,
            "mobile_inference_locked": self.model.mobile_inference_locked,
            "rag_enabled": bool(use_rag),
            "retrieved_chunks": retrieved_chunks,
        }

    @torch.inference_mode()
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 110,
        temperature: float = 0.0,
        use_rag: bool = True,
        rag_top_k: int = 2,
    ) -> Dict[str, Any]:
        """
        Stateful multi-turn chat interface with automatic ChromaDB RAG grounding
        and cross-turn anti-copying n-gram protection.
        """
        return self.reply(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_rag=use_rag,
            rag_top_k=rag_top_k,
            remember_turn=True,
        )

    @torch.inference_mode()
    def benchmark(
        self,
        prompts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 80,
    ) -> Dict[str, Any]:
        """Run a multi-prompt mobile inference benchmark."""
        if prompts is None:
            prompts = [
                "Who are you?",
                "Do you train on my phone?",
                "Write a Python function to check if a number is prime.",
                "What processor and battery does the Poco M3 have?",
            ]
        results = [
            self.reply(p, max_new_tokens=max_new_tokens, temperature=0.0)
            for p in prompts
        ]
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_ms = sum(r["latency_ms"] for r in results)
        return {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "parameters": self.model.num_parameters(),
            "mobile_inference_locked": self.model.mobile_inference_locked,
            "num_prompts": len(results),
            "total_tokens_generated": total_tokens,
            "avg_ms_per_token": round(total_ms / max(1, total_tokens), 2),
            "avg_tokens_per_sec": round((total_tokens * 1000.0) / max(1e-3, total_ms), 2),
            "samples": results,
        }
