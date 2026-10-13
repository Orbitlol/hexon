"""
Hexon On-Device Mobile Inference Engine.

Loads either a quantized `.gguf` v3 file or a `.pt` checkpoint in strict
inference-only mode (no gradients, no training hooks) and performs fast
KV-cached generation.
"""

from __future__ import annotations

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
    Inference-only runtime for Hexon models on mobile/edge devices.
    Accepts `.gguf` files (F32, F16, Q8_0, Q4_0) or `.pt` checkpoints.
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

    def attach_rag(self, rag: Any) -> "HexonMobileEngine":
        """Attach a `HexonChromaRAG` instance to this inference engine."""
        self.rag = rag
        return self

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

    @torch.inference_mode()
    def reply(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: Optional[float] = None,
        use_rag: bool = False,
        rag_top_k: int = 3,
    ) -> Dict[str, Any]:
        """Generate an assistant response to a user message with optional ChromaDB RAG."""
        retrieved_chunks: List[Dict[str, Any]] = []
        effective_prompt = user_prompt

        if use_rag:
            if self.rag is None:
                from .rag import HexonChromaRAG
                self.rag = HexonChromaRAG()
            aug_prompt, chunks = self.rag.build_rag_prompt(user_prompt, top_k=rag_top_k)
            retrieved_chunks = [c.to_dict() for c in chunks]
            if self.model.config.vocab_size >= 16000 and chunks:
                effective_prompt = aug_prompt

        if repetition_penalty is None:
            repetition_penalty = 1.0 if self.model.config.vocab_size >= 16000 else 1.05

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": effective_prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(formatted, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        t0 = time.perf_counter()
        output_tensor = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0

        new_ids = output_tensor[0, len(prompt_ids) :].tolist()
        if self.tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(self.tokenizer.eos_token_id)]

        response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        num_generated = max(1, len(new_ids))

        return {
            "prompt": user_prompt,
            "response": response_text,
            "tokens_generated": len(new_ids),
            "latency_ms": round(elapsed * 1000.0, 2),
            "ms_per_token": round((elapsed * 1000.0) / num_generated, 2),
            "tokens_per_sec": round(num_generated / max(elapsed, 1e-6), 2),
            "source_format": self.source_format,
            "mobile_inference_locked": self.model.mobile_inference_locked,
            "rag_enabled": use_rag,
            "retrieved_chunks": retrieved_chunks,
        }

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
                "How does Hexon save memory on mobile?",
                "What export formats does Hexon support?",
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
