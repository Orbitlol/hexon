"""
Hexon Offline Server-Side Training Pipeline.

Trains a Hexon model on a server or workstation (never on the mobile device),
then freezes the model and saves a clean checkpoint ready for GGUF v3 and
PyTorch Mobile export.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .config import HexonConfig
from .dataset import (
    HexonInstructionDataset,
    build_mobile_assistant_corpus,
    load_jsonl_conversations,
)
from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer


def save_hexon_checkpoint(
    model: HexonForCausalLM,
    tokenizer: HexonTokenizer,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a Hexon PyTorch checkpoint containing model weights, config, and tokenizer."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        sd[k] = v.detach().cpu().to(torch.float16)
    payload = {
        "config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "state_dict": sd,
        "mobile_inference_only": True,
        "metadata": metadata or {},
    }
    torch.save(payload, str(out_path))
    return out_path


def load_hexon_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    lock_for_mobile: bool = True,
) -> Tuple[HexonForCausalLM, HexonTokenizer]:
    """Load a Hexon PyTorch checkpoint and optionally lock it for inference-only deployment."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    config = HexonConfig.from_dict(payload["config"])
    tokenizer = HexonTokenizer.from_dict(payload["tokenizer"])
    model = HexonForCausalLM(config)
    sd = {k: v.to(torch.float32) for k, v in payload["state_dict"].items()}
    if config.tie_word_embeddings and "lm_head.weight" not in sd and "embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    model.load_state_dict(sd, strict=True)
    model.to(device)
    if lock_for_mobile:
        model.freeze_for_mobile_inference()
    return model, tokenizer


def train_hexon_offline(
    preset: str = "hexon-nano",
    vocab_size: int = 1024,
    epochs: int = 45,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    max_seq_len: int = 96,
    data_jsonl: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[HexonForCausalLM, HexonTokenizer, Dict[str, Any]]:
    """
    Train a Hexon model offline on the server/workstation, then freeze it for
    mobile inference export.
    """
    torch.set_num_threads(2)
    torch.manual_seed(seed)

    conversations = (
        load_jsonl_conversations(data_jsonl)
        if data_jsonl is not None
        else build_mobile_assistant_corpus()
    )

    # Fit compact domain-adapted BPE tokenizer on corpus texts
    corpus_texts: List[str] = [item["full"] for item in conversations]
    tokenizer = HexonTokenizer.train_from_texts(corpus_texts, vocab_size=vocab_size)

    config = HexonConfig.from_preset(preset, vocab_size=tokenizer.vocab_size)
    model = HexonForCausalLM(config)

    dataset = HexonInstructionDataset(
        conversations=conversations,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        mask_user_prompt=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    total_steps = max(1, epochs * len(loader))
    warmup_steps = max(1, int(0.08 * total_steps))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return learning_rate * (0.05 + 0.95 * cosine)

    start_time = time.perf_counter()
    step = 0
    loss_history: List[float] = []

    model.train(True)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = out["loss"]
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            step += 1

        avg_epoch_loss = epoch_loss / len(loader)
        loss_history.append(avg_epoch_loss)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"[Hexon Offline Train] Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {avg_epoch_loss:.4f} | LR: {get_lr(step - 1):.5f}"
            )

    elapsed = time.perf_counter() - start_time

    # Permanently lock model into inference-only mode for mobile deployment
    model.freeze_for_mobile_inference()

    metrics = {
        "preset": preset,
        "parameters": model.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "epochs": epochs,
        "total_steps": step,
        "initial_loss": round(loss_history[0], 4) if loss_history else None,
        "final_loss": round(loss_history[-1], 4) if loss_history else None,
        "training_seconds": round(elapsed, 2),
        "mobile_inference_locked": model.mobile_inference_locked,
    }

    if checkpoint_path is not None:
        save_hexon_checkpoint(model, tokenizer, checkpoint_path, metadata=metrics)

    return model, tokenizer, metrics
