"""Hexon Stage 1: Base Model Pre-training on Curated Common Crawl Data.

Takes the cleaned, deduplicated Common Crawl subset produced by `hexon.curation`,
packs documents into continuous token streams up to `max_seq_len` so Rotary
Position Embeddings (RoPE) are trained across all context positions, and
pre-trains the base `HexonForCausalLM` causal language model before SFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import HexonConfig
from .curation import CuratedDocument, CurationConfig, CurationReport, run_curation_pipeline
from .model import HexonForCausalLM
from .tokenizer import HexonTokenizer
from .train import save_hexon_checkpoint


@dataclass
class HexonPretrainConfig:
    """Hyperparameters for Common Crawl base model pre-training."""

    raw_cc_dir: str = "data/raw_cc"
    curated_jsonl_path: str = "data/curated/common_crawl_curated.jsonl"
    max_cc_documents: int = 120
    max_words_per_cc_doc: int = 140
    max_seq_len: int = 224
    pack_stride: int = 160
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 3.5e-3
    min_learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


class HexonPackedPretrainDataset(Dataset):
    """Packs curated Common Crawl & foundational English texts into dense blocks."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: HexonTokenizer,
        max_seq_len: int = 224,
        stride: int = 160,
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Build a continuous packed token stream across documents so every RoPE
        # position [0 .. max_seq_len - 1] receives dense gradient signal.
        token_stream: List[int] = []
        for doc_text in texts:
            clean = doc_text.strip()
            if not clean:
                continue
            doc_ids = tokenizer.encode(clean, add_bos=True, add_eos=True)
            token_stream.extend(doc_ids)

        if len(token_stream) <= max_seq_len + 1:
            padded = token_stream + [tokenizer.pad_token_id] * (max_seq_len + 1 - len(token_stream))
            inp = torch.tensor(padded[:max_seq_len], dtype=torch.long)
            tgt = torch.tensor(padded[1 : max_seq_len + 1], dtype=torch.long)
            tgt[inp == tokenizer.pad_token_id] = -100
            self.samples.append((inp, tgt))
            return

        step = max(32, min(stride, max_seq_len))
        for start in range(0, len(token_stream) - max_seq_len - 1, step):
            chunk = token_stream[start : start + max_seq_len + 1]
            inp = torch.tensor(chunk[:-1], dtype=torch.long)
            tgt = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def select_targeted_cc_subset(
    curated_docs: Sequence[CuratedDocument],
    max_documents: int = 120,
    max_words_per_doc: int = 140,
) -> List[str]:
    """Select the highest-quality, highest-stopword-density Common Crawl passages."""
    ranked = sorted(
        curated_docs,
        key=lambda d: (d.stopword_ratio * 0.6 + d.alpha_ratio * 0.4),
        reverse=True,
    )
    selected: List[str] = []
    for doc in ranked[:max_documents]:
        words = doc.text.split()
        if len(words) > max_words_per_doc:
            snippet = " ".join(words[:max_words_per_doc])
            # Cut cleanly at last sentence boundary if available
            last_dot = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "))
            if last_dot > len(snippet) // 2:
                snippet = snippet[: last_dot + 1]
        else:
            snippet = doc.text
        selected.append(snippet.strip())
    return selected


def pretrain_base_model(
    model_config: HexonConfig,
    pretrain_config: Optional[HexonPretrainConfig] = None,
    tokenizer: Optional[HexonTokenizer] = None,
    extra_foundational_texts: Optional[Sequence[str]] = None,
    curated_docs: Optional[Sequence[CuratedDocument]] = None,
    save_base_checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[HexonForCausalLM, HexonTokenizer, Dict[str, object]]:
    """Run Stage 1 Common Crawl curation and base causal LM pre-training."""
    pt_cfg = pretrain_config or HexonPretrainConfig()
    torch.manual_seed(pt_cfg.seed)

    # 1. Run Common Crawl Curation Pipeline if curated_docs not already provided
    if curated_docs is None:
        curated_docs, curation_report = run_curation_pipeline(
            raw_dir=pt_cfg.raw_cc_dir,
            output_jsonl=pt_cfg.curated_jsonl_path,
        )
    else:
        curation_report = CurationReport(
            raw_documents=len(curated_docs),
            kept_documents=len(curated_docs),
            total_clean_chars=sum(d.char_count for d in curated_docs),
            total_clean_words=sum(d.word_count for d in curated_docs),
        )

    cc_passages = select_targeted_cc_subset(
        curated_docs,
        max_documents=pt_cfg.max_cc_documents,
        max_words_per_doc=pt_cfg.max_words_per_cc_doc,
    )

    all_pretrain_texts: List[str] = list(cc_passages)
    if extra_foundational_texts:
        all_pretrain_texts.extend(extra_foundational_texts)

    # 2. Train BPE tokenizer on combined Common Crawl + foundational corpus if needed
    if tokenizer is None:
        tokenizer = HexonTokenizer.train_from_texts(
            all_pretrain_texts,
            vocab_size=model_config.vocab_size,
        )
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id

    # 3. Build packed pre-training dataset across full context length
    dataset = HexonPackedPretrainDataset(
        texts=all_pretrain_texts,
        tokenizer=tokenizer,
        max_seq_len=pt_cfg.max_seq_len,
        stride=pt_cfg.pack_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=pt_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device(pt_cfg.device)
    model = HexonForCausalLM(model_config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pt_cfg.learning_rate,
        weight_decay=pt_cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps = max(1, pt_cfg.epochs * len(loader))
    global_step = 0
    loss_history: List[float] = []

    if verbose:
        print(
            f"[Hexon Stage 1 Pre-train] Curated CC docs: {len(curated_docs)} "
            f"(selected {len(cc_passages)} passages) | Packed chunks: {len(dataset)} "
            f"(seq_len={pt_cfg.max_seq_len})"
        )

    for epoch in range(pt_cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            progress = global_step / max(1, total_steps - 1)
            lr = pt_cfg.min_learning_rate + 0.5 * (
                pt_cfg.learning_rate - pt_cfg.min_learning_rate
            ) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, use_cache=False)
            logits = out["logits"]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pt_cfg.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_loss)
        if verbose:
            print(
                f"  [CC Pre-train Epoch {epoch + 1:02d}/{pt_cfg.epochs:02d}] "
                f"Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 15.0)):.2f}"
            )

    metrics: Dict[str, object] = {
        "stage": "common_crawl_base_pretrain",
        "curation_report": curation_report.to_dict(),
        "cc_passages_used": len(cc_passages),
        "packed_sequences": len(dataset),
        "max_seq_len": pt_cfg.max_seq_len,
        "pretrain_epochs": pt_cfg.epochs,
        "initial_loss": loss_history[0] if loss_history else 0.0,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "loss_history": loss_history,
    }

    if save_base_checkpoint_path is not None:
        save_hexon_checkpoint(
            model,
            tokenizer,
            save_base_checkpoint_path,
            metadata=metrics,
        )

    return model, tokenizer, metrics
