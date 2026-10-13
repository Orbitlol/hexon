"""
Hexon Offline Training Dataset & Multi-Template Mobile Assistant Corpus.

Trains Hexon across all standard mobile GGUF prompt formats:
1. ChatML (`<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>`)
2. ChatML with System Prompt
3. `<|user|>...<|assistant|>...<|im_end|>` (used by several Android GGUF frontends)
4. Direct / Raw prompt completion (`Question\\nAnswer<|im_end|>`)

This guarantees coherent, clean responses on Android GGUF apps (PocketPal,
ChatterUI, Maid, Layla, llama.cpp) regardless of which chat template the app applies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import HexonTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are Hexon, a fast, private AI language model running locally on a mobile device."
)

QA_PAIRS: List[Tuple[str, str]] = [
    (
        "Who are you?",
        "I am Hexon, a compact AI language model engineered to run locally on mobile devices with low memory and fast inference.",
    ),
    (
        "What is Hexon?",
        "Hexon is a lightweight transformer language model built in PyTorch and exported to GGUF and TorchScript for offline mobile inference.",
    ),
    (
        "Do you train on my phone?",
        "No. Hexon is trained offline on servers and exported as a frozen GGUF or TorchScript file so it never trains on your mobile device.",
    ),
    (
        "How does Hexon save memory on mobile?",
        "Hexon uses Grouped-Query Attention (GQA), tied token embeddings, RMSNorm, SwiGLU, and 4-bit or 8-bit GGUF quantization to keep RAM usage tiny.",
    ),
    (
        "Can you work offline in airplane mode?",
        "Yes! Because Hexon runs 100% on-device using quantized GGUF or PyTorch Mobile weights, no internet connection is required.",
    ),
    (
        "What is GGUF quantization?",
        "GGUF is a binary model format that stores weights in compact blocks like Q8_0 (8-bit) and Q4_0 (4-bit) for fast CPU and NPU inference on mobile devices.",
    ),
    (
        "Summarize this note: Meeting moved to 3 PM tomorrow with the design team to review the mobile UI.",
        "Summary: Design team mobile UI review meeting is rescheduled to 3 PM tomorrow.",
    ),
    (
        "Set a quick reminder checklist for traveling.",
        "Travel checklist: 1. Phone and charger 2. ID or passport 3. Boarding pass 4. Medications 5. Offline maps downloaded.",
    ),
    (
        "What is 15 percent tip on an 80 dollar dinner?",
        "A 15 percent tip on $80 is $12, making the total bill $92.",
    ),
    (
        "Explain Grouped-Query Attention simply.",
        "Grouped-Query Attention shares key and value heads across multiple query heads, shrinking the KV cache so mobile devices use much less RAM.",
    ),
    (
        "How do I conserve battery while running local AI?",
        "Use quantized Q4_0 or Q8_0 GGUF models, keep context length concise, and run inference in bursts rather than continuous background loops.",
    ),
    (
        "Write a short polite reply saying I am running 5 minutes late.",
        "Hi! I am on my way and running about 5 minutes behind schedule. Thank you for your patience!",
    ),
    (
        "Hello!",
        "Hello! I am Hexon, your on-device mobile AI assistant. How can I help you today?",
    ),
    (
        "Hi",
        "Hello! I am Hexon, your on-device mobile AI assistant. How can I help you today?",
    ),
    (
        "What export formats does Hexon support?",
        "Hexon exports to GGUF v3 (F32, F16, Q8_0, Q4_0) as well as PyTorch Mobile TorchScript and dynamic INT8 checkpoints.",
    ),
]


def build_mobile_assistant_corpus() -> List[Dict[str, str]]:
    """
    Build formatted (prompt_prefix, full_sequence) pairs across all standard
    mobile GGUF chat templates so the model responds accurately in any Android client.
    """
    samples: List[Dict[str, str]] = []
    for user_q, assistant_a in QA_PAIRS:
        # 1. Standard ChatML (no system prompt)
        p1 = f"<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
        samples.append({"prompt": p1, "full": f"{p1}{assistant_a}<|im_end|>"})

        # 2. Standard ChatML (with system prompt)
        p2 = (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
        )
        samples.append({"prompt": p2, "full": f"{p2}{assistant_a}<|im_end|>"})

        # 3. <|user|> / <|assistant|> tag template (compact & newline variants)
        p3 = f"<|user|>{user_q}<|assistant|>"
        samples.append({"prompt": p3, "full": f"{p3}{assistant_a}<|im_end|>"})

        p4 = f"<|user|>\n{user_q}\n<|assistant|>\n"
        samples.append({"prompt": p4, "full": f"{p4}{assistant_a}<|im_end|>"})

        # 4. Direct / Raw prompt completion (for raw prompt mode in mobile apps)
        p5 = f"{user_q}\n"
        samples.append({"prompt": p5, "full": f"{p5}{assistant_a}<|im_end|>"})

    # 5. Multi-turn ChatML conversations so multi-turn mobile chat stays sharp
    n = len(QA_PAIRS)
    for i in range(n):
        q1, a1 = QA_PAIRS[i]
        q2, a2 = QA_PAIRS[(i + 1) % n]
        q3, a3 = QA_PAIRS[(i + 2) % n]

        # 2-turn conversation
        prefix_2 = (
            f"<|im_start|>user\n{q1}<|im_end|>\n"
            f"<|im_start|>assistant\n{a1}<|im_end|>\n"
            f"<|im_start|>user\n{q2}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        samples.append({"prompt": prefix_2, "full": f"{prefix_2}{a2}<|im_end|>"})

    return samples


def load_jsonl_conversations(path: str | Path) -> List[Dict[str, str]]:
    """Load custom offline training conversations from a `.jsonl` file."""
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" in record and "response" in record:
                q = str(record["prompt"]).strip()
                a = str(record["response"]).strip()
                p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                items.append({"prompt": p, "full": f"{p}{a}<|im_end|>"})
            elif "messages" in record:
                msgs = record["messages"]
                user_q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                p = f"<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
                items.append({"prompt": p, "full": f"{p}{asst_a}<|im_end|>"})
    return items


class HexonInstructionDataset(Dataset):
    """
    Tokenized causal language modeling dataset for offline Hexon training.
    Includes each sample both with and without BOS so the model works regardless
    of whether the mobile GGUF client prepends `<|bos|>`.
    """

    def __init__(
        self,
        conversations: Sequence[Dict[str, str]],
        tokenizer: HexonTokenizer,
        max_length: int = 128,
        mask_user_prompt: bool = True,
    ) -> None:
        self.samples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.pad_id = tokenizer.pad_token_id

        for item in conversations:
            prompt_text = item["prompt"]
            full_text = item["full"]

            for add_bos in (True, False):
                full_ids = tokenizer.encode(full_text, add_bos=add_bos, add_eos=False)
                full_ids = full_ids[:max_length]

                seq_t = torch.tensor(full_ids, dtype=torch.long)
                label_t = seq_t.clone()

                if mask_user_prompt:
                    prompt_ids = tokenizer.encode(prompt_text, add_bos=add_bos, add_eos=False)
                    mask_len = min(len(prompt_ids), max_length)
                    if mask_len > 1:
                        label_t[: mask_len - 1] = self.pad_id

                self.samples.append(seq_t)
                self.labels.append(label_t)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.samples[idx],
            "labels": self.labels[idx],
        }

    def collate_fn(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        bsz = len(batch)
        inputs = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["input_ids"].size(0)
            inputs[i, :n] = item["input_ids"]
            labels[i, :n] = item["labels"]
        return {"input_ids": inputs, "labels": labels}
