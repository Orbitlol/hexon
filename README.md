# Hexon — Compact Mobile-First AI Language Model (PyTorch + GGUF v3)

**Hexon** is a lightweight, decoder-only Transformer language model engineered in **PyTorch** specifically for **on-device mobile inference** (Android, iOS, and edge devices).

By design, **Hexon never trains on the mobile device**:
1. **Common Crawl Data Curation Pipeline (`hexon/curation.py`):** Pulls a targeted subset of Common Crawl WARC/WET/JSONL dumps (`data/raw_cc/`) and cleans them through a 7-stage curation pipeline (URL/domain filtering, HTML & boilerplate removal, Unicode NFKC normalization, FineWeb/Gopher quality heuristics, English language detection, PII redaction, and SHA-1 paragraph/document deduplication) into `data/curated/common_crawl_curated.jsonl`.
2. **Two-Stage Offline Training (Common Crawl Base Pre-training + Supervised Fine-Tuning):** First, `hexon/pretrain.py` pre-trains the base language model on packed sequences of curated Common Crawl web text (`exports/hexon-nano-base-pretrained.pt`). Second, `hexon/sft.py` runs Supervised Fine-Tuning (SFT) with prompt loss masking (`ignore_index=-100`) across single-turn and multi-turn mobile conversations (`exports/hexon-nano-checkpoint.pt`).
3. **Inference-Only Mobile Lock (`freeze_for_mobile_inference`):** Before export, all parameters are permanently frozen (`requires_grad=False`), training hooks and loss computation are disabled, and `.train(True)` is blocked.
4. **100% Native `llama.cpp` GGUF v3 & PyTorch Mobile Export:** Models are exported to standalone **GGUF v3** files (`F16`, `Q8_0`, `Q4_0`) with interleaved RoPE (`GGML_ROPE_TYPE_NORMAL`), SentencePiece `▁` (`\u2581`) BPE merge scores (`llm_tokenizer_spm`), `USER_DEFINED` special role markers, and ChatML (`<|im_start|>`, `<|im_end|>`) templates, plus **PyTorch Mobile** artifacts (TorchScript `.pt` and dynamic `INT8` `.pt`).

---

## Architecture Highlights

- **Grouped-Query Attention (GQA):** Shares key/value heads across query heads (3:1 or 4:1 ratio) to reduce the mobile KV-cache memory footprint by up to 75%.
- **Shared Token Embeddings (`tie_word_embeddings=True`):** Ties `embed_tokens` and `lm_head` weights (MobileLLM-style), saving 20–35% of total parameters on compact models.
- **Pre-Norm RMSNorm & SwiGLU FFN:** Fast, bias-free normalization and gated linear units suited for mobile ARM NEON / Metal / Vulkan kernels.
- **Rotary Position Embeddings (RoPE):** Zero-parameter positional encoding with KV-cache support.
- **Compact Byte-Fallback Tokenizer:** Subword vocabulary with 256 `<0xXX>` byte tokens embedded directly in GGUF metadata (`tokenizer.ggml.*`), guaranteeing 0% out-of-vocabulary failures.

### Architecture Presets

| Preset | Layers | Hidden Dim | FFN Dim | Heads (Q / KV) | Context | Params | Q8_0 GGUF | Q4_0 GGUF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hexon-nano` | 6 | 192 | 512 | 6 / 2 | 512 | 2.56M | ~2.8 MB | ~1.5 MB |
| `hexon-micro` | 8 | 384 | 1024 | 6 / 2 | 1024 | 14.9M | ~15.8 MB | ~8.5 MB |
| `hexon-mobile-35m` | 12 | 512 | 1408 | 8 / 2 | 2048 | 41.9M | ~44.5 MB | ~23.8 MB |
| `hexon-mobile-125m` | 16 | 768 | 2048 | 12 / 4 | 2048 | 121.6M | ~129 MB | ~69 MB |

---

## Pretrained & Exported Artifacts (`exports/`)

A trained working `hexon-nano` checkpoint and its mobile export files are included out of the box in `exports/`:

| File | Format | Quantization | Size | Purpose |
| :--- | :--- | :--- | :---: | :--- |
| `exports/hexon-135m-q4_0.gguf` | **GGUF v3** | 4-bit/8-bit (`Q4_1`/`Q8_0`) | **93.81 MiB** | **Flagship 135M (134.5M params) compositional mobile AI + ChromaDB RAG** |
| `exports/hexon-nano-q4_0.gguf` | **GGUF v3** | 4-bit block (`Q4_0`) | **1.63 MB** | Ultra-compact mobile GGUF inference (SFT) |
| `exports/hexon-nano-q8_0.gguf` | **GGUF v3** | 8-bit block (`Q8_0`) | **3.04 MB** | High-accuracy mobile GGUF inference (SFT) |
| `exports/hexon-nano-f16.gguf` | **GGUF v3** | 16-bit float (`F16`) | **5.67 MB** | Half-precision GGUF baseline (SFT) |
| `exports/hexon-nano-base-f16.gguf` | **GGUF v3** | 16-bit float (`F16`) | **5.67 MB** | Pre-trained Common Crawl base model (pre-SFT) |
| `exports/hexon-nano-base-pretrained.pt` | **PyTorch** | `FP16` | **5.37 MB** | Pre-trained Common Crawl base checkpoint |
| `exports/hexon-nano-mobile-int8.pt` | **PyTorch Mobile** | Dynamic `qint8` | **3.82 MB** | Quantized PyTorch inference bundle |
| `exports/hexon-nano-torchscript.pt` | **TorchScript** | Frozen / Traced | **11.36 MB** | C++ / Android JNI / iOS LibTorch runtime |
| `exports/hexon-nano-checkpoint.pt` | **PyTorch** | `FP16` | **5.37 MB** | Offline SFT server checkpoint |

---

## Quickstart

### 1. Run Mobile Inference Directly from a `.gguf` File

```bash
python3 -m hexon.cli generate \
  --model exports/hexon-nano-q4_0.gguf \
  --prompt "Who are you?"
```

Output:
```json
{
  "prompt": "Who are you?",
  "response": "I am Hexon, a compact AI language model engineered to run locally on mobile devices with low memory and fast inference.",
  "tokens_generated": 23,
  "latency_ms": 88.4,
  "ms_per_token": 3.84,
  "tokens_per_sec": 260.18,
  "source_format": "gguf-q4_0",
  "mobile_inference_locked": true
}
```

### 2. Inspect a GGUF v3 Export File

```bash
python3 -m hexon.cli inspect exports/hexon-nano-q4_0.gguf
```

### 3. Run the 3-Stage Offline Pipeline (Common Crawl Curation → Base Pre-training → SFT + Mobile Export)

```bash
# Stage 0: Curate Common Crawl WARC/WET/JSONL snapshots into clean English JSONL
python3 -m hexon.cli curate --raw-dir data/raw_cc --output data/curated/common_crawl_curated.jsonl

# Stage 1: Pre-train the base language model on packed Common Crawl web text
python3 -m hexon.cli pretrain --preset hexon-nano --curated-jsonl data/curated/common_crawl_curated.jsonl

# Stage 2: Run 60-intent + 8-turn Supervised Fine-Tuning (SFT) and export GGUF & PyTorch Mobile files
python3 -m hexon.cli sft --preset hexon-nano --output-dir exports --epochs 14
```

### 4. Python API

```python
from hexon import HexonMobileEngine

# Load directly from a single quantized .gguf file (no external config or tokenizer needed)
engine = HexonMobileEngine.from_file("exports/hexon-nano-q8_0.gguf")
result = engine.reply("Do you train on my phone?")
print(result["response"])
# -> "Never! Training only happens offline on servers, while your phone runs a frozen, read-only GGUF or TorchScript model strictly for fast inference."
```

---

## Repository Structure

```text
hexon/
├── __init__.py        # Public package exports
├── config.py          # HexonConfig & mobile memory estimator + presets
├── curation.py        # 7-stage Common Crawl WARC/WET/JSONL data curation pipeline
├── pretrain.py        # Stage 1 Common Crawl base model pre-training engine
├── sft.py             # Stage 2 60-intent + carrier-frame + 8-turn SFT pipeline
├── tokenizer.py       # SentencePiece '▁' BPE + 256-byte fallback + ChatML tokenizer
├── model.py           # PyTorch GQA + interleaved RoPE + RMSNorm + SwiGLU causal LM
├── export_gguf.py     # Spec-compliant GGUF v3 writer & F32/F16/Q8_0/Q4_0 quantizer
├── gguf_runtime.py    # Standalone GGUF v3 binary parser, dequantizer & loader
├── export_mobile.py   # TorchScript & PyTorch Mobile dynamic INT8 exporter
├── inference.py       # HexonMobileEngine unified inference-only runtime (.gguf & .pt)
├── dataset.py         # Offline instruction tuning dataset & mobile corpus
├── train.py           # Offline server-side training & checkpoint helpers
├── server.py          # Live HTTP mobile playground & GGUF/PyTorch artifact server
└── cli.py             # CLI (`curate`, `pretrain`, `sft`, `train`, `export`, `inspect`, `generate`, `build-all`)
data/
├── raw_cc/            # Raw Common Crawl WARC/WET/JSONL snapshots
└── curated/           # Cleaned & deduplicated Common Crawl corpus (`common_crawl_curated.jsonl`)
exports/               # Pre-trained base + SFT checkpoints & GGUF/TorchScript export files
tests/
└── test_hexon.py      # 11 unit & integration tests (curation, pre-training, SFT, GGUF v3, multi-turn chat)
```
