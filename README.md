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
| `hexon-135m` / `hexon-code` | 30 | 576 | 1536 | 9 / 3 | 2048 | 134.5M | ~143 MB | ~103 MB |
| **`hexon-1.5`** | **32** | 576 | 1536 | 9 / 3 | 2048 | **141.6M** | ~150 MB | **98.22 MiB** (`q4_deep`) |

`hexon-1.5` adds depth only — same width, same 49152 vocabulary, two extra
decoder blocks on top of `hexon-1.2`'s 30. See [How hexon-1.5 was
built](#how-hexon-15-was-built).

**32 layers is not a design choice, it is where the arithmetic stops.** GitHub
rejects files over 100 MiB, and that limit interacts badly with quantisation:
dropping the transformer trunk from `Q4_1` to `Q4_0` to save space costs 0.18
held-out accuracy, which is far more than the extra depth buys. With embeddings
held at `Q8_0` (see below), 32 layers is the deepest model whose trunk still
fits at `Q4_1`. The ablation is in
[`eval/RESULTS.md`](eval/RESULTS.md#the-100-mib-shipping-limit-drove-the-design).

---

## Pretrained & Exported Artifacts (`exports/`)

A trained working `hexon-nano` checkpoint and its mobile export files are included out of the box in `exports/`:

| File | Format | Quantization | Size | Purpose |
| :--- | :--- | :--- | :---: | :--- |
| `exports/hexon-1.5.gguf` | **GGUF v3** | `Q4_DEEP` (`Q8_0` embeddings + `Q4_1` trunk) | **98.22 MiB** | **Flagship. 32 layers / 141.6M params, depth-expanded from hexon-1.2.** |
| `exports/hexon-1.2.gguf` | **GGUF v3** | `Q4_CODE` | **98.42 MiB** | Prior flagship, 30 layers / 134.5M params |
| `exports/hexon-code.gguf` | **GGUF v3** | `Q4_CODE` | **98.42 MiB** | Coding-heavy 30-layer variant |
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

**Every model in this table is committed to the repository and downloadable
straight from GitHub.** The largest is `hexon-1.5.gguf` at 98.22 MiB, which
sits just under GitHub's 100 MiB per-file limit — that limit is the reason
hexon-1.5 is 32 layers deep rather than the 48 it was first trained at.

---

## Evaluation

Honest numbers live in **[`eval/RESULTS.md`](eval/RESULTS.md)**. Summary of the
50-question held-out set, with ChromaDB grounding enabled:

| Model | Accuracy | Hallucination flags | Grounded accuracy | Ungrounded accuracy |
| :--- | :---: | :---: | :---: | :---: |
| hexon-135m-q4_0 | **0.720** | 0.000 | 0.735 | **0.688** |
| hexon-code | **0.720** | 0.000 | **0.765** | 0.625 |
| **hexon-1.5 (shipped)** | 0.700 | 0.020 | 0.735 | 0.625 |
| hexon-1.2 | 0.660 | 0.000 | 0.676 | 0.625 |

Four things worth knowing before quoting any of it:

0. **The shipped hexon-1.5 is not the best hexon-1.5 we trained.** The
   48-layer version scored **0.760** held-out — the best number any model in
   this repo has produced — but at `Q4_CODE` it is 136.48 MiB and GitHub will
   not accept it. The 32-layer model that *does* fit scores 0.700. Full
   ablation in [`eval/RESULTS.md`](eval/RESULTS.md#the-100-mib-shipping-limit-drove-the-design).

1. **The old 8-item keyword check was not a benchmark.** 4 of its 8 questions
   appear verbatim in training corpora, and all 8 were part of hexon-1.2's
   fine-tune. hexon-1.2's headline 8/8 score was largely memorisation of the
   test set — on held-out questions it is the *weakest* of the four models.
2. **The hallucination-reduction target has not been met as a model
   improvement.** The drop in hallucination flags came from rebuilding the
   ChromaDB knowledge base, not from hexon-1.5. With 50 items, one flag is
   0.020, so the 0.000-vs-0.020 spread across these models is within noise.
3. **Scoring is keyword-based**, so accuracy is a floor rather than a grade,
   and hallucination flags only catch wrong answers that someone anticipated.
4. **The spread between these four models is small.** With n=50, one question
   is 0.020, so 0.720 and 0.660 are roughly two questions apart. Treat the
   ordering as indicative, not settled.

Reproduce with:

```bash
python3 -m pip install --break-system-packages --target /tmp/pydeps \
  "torch==2.2.2" "numpy<2" "chromadb>=1.0.0"
PYTHONPATH=/tmp/pydeps:. python3 scripts/run_eval.py \
  --models hexon-1.5 hexon-1.2 hexon-code hexon-135m-q4_0 \
  --configs rag-on --out-dir eval/results_v2
```

`run_eval.py` refuses to stay quiet about contamination: it runs a Jaccard
check of every eval question against every training corpus on every run.

---

## How hexon-1.5 was built

1. **Depth expansion, not retraining.** `scripts/train_15.py` loads
   `hexon-1.2.gguf`, keeps its 30 layers and adds more on top. Each new block
   has its `o_proj` and `down_proj` weights set to zero; because Hexon is
   pre-norm with bias-free projections, a zeroed output projection makes the
   block an identity function. Verified: before training, the expanded model's
   logits match hexon-1.2's **exactly** (max absolute difference `0.0`). The
   same property means untrained blocks can later be *deleted* with no effect
   on output — `scripts/prune_identity_layers.py` does exactly that and asserts
   the logits still match before it will write a file.
2. **Grounded SFT.** `scripts/build_15_sft.py` produces 126 records (55 facts,
   33 coding, 20 chat, 10 calibrated refusals, 8 multi-turn). Prompts reproduce
   `HexonChromaRAG.build_rag_prompt` byte for byte and are grounded in the real
   ChromaDB chunks, so training matches inference.
3. **Cheap training on CPU.** The frozen 28-layer trunk is precomputed once and
   cached, gradients flow only through the trainable tail, and cross-entropy is
   computed on supervised positions only (never a full
   `[seq_len, vocab_size]` logit tensor). 8 epochs over 126 examples take about
   17 minutes on 2 cores.

## Knowledge base

`data/chromadb` holds 228 chunks, every one 200–300 words
(`CHUNK_SCHEMA_VERSION = v2_200_300_words`). Rebuild with:

```bash
PYTHONPATH=/tmp/pydeps:. python3 scripts/build_knowledge_base.py
```

Composition after the rebuild:

| Source | Chunks |
| :--- | :---: |
| `mined_pypi` — filtered PyPI package documentation | 78 |
| `stanford_alpaca` | 58 |
| `mined_npm` — filtered npm READMEs | 42 |
| `hexon_core_authored` — this project's own reference corpus | 29 |
| `coding_reference`, `hexon_docs`, `encyclopedia`, others | 21 |

The rebuild deleted **146 raw Common Crawl chunks** that made up 67% of the old
index. Sampled, they contained invoice-factoring teleconference spam and 9/11
conspiracy blog posts; with them in place, enabling RAG made every model
*worse* than running with no retrieval at all. Removing them and adding curated
content lifted held-out accuracy by 0.14–0.20 per model.

The retrieval embedder is a hashed bag-of-words model, so its width sets its
ceiling. `DEFAULT_EMBEDDING_DIM` in `hexon/rag.py` was raised 384 → 2048, which
took measured recall@1 from 0.318 to 0.409 on 88 labelled question/chunk pairs.

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
├── curated/           # Cleaned & deduplicated Common Crawl corpus (`common_crawl_curated.jsonl`)
├── knowledge/         # hexon-1.5 corpus: authored reference + mined PyPI/npm docs
│   ├── hexon_core_science.jsonl
│   ├── hexon_core_tech.jsonl
│   └── hexon_mined_docs.jsonl
├── sft/
│   └── hexon_15_sft.jsonl   # 126-record grounded SFT set for hexon-1.5
└── chromadb/          # 228 x 200-300 word chunks (hexon_knowledge collection)
exports/               # Pre-trained base + SFT checkpoints & GGUF/TorchScript export files
scripts/
├── run_eval.py            # Score exported models against the held-out benchmark
├── rescore.py             # Re-apply the scorer to saved responses (no regeneration)
├── build_knowledge_base.py    # Rebuild ChromaDB from curated sources
├── build_core_corpus.py       # Author science/geography/history/maths reference
├── build_tech_corpus.py       # Author computing/Python/JavaScript reference
├── mine_knowledge.py          # Mine + filter PyPI and npm documentation
├── build_15_sft.py            # Build the grounded hexon-1.5 SFT set
├── train_15.py                # Depth-expand hexon-1.2 into hexon-1.5 and train
├── prune_identity_layers.py   # Delete no-op blocks, verified bit-identical
├── reexport_ckpt.py           # Re-export a checkpoint under another quant scheme
└── export_checkpoint.py       # Export any training epoch to GGUF
eval/
├── heldout_eval.json   # The benchmark: 50 held-out + 8 legacy + 3 multi-turn
├── RESULTS.md          # Full measured results and their caveats
├── results/            # Raw responses, pre-rebuild knowledge base
└── results_v2/         # Raw responses, post-rebuild knowledge base
tests/
└── test_hexon.py      # 14 unit & integration tests (curation, pre-training, SFT, GGUF v3, multi-turn chat)
```
