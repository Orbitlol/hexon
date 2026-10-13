"""
Hexon: Compact Mobile-First AI Language Model (PyTorch + GGUF v3 + TorchScript).

Training is performed strictly offline:
1. Common Crawl Snapshot Ingestion & 7-Stage Data Curation (`hexon.curation`)
2. Base Model Causal LM Pre-training (`hexon.pretrain`)
3. Supervised Fine-Tuning (`hexon.sft`)

Exported `.gguf` and `.pt` artifacts are locked for low-memory, on-device mobile inference.
"""

from .config import HexonConfig
from .curation import (
    CuratedDocument,
    CurationConfig,
    CurationReport,
    curate_raw_records,
    pull_common_crawl_snapshot,
    run_curation_pipeline,
)
from .export_gguf import GGMLQuantizationType, export_to_gguf
from .export_mobile import export_to_mobile_int8_pt, export_to_torchscript
from .gguf_runtime import GGUFReader
from .inference import HexonMobileEngine
from .model import HexonForCausalLM
from .pretrain import HexonPretrainConfig, pretrain_base_model
from .rag import (
    HexonChromaEmbeddingFunction,
    HexonChromaRAG,
    RetrievedChunk,
    chunk_text,
)
from .sft import HexonSFTConfig, build_sft_corpus, run_sft_pipeline, save_sft_dataset_jsonl
from .tokenizer import HexonTokenizer
from .train import load_hexon_checkpoint, save_hexon_checkpoint, train_hexon_offline

__all__ = [
    "HexonConfig",
    "HexonForCausalLM",
    "HexonTokenizer",
    "HexonMobileEngine",
    "GGUFReader",
    "GGMLQuantizationType",
    "CurationConfig",
    "CuratedDocument",
    "CurationReport",
    "pull_common_crawl_snapshot",
    "curate_raw_records",
    "run_curation_pipeline",
    "HexonPretrainConfig",
    "pretrain_base_model",
    "HexonChromaRAG",
    "HexonChromaEmbeddingFunction",
    "RetrievedChunk",
    "chunk_text",
    "export_to_gguf",
    "export_to_torchscript",
    "export_to_mobile_int8_pt",
    "train_hexon_offline",
    "HexonSFTConfig",
    "build_sft_corpus",
    "run_sft_pipeline",
    "save_sft_dataset_jsonl",
    "save_hexon_checkpoint",
    "load_hexon_checkpoint",
]

__version__ = "1.0.0"
