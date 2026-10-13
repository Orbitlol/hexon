"""
Hexon Command-Line Interface.

Usage examples:
  # Run Common Crawl data curation + Base Pre-training + SFT + GGUF/Mobile exports:
  python3 -m hexon.cli build-all --output-dir exports

  # Run only the Common Crawl data curation pipeline:
  python3 -m hexon.cli curate --raw-dir data/raw_cc --output data/curated/common_crawl_curated.jsonl

  # Inspect an exported GGUF file:
  python3 -m hexon.cli inspect exports/hexon-nano-q4_0.gguf

  # Run mobile inference directly from a quantized GGUF file:
  python3 -m hexon.cli generate --model exports/hexon-nano-q8_0.gguf --prompt "Who are you?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .config import HexonConfig
from .curation import CurationConfig, run_curation_pipeline
from .export_gguf import export_to_gguf
from .export_mobile import export_to_mobile_int8_pt, export_to_torchscript
from .gguf_runtime import GGUFReader
from .inference import HexonMobileEngine
from .pretrain import HexonPretrainConfig, pretrain_base_model
from .rag import HexonChromaRAG
from .sft import HexonSFTConfig, run_sft_pipeline, save_sft_dataset_jsonl
from .train import load_hexon_checkpoint, train_hexon_offline


def cmd_curate(args: argparse.Namespace) -> None:
    docs, report = run_curation_pipeline(
        raw_dir=args.raw_dir,
        output_jsonl=args.output,
        config=CurationConfig(),
    )
    print(json.dumps({"kept_documents": len(docs), "report": report.to_dict()}, indent=2))


def cmd_pretrain(args: argparse.Namespace) -> None:
    cfg = HexonConfig.from_preset(args.preset, vocab_size=args.vocab_size)
    pt_cfg = HexonPretrainConfig(
        raw_cc_dir=args.raw_dir,
        curated_jsonl_path=args.curated_jsonl,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    _, _, metrics = pretrain_base_model(
        model_config=cfg,
        pretrain_config=pt_cfg,
        save_base_checkpoint_path=args.output,
        verbose=True,
    )
    print(json.dumps(metrics, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    model, tokenizer, metrics = train_hexon_offline(
        preset=args.preset,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        data_jsonl=args.data,
        checkpoint_path=args.output,
        seed=args.seed,
        verbose=True,
    )
    print(json.dumps(metrics, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    model, tokenizer = load_hexon_checkpoint(args.checkpoint, lock_for_mobile=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = model.config.model_name

    results: List[Dict[str, Any]] = []
    for q in args.gguf_quants.split(","):
        q = q.strip().lower()
        if not q:
            continue
        gguf_path = out_dir / f"{name}-{q}.gguf"
        res = export_to_gguf(model, tokenizer, gguf_path, quant_type=q)
        results.append(res)

    if args.export_torchscript:
        ts_path = out_dir / f"{name}-torchscript.pt"
        results.append(export_to_torchscript(model, ts_path))

    if args.export_int8_pt:
        int8_path = out_dir / f"{name}-mobile-int8.pt"
        results.append(export_to_mobile_int8_pt(model, tokenizer, int8_path))

    print(json.dumps({"exports": results}, indent=2))


def cmd_inspect(args: argparse.Namespace) -> None:
    reader = GGUFReader(args.gguf_path)
    summary = reader.summary()
    if args.show_tensors:
        summary["tensors"] = [
            {
                "name": t.name,
                "torch_shape": list(t.torch_shape),
                "ggml_dims": list(t.ggml_dims),
                "quant_type": t.ggml_type.name,
                "byte_size": t.byte_size,
                "offset": t.offset,
            }
            for t in reader.tensors.values()
        ]
    print(json.dumps(summary, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    engine = HexonMobileEngine.from_file(args.model)
    result = engine.reply(
        user_prompt=args.prompt,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_rag=args.rag,
        rag_top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


def cmd_rag(args: argparse.Namespace) -> None:
    rag = HexonChromaRAG(persist_dir=args.persist_dir)
    if args.add_doc:
        added = rag.add_documents(
            [args.add_doc],
            metadatas=[{"source": "cli", "category": "user_added", "title": args.add_doc[:60]}],
        )
        print(json.dumps({"added_chunks": added, "total_chunks": rag.count()}, indent=2))
        return

    engine = HexonMobileEngine.from_file(args.model, rag=rag) if args.model else None
    result = rag.query(
        question=args.query,
        engine=engine,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2))


def cmd_sft(args: argparse.Namespace) -> None:
    sft_cfg = HexonSFTConfig(
        preset=args.preset,
        vocab_size=args.vocab_size,
        pretrain_epochs=args.pretrain_epochs,
        sft_epochs=args.sft_epochs,
        batch_size=args.batch_size,
        sft_lr=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    _, _, metrics = run_sft_pipeline(
        sft_config=sft_cfg,
        checkpoint_path=args.output,
        verbose=True,
    )
    print(json.dumps(metrics, indent=2))


def cmd_build_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt_path = out_dir / f"{args.preset}-base-pretrained.pt"
    ckpt_path = out_dir / f"{args.preset}-checkpoint.pt"
    sft_jsonl_path = out_dir / f"{args.preset}-sft-dataset.jsonl"

    num_sft_rows = save_sft_dataset_jsonl(sft_jsonl_path)
    print(
        f"=== [1/4] Common Crawl Curation + Base Pre-training + SFT "
        f"({num_sft_rows} Base Intents) for {args.preset} Offline ==="
    )
    sft_cfg = HexonSFTConfig(
        preset=args.preset,
        vocab_size=args.vocab_size,
        pretrain_epochs=args.pretrain_epochs,
        sft_epochs=args.epochs,
        batch_size=args.batch_size,
        sft_lr=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    model, tokenizer, train_metrics = run_sft_pipeline(
        sft_config=sft_cfg,
        checkpoint_path=ckpt_path,
        base_checkpoint_path=base_ckpt_path,
        verbose=True,
    )
    tokenizer.save(out_dir / f"{args.preset}-tokenizer.json")
    curation_report_path = out_dir / "hexon-curation-report.json"
    with open(curation_report_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics.get("curation_report", {}), f, indent=2)

    print("\n=== [2/4] Exporting GGUF v3 Mobile Artifacts (F16, Q8_0, Q4_0) ===")
    if base_ckpt_path.exists():
        base_m, base_t = load_hexon_checkpoint(base_ckpt_path, lock_for_mobile=True)
        export_to_gguf(base_m, base_t, out_dir / f"{args.preset}-base-f16.gguf", quant_type="f16")
    gguf_exports: Dict[str, Any] = {}
    for q in ("f16", "q8_0", "q4_0"):
        gguf_file = out_dir / f"{args.preset}-{q}.gguf"
        info = export_to_gguf(model, tokenizer, gguf_file, quant_type=q)
        gguf_exports[q] = info
        print(
            f"  -> {gguf_file.name}: {info['file_size_mb']:.3f} MB "
            f"({info['tensor_count']} tensors, GGUF v{info['gguf_version']})"
        )

    print("\n=== [3/4] Exporting PyTorch Mobile Artifacts (TorchScript & INT8) ===")
    ts_info = export_to_torchscript(model, out_dir / f"{args.preset}-torchscript.pt")
    int8_info = export_to_mobile_int8_pt(
        model, tokenizer, out_dir / f"{args.preset}-mobile-int8.pt"
    )
    print(f"  -> {Path(ts_info['output_path']).name}: {ts_info['file_size_mb']:.3f} MB")
    print(f"  -> {Path(int8_info['output_path']).name}: {int8_info['file_size_mb']:.3f} MB")

    print("\n=== [4/4] Verifying Standalone GGUF Mobile Inference (Q8_0 & Q4_0) ===")
    q8_engine = HexonMobileEngine.from_file(out_dir / f"{args.preset}-q8_0.gguf")
    q4_engine = HexonMobileEngine.from_file(out_dir / f"{args.preset}-q4_0.gguf")

    bench_q8 = q8_engine.benchmark(max_new_tokens=80)
    bench_q4 = q4_engine.benchmark(max_new_tokens=80)

    for sample in bench_q8["samples"]:
        print(f"  [Q8_0 GGUF] Q: {sample['prompt']}")
        print(
            f"              A: {sample['response']} "
            f"({sample['ms_per_token']} ms/tok, {sample['tokens_per_sec']} tok/s)"
        )

    report = {
        "training": train_metrics,
        "estimated_mobile_ram_mb": model.config.estimate_mobile_memory_mb(),
        "gguf_exports": gguf_exports,
        "pytorch_mobile_exports": {
            "torchscript": ts_info,
            "dynamic_int8": int8_info,
        },
        "benchmarks": {
            "gguf_q8_0": bench_q8,
            "gguf_q4_0": bench_q4,
        },
    }
    report_path = out_dir / "hexon_export_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved complete export & benchmark report to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexon",
        description="Hexon: Compact Mobile-First AI Language Model (PyTorch + GGUF v3)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_curate = sub.add_parser("curate", help="Pull and filter Common Crawl snapshot dumps")
    p_curate.add_argument("--raw-dir", default="data/raw_cc")
    p_curate.add_argument("--output", default="data/curated/common_crawl_curated.jsonl")
    p_curate.set_defaults(func=cmd_curate)

    p_pt = sub.add_parser("pretrain", help="Pre-train Hexon base model on curated Common Crawl data")
    p_pt.add_argument("--preset", default="hexon-nano")
    p_pt.add_argument("--vocab-size", type=int, default=1536)
    p_pt.add_argument("--raw-dir", default="data/raw_cc")
    p_pt.add_argument("--curated-jsonl", default="data/curated/common_crawl_curated.jsonl")
    p_pt.add_argument("--epochs", type=int, default=4)
    p_pt.add_argument("--batch-size", type=int, default=24)
    p_pt.add_argument("--lr", type=float, default=3.5e-3)
    p_pt.add_argument("--max-seq-len", type=int, default=224)
    p_pt.add_argument("--output", default="exports/hexon-nano-base-pretrained.pt")
    p_pt.add_argument("--seed", type=int, default=42)
    p_pt.set_defaults(func=cmd_pretrain)

    p_train = sub.add_parser("train", help="Train a Hexon checkpoint offline on server")
    p_train.add_argument("--preset", default="hexon-nano")
    p_train.add_argument("--vocab-size", type=int, default=1024)
    p_train.add_argument("--epochs", type=int, default=45)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--max-seq-len", type=int, default=96)
    p_train.add_argument("--data", default=None, help="Optional custom JSONL file")
    p_train.add_argument("--output", default="exports/hexon-nano-checkpoint.pt")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_export = sub.add_parser("export", help="Export checkpoint to GGUF & PyTorch Mobile")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output-dir", default="exports")
    p_export.add_argument("--gguf-quants", default="f16,q8_0,q4_0")
    p_export.add_argument("--export-torchscript", action="store_true", default=True)
    p_export.add_argument("--export-int8-pt", action="store_true", default=True)
    p_export.set_defaults(func=cmd_export)

    p_inspect = sub.add_parser("inspect", help="Inspect a GGUF v3 model file")
    p_inspect.add_argument("gguf_path", help="Path to .gguf file")
    p_inspect.add_argument("--show-tensors", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    p_gen = sub.add_parser("generate", help="Run inference from a .gguf or .pt file")
    p_gen.add_argument("--model", default="exports/hexon-135m-q4_0.gguf", help="Path to .gguf or .pt file")
    p_gen.add_argument("--prompt", required=True, help="User prompt")
    p_gen.add_argument("--system", default=None, help="Optional system prompt")
    p_gen.add_argument("--max-new-tokens", type=int, default=80)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--rag", action="store_true", help="Enable ChromaDB RAG retrieval")
    p_gen.add_argument("--top-k", type=int, default=3, help="Number of ChromaDB chunks to retrieve")
    p_gen.set_defaults(func=cmd_generate)

    p_rag = sub.add_parser("rag", help="Query or ingest into the ChromaDB RAG knowledge base")
    p_rag.add_argument("--query", default="How does Hexon use ChromaDB for RAG?", help="Question to retrieve/answer")
    p_rag.add_argument("--model", default="exports/hexon-135m-q4_0.gguf", help="Model path for RAG synthesis")
    p_rag.add_argument("--persist-dir", default="data/chromadb", help="ChromaDB persistence directory")
    p_rag.add_argument("--top-k", type=int, default=3, help="Top-k passages to retrieve")
    p_rag.add_argument("--max-new-tokens", type=int, default=90)
    p_rag.add_argument("--add-doc", default=None, help="Optional document text to index into ChromaDB")
    p_rag.set_defaults(func=cmd_rag)

    p_sft = sub.add_parser("sft", help="Run two-stage CC Pretrain + SFT pipeline offline")
    p_sft.add_argument("--preset", default="hexon-nano")
    p_sft.add_argument("--vocab-size", type=int, default=1536)
    p_sft.add_argument("--pretrain-epochs", type=int, default=4)
    p_sft.add_argument("--sft-epochs", type=int, default=24)
    p_sft.add_argument("--batch-size", type=int, default=24)
    p_sft.add_argument("--lr", type=float, default=2.8e-3)
    p_sft.add_argument("--max-seq-len", type=int, default=224)
    p_sft.add_argument("--output", default="exports/hexon-nano-checkpoint.pt")
    p_sft.add_argument("--seed", type=int, default=42)
    p_sft.set_defaults(func=cmd_sft)

    p_all = sub.add_parser(
        "build-all",
        help="Run Common Crawl Curation + Base Pretrain + SFT offline, export GGUF & Mobile PT",
    )
    p_all.add_argument("--preset", default="hexon-nano")
    p_all.add_argument("--vocab-size", type=int, default=1536)
    p_all.add_argument("--pretrain-epochs", type=int, default=4)
    p_all.add_argument("--epochs", type=int, default=14)
    p_all.add_argument("--batch-size", type=int, default=14)
    p_all.add_argument("--lr", type=float, default=3.4e-3)
    p_all.add_argument("--max-seq-len", type=int, default=208)
    p_all.add_argument("--output-dir", default="exports")
    p_all.add_argument("--seed", type=int, default=42)
    p_all.set_defaults(func=cmd_build_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
