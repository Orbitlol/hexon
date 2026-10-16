#!/usr/bin/env python3
"""
Rebuild the Hexon ChromaDB knowledge base for hexon-1.5.

What this changes and why:

*   **Drops the raw Common Crawl chunks.** The previous 225 chunk index drew
    146 of its chunks from `cc_*.jsonl` web scrape dumps. Sampled content
    included invoice factoring teleconferences and conspiracy blog posts.
    Retrieval measurement showed enabling RAG made held-out accuracy *worse*
    on every model (hexon-code: 0.680 without RAG, 0.520 with), which is what
    you would expect when retrieval mostly returns irrelevant web text.

*   **Adds the hexon-1.5 corpus**: 29 hand-authored reference chunks covering
    science, geography, history, maths, computing, mobile hardware, Python and
    JavaScript, plus 120 filtered chunks mined from PyPI and npm package
    documentation.

*   **Keeps** the curated parts of the old index (stanford_alpaca,
    coding_reference, hexon_docs, encyclopedia, hardware_specs, seed_tasks).

Every chunk written here is 200-300 words, matching CHUNK_SCHEMA_VERSION
`v2_200_300_words`, and the script fails loudly if any chunk falls outside.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexon.rag import HexonChromaRAG  # noqa: E402

PERSIST_DIR = ROOT / "data" / "chromadb"

NEW_CORPORA = [
    ROOT / "data" / "knowledge" / "hexon_core_science.jsonl",
    ROOT / "data" / "knowledge" / "hexon_core_tech.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_docs.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_bulk.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_general.jsonl",
    ROOT / "data" / "knowledge" / "hexon_general.jsonl",
]

# Curated sources worth carrying over from the pre-existing index. Everything
# else - the cc_* web scrape dumps - is dropped.
#
# Note: `hexon_core_authored` and `mined_*` are deliberately NOT in this list.
# Those come from the corpus files on disk, so carrying them over from the
# database as well would duplicate every chunk on a re-run.
KEEP_SOURCES_PREFIXES = (
    "stanford_alpaca",
    "coding_reference",
    "hexon_docs",
    "encyclopedia",
    "hardware_specs",
    "seed_tasks",
    "fineweb_fixture",
)


def load_new_docs(exclude=()) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for path in NEW_CORPORA:
        if any(e and e in path.name for e in exclude):
            print(f"  - excluding {path.name}")
            continue
        if not path.exists():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                docs.append(row)
    return docs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    # Snapshot the curated chunks out of the existing index before rebuilding.
    # The index has to be rebuilt from scratch whenever DEFAULT_EMBEDDING_DIM
    # changes, because every stored vector is that wide.
    probe = HexonChromaRAG(persist_dir=str(PERSIST_DIR))
    before = probe.count()
    print(f"chunks before: {before}")

    existing = probe.collection.get()
    keep_ids: List[str] = []
    keep_docs: List[str] = []
    keep_metas: List[Dict[str, Any]] = []
    dropped = 0
    for cid, doc, meta in zip(
        existing.get("ids", []),
        existing.get("documents", []),
        existing.get("metadatas", []),
    ):
        src = str((meta or {}).get("source", ""))
        if src.startswith(KEEP_SOURCES_PREFIXES):
            keep_ids.append(cid)
            keep_docs.append(doc)
            keep_metas.append(dict(meta or {}))
        else:
            dropped += 1
    print(f"  keeping  : {len(keep_ids)} curated chunks")
    print(f"  dropping : {dropped} (raw Common Crawl web scrape)")
    del probe

    # Recreate the collection so the new embedding width takes effect.
    import chromadb

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        client.delete_collection("hexon_knowledge")
        print("  recreated collection 'hexon_knowledge'")
    except Exception as exc:
        print(f"  ! could not delete collection: {exc}")

    # auto_ingest_defaults=False is essential: constructing a RAG over an empty
    # collection would otherwise re-run the default ingest, which pulls the
    # cc_* web scrape dumps straight back in.
    rag = HexonChromaRAG(persist_dir=str(PERSIST_DIR), auto_ingest_defaults=False)
    print(f"  embedding_dim: {rag.embedding_fn.dim}")

    if keep_docs:
        rag.add_documents(
            documents=keep_docs, metadatas=keep_metas, ids=keep_ids, chunk=False
        )
        print(f"  re-added : {len(keep_docs)} curated chunks")

    new_docs = load_new_docs(exclude=args.exclude)
    texts = [d["text"] for d in new_docs]
    metas = [
        {
            "source": d.get("source", "unknown"),
            "category": d.get("category", "general"),
            "title": d.get("title", d.get("id", "")),
        }
        for d in new_docs
    ]
    ids = [d.get("id") or f"kb-{i}" for i, d in enumerate(new_docs)]
    added = rag.add_documents(documents=texts, metadatas=metas, ids=ids, chunk=True)
    print(f"  added    : {added} new chunks")

    after = rag.count()
    stats = rag.chunk_word_count_stats()
    got = rag.collection.get()
    src_counts = Counter(str((m or {}).get("source", "?")) for m in got.get("metadatas", []))
    cat_counts = Counter(str((m or {}).get("category", "?")) for m in got.get("metadatas", []))

    print()
    print(f"chunks after : {after}")
    print(f"word counts  : min={stats['min_words']} max={stats['max_words']} avg={stats['avg_words']}")
    print("by source:")
    for k, v in src_counts.most_common():
        print(f"   {k:<28} {v}")
    print("by category:")
    for k, v in cat_counts.most_common():
        print(f"   {k:<28} {v}")

    bad = rag.collection.get()
    out_of_range = [
        (i, len(d.split()))
        for i, d in zip(bad.get("ids", []), bad.get("documents", []))
        if not (200 <= len(d.split()) <= 300)
    ]
    if out_of_range:
        print(f"\nFAIL: {len(out_of_range)} chunks outside 200-300 words: {out_of_range[:10]}")
        return 1
    print("\nOK: every chunk is 200-300 words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
