#!/usr/bin/env python3
"""Build the general-knowledge half of the hexon-1.8 corpus.

Mining PyPI, npm and GitHub produced thousands of chunks that are almost
entirely package documentation. That is a good corpus for coding questions and
a weak one for everything else, and the held-out benchmark is 74% non-coding
(science 12, coding 13, tech 8, geography 7, math 4, general 4, history 2).

So this builds the missing half from curated sources already in the repo:

*   alpaca_knowledge.jsonl - 225 Q&A pairs evenly split across science,
    coding, math, history_geography and health. Short pairs, so they are
    grouped by category into 200-300 word documents rather than chunked
    individually.
*   educational_text_chunks.json - expository prose.
*   common_crawl_curated.jsonl - curated web text, filtered so the
    invoice-factoring and conspiracy-blog material that got the old Common
    Crawl chunks deleted does not come back.

    python3 scripts/build_general_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hexon.rag import chunk_text  # noqa: E402
import mine_knowledge as mk  # noqa: E402

OUT = ROOT / "data" / "knowledge" / "hexon_general.jsonl"

ALPACA = ROOT / "data" / "curated" / "alpaca_knowledge.jsonl"
EDUCATIONAL = ROOT / "data" / "raw_cc" / "educational_text_chunks.json"
COMMON_CRAWL = ROOT / "data" / "curated" / "common_crawl_curated.jsonl"

TARGET_MIN, TARGET_MAX = 200, 300


def _words(text: str) -> int:
    return len(text.split())


def emit(records: List[Dict[str, Any]], title: str, category: str,
         source: str, text: str, seq: int) -> None:
    text = " ".join(text.split()).strip()
    w = _words(text)
    if not (TARGET_MIN <= w <= TARGET_MAX):
        return
    if mk.prose_density(text) < 0.45:
        return
    slug = re.sub(r"[^a-z0-9]+", "", title.lower())[:40] or "doc"
    records.append({
        "id": f"general-{source}-{slug}-{seq}",
        "title": title,
        "category": category,
        "source": source,
        "text": text,
        "word_count": w,
    })


def from_alpaca() -> List[Dict[str, Any]]:
    if not ALPACA.exists():
        return []
    rows = [json.loads(l) for l in open(ALPACA, encoding="utf-8") if l.strip()]
    by_cat: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "general"), []).append(r)

    out: List[Dict[str, Any]] = []
    for cat, items in by_cat.items():
        buf: List[str] = []
        seq = 0
        for r in items:
            q = (r.get("instruction", "") or "").strip()
            a = (r.get("output", "") or "").strip()
            if not q or not a:
                continue
            pair = f"{q} {a}"
            if buf and _words(" ".join(buf) + " " + pair) > TARGET_MAX:
                emit(out, f"{cat.replace('_', ' ').title()} reference "
                          f"(part {seq + 1})", cat, "alpaca_knowledge",
                     " ".join(buf), seq)
                seq += 1
                buf = []
            buf.append(pair)
        if buf:
            emit(out, f"{cat.replace('_', ' ').title()} reference "
                      f"(part {seq + 1})", cat, "alpaca_knowledge",
                 " ".join(buf), seq)
    return out


def from_educational() -> List[Dict[str, Any]]:
    if not EDUCATIONAL.exists():
        return []
    data = json.loads(EDUCATIONAL.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    seq = 0
    for i, rec in enumerate(data):
        text = rec.get("text", "") if isinstance(rec, dict) else str(rec)
        if _words(text) < 40:
            continue
        for piece in chunk_text(text, min_words=TARGET_MIN,
                                max_words=TARGET_MAX, overlap_words=35):
            emit(out, f"Educational reference {i + 1}", "science",
                 "educational_text", piece, seq)
            seq += 1
    return out


def from_common_crawl() -> List[Dict[str, Any]]:
    if not COMMON_CRAWL.exists():
        return []
    out: List[Dict[str, Any]] = []
    seq = 0
    for line in open(COMMON_CRAWL, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        text = doc.get("text", "")
        if _words(text) < 60 or mk.changelog_score(text) >= 3:
            continue
        for piece in chunk_text(text, min_words=TARGET_MIN,
                                max_words=TARGET_MAX, overlap_words=35):
            emit(out, f"Reference: {doc.get('source', 'curated web')}",
                 "general", "common_crawl_curated", piece, seq)
            seq += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    records: List[Dict[str, Any]] = []
    for name, fn in (("alpaca_knowledge", from_alpaca),
                     ("educational_text", from_educational),
                     ("common_crawl_curated", from_common_crawl)):
        got = fn()
        print(f"  {name:<22} {len(got):>5} chunks")
        records.extend(got)

    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for r in records:
        key = re.sub(r"[^a-z0-9 ]", "", r["text"].lower())[:180]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    w = [r["word_count"] for r in unique] or [0]
    print()
    print(f"wrote {len(unique)} general-knowledge chunks -> {out_path}")
    print(f"word count: min={min(w)} max={max(w)} avg={sum(w)/len(w):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
