"""Measure retrieval recall of the ChromaDB index against the held-out eval set.

Recall is the ceiling on grounded accuracy: if the right chunk is not retrieved,
no amount of training will let the model answer correctly from the database.

Usage:
    PYTHONPATH=/home/user/hexon/.deps:/home/user/hexon python3 scripts/recall_check.py
    python3 scripts/recall_check.py --set heldout_v1 --top-k 3
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexon.rag import HexonChromaRAG  # noqa: E402

DEFAULT_SET = "heldout_v1"
DEFAULT_TOPK = 3


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _keywords(item: dict) -> list:
    """Salient terms from the reference answer, used to judge a hit."""
    words = _norm(item.get("reference", "")).split()
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
        "for", "and", "or", "by", "with", "it", "its", "that", "this", "from", "as",
        "be", "can", "has", "have", "you", "your", "when", "which", "they", "than",
    }
    return [w for w in words if w not in stop and len(w) > 2]


def _hit(item: dict, texts: list) -> bool:
    """True if any retrieved chunk contains the reference answer's keywords."""
    kws = _keywords(item)
    if not kws:
        return False
    for t in texts:
        n = _norm(t)
        if all(k in n for k in kws):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist-dir", default=str(ROOT / "data" / "chromadb"))
    ap.add_argument("--set", dest="set_id", default=DEFAULT_SET)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--threshold", type=int, default=60,
                    help="majority-of-keywords threshold as a percentage")
    args = ap.parse_args()

    data = json.loads((ROOT / "eval" / "heldout_eval.json").read_text())
    target = next((s for s in data["sets"] if s["id"] == args.set_id), None)
    if target is None:
        print(f"no set {args.set_id!r}; available: {[s['id'] for s in data['sets']]}")
        return 1
    items = target.get("items") or []
    if not items:
        print(f"set {args.set_id!r} has no items")
        return 1

    rag = HexonChromaRAG(persist_dir=args.persist_dir)
    print(f"index: {args.persist_dir} ({rag.count()} chunks)")
    print(f"set  : {args.set_id} ({len(items)} items), top_k={args.top_k}\n")

    at1 = atk = scored = 0
    per_cat: dict = {}

    for item in items:
        chunks = rag.query(item["question"], top_k=args.top_k).get("retrieved_chunks") or []
        texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in chunks]
        if not texts:
            continue
        scored += 1
        kw = _keywords(item)
        need = max(1, len(kw) * args.threshold // 100)

        def enough(texts, need=need, kw=kw):
            for t in texts:
                n = _norm(t)
                if sum(1 for k in kw if k in n) >= need:
                    return True
            return False

        top1 = enough(texts[:1])
        topk = enough(texts)
        at1 += int(top1)
        atk += int(topk)
        cat = item.get("category", "?")
        s = per_cat.setdefault(cat, {"n": 0, "at1": 0, "atk": 0})
        s["n"] += 1
        s["at1"] += int(top1)
        s["atk"] += int(topk)

    print(f"{'category':<12} {'n':>3} {'@1':>6} {'@k':>6}")
    print("-" * 30)
    for cat in sorted(per_cat):
        s = per_cat[cat]
        print(f"{cat:<12} {s['n']:>3} {s['at1']/s['n']:>6.3f} {s['atk']/s['n']:>6.3f}")
    print("-" * 30)
    if scored:
        print(f"{'OVERALL':<12} {scored:>3} {at1/scored:>6.3f} {atk/scored:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
