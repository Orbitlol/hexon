#!/usr/bin/env python3
"""
Re-score saved evaluation runs against the current eval set and scorer.

Eval results store every raw model response, so a scoring bug can be fixed and
all previous runs re-scored without paying for generation again. That matters
here: the first hallucination metric flagged correct contrastive answers such
as "It is not Toronto or Montreal", which inflated the hallucination rate and
would have made the models look worse the more careful they became.

Usage
-----
    PYTHONPATH=/tmp/pydeps:/home/user/hexon python3 scripts/rescore.py \
        --dirs eval/results eval/results_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexon.evalbench import load_eval_set, score_item  # noqa: E402


def _rate(recs: Sequence[Dict[str, Any]], field: str,
          keyed_by: str | None = None, want: bool = True) -> float:
    subset = [r for r in recs if r.get(keyed_by) is want] if keyed_by else list(recs)
    if not subset:
        return 0.0
    return round(sum(1 for r in subset if r.get(field)) / len(subset), 4)


def rescore_file(path: Path, items_by_id: Dict[str, Dict[str, Any]],
                 write: bool) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for set_id, block in data.get("sets", {}).items():
        recs = block.get("records", [])
        for rec in recs:
            item = items_by_id.get(rec["id"])
            if item is None:
                continue
            old_correct = rec.get("correct")
            old_hall = rec.get("hallucination")
            fresh = score_item(item, rec.get("response", ""))
            rec["correct"] = fresh["correct"]
            rec["groups_matched"] = fresh["groups_matched"]
            rec["group_coverage"] = fresh["group_coverage"]
            rec["hallucination"] = fresh["hallucination"]
            rec["hallucination_matches"] = fresh["hallucination_matches"]
            rec["unsupported_numbers"] = fresh["unsupported_numbers"]
            if old_correct != fresh["correct"] or old_hall != fresh["hallucination"]:
                changed += 1

        n = len(recs)
        if not n:
            continue
        block["correct"] = sum(1 for r in recs if r["correct"])
        block["accuracy"] = round(block["correct"] / n, 4)
        block["group_coverage"] = round(
            sum(r["group_coverage"] for r in recs) / n, 4)
        block["hallucinations"] = sum(1 for r in recs if r["hallucination"])
        block["hallucination_rate"] = round(block["hallucinations"] / n, 4)
        block["unsupported_number_items"] = sum(
            1 for r in recs if r["unsupported_numbers"])
        block["grounded_items"] = sum(1 for r in recs if r.get("context_covers_answer"))
        block["accuracy_grounded"] = _rate(
            recs, "correct", "context_covers_answer", True)
        block["accuracy_ungrounded"] = _rate(
            recs, "correct", "context_covers_answer", False)

    if write and changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["_rescored_changed_records"] = changed
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["eval/results", "eval/results_v2"])
    ap.add_argument("--write", action="store_true", default=True,
                    help="Rewrite the result files in place (default).")
    args = ap.parse_args()

    ev = load_eval_set()
    items_by_id = {item["id"]: item for _, item in (
        (sid, it) for sid, it in (
            (s.get("id"), i) for s in ev.get("sets", []) for i in s.get("items", [])
        )
    )}

    rows: List[Dict[str, Any]] = []
    for d in args.dirs:
        directory = ROOT / d
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*__*.json")):
            data = rescore_file(f, items_by_id, args.write)
            rows.append((d, data))
            print(f"{d}/{f.name}: {data['_rescored_changed_records']} records changed")

    print()
    header = f"{'run':<16}{'model':<20}{'rag':<6}{'set':<22}{'acc':>7}{'hall':>7}{'grnd':>7}{'acc_g':>7}{'acc_u':>7}"
    print(header)
    print("-" * len(header))
    for d, data in rows:
        for set_id, s in data.get("sets", {}).items():
            g = s.get("accuracy_grounded")
            u = s.get("accuracy_ungrounded")
            print(
                f"{d.split('/')[-1]:<16}{data['model']:<20}"
                f"{('on' if data.get('use_rag') else 'off'):<6}{set_id:<22}"
                f"{s['accuracy']:>7.3f}{s['hallucination_rate']:>7.3f}"
                f"{str(s['grounded_items'])+'/'+str(s['items']):>7}"
                f"{(f'{g:.3f}' if isinstance(g, float) else '-'):>7}"
                f"{(f'{u:.3f}' if isinstance(u, float) else '-'):>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
