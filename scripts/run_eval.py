#!/usr/bin/env python3
"""
Run the Hexon held-out benchmark against one or more exported models.

Usage
-----
    PYTHONPATH=/tmp/pydeps:/home/user/hexon python3 scripts/run_eval.py \
        --models hexon-135m-q4_0 hexon-code hexon-1.2 \
        --configs rag-on rag-off \
        --sets heldout_v1 legacy_keyword_check multiturn_anti_copy \
        --out-dir eval/results

One JSON result file is written per (model, config) pair. Re-running an
existing pair overwrites it, so partial runs can be resumed by dropping the
models you have already scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexon import HexonMobileEngine  # noqa: E402
from hexon.evalbench import (  # noqa: E402
    check_contamination,
    load_eval_set,
    run_eval,
    summarise,
)

EXPORTS = ROOT / "exports"

DEFAULT_CORPORA = [
    "data/curated/alpaca_knowledge.jsonl",
    "data/curated/common_crawl_curated.jsonl",
    "exports/hexon-nano-sft-dataset.jsonl",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["hexon-135m-q4_0", "hexon-code", "hexon-1.2"],
        help="Model stems in exports/ (without the .gguf suffix).",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["rag-on"],
        choices=["rag-on", "rag-off"],
        help="Whether ChromaDB grounding is enabled.",
    )
    ap.add_argument(
        "--sets",
        nargs="+",
        default=["heldout_v1", "legacy_keyword_check", "multiturn_anti_copy"],
        help="Eval subsets to run.",
    )
    ap.add_argument("--out-dir", default="eval/results")
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument(
        "--skip-contamination",
        action="store_true",
        help="Skip the eval-vs-training-corpus leakage check.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = load_eval_set()

    if not args.skip_contamination:
        print("== contamination check ==")
        rep = check_contamination(ev, DEFAULT_CORPORA)
        print(f"   items checked  : {rep['eval_items_checked']}")
        print(f"   corpora        : {len(rep['corpora_checked'])}")
        print(f"   contaminated   : {rep['contaminated_items'] or 'none'}")
        if not rep["clean"]:
            print("   WARNING: eval items overlap training corpora:")
            for f in rep["findings"]:
                print(f"     - {f['item_id']} ({f['set']}) jaccard={f['jaccard']} in {f['corpus']}")
        (out_dir / "contamination_report.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8"
        )
        print()

    all_results = []
    for model in args.models:
        path = EXPORTS / f"{model}.gguf"
        if not path.exists():
            print(f"!! skipping {model}: {path} not found")
            continue
        print(f"== loading {model} ==")
        t0 = time.time()
        engine = HexonMobileEngine.from_file(str(path))
        print(f"   loaded in {time.time() - t0:.1f}s (vocab={engine.model.config.vocab_size})")

        for cfg in args.configs:
            use_rag = cfg == "rag-on"
            print(f"-- {model} [{cfg}] --")
            t1 = time.time()
            res = run_eval(
                engine,
                model_label=model,
                eval_set=ev,
                set_ids=args.sets,
                use_rag=use_rag,
                max_new_tokens=args.max_new_tokens,
                rag_top_k=args.rag_top_k,
                verbose=not args.quiet,
            )
            res["wall_clock_s"] = round(time.time() - t1, 1)
            out_file = out_dir / f"{model}__{cfg}.json"
            out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
            all_results.append(res)
            print(f"   -> {out_file.relative_to(ROOT)}  ({res['wall_clock_s']}s)")
            for set_id, s in res["sets"].items():
                print(
                    f"      {set_id:<22} acc={s['accuracy']:.3f} "
                    f"hall={s['hallucination_rate']:.3f} "
                    f"unsupported={s['unsupported_number_items']}"
                )
            if "multiturn" in res:
                mt = res["multiturn"]
                print(
                    f"      {'multiturn_anti_copy':<22} copy={mt['copying_rate']:.3f} "
                    f"expect_met={mt['expectation_met_rate']:.3f}"
                )
            print()

    if all_results:
        print("== summary ==")
        print(summarise(all_results))
        (out_dir / "summary.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
