"""
Hexon Held-Out Evaluation Benchmark.

This module exists because accuracy claims are worthless without a benchmark
that the model was not trained on. Two things it enforces:

1.  **Held-out scoring (``score_item``)** - keyword/regex scoring over the
    committed eval set in ``eval/heldout_eval.json``. Every item carries
    ``must_include`` alias groups, optional ``must_not`` hallucination
    patterns and an optional unsupported-number probe.

2.  **Contamination checking (``check_contamination``)** - verifies that none
    of the eval questions appear in any training corpus. This is the guard
    that stops a future training run from silently poisoning the benchmark.

The scoring is keyword-based, not a judge model. That is deliberate: it is
deterministic, reproducible and runs on CPU. It is also blunt - it cannot
tell a fluent correct answer from a fluent wrong one unless the wrong one
contains a flagged pattern. Treat the numbers as a lower bound on quality
and a direct measure of hallucination, not as a general quality score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "heldout_eval.json"

_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_WORD_RE = re.compile(r"[a-z0-9]+")


def load_eval_set(path: str | Path = DEFAULT_EVAL_PATH) -> Dict[str, Any]:
    """Load the committed held-out eval set."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval set not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_items(eval_set: Dict[str, Any], set_ids: Optional[Sequence[str]] = None):
    """Yield ``(set_id, item)`` pairs, optionally restricted to given set ids."""
    for block in eval_set.get("sets", []):
        sid = block.get("id")
        if set_ids is not None and sid not in set_ids:
            continue
        for item in block.get("items", []):
            yield sid, item


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _matches_any(patterns: Iterable[str], text: str) -> bool:
    for pat in patterns:
        try:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        except re.error:
            if pat.lower() in text:
                return True
    return False


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """Fraction of ``a``'s n-grams that also appear in ``b`` (0..1)."""
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) < n or len(tb) < n:
        return 0.0
    ga = {tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    gb = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def find_unsupported_numbers(response: str, question: str, reference: str) -> List[str]:
    """Numbers asserted in ``response`` that appear neither in the question
    nor in the reference answer.

    This is a noisy proxy for fabrication: a correct answer can legitimately
    restate a number in a different form, and some items have several valid
    numbers. It is reported as a separate rate and is never folded into
    accuracy, and every hit is meant to be eyeballed before it is believed.
    """
    resp_nums = set(_NUMBER_RE.findall(response))
    allowed = set(_NUMBER_RE.findall(question)) | set(_NUMBER_RE.findall(reference))
    # Normalise thousands separators before comparing
    allowed |= {a.replace(",", "") for a in allowed}
    out = []
    for num in resp_nums:
        if num in allowed or num.replace(",", "") in allowed:
            continue
        out.append(num)
    return sorted(out)


def context_covers_item(item: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> bool:
    """True when the retrieved ChromaDB context already contains the answer.

    An item whose answer is sitting in the retrieved context is a *grounded*
    item: getting it right mostly demonstrates retrieval plus faithful copying,
    not knowledge. Splitting the score by this flag is what keeps the RAG-on
    numbers honest once the knowledge base grows to cover the test domains.
    """
    if not chunks:
        return False
    blob = "\n".join(str(c.get("text", "")) for c in chunks).lower()
    groups: List[List[str]] = item.get("must_include") or []
    if not groups:
        return False
    for group in groups:
        patterns = group if isinstance(group, list) else [group]
        if not _matches_any(patterns, blob):
            return False
    return True


def score_item(item: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Score one response against one eval item.

    Returns a dict with ``correct`` (all alias groups satisfied),
    ``groups_matched``, ``hallucination`` (a ``must_not`` pattern fired),
    ``hallucination_matches`` and ``unsupported_numbers``.
    """
    text = (response or "").strip()
    low = text.lower()

    groups: List[List[str]] = item.get("must_include") or []
    groups_matched: List[bool] = []
    for group in groups:
        patterns = group if isinstance(group, list) else [group]
        groups_matched.append(_matches_any(patterns, low))

    correct = all(groups_matched) if groups_matched else False

    must_not: List[str] = item.get("must_not") or []
    hall_matches: List[str] = []
    for pat in must_not:
        try:
            m = re.search(pat, low, flags=re.IGNORECASE)
        except re.error:
            m = None
        if m:
            hall_matches.append(m.group(0))

    unsupported: List[str] = []
    if item.get("check_unsupported_number"):
        unsupported = find_unsupported_numbers(
            low, item.get("question", ""), item.get("reference", "")
        )

    return {
        "correct": bool(correct),
        "groups_matched": groups_matched,
        "groups_total": len(groups_matched),
        "group_coverage": (
            sum(1 for g in groups_matched if g) / len(groups_matched)
            if groups_matched
            else 0.0
        ),
        "hallucination": bool(hall_matches),
        "hallucination_matches": hall_matches,
        "unsupported_numbers": unsupported,
        "empty": len(text) == 0,
    }


# --------------------------------------------------------------------------
# Running the benchmark against an engine
# --------------------------------------------------------------------------


def _rate(
    recs: Sequence[Dict[str, Any]],
    field: str,
    keyed_by: Optional[str] = None,
    want: bool = True,
) -> float:
    """Mean of a boolean field over ``recs``, optionally filtered on another flag."""
    subset = [r for r in recs if r.get(keyed_by) is want] if keyed_by else list(recs)
    if not subset:
        return 0.0
    return round(sum(1 for r in subset if r.get(field)) / len(subset), 4)


def run_eval(
    engine: Any,
    model_label: str,
    eval_set: Optional[Dict[str, Any]] = None,
    set_ids: Optional[Sequence[str]] = None,
    use_rag: bool = True,
    max_new_tokens: int = 110,
    rag_top_k: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run every item in the eval set through ``engine.reply`` and score it."""
    ev = eval_set or load_eval_set()

    per_set: Dict[str, List[Dict[str, Any]]] = {}
    for set_id, item in iter_items(ev, set_ids):
        question = item["question"]
        out: Dict[str, Any] = {}
        try:
            out = engine.reply(
                question,
                max_new_tokens=max_new_tokens,
                use_rag=use_rag,
                rag_top_k=rag_top_k,
            )
            response = out.get("response", "")
            tokens = out.get("tokens_generated", 0)
            latency = out.get("latency_ms", 0.0)
        except Exception as exc:  # pragma: no cover - defensive
            response, tokens, latency = f"<error: {exc}>", 0, 0.0

        chunks = out.get("retrieved_chunks", []) if isinstance(out, dict) else []
        covered = context_covers_item(item, chunks)
        sc = score_item(item, response)
        rec = {
            "id": item["id"],
            "category": item.get("category", ""),
            "question": question,
            "reference": item.get("reference", ""),
            "response": response,
            "tokens_generated": tokens,
            "latency_ms": round(latency, 1),
            "retrieved": len(chunks),
            "context_covers_answer": covered,
            **sc,
        }
        per_set.setdefault(set_id, []).append(rec)
        if verbose:
            flag = "OK  " if sc["correct"] else "MISS"
            if sc["hallucination"]:
                flag = "HALL"
            print(f"  [{flag}] {item['id']:<10} {question[:62]}")
            if not sc["correct"]:
                print(f"          -> {response[:160]!r}")

    results: Dict[str, Any] = {"model": model_label, "use_rag": use_rag, "sets": {}}
    for set_id, recs in per_set.items():
        n = len(recs)
        results["sets"][set_id] = {
            "items": n,
            "correct": sum(1 for r in recs if r["correct"]),
            "accuracy": round(sum(1 for r in recs if r["correct"]) / n, 4) if n else 0.0,
            "group_coverage": (
                round(sum(r["group_coverage"] for r in recs) / n, 4) if n else 0.0
            ),
            "hallucinations": sum(1 for r in recs if r["hallucination"]),
            "hallucination_rate": (
                round(sum(1 for r in recs if r["hallucination"]) / n, 4) if n else 0.0
            ),
            "unsupported_number_items": sum(1 for r in recs if r["unsupported_numbers"]),
            "empty_responses": sum(1 for r in recs if r["empty"]),
            "grounded_items": sum(1 for r in recs if r["context_covers_answer"]),
            "accuracy_grounded": _rate(recs, "correct", keyed_by="context_covers_answer", want=True),
            "accuracy_ungrounded": _rate(recs, "correct", keyed_by="context_covers_answer", want=False),
            "mean_tokens": round(sum(r["tokens_generated"] for r in recs) / n, 1) if n else 0.0,
            "mean_latency_ms": round(sum(r["latency_ms"] for r in recs) / n, 1) if n else 0.0,
            "records": recs,
        }

    # Multi-turn anti-copying sequences
    mt_results: List[Dict[str, Any]] = []
    for block in ev.get("sets", []):
        if block.get("id") != "multiturn_anti_copy":
            continue
        if set_ids is not None and "multiturn_anti_copy" not in set_ids:
            continue
        for seq in block.get("sequences", []):
            turns = seq["turns"]
            history: List[Dict[str, str]] = []
            responses: List[str] = []
            for turn in turns:
                try:
                    out = engine.reply(
                        turn["content"],
                        max_new_tokens=max_new_tokens,
                        use_rag=use_rag,
                        rag_top_k=rag_top_k,
                        history=history,
                    )
                    resp = out.get("response", "")
                except Exception as exc:  # pragma: no cover
                    resp = f"<error: {exc}>"
                responses.append(resp)
                history.append({"role": "user", "content": turn["content"]})
                history.append({"role": "assistant", "content": resp})

            copies = []
            for i in range(1, len(responses)):
                overlap = ngram_overlap(responses[i], responses[i - 1])
                copies.append(
                    {
                        "turn": i + 1,
                        "identical": responses[i].strip() == responses[i - 1].strip(),
                        "ngram_overlap_with_previous": round(overlap, 3),
                        "copying": (responses[i].strip() == responses[i - 1].strip())
                        or overlap >= 0.5,
                    }
                )
            expect = seq.get("expect_turn_2_contains") or []
            last = responses[-1].lower() if responses else ""
            mt_results.append(
                {
                    "id": seq["id"],
                    "responses": responses,
                    "expect_turn_2_contains_met": _matches_any(
                        [p for g in expect for p in (g if isinstance(g, list) else [g])], last
                    ),
                    "copy_checks": copies,
                    "any_copying": any(c["copying"] for c in copies),
                }
            )
    if mt_results:
        results["multiturn"] = {
            "sequences": mt_results,
            "copying_rate": round(
                sum(1 for m in mt_results if m["any_copying"]) / len(mt_results), 4
            ),
            "expectation_met_rate": round(
                sum(1 for m in mt_results if m["expect_turn_2_contains_met"]) / len(mt_results),
                4,
            ),
        }

    return results


# --------------------------------------------------------------------------
# Contamination guard
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _content_tokens(text: str, stopwords: Optional[set] = None) -> set:
    stop = stopwords or set()
    return {t for t in _normalise(text).split() if len(t) > 2 and t not in stop}


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "you",
    "your", "what", "which", "who", "how", "does", "did", "have", "has",
    "from", "into", "than", "then", "them", "they", "their", "there",
    "about", "when", "where", "why", "can", "could", "would", "should",
    "write", "give", "explain", "describe", "list", "name", "tell",
}


def check_contamination(
    eval_set: Optional[Dict[str, Any]] = None,
    corpus_paths: Optional[Sequence[str | Path]] = None,
    jaccard_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Check eval questions against training corpora for leakage.

    Compares each eval question against every line of each corpus file using
    token-set Jaccard similarity. Anything at or above ``jaccard_threshold``
    is reported as suspected contamination. This is a lexical check, so it
    catches verbatim and near-verbatim reuse, not paraphrase.
    """
    ev = eval_set or load_eval_set()
    paths = [Path(p) for p in (corpus_paths or [])]
    corpora: List[Tuple[str, List[str]]] = []
    for p in paths:
        if not p.exists():
            continue
        lines: List[str] = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                if isinstance(obj, dict):
                    parts = []
                    for key in ("instruction", "input", "question", "user", "prompt", "text"):
                        v = obj.get(key)
                        if isinstance(v, str) and v:
                            parts.append(v)
                    lines.append(" ".join(parts) if parts else json.dumps(obj))
                else:
                    lines.append(str(obj))
        corpora.append((str(p), lines))

    findings: List[Dict[str, Any]] = []
    for set_id, item in iter_items(ev):
        q_toks = _content_tokens(item["question"], _STOPWORDS)
        if not q_toks:
            continue
        for corpus_name, lines in corpora:
            for line in lines:
                l_toks = _content_tokens(line, _STOPWORDS)
                if not l_toks:
                    continue
                inter = len(q_toks & l_toks)
                union = len(q_toks | l_toks)
                jac = inter / union if union else 0.0
                if jac >= jaccard_threshold:
                    findings.append(
                        {
                            "item_id": item["id"],
                            "set": set_id,
                            "question": item["question"],
                            "corpus": corpus_name,
                            "jaccard": round(jac, 3),
                            "corpus_line": line[:200],
                        }
                    )
                    break

    return {
        "eval_items_checked": sum(1 for _ in iter_items(ev)),
        "corpora_checked": [name for name, _ in corpora],
        "jaccard_threshold": jaccard_threshold,
        "contaminated_items": sorted({f["item_id"] for f in findings}),
        "findings": findings,
        "clean": not findings,
    }


def summarise(results: Sequence[Dict[str, Any]]) -> str:
    """Render a compact comparison table over several ``run_eval`` results."""
    headers = ["model", "set", "rag", "acc", "grp_cov", "hall%", "unsup#", "tok"]
    lines = [" | ".join(h.rjust(8) for h in headers)]
    lines.append("-" * len(lines[0]))
    for r in results:
        for set_id, s in r.get("sets", {}).items():
            rows = [
                r["model"],
                set_id,
                "on" if r.get("use_rag") else "off",
                f"{s['accuracy']:.3f}",
                f"{s['group_coverage']:.3f}",
                f"{s['hallucination_rate']:.3f}",
                str(s["unsupported_number_items"]),
                f"{s['mean_tokens']:.0f}",
            ]
            lines.append(" | ".join(x.rjust(8) for x in rows))
        mt = r.get("multiturn")
        if mt:
            lines.append(
                " | ".join(
                    x.rjust(8)
                    for x in [
                        r["model"],
                        "multiturn",
                        "on" if r.get("use_rag") else "off",
                        "-",
                        "-",
                        f"copy:{mt['copying_rate']:.2f}",
                        f"exp:{mt['expectation_met_rate']:.2f}",
                        "-",
                    ]
                )
            )
    return "\n".join(lines)
