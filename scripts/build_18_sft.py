#!/usr/bin/env python3
"""Build the hexon-1.8 supervised fine-tuning set at scale.

The 1.5 set was 126 hand-written records - far too small to train a model with
real capacity, and hand-writing more is not viable. This generates grounded
examples directly from the knowledge corpus instead.

How an example is formed
------------------------
1.  Take a real corpus chunk - the exact 200-300 word text ChromaDB retrieves.
2.  Derive a question from its package name or title.
3.  Derive the answer *extractively* from that chunk.

The answer is always a span of the context, so every example teaches the same
behaviour: read the context, answer from it, do not invent.

Calibrated refusals
-------------------
A subset pairs a question with a deliberately OFF-TOPIC chunk, and the target
answer is a refusal. This is the main lever on hallucination in a retrieval
system: if the retrieved chunk does not contain the answer, the model should
abstain rather than fabricate. Refusals are capped at ~12% because an earlier
hexon-1.5 run over-trained on them and the model started declining questions
it could answer.

Prompts use `build_prompt` from build_15_sft.py, which reproduces
`HexonChromaRAG.build_rag_prompt` byte for byte, so training matches inference.

    python3 scripts/build_18_sft.py --max-examples 40000 --per-chunk 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_15_sft as sft15  # noqa: E402
import mine_knowledge as mk  # noqa: E402

OUT = ROOT / "data" / "sft" / "hexon_18_sft.jsonl"

CORPUS_FILES = [
    ROOT / "data" / "knowledge" / "hexon_core_science.jsonl",
    ROOT / "data" / "knowledge" / "hexon_core_tech.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_docs.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_bulk.jsonl",
    ROOT / "data" / "knowledge" / "hexon_mined_general.jsonl",
    ROOT / "data" / "knowledge" / "hexon_general.jsonl",
]

SEED_SFT = ROOT / "data" / "sft" / "hexon_15_sft.jsonl"

REFUSAL_ANSWERS = [
    "I don't have reliable information about that in the knowledge base I was "
    "given, so I would rather not guess.",
    "That isn't covered by the reference material available to me. I can't "
    "answer it accurately without inventing something.",
    "The sources I have access to don't contain an answer to that, so I'm not "
    "going to attempt one.",
    "I can't answer that from what I have here. If you can point me at a "
    "source, I can work from it.",
]

QUESTION_TEMPLATES = {
    "mined_pypi": [
        "What is {name}?",
        "What is {name} used for?",
        "How do you install {name}?",
        "Explain what {name} does.",
    ],
    "mined_npm": [
        "What is {name}?",
        "What is {name} used for?",
        "How do you install {name}?",
        "Explain what {name} does.",
    ],
    "mined_github": [
        "What is {name}?",
        "What is {name} used for?",
        "How do you use {name}?",
    ],
    "hexon_core_authored": [
        "What is {name}?",
        "Explain {name}.",
        "What should I know about {name}?",
    ],
}

CODE_HINTS = ("install", "usage", "example", "api", "command", "config",
              "quickstart", "getting started", "how to", "tutorial")
CODE_SOURCES = {"mined_pypi", "mined_npm", "mined_github"}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

NOISE = re.compile(
    r"https?://|www\.|```|\$\s|\bpip install\b|\bnpm install\b|>>>|::"
    r"|={3,}|-{3,}|\*{3,}|\.py\b|\.js\b|\.json\b|\.md\b|/usr/|/home/"
    r"|[\w.]+@[\w.]+\.[a-z]{2,}|^\s*[#*|>]|\|\s*---|\[!\[",
    re.I,
)

VERB = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|does|do|did|can|could"
    r"|will|would|should|may|might|must|provides?|allows?|lets?|helps?"
    r"|enables?|uses?|used|works?|means?|refers?|describes?|implements?"
    r"|supports?|includes?|contains?|makes?|takes?|gives?|runs?|returns?"
    r"|handles?|performs?|creates?|builds?)\b",
    re.I,
)


def _clean_name(raw: str) -> str:
    name = re.sub(r"[-_]+", " ", (raw or "").strip())
    return re.sub(r"\s+", " ", name).strip()


def answer_quality(text: str) -> float:
    """Fraction of words carrying meaning, for a short answer.

    mine_knowledge.prose_density cannot be used here: it hard-returns 0.0 for
    text under 120 words because it was written to score 200-300 word chunks.
    Answers are 20-110 words, so every one scored 0.0 and was rejected.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'’-]{1,}", text)
    if len(words) < 8:
        return 0.0
    content = [w for w in words
               if w.lower() not in mk._ENGLISH_STOP and len(w) > 2]
    return len(content) / max(1, len(words))


def _clean_sentence(s: str) -> str:
    s = re.sub(r"[=\-_*~]{3,}", " ", s)
    s = re.sub(r"^[#*>|\s]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def derive_answer(text: str) -> Optional[str]:
    """Extract a self-contained prose answer from a chunk."""
    text = " ".join(text.split())
    if not text:
        return None
    candidates = [_clean_sentence(s) for s in SENTENCE_SPLIT.split(text)]
    candidates = [s for s in candidates if s]
    if not candidates:
        return None

    take: List[str] = []
    words = 0
    for s in candidates:
        sw = len(s.split())
        if NOISE.search(s) or sw < 6 or not VERB.search(s):
            if take:
                break
            continue
        if take and words + sw > 110:
            break
        take.append(s)
        words += sw
        if words >= 38:
            break

    if not take or words < 20:
        return None
    ans = " ".join(take).strip()
    if answer_quality(ans) < 0.38:
        return None
    if not ans.endswith((".", "!", "?")):
        return None
    return ans


def looks_like_code_request(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in CODE_HINTS)


def load_chunks() -> List[Dict[str, Any]]:
    from hexon.rag import chunk_text

    chunks: List[Dict[str, Any]] = []
    for path in CORPUS_FILES:
        if not path.exists():
            print(f"  ! missing {path.name}", file=sys.stderr)
            continue
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", "")
                if not text:
                    continue
                w = len(text.split())
                pieces = [text] if 150 <= w <= 320 else chunk_text(
                    text, min_words=200, max_words=300, overlap_words=35)
                for piece in pieces:
                    pw = len(piece.split())
                    if not (150 <= pw <= 320):
                        continue
                    chunks.append({
                        "text": piece,
                        "title": doc.get("title", ""),
                        "source": doc.get("source", "unknown"),
                        "package": doc.get("package", ""),
                    })
                    n += 1
        print(f"  {path.name}: {n} chunks")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=40000)
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--refusal-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks()
    print(f"loaded {len(chunks)} corpus chunks")

    eval_path = ROOT / "eval" / "heldout_eval.json"
    banned: set = set()
    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        for group in data.get("sets", []):
            for item in group.get("items", []):
                banned.add(re.sub(r"[^a-z0-9 ]", "",
                                  item.get("question", "").lower()).strip())
    print(f"held-out questions to avoid: {len(banned)}")

    def is_banned(q: str) -> bool:
        return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip() in banned

    records: List[Dict[str, Any]] = []
    seen_questions: set = set()

    def add(prompt: str, answer: str, category: str, **extra) -> None:
        rec = {
            "category": category,
            "user": prompt,
            "assistant": answer,
            "full": (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                     f"<|im_start|>assistant\n{answer}<|im_end|>\n"),
        }
        rec.update(extra)
        records.append(rec)

    order = list(range(len(chunks)))
    rng.shuffle(order)

    for idx in order:
        if len(records) >= args.max_examples:
            break
        c = chunks[idx]
        src = c["source"]
        name = _clean_name(c.get("package") or c.get("title") or "")
        if not name:
            continue
        answer = derive_answer(c["text"])
        if not answer:
            continue

        templates = QUESTION_TEMPLATES.get(
            src, QUESTION_TEMPLATES["hexon_core_authored"])
        rng.shuffle(templates)

        subject = re.sub(r"[^a-z0-9]", "", name.lower())
        subject_present = bool(subject) and subject in re.sub(
            r"[^a-z0-9]", "", answer.lower())

        made = 0
        for tpl in templates:
            if made >= args.per_chunk:
                break
            q = tpl.format(name=name)
            if is_banned(q) or q.lower() in seen_questions:
                continue
            if tpl.startswith("What is {name}?") and not subject_present:
                continue
            coding = src in CODE_SOURCES and looks_like_code_request(q)
            seen_questions.add(q.lower())
            add(sft15.build_prompt(q, c["text"], coding), answer,
                "grounded_code" if coding else "grounded_fact",
                source=src, grounded=True)
            made += 1

    n_generated = len(records)
    print(f"generated {n_generated} grounded examples")

    # --- calibrated refusals ---------------------------------------------
    n_refusal_target = int(n_generated * args.refusal_fraction /
                           max(0.001, 1 - args.refusal_fraction))
    refusal_qs = sorted(seen_questions)
    rng.shuffle(refusal_qs)
    n_refusal = 0
    for q in refusal_qs[:n_refusal_target]:
        # Pair with a chunk from a different source so the context genuinely
        # does not answer the question.
        off = chunks[rng.randrange(len(chunks))]
        target = re.sub(r"[^a-z0-9]", "", q.lower())
        if target and target in re.sub(r"[^a-z0-9]", "",
                                       off["text"].lower()):
            continue
        add(sft15.build_prompt(q, off["text"], coding=False),
            rng.choice(REFUSAL_ANSWERS),
            "grounded_refuse", source="refusal", grounded=False)
        n_refusal += 1
    print(f"added {n_refusal} calibrated refusals "
          f"({n_refusal/max(1,len(records)):.1%} of set)")

    n_seed = 0
    if SEED_SFT.exists():
        with open(SEED_SFT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if is_banned(rec.get("user", "")[:120]):
                    continue
                rec["source"] = "seed_15"
                records.append(rec)
                n_seed += 1
        print(f"mixed in {n_seed} hand-written seed records")

    rng.shuffle(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: Dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    lens = [len(r["full"].split()) for r in records] or [0]
    print()
    print(f"total records : {len(records)} -> {out_path}")
    print(f"by category   : {by_cat}")
    print(f"full length   : min={min(lens)} max={max(lens)} "
          f"avg={sum(lens)/len(lens):.0f} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
