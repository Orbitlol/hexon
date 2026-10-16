#!/usr/bin/env python3
"""
Mine the "breadth" half of the hexon-1.5 knowledge corpus.

The sandbox can only reach github.com, registry.npmjs.org and pypi.org, so
this script pulls package long descriptions from PyPI's JSON API and READMEs
from the npm registry, cleans the markdown and badge noise out of them, and
emits 200-300 word chunks ready for ChromaDB.

Mined text is noisier than the hand-authored corpus: it contains install
commands, badge SVG urls, licence boilerplate and contributor lists. The
cleaner below strips most of that and drops any chunk whose English prose
density is too low, so what survives is mostly explanatory prose. Anything
that survives is still tagged `source=mined_pypi` / `source=mined_npm` so it
can be traced or dropped later.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "knowledge" / "hexon_mined_docs.jsonl"

PYPI_PACKAGES = [
    "requests", "flask", "fastapi", "django", "numpy", "pandas", "scipy",
    "matplotlib", "pytest", "click", "pydantic", "sqlalchemy", "celery",
    "pillow", "beautifulsoup4", "jinja2", "boto3", "uvicorn", "httpx",
    "rich", "typer", "black", "scikit-learn", "torch", "tqdm", "pyyaml",
    "python-dotenv", "alembic", "redis", "psycopg2-binary",
]

NPM_PACKAGES = [
    "express", "react", "vue", "axios", "lodash", "webpack", "vite",
    "typescript", "eslint", "jest", "moment", "dayjs", "chalk", "commander",
    "zod", "redux", "mongoose", "socket.io", "prettier", "tailwindcss",
]

UA = {"User-Agent": "hexon-corpus-miner/1.0 (offline research build)"}

# --- cleaning -------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]{1,80}`")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]{1,80})\]\(([^)]+)\)")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BADGE_LINE = re.compile(
    r"^\s*\[?!?\[[^\]]*\]\s*\(?[^)]*\)?.*$", re.MULTILINE
)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_MULTI_NL = re.compile(r"\n{3,}")

_NOISE_SECTIONS = re.compile(
    r"(?is)\n\s*(#{1,6}\s*)?"
    r"(licen[cs]e|legal|contributing|code of conduct|changelog|release notes|"
    r"sponsors?|backers?|credits?|acknowledg|authors?|maintainers?|security policy|"
    r"badges?|table of contents|quick ?start|installation notes|donat)"
    r"\s*[:\n].*?(?=\n\s*#{1,6}\s|\Z)"
)

_ENGLISH_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "you", "your",
    "are", "was", "were", "have", "has", "not", "but", "can", "will",
    "all", "any", "use", "using", "used", "one", "two", "see", "how",
    "what", "when", "where", "which", "who", "why", "get", "set",
}


def clean_markdown(text: str) -> str:
    """Strip badges, code, links, images and boilerplate from README markdown."""
    if not text:
        return ""
    t = _HTML_COMMENT.sub(" ", text)
    t = _CODE_BLOCK.sub(" ", t)
    t = _NOISE_SECTIONS.sub(" ", t)
    t = _IMAGE.sub(" ", t)
    t = _HTML_TAG.sub(" ", t)
    t = _BADGE_LINE.sub(" ", t)
    t = _TABLE_ROW.sub(" ", t)
    t = _LINK.sub(r"\1", t)
    t = _INLINE_CODE.sub(lambda m: m.group(0).strip("`"), t)
    t = _HEADING.sub("", t)
    t = _BULLET.sub("", t)
    t = _URL.sub(" ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


_CHANGELOG_MARKERS = [
    re.compile(r"#\d{3,}"),                      # PR / issue numbers
    re.compile(r"(?<![\w.])@[A-Za-z][\w-]{2,}"),  # github handles
    re.compile(r"\bv?\d+\.\d+\.\d+\b"),          # semver strings
    re.compile(r"made their first contribution", re.I),
    re.compile(r"what'?s changed", re.I),
    re.compile(r"full changelog", re.I),
    re.compile(r"\bcompare\s+v?\d", re.I),
]


def changelog_score(text: str) -> int:
    """Count release-note artefacts in a chunk. High scores mean it is a
    changelog or contributor list, not reference prose."""
    return sum(len(p.findall(text)) for p in _CHANGELOG_MARKERS)


def is_reference_prose(text: str) -> bool:
    """Heuristics for 'this chunk explains something' vs 'this chunk is noise'."""
    words = text.split()
    if len(words) < 200:
        return False
    if changelog_score(text) >= 3:
        return False
    # Real explanation has sentences that end in full stops.
    sentences = re.findall(r"[A-Za-z][^.!?]{25,}[.!?]", text)
    if len(sentences) < 4:
        return False
    # A wall of identifiers or paths is documentation soup, not prose.
    words_alpha = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not words_alpha:
        return False
    weird = sum(
        1 for w in words_alpha
        if re.search(r"[_/\\]{1,}", w) or re.fullmatch(r"[A-Za-z]{1,3}\d+", w)
    )
    if weird / len(words_alpha) > 0.06:
        return False
    return True


def prose_density(text: str) -> float:
    """Fraction of words that are real English-ish content words (0..1)."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]{1,}", text)
    if len(words) < 120:
        return 0.0
    content = [
        w for w in words
        if w.lower() not in _ENGLISH_STOP and len(w) > 2
    ]
    return len(content) / max(1, len(words))


# --- fetching -------------------------------------------------------------


def _get_json(url: str, timeout: float = 20.0) -> Optional[Any]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception as exc:
        print(f"    ! {url} -> {exc}", file=sys.stderr)
        return None


def fetch_pypi(pkg: str) -> Optional[Dict[str, Any]]:
    data = _get_json(f"https://pypi.org/pypi/{pkg}/json")
    if not data or not isinstance(data, dict):
        return None
    info = data.get("info") or {}
    body = info.get("description") or ""
    summary = info.get("summary") or ""
    return {
        "name": pkg,
        "title": f"{info.get('name', pkg)}: {summary}".strip(": ").strip(),
        "text": clean_markdown(body),
        "category": "coding",
        "source": "mined_pypi",
        "raw_words": len((summary + " " + body).split()),
    }


def fetch_npm(pkg: str) -> Optional[Dict[str, Any]]:
    data = _get_json(f"https://registry.npmjs.org/{pkg}")
    if not data or not isinstance(data, dict):
        return None
    readme = data.get("readme") or ""
    desc = data.get("description") or ""
    latest = data.get("dist-tags", {}).get("latest") if isinstance(data.get("dist-tags"), dict) else None
    return {
        "name": pkg,
        "title": f"{pkg}: {desc}".strip(": ").strip(),
        "text": clean_markdown(readme),
        "category": "coding",
        "source": "mined_npm",
        "raw_words": len((desc + " " + readme).split()),
    }


# --- chunking -------------------------------------------------------------


def build_chunks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from hexon.rag import chunk_text

    text = doc["text"]
    if not text:
        return []
    pieces = chunk_text(text, min_words=200, max_words=300, overlap_words=35)
    out = []
    for i, piece in enumerate(pieces):
        w = len(piece.split())
        if not (200 <= w <= 300):
            continue
        if prose_density(piece) < 0.45:
            continue
        if not is_reference_prose(piece):
            continue
        out.append(
            {
                "id": f"mined-{doc['source']}-{re.sub(r'[^a-z0-9]', '', doc['name'].lower())}-{i}",
                "title": doc["title"],
                "category": doc["category"],
                "source": doc["source"],
                "package": doc["name"],
                "text": piece,
                "word_count": w,
            }
        )
    return out


def main() -> int:
    docs: List[Dict[str, Any]] = []

    print(f"fetching {len(PYPI_PACKAGES)} packages from PyPI...")
    for pkg in PYPI_PACKAGES:
        d = fetch_pypi(pkg)
        if d and d["text"]:
            docs.append(d)
            print(f"  + {pkg:<20} {d['raw_words']:>6} raw words")
        time.sleep(0.1)

    print(f"fetching {len(NPM_PACKAGES)} packages from npm...")
    for pkg in NPM_PACKAGES:
        d = fetch_npm(pkg)
        if d and d["text"]:
            docs.append(d)
            print(f"  + {pkg:<20} {d['raw_words']:>6} raw words")
        time.sleep(0.1)

    chunks: List[Dict[str, Any]] = []
    for d in docs:
        chunks.extend(build_chunks(d))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_source: Dict[str, int] = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    wc = [c["word_count"] for c in chunks] or [0]
    print()
    print(f"docs fetched     : {len(docs)}")
    print(f"chunks kept      : {len(chunks)} -> {OUT.relative_to(ROOT)}")
    print(f"by source        : {by_source}")
    print(f"chunk word count : min={min(wc)} max={max(wc)} avg={sum(wc)/len(wc):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
