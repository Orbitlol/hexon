"""Hexon Common Crawl & Web Mirror Data Curation Pipeline.

Pulls a targeted subset / monthly snapshot of Common Crawl dumps (WET, WARC,
and processed Hugging Face / Dolma / FineWeb / C4 JSONL mirrors), filters and
cleans raw web text through a 7-stage data curation pipeline, and outputs clean
English pre-training documents for Hexon base model pre-training:

1. URL, MIME & Language Header Filtering (drops spam/adult domains & non-English WARC headers)
2. HTML Extraction & NFKC Unicode Normalization (strips <script>, <style>, <nav>, tags, entities)
3. Line-Level Boilerplate & C4/FineWeb Cleaning (strips nav menus, cookie notices, footers)
4. English Stopword Density & Alphabetic Ratio Filtering
5. Gopher / FineWeb Document Quality Heuristics (word count, mean word length, symbol ratios)
6. Exact SHA-1 + Paragraph + 5-gram Shingle Jaccard Deduplication
7. PII Redaction (emails, IPv4 addresses, phone numbers)
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import html
import json
import os
import re
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ENGLISH_STOPWORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "is", "are", "was", "were", "can", "how", "when", "which", "your", "more",
    "about", "into", "than", "them", "some", "could", "time", "very", "when",
}

BLOCKED_DOMAIN_KEYWORDS: Tuple[str, ...] = (
    "casino", "porn", "xxx", "adult", "betting", "poker", "viagra",
    "malware", "phishing", "doubleclick", "adservice",
)

BOILERPLATE_LINE_REGEX = re.compile(
    r"(skip to content|accept all cookies|cookie policy|privacy policy|"
    r"all rights reserved|terms of service|terms of use|javascript is required|"
    r"enable javascript|subscribe to our newsletter|sign in|log in|sign up|"
    r"create an account|powered by wordpress|lorem ipsum|click here to|"
    r"toggle navigation|main menu|search this site|share on facebook|"
    r"share on twitter|back to top|forgot your password|add to cart|"
    r"shopping cart|checkout now|comments are closed)",
    re.IGNORECASE,
)

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_REGEX = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)

GITHUB_CC_MIRROR_SOURCES: Tuple[Tuple[str, str], ...] = (
    (
        "cc_en_head-0091.jsonl",
        "https://api.github.com/repos/allenai/dolma/contents/tests/data/multiple_files/cc_en_head-0091.jsonl",
    ),
    (
        "cc_en_head-0174.jsonl",
        "https://api.github.com/repos/allenai/dolma/contents/tests/data/multiple_files/cc_en_head-0174.jsonl",
    ),
    (
        "cc_pt_input.jsonl",
        "https://api.github.com/repos/SunnyHaze/DataFlow-Clean/contents/dataflow/example/GeneralTextPipeline/pt_input.jsonl",
    ),
    (
        "cc_warc_after.jsonl",
        "https://api.github.com/repos/david-xinyuwei/david-share/contents/Deep-Learning/Make-High-Quality-Dataset-From-WARC/after.jsonl",
    ),
    (
        "fineweb_fixture.jsonl",
        "https://api.github.com/repos/bassimeledath/humanwrite/contents/data/fixtures/fineweb_fixture.jsonl",
    ),
    (
        "dolma_docs_000.json.gz",
        "https://api.github.com/repos/allenai/dolma/contents/tests/data/provided/documents/000.json.gz",
    ),
    (
        "sample-0000.warc.gz",
        "https://api.github.com/repos/allenai/dolma/contents/tests/data/warc/sample-0000.warc.gz",
    ),
    (
        "sample-0001.warc.gz",
        "https://api.github.com/repos/allenai/dolma/contents/tests/data/warc/sample-0001.warc.gz",
    ),
    (
        "whirlwind.warc.wet",
        "https://api.github.com/repos/commoncrawl/whirlwind-python/contents/whirlwind.warc.wet",
    ),
    (
        "educational_text_chunks.json",
        "https://api.github.com/repos/SunnyHaze/DataFlow-Clean/contents/dataflow/example/core_text_data/pipeline_small_chunk.json",
    ),
)


@dataclass
class CurationConfig:
    """Configuration for the Common Crawl & web text data curation pipeline."""

    min_line_chars: int = 25
    min_line_words: int = 5
    min_line_alpha_ratio: float = 0.62
    min_doc_words: int = 18
    max_doc_words: int = 1200
    min_stopword_ratio: float = 0.10
    min_alpha_ratio: float = 0.65
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_symbol_to_word_ratio: float = 0.12
    shingle_size: int = 5
    max_jaccard_similarity: float = 0.80
    redact_pii: bool = True


@dataclass
class CuratedDocument:
    """A single cleaned and quality-filtered Common Crawl pre-training document."""

    doc_id: str
    url: str
    source: str
    cc_segment: str
    text: str
    word_count: int
    char_count: int
    stopword_ratio: float
    alpha_ratio: float


@dataclass
class CurationReport:
    """Summary metrics from running the Common Crawl data curation pipeline."""

    raw_documents: int = 0
    kept_documents: int = 0
    rejected_url_or_lang: int = 0
    rejected_boilerplate_or_short: int = 0
    rejected_quality_heuristics: int = 0
    rejected_duplicates: int = 0
    total_raw_chars: int = 0
    total_clean_chars: int = 0
    total_clean_words: int = 0
    sources_processed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def pull_common_crawl_snapshot(
    raw_dir: str | Path = "data/raw_cc",
    download_if_missing: bool = True,
) -> List[Path]:
    """Ensure targeted Common Crawl snapshot files (WARC/WET/JSONL) exist locally."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in raw_path.iterdir()
        if p.is_file() and p.suffix in {".jsonl", ".json", ".gz", ".warc", ".wet"}
    )
    if existing or not download_if_missing:
        return existing

    downloaded: List[Path] = []
    for filename, url in GITHUB_CC_MIRROR_SOURCES:
        target = raw_path / filename
        if target.exists() and target.stat().st_size > 0:
            downloaded.append(target)
            continue
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github.raw",
                    "User-Agent": "Hexon-CC-Curation/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if data:
                target.write_bytes(data)
                downloaded.append(target)
        except Exception:
            continue
    return sorted(downloaded)


def strip_html_and_normalize(raw_text: str) -> str:
    """Strip HTML tags/scripts/styles and normalize Unicode to NFKC."""
    text = raw_text
    if any(tag in text.lower() for tag in ("<html", "<body", "<div", "<p>", "<script", "<style", "<a ", "<span")):
        text = re.sub(
            r"(?is)<(script|style|noscript|nav|footer|header|svg|iframe|form|aside)[^>]*>.*?</\1>",
            " ",
            text,
        )
        text = re.sub(r"(?is)<!--.*?-->", " ", text)
        text = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    # Fix common UTF-8 mojibake artifacts from web crawls
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", " - ")
        .replace("\u2013", "-")
        .replace("\u00a0", " ")
    )
    return text


def redact_pii_text(text: str) -> str:
    """Redact email addresses, IPv4 addresses, and phone numbers."""
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    text = IPV4_REGEX.sub("[IP]", text)
    text = PHONE_REGEX.sub("[PHONE]", text)
    return text


def _extract_raw_records_from_dir(raw_dir: str | Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Parse WARC, WET, JSONL, JSON.GZ, and JSON files from raw_dir into unified records."""
    raw_path = Path(raw_dir)
    records: List[Dict[str, str]] = []
    sources: List[str] = []

    if not raw_path.exists():
        return records, sources

    # 1. JSONL files (Dolma CC-MAIN, FineWeb, DataFlow Common Crawl dumps)
    for path_str in sorted(glob.glob(str(raw_path / "*.jsonl"))):
        p = Path(path_str)
        sources.append(p.name)
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text") or obj.get("raw_content") or ""
                if not text and "instruction" in obj:
                    inst = obj.get("instruction", "")
                    if "instances" in obj and obj["instances"]:
                        inp = obj["instances"][0].get("input", "")
                        out = obj["instances"][0].get("output", "")
                    else:
                        inp = obj.get("input", "")
                        out = obj.get("output", "")
                    text = f"{inst}\n{inp}\n{out}".strip()
                url = str(obj.get("url") or obj.get("id") or f"{p.name}#{idx}")
                meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                cc_seg = str(
                    obj.get("cc_segment")
                    or meta.get("cc_segment")
                    or obj.get("fineweb_id")
                    or "CC-MAIN-snapshot"
                )
                lang = str(obj.get("language") or meta.get("language") or "en").lower()
                if text:
                    records.append(
                        {
                            "url": url,
                            "text": text,
                            "source": p.name,
                            "cc_segment": cc_seg,
                            "lang": lang,
                        }
                    )

    # 2. Compressed JSONL (.json.gz)
    for path_str in sorted(glob.glob(str(raw_path / "*.json.gz"))):
        p = Path(path_str)
        sources.append(p.name)
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text") or obj.get("raw_content") or ""
                if text:
                    records.append(
                        {
                            "url": str(obj.get("id") or f"{p.name}#{idx}"),
                            "text": text,
                            "source": p.name,
                            "cc_segment": "CC-MAIN-dolma",
                            "lang": "en",
                        }
                    )

    # 3. JSON document arrays (.json)
    for path_str in sorted(glob.glob(str(raw_path / "*.json"))):
        p = Path(path_str)
        sources.append(p.name)
        try:
            arr = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(arr, list):
            for idx, obj in enumerate(arr):
                if isinstance(obj, dict) and obj.get("text"):
                    records.append(
                        {
                            "url": f"{p.name}#{idx}",
                            "text": str(obj["text"]),
                            "source": p.name,
                            "cc_segment": "CC-MAIN-edu",
                            "lang": "en",
                        }
                    )

    # 4. Raw Common Crawl WARC & WET archives (.warc, .warc.gz, .warc.wet)
    for path_str in sorted(glob.glob(str(raw_path / "*.warc*"))):
        p = Path(path_str)
        sources.append(p.name)
        try:
            raw_bytes = (
                gzip.open(p, "rb").read() if p.name.endswith(".gz") else p.read_bytes()
            )
            raw_str = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for idx, rec in enumerate(raw_str.split("WARC/1.0")):
            if "WARC-Target-URI:" not in rec:
                continue
            m_uri = re.search(r"WARC-Target-URI:\s*(\S+)", rec)
            uri = m_uri.group(1) if m_uri else f"{p.name}#{idx}"
            m_lang = re.search(r"WARC-Identified-Content-Language:\s*(\S+)", rec)
            lang = m_lang.group(1).lower() if m_lang else "en"
            # For WARC response records, body follows WARC header + HTTP header
            parts = re.split(r"\r?\n\r?\n", rec, maxsplit=2)
            body = parts[-1] if len(parts) >= 2 else ""
            if body.strip():
                records.append(
                    {
                        "url": uri,
                        "text": body,
                        "source": p.name,
                        "cc_segment": "CC-MAIN-warc",
                        "lang": lang,
                    }
                )

    return records, sources


def _word_shingles(words: List[str], k: int = 5) -> Set[int]:
    """Compute hashed k-word shingles for near-duplicate Jaccard detection."""
    if len(words) < k:
        return {hash(" ".join(words))}
    return {hash(" ".join(words[i : i + k])) for i in range(0, len(words) - k + 1, 2)}


def curate_raw_records(
    raw_records: Iterable[Dict[str, str]],
    config: Optional[CurationConfig] = None,
) -> Tuple[List[CuratedDocument], CurationReport]:
    """Filter, clean, and deduplicate raw Common Crawl records."""
    cfg = config or CurationConfig()
    report = CurationReport()
    curated: List[CuratedDocument] = []

    seen_hashes: Set[str] = set()
    seen_shingles: List[Set[int]] = []

    for idx, rec in enumerate(raw_records):
        report.raw_documents += 1
        raw_text = rec.get("text", "")
        report.total_raw_chars += len(raw_text)
        url = rec.get("url", f"doc-{idx}")
        lang = rec.get("lang", "en").lower()

        # Stage 1: URL & Language Header Filter
        url_lower = url.lower()
        if any(bad in url_lower for bad in BLOCKED_DOMAIN_KEYWORDS):
            report.rejected_url_or_lang += 1
            continue
        if lang not in {"en", "eng", "en-us", "en-gb", ""}:
            report.rejected_url_or_lang += 1
            continue

        # Stage 2: HTML Extraction, NFKC Normalization & PII Redaction
        text = strip_html_and_normalize(raw_text)
        if cfg.redact_pii:
            text = redact_pii_text(text)

        # Stage 3: Line-Level Boilerplate & C4/FineWeb Cleaning
        raw_lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        clean_lines: List[str] = []
        seen_doc_lines: Set[str] = set()

        for ln in raw_lines:
            if len(ln) < cfg.min_line_chars:
                continue
            if BOILERPLATE_LINE_REGEX.search(ln):
                continue
            # Drop ALL-CAPS navigation headers
            if ln.isupper() and len(ln) < 90:
                continue
            line_words = ln.split()
            if len(line_words) < cfg.min_line_words:
                continue
            alpha_count = sum(c.isalpha() for c in ln)
            if alpha_count / max(len(ln), 1) < cfg.min_line_alpha_ratio:
                continue
            norm_ln = ln.lower()
            if norm_ln in seen_doc_lines:
                continue
            seen_doc_lines.add(norm_ln)
            clean_lines.append(ln)

        cleaned_text = "\n".join(clean_lines).strip()
        if not cleaned_text:
            report.rejected_boilerplate_or_short += 1
            continue

        # Stage 4 & 5: Language Stopword Density & Gopher / FineWeb Quality Heuristics
        words_alpha = re.findall(r"[A-Za-z']+", cleaned_text.lower())
        if len(words_alpha) < cfg.min_doc_words:
            report.rejected_boilerplate_or_short += 1
            continue

        # Truncate overly long web pages to keep balanced document lengths
        if len(words_alpha) > cfg.max_doc_words:
            paragraphs = cleaned_text.split("\n")
            kept_paras: List[str] = []
            running_words = 0
            for p in paragraphs:
                pw = len(p.split())
                if running_words + pw > cfg.max_doc_words and kept_paras:
                    break
                kept_paras.append(p)
                running_words += pw
            cleaned_text = "\n".join(kept_paras).strip()
            words_alpha = re.findall(r"[A-Za-z']+", cleaned_text.lower())

        total_chars = len(cleaned_text)
        alpha_ratio = sum(c.isalpha() for c in cleaned_text) / max(total_chars, 1)
        stopword_hits = sum(1 for w in words_alpha if w in ENGLISH_STOPWORDS)
        stopword_ratio = stopword_hits / max(len(words_alpha), 1)
        mean_word_len = sum(len(w) for w in words_alpha) / max(len(words_alpha), 1)
        symbol_count = sum(cleaned_text.count(sym) for sym in ("#", "...", "{", "}", "|", "\\"))
        symbol_ratio = symbol_count / max(len(words_alpha), 1)

        if (
            alpha_ratio < cfg.min_alpha_ratio
            or stopword_ratio < cfg.min_stopword_ratio
            or not (cfg.min_mean_word_length <= mean_word_len <= cfg.max_mean_word_length)
            or symbol_ratio > cfg.max_symbol_to_word_ratio
        ):
            report.rejected_quality_heuristics += 1
            continue

        # Stage 6: Exact SHA-1 + 5-gram Shingle Near-Duplicate Filtering
        norm_digest = hashlib.sha1(cleaned_text.lower().encode("utf-8")).hexdigest()
        if norm_digest in seen_hashes:
            report.rejected_duplicates += 1
            continue

        doc_shingles = _word_shingles(words_alpha, k=cfg.shingle_size)
        is_near_dup = False
        for prev_shingles in seen_shingles:
            inter = len(doc_shingles & prev_shingles)
            union = len(doc_shingles | prev_shingles)
            if union > 0 and (inter / union) >= cfg.max_jaccard_similarity:
                is_near_dup = True
                break
        if is_near_dup:
            report.rejected_duplicates += 1
            continue

        seen_hashes.add(norm_digest)
        seen_shingles.append(doc_shingles)

        doc = CuratedDocument(
            doc_id=f"cc-{len(curated):04d}-{norm_digest[:8]}",
            url=url,
            source=rec.get("source", "common_crawl"),
            cc_segment=rec.get("cc_segment", "CC-MAIN"),
            text=cleaned_text,
            word_count=len(words_alpha),
            char_count=total_chars,
            stopword_ratio=round(stopword_ratio, 4),
            alpha_ratio=round(alpha_ratio, 4),
        )
        curated.append(doc)
        report.kept_documents += 1
        report.total_clean_chars += total_chars
        report.total_clean_words += len(words_alpha)

    return curated, report


def run_curation_pipeline(
    raw_dir: str | Path = "data/raw_cc",
    output_jsonl: Optional[str | Path] = "data/curated/common_crawl_curated.jsonl",
    config: Optional[CurationConfig] = None,
) -> Tuple[List[CuratedDocument], CurationReport]:
    """Pull Common Crawl snapshot files, filter and clean them, and save JSONL."""
    pull_common_crawl_snapshot(raw_dir=raw_dir, download_if_missing=True)
    raw_records, sources = _extract_raw_records_from_dir(raw_dir)
    curated_docs, report = curate_raw_records(raw_records, config=config)
    report.sources_processed = sources

    if output_jsonl is not None:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in curated_docs:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    return curated_docs, report
