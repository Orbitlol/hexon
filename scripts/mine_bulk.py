#!/usr/bin/env python3
"""Bulk-mine a large knowledge corpus from the reachable sources.

The sandbox can only reach github.com, registry.npmjs.org and pypi.org, so
this script discovers package names automatically instead of using a
hand-written list, then pulls and chunks the documentation for each one.

Sources, in the order they are tried:

1.  **PyPI** long descriptions. Package names come from the
    `hugovk/top-pypi-packages` list (fetched through the GitHub API, which is
    reachable); if that fails we fall back to a hardcoded seed list.
2.  **npm** READMEs, discovered through the registry search endpoint.
3.  **GitHub** repository READMEs, discovered through `gh api` search. `gh` is
    authenticated, so this gets a 5000/hour rate limit instead of the 60/hour
    anonymous one. Skipped silently if `gh` is unavailable.

Cleaning, prose-density filtering and chunking are reused from
`mine_knowledge.py` so that everything in the corpus is chunked identically
(200-300 words).

    python3 scripts/mine_bulk.py --max-pypi 1200 --max-npm 600 --max-github 800
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mine_knowledge as mk  # noqa: E402

OUT = ROOT / "data" / "knowledge" / "hexon_mined_bulk.jsonl"

# Used only if the top-pypi-packages list cannot be fetched.
PYPI_SEED = [
    "requests", "flask", "fastapi", "django", "numpy", "pandas", "scipy",
    "matplotlib", "pytest", "click", "pydantic", "sqlalchemy", "celery",
    "pillow", "beautifulsoup4", "jinja2", "boto3", "uvicorn", "httpx",
    "rich", "typer", "black", "scikit-learn", "torch", "tqdm", "pyyaml",
    "python-dotenv", "alembic", "redis", "psycopg2-binary", "aiohttp",
    "tensorflow", "keras", "transformers", "tokenizers", "datasets",
    "nltk", "spacy", "gensim", "networkx", "sympy", "statsmodels",
    "seaborn", "plotly", "bokeh", "altair", "polars", "dask", "ray",
    "numba", "cython", "hypothesis", "mypy", "ruff", "flake8", "pylint",
    "isort", "tox", "nox", "poetry", "pipenv", "setuptools", "wheel",
    "twine", "sphinx", "mkdocs", "nbconvert", "jupyter", "ipykernel",
    "opencv-python", "imutils", "pytesseract", "scrapy", "selenium",
    "playwright", "paramiko", "fabric", "invoke", "marshmallow", "attrs",
    "cattrs", "orjson", "ujson", "msgpack", "protobuf", "grpcio",
    "websockets", "socketio", "starlette", "sanic", "tornado", "pyramid",
    "bottle", "cherrypy", "gunicorn", "supervisor", "docker", "kubernetes",
    "ansible", "salt", "terraform", "moto", "responses", "vcrpy",
    "freezegun", "factory-boy", "faker", "sqlmodel", "peewee", "pony",
    "tortoise-orm", "asyncpg", "aiosqlite", "pymongo", "motor", "elasticsearch",
    "opensearch-py", "neo4j", "py2neo", "networkx", "igraph", "python-louvain",
]

NPM_SEED = [
    "express", "react", "vue", "axios", "lodash", "webpack", "vite",
    "typescript", "eslint", "prettier", "jest", "mocha", "chai", "commander",
    "chalk", "mongoose", "socket.io", "tailwindcss", "redux", "mobx",
    "next", "nuxt", "svelte", "angular", "ember", "backbone", "jquery",
    "three", "d3", "chart.js", "moment", "dayjs", "date-fns", "uuid",
    "zod", "yup", "joi", "ajv", "immer", "rxjs", "apollo", "graphql",
    "prisma", "sequelize", "typeorm", "knex", "drizzle-orm", "swr",
    "react-query", "zustand", "jotai", "recoil", "formik", "react-hook-form",
    "styled-components", "emotion", "framer-motion", "gsap", "animejs",
    "puppeteer", "playwright", "cypress", "vitest", "rollup", "esbuild",
    "swc", "parcel", "turbo", "nx", "lerna", "husky", "lint-staged",
    "commitlint", "semantic-release", "storybook", "electron", "tauri",
    "capacitor", "cordova", "ionic", "react-native", "expo", "metro",
]

# Topics used to find GitHub repos with substantial READMEs.
GITHUB_QUERIES = [
    "stars:>20000",
    "stars:>8000 topic:documentation",
    "stars:>5000 topic:python",
    "stars:>5000 topic:javascript",
    "stars:>3000 topic:machine-learning",
    "stars:>3000 topic:database",
    "stars:>3000 topic:web",
    "stars:>2000 topic:devops",
    "stars:>2000 topic:security",
    "stars:>2000 topic:algorithms",
    "stars:>2000 topic:distributed-systems",
    "stars:>1000 topic:compilers",
]


def _gh_json(args: List[str]) -> Optional[Any]:
    """Run `gh api ...` and parse JSON. Returns None on any failure."""
    try:
        proc = subprocess.run(
            ["gh", "api"] + args,
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def discover_pypi(limit: int) -> List[str]:
    """Top PyPI packages by downloads, via the GitHub-hosted ranking."""
    data = _gh_json([
        "repos/hugovk/top-pypi-packages/contents/"
        "top-pypi-packages-30-days.min.json",
    ])
    names: List[str] = []
    if isinstance(data, dict) and data.get("content"):
        import base64
        try:
            raw = base64.b64decode(data["content"]).decode("utf-8", "replace")
            rows = json.loads(raw).get("rows", [])
            names = [r["project"] for r in rows if r.get("project")]
        except Exception as exc:
            print(f"  ! could not decode top-pypi list: {exc}")
    if names:
        print(f"  discovered {len(names)} PyPI packages (top-pypi-packages)")
        return names[:limit]
    print("  ! falling back to hardcoded PyPI seed list")
    return PYPI_SEED[:limit]


def discover_npm(limit: int) -> List[str]:
    """Popular npm packages via the registry search endpoint."""
    names: List[str] = []
    for query in ["", "keywords:framework", "keywords:library",
                  "keywords:tool", "keywords:cli", "keywords:database",
                  "keywords:api", "keywords:frontend", "keywords:backend"]:
        url = ("https://registry.npmjs.org/-/v1/search?text="
               f"{query or 'stars'}&size=250")
        data = mk._get_json(url, timeout=30.0)
        if not data:
            continue
        for obj in data.get("objects", []):
            pkg = (obj.get("package") or {}).get("name")
            if pkg and pkg not in names:
                names.append(pkg)
        time.sleep(0.2)
        if len(names) >= limit:
            break
    if names:
        print(f"  discovered {len(names)} npm packages")
        return names[:limit]
    print("  ! falling back to hardcoded npm seed list")
    return NPM_SEED[:limit]


def discover_github(limit: int) -> List[str]:
    """Top-starred repos, as 'owner/name' strings."""
    repos: List[str] = []
    for q in GITHUB_QUERIES:
        data = _gh_json([
            "search/repositories",
            "-X", "GET",
            "-f", f"q={q}",
            "-f", "sort=stars",
            "-f", "order=desc",
            "-f", "per_page=100",
        ])
        if not data or not isinstance(data.get("items"), list):
            continue
        for item in data["items"]:
            full = item.get("full_name")
            if full and full not in repos:
                repos.append(full)
        if len(repos) >= limit:
            break
    if repos:
        print(f"  discovered {len(repos)} GitHub repositories")
    else:
        print("  ! no GitHub repos discovered (gh unavailable or rate limited)")
    return repos[:limit]


def fetch_github_readme(full_name: str) -> Optional[Dict[str, Any]]:
    """Fetch a repo README through the authenticated GitHub API."""
    data = _gh_json([f"repos/{full_name}/readme"])
    if not isinstance(data, dict) or not data.get("content"):
        return None
    import base64
    try:
        body = base64.b64decode(data["content"]).decode("utf-8", "replace")
    except Exception:
        return None
    cleaned = mk.clean_markdown(body)
    if not cleaned:
        return None
    slug = re.sub(r"[^a-z0-9]", "", full_name.lower())
    return {
        "name": slug,
        "title": f"{full_name}: README",
        "text": cleaned,
        "category": "coding",
        "source": "mined_github",
        "raw_words": len(body.split()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pypi", type=int, default=1200)
    ap.add_argument("--max-npm", type=int, default=600)
    ap.add_argument("--max-github", type=int, default=800)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs: List[Dict[str, Any]] = []
    seen: set = set()

    def add(d: Optional[Dict[str, Any]]) -> bool:
        if not d or not d.get("text"):
            return False
        key = (d["source"], d["name"])
        if key in seen:
            return False
        seen.add(key)
        docs.append(d)
        return True

    # --- PyPI -------------------------------------------------------------
    pypi_names = discover_pypi(args.max_pypi)
    print(f"fetching up to {len(pypi_names)} packages from PyPI...")
    n_ok = 0
    for i, pkg in enumerate(pypi_names, 1):
        if add(mk.fetch_pypi(pkg)):
            n_ok += 1
        if i % 100 == 0:
            print(f"    {i}/{len(pypi_names)} -> {n_ok} kept")
        time.sleep(0.05)
    print(f"  PyPI: {n_ok} docs with usable text")

    # --- npm --------------------------------------------------------------
    npm_names = discover_npm(args.max_npm)
    print(f"fetching up to {len(npm_names)} packages from npm...")
    n_ok = 0
    for i, pkg in enumerate(npm_names, 1):
        if add(mk.fetch_npm(pkg)):
            n_ok += 1
        if i % 100 == 0:
            print(f"    {i}/{len(npm_names)} -> {n_ok} kept")
        time.sleep(0.05)
    print(f"  npm: {n_ok} docs with usable text")

    # --- GitHub -----------------------------------------------------------
    repos = discover_github(args.max_github)
    print(f"fetching up to {len(repos)} READMEs from GitHub...")
    n_ok = 0
    for i, full in enumerate(repos, 1):
        if add(fetch_github_readme(full)):
            n_ok += 1
        if i % 100 == 0:
            print(f"    {i}/{len(repos)} -> {n_ok} kept")
    print(f"  GitHub: {n_ok} docs with usable text")

    # --- chunk ------------------------------------------------------------
    print(f"chunking {len(docs)} documents...")
    chunks: List[Dict[str, Any]] = []
    for d in docs:
        try:
            chunks.extend(mk.build_chunks(d))
        except Exception as exc:
            print(f"    ! chunk failed for {d['name']}: {exc}")

    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_source: Dict[str, int] = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    wc = [c["word_count"] for c in chunks] or [0]
    total_words = sum(wc)
    print()
    print(f"docs fetched      : {len(docs)}")
    print(f"chunks kept       : {len(chunks)} -> {out_path}")
    print(f"by source         : {by_source}")
    print(f"total words       : {total_words:,}")
    print(f"approx text size  : {total_words * 6 / 1e6:.1f} MB")
    print(f"chunk word count  : min={min(wc)} max={max(wc)} "
          f"avg={sum(wc)/len(wc):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
