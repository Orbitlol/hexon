#!/usr/bin/env python3
"""Bulk-mine a large knowledge corpus from the reachable sources.

The sandbox can only reach github.com, registry.npmjs.org and pypi.org, so
this discovers package names automatically instead of using a hand-written
list, then pulls and chunks the documentation for each one.

Sources:

1.  **PyPI** long descriptions. Names come from the hugovk/top-pypi-packages
    list via the GitHub API; falls back to a seed list if unreachable.
2.  **npm** READMEs, discovered through the registry search endpoint.
3.  **GitHub** repository READMEs via `gh api` (authenticated, so 5000/hr
    instead of the 60/hr anonymous limit). Skipped if `gh` is unavailable.

Two behaviours exist specifically because the sandbox re-clones from scratch
without warning and kills running jobs:

*   `--resume` appends to the output file and skips packages already present,
    so a reset costs the in-flight request rather than the whole run.
*   Every chunk is written and flushed the moment it is produced, so partial
    output survives a kill even without `--resume`.

    python3 scripts/mine_bulk.py --max-pypi 1500 --max-github 800 --resume
    python3 scripts/mine_bulk.py --skip-pypi --skip-npm --max-github 1200 \
        --general --out data/knowledge/hexon_mined_general.jsonl --resume
"""

from __future__ import annotations

import argparse
import base64
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
    "opencv-python", "scrapy", "selenium", "playwright", "paramiko",
    "fabric", "invoke", "marshmallow", "attrs", "cattrs", "orjson",
    "ujson", "msgpack", "protobuf", "grpcio", "websockets", "starlette",
    "sanic", "tornado", "pyramid", "bottle", "gunicorn", "supervisor",
    "docker", "kubernetes", "ansible", "moto", "responses", "vcrpy",
    "freezegun", "factory-boy", "faker", "sqlmodel", "peewee",
    "tortoise-orm", "asyncpg", "aiosqlite", "pymongo", "motor",
    "elasticsearch", "neo4j", "igraph", "python-louvain",
]

NPM_SEED = [
    "express", "react", "vue", "axios", "lodash", "webpack", "vite",
    "typescript", "eslint", "prettier", "jest", "mocha", "chai", "commander",
    "chalk", "mongoose", "socket.io", "tailwindcss", "redux", "mobx",
    "next", "nuxt", "svelte", "angular", "ember", "backbone", "jquery",
    "three", "d3", "chart.js", "moment", "dayjs", "date-fns", "uuid",
    "zod", "yup", "joi", "ajv", "immer", "rxjs", "apollo", "graphql",
    "prisma", "sequelize", "typeorm", "knex", "swr", "zustand", "jotai",
    "recoil", "formik", "react-hook-form", "styled-components", "emotion",
    "framer-motion", "gsap", "puppeteer", "playwright", "cypress",
    "vitest", "rollup", "esbuild", "parcel", "turbo", "nx", "lerna",
    "husky", "lint-staged", "commitlint", "semantic-release", "storybook",
    "electron", "tauri", "capacitor", "ionic", "react-native", "expo",
]

# Coding-oriented: find repositories with substantial READMEs.
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

# General-knowledge: the held-out benchmark is 74% non-coding, and package
# documentation does nothing for those questions, so the corpus needs a second
# pass aimed at expository and educational repositories.
GENERAL_QUERIES = [
    "stars:>1000 topic:education",
    "stars:>500 topic:textbook",
    "stars:>500 topic:study-notes",
    "stars:>500 topic:cheatsheet",
    "stars:>300 topic:knowledge-base",
    "stars:>300 topic:encyclopedia",
    "stars:>300 topic:geography",
    "stars:>300 topic:history",
    "stars:>500 topic:science",
    "stars:>300 topic:biology",
    "stars:>300 topic:physics",
    "stars:>300 topic:chemistry",
    "stars:>500 topic:interview-questions",
    "stars:>300 topic:algorithms",
    "stars:>300 topic:data-structures",
    "stars:>1000 topic:books",
    "stars:>300 topic:trivia",
    "stars:>200 topic:world-facts",
]


def _gh_json(args: List[str]) -> Optional[Any]:
    try:
        proc = subprocess.run(["gh", "api"] + args, capture_output=True,
                              text=True, timeout=60)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def discover_pypi(limit: int) -> List[str]:
    data = _gh_json([
        "repos/hugovk/top-pypi-packages/contents/"
        "top-pypi-packages-30-days.min.json",
    ])
    names: List[str] = []
    if isinstance(data, dict) and data.get("content"):
        try:
            raw = base64.b64decode(data["content"]).decode("utf-8", "replace")
            names = [r["project"] for r in json.loads(raw).get("rows", [])
                     if r.get("project")]
        except Exception as exc:
            print(f"  ! could not decode top-pypi list: {exc}")
    if names:
        print(f"  discovered {len(names)} PyPI packages")
        return names[:limit]
    print("  ! falling back to hardcoded PyPI seed list")
    return PYPI_SEED[:limit]


def discover_npm(limit: int) -> List[str]:
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


def discover_github(limit: int, general: bool = False) -> List[str]:
    repos: List[str] = []
    for q in (GENERAL_QUERIES if general else GITHUB_QUERIES):
        data = _gh_json([
            "search/repositories", "-X", "GET",
            "-f", f"q={q}", "-f", "sort=stars", "-f", "order=desc",
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
    data = _gh_json([f"repos/{full_name}/readme"])
    if not isinstance(data, dict) or not data.get("content"):
        return None
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
    ap.add_argument("--max-pypi", type=int, default=1500)
    ap.add_argument("--max-npm", type=int, default=700)
    ap.add_argument("--max-github", type=int, default=800)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-pypi", action="store_true")
    ap.add_argument("--skip-npm", action="store_true")
    ap.add_argument("--skip-github", action="store_true")
    ap.add_argument("--general", action="store_true",
                    help="Use educational/general-knowledge GitHub queries.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    if out_path.exists() and args.resume:
        n = 0
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except Exception:
                    continue
                seen.add((c.get("source"), c.get("package")))
                n += 1
        print(f"resuming: {n} chunks, {len(seen)} packages already done")

    fh = out_path.open("a" if args.resume else "w", encoding="utf-8")
    docs: List[Dict[str, Any]] = []

    def add(d: Optional[Dict[str, Any]]) -> bool:
        if not d or not d.get("text"):
            return False
        key = (d["source"], d["name"])
        if key in seen:
            return False
        seen.add(key)
        docs.append(d)
        # Flush immediately: a sandbox reset should cost one request, not
        # the entire run.
        try:
            for c in mk.build_chunks(d):
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            fh.flush()
        except Exception as exc:
            print(f"    ! chunk failed for {d['name']}: {exc}")
        return True

    if not args.skip_pypi:
        names = discover_pypi(args.max_pypi)
        print(f"fetching up to {len(names)} packages from PyPI...")
        n_ok = 0
        for i, pkg in enumerate(names, 1):
            if add(mk.fetch_pypi(pkg)):
                n_ok += 1
            if i % 100 == 0:
                print(f"    {i}/{len(names)} -> {n_ok} kept")
            time.sleep(0.05)
        print(f"  PyPI: {n_ok} docs with usable text")

    if not args.skip_npm:
        names = discover_npm(args.max_npm)
        print(f"fetching up to {len(names)} packages from npm...")
        n_ok = 0
        for i, pkg in enumerate(names, 1):
            if add(mk.fetch_npm(pkg)):
                n_ok += 1
            if i % 100 == 0:
                print(f"    {i}/{len(names)} -> {n_ok} kept")
            time.sleep(0.05)
        print(f"  npm: {n_ok} docs with usable text")

    if not args.skip_github:
        repos = discover_github(args.max_github, general=args.general)
        print(f"fetching up to {len(repos)} READMEs from GitHub...")
        n_ok = 0
        for i, full in enumerate(repos, 1):
            if add(fetch_github_readme(full)):
                n_ok += 1
            if i % 100 == 0:
                print(f"    {i}/{len(repos)} -> {n_ok} kept")
        print(f"  GitHub: {n_ok} docs with usable text")

    fh.close()

    chunks: List[Dict[str, Any]] = []
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    by_source: Dict[str, int] = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    wc = [c["word_count"] for c in chunks] or [0]
    print()
    print(f"docs fetched      : {len(docs)}")
    print(f"chunks kept       : {len(chunks)} -> {out_path}")
    print(f"by source         : {by_source}")
    print(f"total words       : {sum(wc):,}")
    print(f"approx text size  : {sum(wc) * 6 / 1e6:.1f} MB")
    print(f"chunk word count  : min={min(wc)} max={max(wc)} "
          f"avg={sum(wc)/len(wc):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
