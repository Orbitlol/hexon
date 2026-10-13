"""
Hexon ChromaDB Retrieval-Augmented Generation (RAG) Pipeline.

Provides an offline-capable, mobile-friendly RAG pipeline powered by ChromaDB:
1. `HexonChromaEmbeddingFunction`: 384-dimensional L2-normalized semantic + stem +
   character n-gram + concept-synonym embedding function implementing ChromaDB's
   `EmbeddingFunction` interface (zero external network calls required).
2. `HexonChromaRAG`: Persistent (`chromadb.PersistentClient`) or in-memory
   (`chromadb.EphemeralClient`) vector database manager that:
   - Chunks and indexes curated Common Crawl snapshots, Stanford Alpaca factual
     instruction records, and custom user documents into ChromaDB (`hnsw:space="cosine"`).
   - Retrieves top-k passages via ChromaDB HNSW vector search + lexical reranking.
   - Augments ChatML prompts with retrieved ChromaDB context for grounded generation
     with `HexonMobileEngine` (`hexon-135m` or `hexon-nano`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "as", "into", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "and", "or", "but", "if", "while", "because", "until", "that", "which",
    "who", "whom", "this", "these", "those", "am", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall", "what", "how", "why", "where", "when", "tell",
    "me", "please", "explain", "give", "describe", "you", "your", "i", "my",
}

SYNONYM_CANONICAL_MAP: Dict[str, str] = {
    "processor": "cpu",
    "chipset": "cpu",
    "snapdragon": "cpu",
    "memory": "ram",
    "battery": "power",
    "mah": "power",
    "smartphone": "mobile",
    "phone": "mobile",
    "android": "mobile",
    "poco": "mobile",
    "offline": "local",
    "internet": "network",
    "wifi": "network",
    "vector": "embedding",
    "vectors": "embedding",
    "embeddings": "embedding",
    "database": "db",
    "chromadb": "chroma",
    "retrieval": "rag",
    "retrieved": "rag",
    "plants": "plant",
    "leaves": "plant",
    "sunlight": "light",
    "chlorophyll": "photosynthesis",
    "dogs": "dog",
    "puppy": "dog",
    "canine": "dog",
    "cats": "cat",
    "kitten": "cat",
    "feline": "cat",
    "function": "code",
    "python": "code",
    "algorithm": "code",
}

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _stem_word(w: str) -> str:
    w = SYNONYM_CANONICAL_MAP.get(w, w)
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("tion"):
        w = w[:-4]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return SYNONYM_CANONICAL_MAP.get(w, w)


def extract_content_tokens(text: str) -> List[str]:
    raw_tokens = TOKEN_RE.findall(text.lower())
    stems = [_stem_word(t) for t in raw_tokens if t not in STOPWORDS and len(t) > 1]
    return stems if stems else [_stem_word(t) for t in raw_tokens if len(t) > 1]


class HexonChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    384-dimensional hybrid semantic embedding function for ChromaDB.
    Combines stemmed unigrams, bigrams, character 3/4-grams, and domain concept
    anchors via signed feature hashing with L2 unit normalization.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hexon_chroma_embedding_v1"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HexonChromaEmbeddingFunction":
        return HexonChromaEmbeddingFunction(dim=int(config.get("dim", 384)))

    @staticmethod
    def _hash_feature(feat: str, dim: int) -> Tuple[int, int, float]:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx1 = h % dim
        idx2 = ((h >> 16) % dim)
        sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
        return idx1, idx2, sign

    def _embed_single(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        stems = extract_content_tokens(text)
        if not stems:
            vec[0] = 1.0
            return vec.tolist()

        # 1. Stemmed content words (high weight: 3.0, plus sublinear TF)
        counts: Dict[str, int] = {}
        for s in stems:
            counts[s] = counts.get(s, 0) + 1

        for s, cnt in counts.items():
            tf_weight = 3.0 * (1.0 + math.log(cnt))
            i1, i2, sign = self._hash_feature(f"w:{s}", self.dim)
            vec[i1] += tf_weight * sign
            vec[i2] += (0.5 * tf_weight) * (-sign)

            # Character 3-grams and 4-grams of content words (weight: 0.65)
            padded = f"<{s}>"
            for n in (3, 4):
                if len(padded) >= n:
                    for j in range(len(padded) - n + 1):
                        ng = padded[j : j + n]
                        ni1, _, nsign = self._hash_feature(f"c{n}:{ng}", self.dim)
                        vec[ni1] += 0.65 * nsign

        # 2. Content word bigrams (weight: 1.5)
        for i in range(len(stems) - 1):
            bg = f"bg:{stems[i]}_{stems[i + 1]}"
            bi1, _, bsign = self._hash_feature(bg, self.dim)
            vec[bi1] += 1.5 * bsign

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_single(doc) for doc in input]


@dataclass
class RetrievedChunk:
    """A single document chunk retrieved from ChromaDB."""

    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CORE_KNOWLEDGE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "hexon-arch-01",
        "title": "Hexon Model Architecture & GGUF v3 Mobile Engine",
        "category": "hexon_architecture",
        "source": "hexon_docs",
        "text": (
            "Hexon is a compact, mobile-first Transformer language model engineered in PyTorch "
            "and exported to GGUF v3 (Q4_0, Q4_1, Q8_0, F16) and TorchScript (.pt). The flagship "
            "hexon-135m preset has 134.5 million parameters across 30 Transformer decoder layers "
            "with hidden_size=576, intermediate_size=1536, Grouped-Query Attention (9 query heads "
            "and 3 key-value heads for a 3:1 GQA ratio), RMSNorm pre-normalization, SwiGLU "
            "feed-forward networks, interleaved Rotary Position Embeddings (RoPE, theta=100000), "
            "and tied input/output token embeddings across a 49,152-token BPE vocabulary."
        ),
    },
    {
        "id": "hexon-mobile-lock-02",
        "title": "Hexon Mobile Inference-Only Lock (No On-Device Training)",
        "category": "hexon_mobile",
        "source": "hexon_docs",
        "text": (
            "Hexon never trains on mobile devices. All pre-training on curated Common Crawl "
            "snapshots and Supervised Fine-Tuning (SFT) happen offline on servers. Before "
            "exporting to GGUF or TorchScript, freeze_for_mobile_inference() permanently sets "
            "requires_grad=False on all parameters, disables loss computation, and blocks "
            ".train(True) so mobile devices like the Poco M3 only execute fast, read-only inference."
        ),
    },
    {
        "id": "hexon-chromadb-rag-03",
        "title": "Hexon ChromaDB Retrieval-Augmented Generation (RAG) Pipeline",
        "category": "hexon_rag",
        "source": "hexon_docs",
        "text": (
            "Hexon integrates a persistent ChromaDB vector database (HexonChromaRAG) using "
            "HNSW cosine similarity search and 384-dimensional hybrid semantic embeddings. "
            "Documents from curated Common Crawl snapshots, Stanford Alpaca encyclopedic "
            "knowledge, and user files are chunked and indexed in ChromaDB. At query time, "
            "Hexon retrieves the top-k relevant passages from ChromaDB and injects them into "
            "the ChatML prompt so the language model generates accurate, grounded answers."
        ),
    },
    {
        "id": "poco-m3-specs-04",
        "title": "Poco M3 Smartphone Hardware & Local GGUF Inference",
        "category": "mobile_hardware",
        "source": "hardware_specs",
        "text": (
            "The Xiaomi Poco M3 is an Android smartphone featuring a Qualcomm Snapdragon 662 "
            "octa-core ARM processor (4x Kryo 260 Gold at 2.0 GHz and 4x Kryo 260 Silver at "
            "1.8 GHz), an Adreno 610 GPU, 4 GB or 6 GB LPDDR4X RAM, and a 6000 mAh battery. "
            "It runs quantized GGUF v3 models (such as hexon-135m-q4_0.gguf at ~77 MB or "
            "hexon-nano-q4_0.gguf at 1.63 MB) completely offline via llama.cpp-compatible apps."
        ),
    },
    {
        "id": "science-photosynthesis-05",
        "title": "Photosynthesis and Plant Biology",
        "category": "science",
        "source": "encyclopedia",
        "text": (
            "Photosynthesis is the biochemical process in which green plants, algae, and "
            "cyanobacteria convert light energy from the sun, water (H2O), and carbon dioxide "
            "(CO2) into chemical energy stored in glucose (C6H12O6) while releasing oxygen (O2). "
            "It occurs inside chloroplasts using the green pigment chlorophyll."
        ),
    },
    {
        "id": "science-rayleigh-06",
        "title": "Rayleigh Scattering and Why the Sky Is Blue",
        "category": "science",
        "source": "encyclopedia",
        "text": (
            "Earth's sky appears blue because of Rayleigh scattering. Sunlight is composed of "
            "all visible wavelengths, and as it enters Earth's atmosphere, nitrogen and oxygen "
            "molecules scatter shorter wavelengths (blue and violet light, ~450 nm) in all "
            "directions much more strongly than longer wavelengths (red light, ~650 nm)."
        ),
    },
    {
        "id": "cs-python-algorithms-07",
        "title": "Python Algorithms: Prime Numbers, Binary Search, and Fibonacci",
        "category": "coding",
        "source": "coding_reference",
        "text": (
            "In Python, you can check if an integer n is prime in O(sqrt(n)) time by checking "
            "divisibility from 2 up to int(n ** 0.5) + 1. Binary search locates a target in a "
            "sorted list in O(log n) time by repeatedly halving the search interval [low, high]. "
            "A list can be reversed using slicing `items[::-1]` or `reversed(items)`."
        ),
    },
    {
        "id": "cs-chromadb-hnsw-08",
        "title": "How ChromaDB HNSW Vector Search Works",
        "category": "computer_science",
        "source": "database_docs",
        "text": (
            "ChromaDB is an open-source AI application database that stores dense vector "
            "embeddings, document text, and metadata inside collections. It indexes embeddings "
            "using Hierarchical Navigable Small World (HNSW) graphs to perform fast approximate "
            "nearest-neighbor (ANN) search using cosine, L2, or inner-product distance metrics."
        ),
    },
]


def chunk_text(text: str, chunk_size: int = 420, chunk_overlap: int = 64) -> List[str]:
    """Split long text into sentence-aware overlapping chunks for ChromaDB indexing."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip() if current else sent
        else:
            if current:
                chunks.append(current)
                overlap_tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = f"{overlap_tail} {sent}".strip()
            else:
                # Single very long sentence: slice by character window
                step = max(64, chunk_size - chunk_overlap)
                for idx in range(0, len(sent), step):
                    piece = sent[idx : idx + chunk_size].strip()
                    if len(piece) >= 30:
                        chunks.append(piece)
                current = ""

    if current and len(current) >= 25:
        chunks.append(current)
    return chunks


class HexonChromaRAG:
    """
    ChromaDB-backed Retrieval-Augmented Generation (RAG) manager for Hexon.
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = "data/chromadb",
        collection_name: str = "hexon_knowledge",
        embedding_dim: int = 384,
        auto_ingest_defaults: bool = True,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.collection_name = collection_name
        self.embedding_fn = HexonChromaEmbeddingFunction(dim=embedding_dim)

        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "description": "Hexon Mobile RAG Knowledge Base"},
        )

        if auto_ingest_defaults and self.collection.count() == 0:
            self.ingest_default_knowledge_base()

    def count(self) -> int:
        """Return total number of indexed chunks in the ChromaDB collection."""
        return int(self.collection.count())

    def add_documents(
        self,
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        chunk: bool = True,
        chunk_size: int = 420,
        chunk_overlap: int = 64,
    ) -> int:
        """
        Chunk and upsert documents into the ChromaDB collection.
        Returns the number of chunks indexed.
        """
        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for idx, doc_text in enumerate(documents):
            base_meta = dict(metadatas[idx]) if metadatas and idx < len(metadatas) else {}
            base_id = (
                str(ids[idx])
                if ids and idx < len(ids)
                else hashlib.sha1(doc_text.encode("utf-8")).hexdigest()[:12]
            )
            pieces = (
                chunk_text(doc_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if chunk
                else [doc_text.strip()]
            )
            for c_idx, piece in enumerate(pieces):
                if not piece:
                    continue
                cid = f"{base_id}-c{c_idx}" if len(pieces) > 1 else base_id
                meta = {
                    "source": str(base_meta.get("source", "user")),
                    "category": str(base_meta.get("category", "general")),
                    "title": str(base_meta.get("title", base_id)),
                    "chunk_index": int(c_idx),
                }
                for k, v in base_meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                batch_ids.append(cid)
                batch_docs.append(piece)
                batch_metas.append(meta)

        if not batch_docs:
            return 0

        # Upsert in batches of 128 for ChromaDB efficiency
        for start in range(0, len(batch_docs), 128):
            self.collection.upsert(
                ids=batch_ids[start : start + 128],
                documents=batch_docs[start : start + 128],
                metadatas=batch_metas[start : start + 128],
            )
        return len(batch_docs)

    def ingest_core_knowledge(self) -> int:
        """Ingest built-in Hexon architecture, RAG, hardware, science, and coding docs."""
        docs = [f"{item['title']}: {item['text']}" for item in CORE_KNOWLEDGE_DOCUMENTS]
        metas = [
            {
                "source": item["source"],
                "category": item["category"],
                "title": item["title"],
            }
            for item in CORE_KNOWLEDGE_DOCUMENTS
        ]
        ids = [item["id"] for item in CORE_KNOWLEDGE_DOCUMENTS]
        return self.add_documents(docs, metadatas=metas, ids=ids, chunk=False)

    def ingest_curated_common_crawl(
        self,
        jsonl_path: str | Path = "data/curated/common_crawl_curated.jsonl",
        max_docs: int = 80,
    ) -> int:
        """Ingest cleaned Common Crawl web documents into ChromaDB."""
        p = Path(jsonl_path)
        if not p.exists():
            return 0

        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                if len(text) < 80:
                    continue
                doc_id = str(row.get("doc_id", f"cc-{len(docs)}"))
                url = str(row.get("url", ""))
                first_line = text.split("\n", 1)[0][:90].strip()
                docs.append(text[:900])
                metas.append(
                    {
                        "source": str(row.get("source", "common_crawl")),
                        "category": "common_crawl",
                        "title": first_line,
                        "url": url,
                    }
                )
                ids.append(f"cc-{doc_id}")
                if len(docs) >= max_docs:
                    break

        return self.add_documents(docs, metadatas=metas, ids=ids, chunk=True, chunk_size=420)

    def ingest_alpaca_knowledge(
        self,
        jsonl_path: str | Path = "data/curated/alpaca_knowledge.jsonl",
        max_docs: int = 200,
    ) -> int:
        """Ingest curated Stanford Alpaca factual and technical QA records into ChromaDB."""
        p = Path(jsonl_path)
        if not p.exists():
            return 0

        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                inst = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                out = row.get("output", "").strip()
                if not inst or not out:
                    continue
                combined = f"{inst} {inp}\n{out}".strip() if inp else f"{inst}\n{out}"
                docs.append(combined)
                metas.append(
                    {
                        "source": "stanford_alpaca",
                        "category": str(row.get("category", "knowledge")),
                        "title": inst[:90],
                    }
                )
                ids.append(f"alpaca-{idx:04d}")
                if len(docs) >= max_docs:
                    break

        return self.add_documents(docs, metadatas=metas, ids=ids, chunk=False)

    def ingest_default_knowledge_base(self) -> Dict[str, int]:
        """Populate ChromaDB with Core Docs + Common Crawl + Alpaca Factual Knowledge."""
        c_core = self.ingest_core_knowledge()
        c_cc = self.ingest_curated_common_crawl()
        c_alpaca = self.ingest_alpaca_knowledge()
        return {
            "core_chunks": c_core,
            "common_crawl_chunks": c_cc,
            "alpaca_chunks": c_alpaca,
            "total_indexed": self.count(),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.15,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Query the ChromaDB HNSW vector index and apply hybrid semantic + lexical
        overlap reranking to return the top-k relevant document chunks.
        """
        total = self.count()
        if total == 0 or not query.strip():
            return []

        n_candidates = min(total, max(top_k * 4, 12))
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_candidates,
        }
        if where:
            query_kwargs["where"] = where

        raw = self.collection.query(**query_kwargs)
        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]

        query_stems = set(extract_content_tokens(query))
        candidates: List[RetrievedChunk] = []

        for cid, doc_text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            # Cosine distance in ChromaDB is (1 - cosine_similarity)
            cos_sim = max(0.0, 1.0 - float(dist))
            doc_stems = set(extract_content_tokens(doc_text))
            overlap = (
                len(query_stems & doc_stems) / max(1, len(query_stems))
                if query_stems
                else 0.0
            )
            hybrid_score = round(0.65 * cos_sim + 0.35 * overlap, 4)
            if hybrid_score >= min_score:
                candidates.append(
                    RetrievedChunk(
                        id=str(cid),
                        text=str(doc_text),
                        metadata=dict(meta) if meta else {},
                        distance=round(float(dist), 4),
                        score=hybrid_score,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def build_rag_prompt(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.18,
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Retrieve relevant passages from ChromaDB and construct a grounded RAG prompt.
        """
        chunks = self.retrieve(query, top_k=top_k, min_score=min_score)
        if not chunks:
            return query, []

        ctx_lines = ["Context from ChromaDB:"]
        for c in chunks:
            ctx_lines.append(f"- {c.text}")
        ctx_block = "\n".join(ctx_lines)
        augmented = (
            f"{ctx_block}\n\n"
            f"Based on the context above, answer the following question clearly and concisely:\n"
            f"{query}"
        )
        return augmented, chunks

    def query(
        self,
        question: str,
        engine: Any = None,
        top_k: int = 3,
        min_score: float = 0.18,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute the full ChromaDB RAG pipeline:
        1. Retrieve top-k chunks from ChromaDB.
        2. Build grounded prompt (or fall back to direct prompt if no chunks match).
        3. Generate response with `HexonMobileEngine` (if provided).
        """
        t0 = time.perf_counter()
        augmented_prompt, chunks = self.build_rag_prompt(
            question, top_k=top_k, min_score=min_score
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "question": question,
            "rag_enabled": True,
            "chromadb_collection": self.collection_name,
            "chromadb_total_chunks": self.count(),
            "retrieved_count": len(chunks),
            "retrieved_chunks": [c.to_dict() for c in chunks],
            "augmented_prompt": augmented_prompt,
            "retrieval_latency_ms": round(retrieval_ms, 2),
        }

        if engine is not None:
            # For 135M+ compositional models, pass the full RAG-augmented prompt;
            # for ultra-tiny models (<10M params), pass the question directly if needed
            param_count = getattr(engine.model.config, "vocab_size", 1024)
            prompt_to_send = augmented_prompt if param_count >= 16000 and chunks else question
            gen_out = engine.reply(
                prompt_to_send,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            result["response"] = gen_out["response"]
            result["tokens_generated"] = gen_out["tokens_generated"]
            result["generation_latency_ms"] = gen_out["latency_ms"]
            result["source_format"] = gen_out["source_format"]
            result["mobile_inference_locked"] = gen_out["mobile_inference_locked"]

        return result
