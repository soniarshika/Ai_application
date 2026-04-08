"""
ChromaDB-based vector store with hybrid retrieval pipeline.

Collection naming: every collection is stored as "{username}_{doc_id}".
This gives hard per-user isolation — users only ever see and query their
own collections. The doc_id exposed through the API is the UUID only;
the username prefix is purely internal.

Retrieval strategy:
  1. Dense search    — OpenAI text-embedding-3-small via ChromaDB
                       Run for: original query + expanded queries + HyDE passage
  2. BM25 search     — Keyword matching via rank_bm25
                       Run for: original query + expanded queries
  3. RRF fusion      — Reciprocal Rank Fusion merges all ranked lists into one
  4. Reranking       — text-embedding-3-large re-embeds query + RRF candidates at
                       higher fidelity (3072-dim vs 1536-dim) and reorders by
                       cosine similarity before passing to the LLM.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from rank_bm25 import BM25Okapi

log = logging.getLogger("logistics.retriever")

SIMILARITY_THRESHOLD = 0.25
DEFAULT_TOP_K        = 6
FULL_TEXT_LIMIT      = 20_000
RRF_K                = 60


def _to_similarities(distances: List[float]) -> np.ndarray:
    return np.clip(1.0 - np.array(distances, dtype=float), 0.0, 1.0)


def compute_confidence(similarities: np.ndarray) -> float:
    top_k = np.clip(similarities[:5], 0.0, 1.0)
    if len(top_k) == 0:
        return 0.0
    return round(float(np.clip(float(top_k[0]) * 0.6 + float(np.mean(top_k)) * 0.4, 0.0, 1.0)), 4)


# _DEFAULT_CHROMA_DIR = os.environ.get(
#     "CHROMA_DIR",
#     str(Path(__file__).parent.parent / "chroma_db"),
# )

_DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_db")

class DocumentRetriever:
    def __init__(self, persist_dir: str = _DEFAULT_CHROMA_DIR):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.embed_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._bm25_cache: Dict[str, dict] = {}  # col_name → {bm25, chunks}
        self._rerank_api_key = api_key

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _col(username: str, doc_id: str) -> str:
        """Internal collection name = username prefix + doc_id UUID."""
        return f"{username}_{doc_id}"

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    def index_document(
        self,
        doc_id: str,
        chunks: List[Dict],
        full_text: str,
        filename: str,
        page_count: int,
        username: str,
    ) -> int:
        col_name = self._col(username, doc_id)

        # Drop stale collection + BM25 cache
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass
        self._bm25_cache.pop(col_name, None)

        collection = self.client.create_collection(
            name=col_name,
            embedding_function=self.embed_fn,
            metadata={
                "hnsw:space":       "cosine",
                "username":         username,
                "doc_id":           doc_id,
                "filename":         filename,
                "chunk_count":      len(chunks),
                "page_count":       page_count,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "full_text":        full_text[:FULL_TEXT_LIMIT],
            },
        )

        collection.add(
            ids=[f"{col_name}_{c['chunk_index']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "chunk_index": c["chunk_index"],
                    "chunk_type":  c["chunk_type"],
                    "page_number": c["page_number"],
                }
                for c in chunks
            ],
        )
        return len(chunks)

    # ──────────────────────────────────────────────────────────────────────
    # Hybrid retrieval pipeline
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        doc_id: str,
        username: str,
        question: str,
        expanded_queries: Optional[List[str]] = None,
        hyde_text: Optional[str] = None,
        k: int = DEFAULT_TOP_K,
    ) -> Tuple[List[Dict], np.ndarray, bool]:
        """
        Hybrid retrieval: Dense + BM25 → RRF → rerank (text-embedding-3-large).
        Returns (chunks, rerank_similarities, guardrail_triggered).
        """
        col_name = self._col(username, doc_id)
        try:
            collection = self.client.get_collection(name=col_name, embedding_function=self.embed_fn)
        except Exception:
            return [], np.array([]), True

        n_total = collection.count()
        if n_total == 0:
            return [], np.array([]), True

        fetch_k = min(k * 4, n_total)

        # ── 1. Dense search ───────────────────────────────────────────────
        dense_queries = [question]
        if expanded_queries:
            dense_queries.extend(expanded_queries[:2])
        if hyde_text and hyde_text != question:
            dense_queries.append(hyde_text)

        sim_map: Dict[int, float] = {}
        dense_rankings: List[List[int]] = []

        for q in dense_queries:
            try:
                res   = collection.query(query_texts=[q], n_results=fetch_k,
                                         include=["metadatas", "distances"])
                metas = res["metadatas"][0]
                sims  = _to_similarities(res["distances"][0])
                ranking = []
                for meta, sim in zip(metas, sims):
                    ci = int(meta.get("chunk_index", 0))
                    sim_map[ci] = max(sim_map.get(ci, 0.0), float(sim))
                    ranking.append(ci)
                dense_rankings.append(ranking)
                log.debug(f"  Dense [{q[:50]}] → top_sim={float(sims[0]):.3f}  top={ranking[:5]}")
            except Exception as e:
                log.warning(f"  Dense query failed '{q[:40]}': {e}")

        best_dense_sim = max(sim_map.values()) if sim_map else 0.0

        # ── Guardrail ─────────────────────────────────────────────────────
        if best_dense_sim < SIMILARITY_THRESHOLD:
            log.info(f"  Guardrail: best_dense_sim={best_dense_sim:.3f} — blocked")
            return [], np.array([best_dense_sim]), True

        # ── 2. BM25 search ────────────────────────────────────────────────
        bm25_rankings: List[List[int]] = []
        try:
            bm25_data   = self._get_bm25_index(col_name)
            bm25_index  = bm25_data["bm25"]
            bm25_chunks = bm25_data["chunks"]

            for q in [question] + (expanded_queries or [])[:2]:
                scores     = bm25_index.get_scores(q.lower().split())
                sorted_pos = np.argsort(scores)[::-1][:fetch_k]
                ranking    = [bm25_chunks[int(p)]["chunk_index"]
                              for p in sorted_pos if float(scores[int(p)]) > 0]
                bm25_rankings.append(ranking)
                log.debug(f"  BM25  [{q[:50]}] → hits={len(ranking)}  top={ranking[:5]}")
        except Exception as e:
            log.warning(f"  BM25 failed: {e}")

        # ── 3. RRF fusion ─────────────────────────────────────────────────
        rrf_scores: Dict[int, float] = {}
        for ranking in dense_rankings + bm25_rankings:
            for rank, ci in enumerate(ranking):
                rrf_scores[ci] = rrf_scores.get(ci, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_indices = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:fetch_k]
        log.debug(f"  RRF → {len(top_indices)} candidates  top={top_indices[:8]}")

        idx_to_chunk = {c["chunk_index"]: c for c in bm25_data["chunks"]}
        candidates   = [idx_to_chunk[ci] for ci in top_indices if ci in idx_to_chunk]

        if not candidates:
            return [], np.array([best_dense_sim]), True

        # ── 4. Rerank ─────────────────────────────────────────────────────
        candidates, rerank_sims = self._rerank(question, candidates, top_k=k)

        log.info(f"  Done: {len(candidates)} chunks  best_dense={best_dense_sim:.3f}  "
                 f"top_rerank={float(rerank_sims[0]) if len(rerank_sims) else 0:.3f}")
        return candidates, rerank_sims, False

    # ──────────────────────────────────────────────────────────────────────
    # BM25 index (lazy, cached by col_name)
    # ──────────────────────────────────────────────────────────────────────

    def _get_bm25_index(self, col_name: str) -> dict:
        if col_name not in self._bm25_cache:
            col  = self.client.get_collection(name=col_name, embedding_function=self.embed_fn)
            data = col.get(include=["documents", "metadatas"])

            pairs = sorted(
                zip(data["documents"], data["metadatas"]),
                key=lambda x: int(x[1].get("chunk_index", 0)),
            )
            tokenized = [doc.lower().split() for doc, _ in pairs]
            bm25      = BM25Okapi(tokenized)
            chunks    = [
                {
                    "text":        doc,
                    "chunk_type":  meta.get("chunk_type", "narrative"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_index": int(meta.get("chunk_index", i)),
                }
                for i, (doc, meta) in enumerate(pairs)
            ]
            self._bm25_cache[col_name] = {"bm25": bm25, "chunks": chunks}
            log.debug(f"  BM25 index built for {col_name[:20]}… — {len(chunks)} docs")

        return self._bm25_cache[col_name]

    # ──────────────────────────────────────────────────────────────────────
    # Reranking via text-embedding-3-large
    # ──────────────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> Tuple[List[Dict], np.ndarray]:
        if len(candidates) <= top_k:
            sims = np.ones(len(candidates), dtype=float) * 0.5
            try:
                client = OpenAI(api_key=self._rerank_api_key)
                texts  = [query] + [c["text"] for c in candidates]
                resp   = client.embeddings.create(model="text-embedding-3-large", input=texts)
                vecs   = np.array([e.embedding for e in resp.data], dtype=float)
                sims   = np.clip(vecs[1:] @ vecs[0], 0.0, 1.0)
            except Exception as e:
                log.warning(f"  Reranking (small set) failed: {e}")
            return candidates, sims

        try:
            client = OpenAI(api_key=self._rerank_api_key)
            texts  = [query] + [c["text"] for c in candidates]
            resp   = client.embeddings.create(model="text-embedding-3-large", input=texts)
            vecs   = np.array([e.embedding for e in resp.data], dtype=float)
            scores = np.clip(vecs[1:] @ vecs[0], 0.0, 1.0)
            ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
            result   = [c for c, _ in ranked[:top_k]]
            result_s = np.array([s for _, s in ranked[:top_k]])
            log.debug(f"  Rerank (3-large): {len(candidates)} → {len(result)}  top_sim={float(result_s[0]):.3f}")
            return result, result_s
        except Exception as e:
            log.warning(f"  Reranking failed: {e} — RRF order")
            return candidates[:top_k], np.array([0.5] * min(top_k, len(candidates)))

    # ──────────────────────────────────────────────────────────────────────
    # Collection management
    # ──────────────────────────────────────────────────────────────────────

    def list_collections(self, username: str) -> List[Dict]:
        """Return all collections belonging to this user, newest first."""
        prefix = f"{username}_"
        result = []
        for ref in self.client.list_collections():
            if not ref.name.startswith(prefix):
                continue
            try:
                col  = self.client.get_collection(name=ref.name, embedding_function=self.embed_fn)
                meta = col.metadata or {}
                # doc_id exposed to API = UUID without the username prefix
                doc_id = meta.get("doc_id", ref.name[len(prefix):])
                result.append({
                    "doc_id":           doc_id,
                    "filename":         meta.get("filename", doc_id),
                    "chunk_count":      meta.get("chunk_count", col.count()),
                    "page_count":       meta.get("page_count", 0),
                    "upload_timestamp": meta.get("upload_timestamp"),
                })
            except Exception:
                pass
        result.sort(key=lambda x: x["upload_timestamp"] or "", reverse=True)
        return result

    def get_collection_info(self, doc_id: str, username: str) -> Optional[Dict]:
        col_name = self._col(username, doc_id)
        try:
            col = self.client.get_collection(name=col_name, embedding_function=self.embed_fn)
        except Exception:
            return None

        meta     = col.metadata or {}
        all_data = col.get(include=["documents", "metadatas"])

        chunks = []
        for doc, m in zip(all_data["documents"], all_data["metadatas"]):
            chunks.append({
                "chunk_index":  int(m.get("chunk_index", 0)),
                "chunk_type":   m.get("chunk_type", "narrative"),
                "page_number":  int(m.get("page_number", 1)),
                "text_preview": doc[:120] + ("…" if len(doc) > 120 else ""),
            })
        chunks.sort(key=lambda x: x["chunk_index"])

        return {
            "doc_id":           doc_id,
            "filename":         meta.get("filename", "unknown"),
            "chunk_count":      col.count(),
            "page_count":       meta.get("page_count", 0),
            "upload_timestamp": meta.get("upload_timestamp", ""),
            "chunks":           chunks,
        }

    def delete(self, doc_id: str, username: str) -> bool:
        """Delete a single document collection."""
        col_name = self._col(username, doc_id)
        try:
            self.client.delete_collection(col_name)
            self._bm25_cache.pop(col_name, None)
            log.info(f"  Deleted collection {col_name[:30]}…")
            return True
        except Exception:
            return False

    def delete_all(self, username: str) -> int:
        """
        Permanently delete ALL collections belonging to this user.
        Returns the number of collections deleted.
        """
        prefix  = f"{username}_"
        targets = [ref.name for ref in self.client.list_collections()
                   if ref.name.startswith(prefix)]
        count = 0
        for name in targets:
            try:
                self.client.delete_collection(name)
                self._bm25_cache.pop(name, None)
                count += 1
            except Exception as e:
                log.warning(f"  Failed to delete {name}: {e}")
        log.info(f"  Deleted all {count} collection(s) for user '{username}'")
        return count

    def exists(self, doc_id: str, username: str) -> bool:
        try:
            self.client.get_collection(name=self._col(username, doc_id),
                                       embedding_function=self.embed_fn)
            return True
        except Exception:
            return False

    def get_full_text(self, doc_id: str, username: str) -> Optional[str]:
        try:
            col = self.client.get_collection(name=self._col(username, doc_id),
                                             embedding_function=self.embed_fn)
            return col.metadata.get("full_text")
        except Exception:
            return None
