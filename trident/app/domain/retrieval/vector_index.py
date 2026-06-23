"""
VectorIndex protocol + FAISS implementation for ANN retrieval.

The VectorIndex protocol defines the interface for approximate nearest-neighbor
search so it can be swapped for a hosted vector DB (Pinecone, Weaviate, Qdrant)
later without touching calling code.

The FAISS implementation builds a flat (brute-force) or IVF index from text
embeddings and supports top-K retrieval.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from app.domain.models import TextView

log = logging.getLogger(__name__)


class VectorIndex(Protocol):
    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        ...

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...

    @property
    def size(self) -> int:
        ...


class FAISSIndex:
    def __init__(self, dim: int = 384, index_path: str | None = None, metadata_path: str | None = None):
        self._dim = dim
        self._index: faiss.Index | None = None
        self._id_map: list[str] = []
        self._index_path = index_path
        self._metadata_path = metadata_path

    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        if self._index is None or self._index.ntotal == 0:
            return [], []
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        distances, indices = self._index.search(query, min(k, self._index.ntotal))
        results = []
        scores = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                results.append(self._id_map[int(idx)])
                # Convert L2 distance to cosine similarity score in [0,1]
                sim = 1.0 / (1.0 + float(distances[0][i]))
                scores.append(sim)
        return results, scores

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self._index is None:
            self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(embeddings)
        self._id_map.extend(ids)

    def save(self, path: str | None = None) -> None:
        p = path or self._index_path
        if p is None or self._index is None:
            return
        faiss.write_index(self._index, p)
        mp = self._metadata_path
        if mp:
            with open(mp, "w") as f:
                json.dump({"id_map": self._id_map}, f)
        log.info("Saved index with %d vectors to %s", self._index.ntotal, p)

    def load(self, path: str | None = None) -> None:
        p = path or self._index_path
        if p is None or not Path(p).exists():
            log.warning("Index file not found at %s", p)
            return
        self._index = faiss.read_index(p)
        mp = self._metadata_path
        if mp and Path(mp).exists():
            with open(mp) as f:
                data = json.load(f)
                self._id_map = data.get("id_map", [])
        log.info("Loaded index with %d vectors from %s", self._index.ntotal, p)

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
