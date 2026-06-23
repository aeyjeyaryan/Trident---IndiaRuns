"""
Model loading and embedding generation wrapper.

Models are loaded once at FastAPI startup (via the lifespan) and passed into
services that need them. This avoids per-request model loading and keeps the
model state behind the service/repository abstraction.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

log = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: Any | None = None, model_name: str = "all-MiniLM-L6-v2"):
        self._model = model
        self._model_name = model_name

    @property
    def model(self):
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return [self._dummy_embedding() for _ in texts]
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            log.warning("Embedding failed (%s), using dummy embeddings", e)
            return [self._dummy_embedding() for _ in texts]

    async def embed_text(self, text: str) -> list[float]:
        embs = await self.embed_texts([text])
        return embs[0] if embs else self._dummy_embedding()

    def _dummy_embedding(self) -> list[float]:
        return [0.0] * 384
