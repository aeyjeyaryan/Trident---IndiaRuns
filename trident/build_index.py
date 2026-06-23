"""
Pre-compute embeddings and build FAISS index.
Run once before rank.py. Can exceed 5 minutes.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.domain.retrieval.vector_index import FAISSIndex
from app.infra.persistence import InMemoryCandidateRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    cfg = Settings()

    log.info("Loading embedding model: %s", cfg.embedding_model_name)
    model = SentenceTransformer(cfg.embedding_model_name)

    repo = InMemoryCandidateRepository()
    import asyncio
    asyncio.run(repo.load_from_jsonl(cfg.data_path))
    candidates = asyncio.run(repo.get_all())
    log.info("Loaded %d candidates", len(candidates))

    short_texts = [
        f"{c.headline} {' '.join(s.name[:5] for s in c.skills)} {c.current_title} {c.current_company}"
        for c in candidates
    ]

    t0 = time.monotonic()
    log.info("Embedding %d texts...", len(short_texts))
    embeddings = model.encode(short_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
    log.info("Embedded %d texts in %.1fs", len(embeddings), time.monotonic() - t0)

    idx = FAISSIndex(dim=cfg.embedding_dim, index_path=cfg.index_path, metadata_path=cfg.metadata_path)
    emb_array = np.array(embeddings, dtype=np.float32)
    ids = [c.candidate_id for c in candidates]
    idx.add(emb_array, ids)
    idx.save()
    log.info("Index saved with %d vectors to %s", idx.size, cfg.index_path)


if __name__ == "__main__":
    main()
