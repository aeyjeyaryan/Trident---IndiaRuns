"""
Generate submission CSV by running the TRIDENT pipeline on the full dataset.
Optimized for faster embedding with shorter retrieval texts.
"""

import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import Settings
from app.domain.pipeline import RankingPipeline
from app.domain.gate.fixed_weight_gate import FixedWeightGate
from app.domain.retrieval.vector_index import FAISSIndex
from app.domain.rerank.cross_encoder import CrossEncoderReranker
from app.domain.explanation.explainer import Explainer
from app.infra.embeddings import EmbeddingService
from app.infra.persistence import InMemoryCandidateRepository
from app.domain.models import JobPosting
from app.domain.experts.career import CareerExpert

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


async def main():
    log.info("Loading models...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    cfg = Settings()
    cfg.retrieval_top_k = 400
    cfg.rerank_k = 100

    repo = InMemoryCandidateRepository()
    emb = EmbeddingService(model=emb_model)
    idx = FAISSIndex(dim=384, index_path=cfg.index_path, metadata_path=cfg.metadata_path)
    ce = CrossEncoderReranker(model=ce_model)
    expl = Explainer()
    gate = FixedWeightGate(weights=cfg.gate_fallback_weights)

    pipeline = RankingPipeline(
        config=cfg, candidate_repo=repo, embedding_service=emb,
        vector_index=idx, cross_encoder=ce, explainer=expl, gate=gate,
    )

    # Pre-load candidates into repo (pipeline will skip _stage_0)
    t0 = time.monotonic()
    log.info("Loading candidates...")
    await repo.load_from_jsonl(cfg.data_path)
    candidates = await repo.get_all()
    log.info("Loaded %d candidates in %.1fs", len(candidates), time.monotonic() - t0)

    # Try to load existing index, build if not found
    import numpy as np
    from pathlib import Path
    index_file = Path(cfg.index_path)
    if index_file.exists():
        log.info("Loading existing index from %s", cfg.index_path)
        idx.load()
    else:
        log.info("Building ANN index with short texts...")
        short_texts = [
            f"{c.headline} {' '.join(s.name[:5] for s in c.skills)} {c.current_title} {c.current_company}"
            for c in candidates
        ]
        t0 = time.monotonic()
        embeddings = emb_model.encode(short_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
        log.info("Embedded %d texts in %.1fs", len(embeddings), time.monotonic() - t0)

        emb_array = np.array(embeddings, dtype=np.float32)
        ids = [c.candidate_id for c in candidates]
        idx.add(emb_array, ids)
        idx.save()
        log.info("Index built with %d vectors", idx.size)

    # Run pipeline
    job = JobPosting(
        job_id="submission",
        title="AI/ML Engineer",
        description=(
            "We are looking for an AI/ML Engineer to design, build, and deploy "
            "machine learning models and AI-powered features. The ideal candidate has "
            "strong Python skills, experience with deep learning frameworks (PyTorch or "
            "TensorFlow), and hands-on experience with NLP, LLMs, or computer vision. "
            "You should be comfortable with the full ML lifecycle: data processing, model "
            "training, evaluation, deployment, and monitoring. Experience with MLOps, "
            "Kubernetes, and cloud platforms (AWS/GCP) is highly valued."
        ),
        required_skills=[
            "Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow",
            "NLP", "MLOps",
        ],
        preferred_skills=["Kubernetes", "AWS", "GCP", "Spark", "Computer Vision"],
        seniority_band="senior",
        role_family="data_science",
    )

    t0 = time.monotonic()
    log.info("Running ranking pipeline...")
    ranked = await pipeline.rank(job)
    elapsed = time.monotonic() - t0
    log.info("Pipeline completed in %.1fs with %d ranked candidates", elapsed, len(ranked))

    if len(ranked) < 100:
        log.warning("Only got %d candidates, filling with unranked...", len(ranked))
        ranked_ids = {r.candidate_id for r in ranked}
        for c in candidates:
            if len(ranked) >= 100:
                break
            if c.candidate_id not in ranked_ids:
                from types import SimpleNamespace
                ranked.append(SimpleNamespace(
                    candidate_id=c.candidate_id, score=0.0, rank=0,
                    reasoning=f"{c.current_title} with {c.years_of_experience:.1f} yrs"
                ))

    # Sort by score desc, then by candidate_id asc for tie-breaking
    ranked_sorted = sorted(ranked[:100], key=lambda r: (-r.score, r.candidate_id))
    for i, r in enumerate(ranked_sorted):
        r.rank = i + 1

    # Format reasoning consistently
    cand_map = {c.candidate_id: c for c in candidates}
    for r in ranked_sorted:
        c = cand_map.get(r.candidate_id)
        if c and not getattr(r, 'reasoning', None):
            r.reasoning = f"{c.current_title} with {c.years_of_experience:.1f} yrs"

    output_path = Path(__file__).parent / "submission.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in ranked_sorted:
            reasoning = getattr(r, 'reasoning', '') or ''
            writer.writerow([r.candidate_id, r.rank, r.score, reasoning])

    log.info("Submission written to %s (%d candidates)", output_path, len(ranked_sorted))

    # Validate
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset.validate_submission import validate_submission
    errors = validate_submission(str(output_path))
    if errors:
        log.error("Validation FAILED (%d errors):", len(errors))
        for e in errors:
            log.error("  - %s", e)
    else:
        log.info("Submission is VALID!")


if __name__ == "__main__":
    asyncio.run(main())
