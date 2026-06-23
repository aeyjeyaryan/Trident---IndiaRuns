"""
Rank candidates and produce submission CSV.
Must complete within 5 minutes on CPU only (pre-computed FAISS index required).
Usage: python rank.py [--candidates PATH] [--out PATH]
"""

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="TRIDENT ranking pipeline")
    parser.add_argument("--candidates", help="Path to candidates.jsonl (optional, uses config default)")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    cfg = Settings()
    if args.candidates:
        cfg.data_path = args.candidates
    cfg.retrieval_top_k = 400
    cfg.rerank_k = 100

    from sentence_transformers import SentenceTransformer, CrossEncoder
    emb_model = SentenceTransformer(cfg.embedding_model_name)
    ce_model = CrossEncoder(cfg.cross_encoder_model_name)

    repo = InMemoryCandidateRepository()
    emb = EmbeddingService(model=emb_model)
    idx = FAISSIndex(dim=cfg.embedding_dim, index_path=cfg.index_path, metadata_path=cfg.metadata_path)
    ce = CrossEncoderReranker(model=ce_model)
    expl = Explainer()
    gate = FixedWeightGate(weights=cfg.gate_fallback_weights)

    pipeline = RankingPipeline(
        config=cfg, candidate_repo=repo, embedding_service=emb,
        vector_index=idx, cross_encoder=ce, explainer=expl, gate=gate,
    )

    # Load index (must be pre-built by build_index.py)
    idx.load()
    if idx.size == 0:
        log.error("FAISS index is empty. Run build_index.py first.")
        sys.exit(1)

    # Load candidates
    t0 = time.monotonic()
    log.info("Loading candidates...")
    await repo.load_from_jsonl(cfg.data_path)
    log.info("Loaded %d candidates in %.1fs", repo.size, time.monotonic() - t0)

    job = JobPosting(
        job_id="submission",
        title="Senior AI Engineer — Founding Team",
        description=(
            "Senior AI Engineer at a Series A AI-native talent intelligence platform. "
            "Seeking someone with deep technical depth in modern ML systems — embeddings, "
            "retrieval, ranking, LLMs, fine-tuning — who also has a scrappy product-engineering "
            "attitude. Must have production experience with embeddings-based retrieval systems "
            "(sentence-transformers, BGE, E5) deployed to real users, vector databases or "
            "hybrid search infrastructure (FAISS, Pinecone, Weaviate, Qdrant, Milvus, "
            "Elasticsearch, OpenSearch), strong Python, and hands-on experience designing "
            "evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing). "
            "5-9 years experience preferred. Location: Pune/Noida (hybrid). "
            "Disqualifiers: pure research without production, consulting-only background, "
            "LangChain-only AI experience, no production code in 18 months."
        ),
        required_skills=[
            "Python", "PyTorch", "TensorFlow", "NLP", "Embeddings",
            "FAISS", "Machine Learning",
        ],
        preferred_skills=[
            "Vector Database", "Elasticsearch", "MLOps", "Kubernetes",
            "Fine-tuning", "LLM", "NDCG", "Information Retrieval",
        ],
        seniority_band="senior",
        role_family="data_science",
        location="Pune/Noida",
        country="India",
        min_years_experience=5,
        max_years_experience=9,
    )

    t0 = time.monotonic()
    log.info("Running ranking pipeline...")
    ranked = await pipeline.rank(job)
    elapsed = time.monotonic() - t0
    log.info("Pipeline completed in %.1fs with %d ranked candidates", elapsed, len(ranked))

    # Fill if short
    if len(ranked) < 100:
        log.warning("Only got %d candidates, filling with unranked...", len(ranked))
        all_cands = await repo.get_all()
        ranked_ids = {r.candidate_id for r in ranked}
        for c in all_cands:
            if len(ranked) >= 100:
                break
            if c.candidate_id not in ranked_ids:
                from types import SimpleNamespace
                ranked.append(SimpleNamespace(
                    candidate_id=c.candidate_id, score=0.0, rank=0,
                    reasoning=f"{c.current_title} with {c.years_of_experience:.1f} yrs"
                ))

    # Sort by score desc, then candidate_id asc for tie-breaking
    ranked_sorted = sorted(ranked[:100], key=lambda r: (-r.score, r.candidate_id))
    for i, r in enumerate(ranked_sorted):
        r.rank = i + 1

    # Write CSV
    output_path = Path(args.out)
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
        sys.exit(1)
    else:
        log.info("Submission is VALID!")


if __name__ == "__main__":
    asyncio.run(main())
