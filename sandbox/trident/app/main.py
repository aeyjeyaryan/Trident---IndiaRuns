"""
TRIDENT — Tri-Expert Decision Network for Talent Ranking
FastAPI service entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

try:
    import faiss  # noqa: F401
    _faiss_available = True
except ImportError:
    _faiss_available = False

try:
    from sentence_transformers import SentenceTransformer
    _st_available = True
except ImportError:
    _st_available = False

try:
    from sentence_transformers import CrossEncoder
    _ce_available = True
except ImportError:
    _ce_available = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import rank as rank_router
from app.core.config import Settings
from app.core.logging import setup_logging, get_logger
from app.domain.pipeline import RankingPipeline
from app.domain.gate.fixed_weight_gate import FixedWeightGate
from app.domain.gate.learned_gate import LearnedGate
from app.domain.retrieval.vector_index import FAISSIndex
from app.domain.rerank.cross_encoder import CrossEncoderReranker
from app.domain.explanation.explainer import Explainer
from app.infra.embeddings import EmbeddingService
from app.infra.persistence import InMemoryCandidateRepository

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("TRIDENT service starting up")

    config = Settings()

    # Load embedding model
    embedding_model = None
    cross_encoder_model = None
    if _st_available:
        try:
            log.info("Loading embedding model: %s", config.embedding_model_name)
            embedding_model = SentenceTransformer(config.embedding_model_name)
            log.info("Loading cross-encoder model: %s", config.cross_encoder_model_name)
            cross_encoder_model = CrossEncoder(config.cross_encoder_model_name)
        except Exception as e:
            log.warning("Failed to load models (%s), running with fallbacks", e)
    else:
        log.warning("sentence-transformers not available; using fallback embeddings")

    # Infrastructure
    candidate_repo = InMemoryCandidateRepository()
    embedding_service = EmbeddingService(
        model=embedding_model,
        model_name=config.embedding_model_name,
    )
    vector_index = FAISSIndex(
        dim=config.embedding_dim,
        index_path=config.index_path,
        metadata_path=config.metadata_path,
    )
    cross_encoder = CrossEncoderReranker(
        model=cross_encoder_model,
        model_name=config.cross_encoder_model_name,
    )

    # Gate
    learned_gate = LearnedGate(config=config)
    gate = learned_gate if learned_gate.is_trained else FixedWeightGate(weights=config.gate_fallback_weights)
    log.info("Using gate: %s", "learned" if learned_gate.is_trained else "fixed-weight fallback")

    # LLM client for explanations
    llm_client = None
    if config.use_llm_explanations and config.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.gemini_api_key)
            llm_client = genai.GenerativeModel("gemini-1.5-flash")
            log.info("Gemini LLM client configured for explanations")
        except Exception as e:
            log.warning("Failed to configure Gemini client (%s)", e)

    explainer = Explainer(llm_client=llm_client)

    # Pipeline
    pipeline = RankingPipeline(
        config=config,
        candidate_repo=candidate_repo,
        embedding_service=embedding_service,
        vector_index=vector_index,
        cross_encoder=cross_encoder,
        explainer=explainer,
        gate=gate,
    )
    rank_router.set_pipeline(pipeline)

    yield

    log.info("TRIDENT service shutting down")


app = FastAPI(
    title="TRIDENT — Talent Ranking API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rank_router.router, prefix="/v1")
