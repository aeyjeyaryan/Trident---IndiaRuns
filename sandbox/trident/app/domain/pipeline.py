"""
TRIDENT Pipeline — Orchestrates stages 0–5 of the ranking process.

This is the only place that calls all of the above. Routes contain no business
logic; they delegate entirely to this pipeline.

Pipeline stages:
  0. Ingestion/normalization — parse raw candidates into three views
  1. Coarse retrieval — ANN search narrows pool to top-K
  2. Expert scoring — three experts score in parallel
  3. Gate fusion — mixture-of-rankers combines scores
  4. Rerank + diversity — cross-encoder + MMR
  5. Explanation — structured breakdown + NL explanation
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Any

from app.core.config import Settings
from app.core.logging import StageLogger
from app.domain.models import (
    JobPosting,
    CandidateProfile,
    TextView,
    CareerView,
    BehavioralView,
    RoleTuple,
    ExpertScore,
    FusedScore,
    RankedCandidate,
)
from app.domain.experts.base import Expert
from app.domain.experts.semantic import SemanticExpert
from app.domain.experts.career import CareerExpert
from app.domain.experts.behavioral import BehavioralExpert
from app.domain.gate.fixed_weight_gate import FixedWeightGate
from app.domain.gate.learned_gate import LearnedGate
from app.domain.retrieval.vector_index import FAISSIndex
from app.domain.rerank.cross_encoder import CrossEncoderReranker
from app.domain.rerank.mmr import MMRDiversityReranker
from app.domain.explanation.explainer import Explainer
from app.infra.embeddings import EmbeddingService
from app.infra.persistence import CandidateRepository

log = logging.getLogger(__name__)


class RankingPipeline:
    def __init__(
        self,
        config: Settings,
        candidate_repo: CandidateRepository,
        embedding_service: EmbeddingService,
        vector_index: FAISSIndex,
        cross_encoder: CrossEncoderReranker,
        explainer: Explainer,
        gate: FixedWeightGate | LearnedGate | None = None,
    ):
        self._config = config
        self._candidate_repo = candidate_repo
        self._embedding_service = embedding_service
        self._vector_index = vector_index
        self._cross_encoder = cross_encoder
        self._explainer = explainer

        # Experts
        self._experts: list[Expert] = [
            SemanticExpert(temperature=config.semantic_temperature),
            CareerExpert(config=config),
            BehavioralExpert(config=config),
        ]

        # Gate
        if gate is not None:
            self._gate = gate
        else:
            try:
                self._gate = LearnedGate(config=config)
                if not self._gate.is_trained:
                    log.info("LearnedGate not trained; using FixedWeightGate fallback")
                    self._gate = FixedWeightGate(weights=config.gate_fallback_weights)
            except Exception:
                self._gate = FixedWeightGate(weights=config.gate_fallback_weights)

        self._mmr = MMRDiversityReranker(lambda_=config.mmr_lambda)

    async def rank(
        self,
        job: JobPosting,
        candidate_ids: list[str] | None = None,
        request_id: str = "",
    ) -> list[RankedCandidate]:
        slog = StageLogger(log, "pipeline", request_id)
        slog.start({"job_title": job.title})

        # Stage 0: Load & normalize
        slog.log("Stage 0: loading and normalizing candidates")
        await self._stage_0_load_data(job)
        candidates = await self._candidate_repo.get_all()
        if candidate_ids:
            candidates = [c for c in candidates if c.candidate_id in candidate_ids]
        slog.end({"candidates_in_pool": len(candidates)})

        # Stage 1: Coarse retrieval
        slog.start({"stage": "stage_1_retrieval"})
        shortlist_ids, _ = await self._stage_1_retrieval(job, candidates)
        slog.end({"stage": "stage_1_retrieval", "shortlist_size": len(shortlist_ids)})

        shortlist = [c for c in candidates if c.candidate_id in shortlist_ids]
        if not shortlist:
            shortlist = candidates[: self._config.retrieval_top_k]

        # Stage 2: Build three views + score
        slog.start({"stage": "stage_2_views"})
        text_views, career_views, behavioral_views = self._build_views(job, shortlist)
        slog.end({"stage": "stage_2_views", "n_candidates": len(shortlist)})

        # Embed job description
        job_embedding = np.array(
            await self._embedding_service.embed_text(job.description), dtype=np.float32
        )

        # Score shortlist text embeddings (batch)
        texts_to_embed = [(i, tv) for i, tv in enumerate(text_views) if tv.embedding is None]
        if texts_to_embed:
            raw_texts = [tv.text for _, tv in texts_to_embed]
            batch_embeddings = await self._embedding_service.embed_texts(raw_texts)
            for (i, tv), emb in zip(texts_to_embed, batch_embeddings):
                tv.embedding = emb

        # Run experts in parallel
        slog.start({"stage": "stage_2_experts"})
        expert_results = await asyncio.gather(
            self._experts[0].score(job, text_views, job_embedding=job_embedding),
            self._experts[1].score(job, career_views),
            self._experts[2].score(job, behavioral_views),
        )
        slog.end({"stage": "stage_2_experts"})

        sem_scores: list[ExpertScore] = expert_results[0]
        car_scores: list[ExpertScore] = expert_results[1]
        beh_scores: list[ExpertScore] = expert_results[2]

        # Build lookup dicts
        sem_map = {s.candidate_id: s for s in sem_scores}
        car_map = {s.candidate_id: s for s in car_scores}
        beh_map = {s.candidate_id: s for s in beh_scores}

        # Stage 3: Gate fusion
        slog.start({"stage": "stage_3_gate"})
        fused_scores = self._stage_3_fusion(job, sem_map, car_map, beh_map, shortlist)
        slog.end({"stage": "stage_3_gate", "n_fused": len(fused_scores)})

        # Sort by fused score descending, take top-K for rerank
        fused_scores.sort(key=lambda x: x.fused_score, reverse=True)
        stage3_top = fused_scores[: self._config.rerank_k]

        # Stage 4: Rerank + diversity
        slog.start({"stage": "stage_4_rerank"})
        reranked = await self._stage_4_rerank(job, stage3_top, text_views)
        slog.end({"stage": "stage_4_rerank", "n_reranked": len(reranked)})

        # Stage 5: Explanation
        slog.start({"stage": "stage_5_explanation"})
        ranked = self._stage_5_explain(reranked, job, shortlist)
        slog.end({"stage": "stage_5_explanation", "n_ranked": len(ranked)})

        slog.end({"total_ranked": len(ranked)})
        return ranked

    # ─── Stage implementations ──────────────────────────────────────────

    async def _stage_0_load_data(self, job: JobPosting) -> None:
        if self._candidate_repo.size == 0:
            await self._candidate_repo.load_from_jsonl(self._config.data_path)

    async def _stage_1_retrieval(
        self,
        job: JobPosting,
        candidates: list[CandidateProfile],
    ) -> tuple[list[str], list[float]]:
        if self._vector_index.size == 0:
            texts = [
                f"{c.summary} {c.headline} {' '.join(s.name for s in c.skills)}"
                for c in candidates
            ]
            embeddings = await self._embedding_service.embed_texts(texts)
            emb_array = np.array(embeddings, dtype=np.float32)
            ids = [c.candidate_id for c in candidates]
            self._vector_index.add(emb_array, ids)
            self._vector_index.save()

        job_emb = await self._embedding_service.embed_text(job.description)
        ids, scores = self._vector_index.search(np.array(job_emb, dtype=np.float32), self._config.retrieval_top_k)
        return ids, scores

    def _build_views(
        self,
        job: JobPosting,
        candidates: list[CandidateProfile],
    ) -> tuple[list[TextView], list[CareerView], list[BehavioralView]]:
        text_views: list[TextView] = []
        career_views: list[CareerView] = []
        behavioral_views: list[BehavioralView] = []

        for c in candidates:
            # Text view
            skill_text = " ".join(s.name for s in c.skills)
            edu_text = " ".join(f"{e.degree} in {e.field_of_study}" for e in c.education)
            text = (
                f"{c.headline} {c.summary} "
                f"Skills: {skill_text} "
                f"Education: {edu_text} "
                f"Years of experience: {c.years_of_experience}"
            )
            text_views.append(TextView(candidate_id=c.candidate_id, text=text))

            # Career view
            roles = []
            for r in (c.career_history or []):
                roles.append(RoleTuple(
                    title=r.title,
                    seniority=CareerExpert._infer_seniority(r.title),
                    employer=r.company,
                    industry=r.industry or "",
                    tenure_months=r.duration_months,
                    is_current=r.is_current or False,
                    start_date=r.start_date,
                    end_date=r.end_date,
                ))
            career_views.append(CareerView(
                candidate_id=c.candidate_id,
                role_sequence=roles,
                total_experience_years=c.years_of_experience,
                current_title=c.current_title,
                current_industry=c.current_industry,
                skills=[s.name for s in c.skills],
                education=c.education,
            ))

            # Behavioral view
            rs = c.redrob_signals
            behavioral_views.append(BehavioralView(
                candidate_id=c.candidate_id,
                profile_views=rs.profile_views_received_30d,
                applications_submitted=rs.applications_submitted_30d,
                recruiter_response_rate=rs.recruiter_response_rate,
                avg_response_time_hours=rs.avg_response_time_hours,
                connection_count=rs.connection_count,
                search_appearance_30d=rs.search_appearance_30d,
                saved_by_recruiters_30d=rs.saved_by_recruiters_30d,
                interview_completion_rate=rs.interview_completion_rate,
                offer_acceptance_rate=rs.offer_acceptance_rate,
                github_activity_score=rs.github_activity_score,
                profile_completeness=rs.profile_completeness_score,
                open_to_work=rs.open_to_work_flag,
                signup_date=rs.signup_date,
                last_active_date=rs.last_active_date,
                skill_assessment_scores=rs.skill_assessment_scores or {},
                endorsements_received=rs.endorsements_received,
                verified_email=rs.verified_email,
                verified_phone=rs.verified_phone,
            ))

        return text_views, career_views, behavioral_views

    def _stage_3_fusion(
        self,
        job: JobPosting,
        sem_map: dict[str, ExpertScore],
        car_map: dict[str, ExpertScore],
        beh_map: dict[str, ExpertScore],
        candidates: list[CandidateProfile],
    ) -> list[FusedScore]:
        _default = lambda cid: ExpertScore(candidate_id=cid, score=0.5, low_confidence=True)
        cids = [c.candidate_id for c in candidates]
        sem_vals = [sem_map.get(cid, _default(cid)).score for cid in cids]
        car_vals = [car_map.get(cid, _default(cid)).score for cid in cids]
        beh_vals = [beh_map.get(cid, _default(cid)).score for cid in cids]

        fused_list = self._gate.fuse(job, sem_vals, car_vals, beh_vals, cids)

        results: list[FusedScore] = []
        for i, cid in enumerate(cids):
            _, fused, weights = fused_list[i] if i < len(fused_list) else (cid, 0.0, [0.4, 0.35, 0.25])
            results.append(FusedScore(
                candidate_id=cid,
                fused_score=fused,
                semantic_score=sem_vals[i],
                career_score=car_vals[i],
                behavioral_score=beh_vals[i],
                gate_weights=weights,
                low_confidence=(
                    sem_map.get(cid, _default(cid)).low_confidence
                    or car_map.get(cid, _default(cid)).low_confidence
                    or beh_map.get(cid, _default(cid)).low_confidence
                ),
            ))
        return results

    async def _stage_4_rerank(
        self,
        job: JobPosting,
        fused_scores: list[FusedScore],
        text_views: list[TextView],
    ) -> list[FusedScore]:
        tv_map = {tv.candidate_id: tv.text for tv in text_views}

        # Build (cid, text) pairs for cross-encoder
        ce_pairs: list[tuple[str, str]] = []
        for fs in fused_scores:
            txt = tv_map.get(fs.candidate_id, "")
            ce_pairs.append((fs.candidate_id, txt))

        reranked_pairs = await self._cross_encoder.rerank(job.description, ce_pairs)

        # Reorder fused_scores by cross-encoder results
        rerank_order = {cid: i for i, (cid, _) in enumerate(reranked_pairs)}
        fused_scores.sort(key=lambda x: rerank_order.get(x.candidate_id, 999))

        # Build embedding dict for MMR
        embeddings: dict[str, np.ndarray] = {}
        for tv in text_views:
            if tv.embedding is not None:
                embeddings[tv.candidate_id] = np.array(tv.embedding, dtype=np.float32)

        mmr_input = [(fs.candidate_id, fs.fused_score) for fs in fused_scores]

        mmr_cids = [cid for cid, _ in mmr_input]
        mmr_embeddings = {}
        for cid in mmr_cids:
            if cid in embeddings:
                mmr_embeddings[cid] = embeddings[cid]

        mmr_results = self._mmr.rerank(mmr_input, mmr_embeddings)
        mmr_order = {cid: i for i, (cid, _) in enumerate(mmr_results)}
        fused_scores.sort(key=lambda x: mmr_order.get(x.candidate_id, 999))

        return fused_scores

    def _stage_5_explain(
        self,
        fused_scores: list[FusedScore],
        job: JobPosting,
        candidates: list[CandidateProfile] | None = None,
    ) -> list[RankedCandidate]:
        profile_map: dict[str, CandidateProfile] = {}
        if candidates:
            profile_map = {c.candidate_id: c for c in candidates}
        ranked: list[RankedCandidate] = []
        for rank, fs in enumerate(fused_scores, start=1):
            rc = RankedCandidate(
                candidate_id=fs.candidate_id,
                rank=rank,
                score=fs.fused_score,
                fused_score=fs.fused_score,
                semantic_score=fs.semantic_score,
                career_score=fs.career_score,
                behavioral_score=fs.behavioral_score,
                gate_weights=fs.gate_weights,
                low_confidence=fs.low_confidence,
            )
            rc.reasoning = self._explainer.explain(
                rc, job, profile=profile_map.get(fs.candidate_id)
            )
            ranked.append(rc)
        return ranked

    async def build_index(
        self,
        job_description: str | None = None,
    ) -> dict:
        candidates = await self._candidate_repo.get_all()
        if not candidates:
            await self._candidate_repo.load_from_jsonl(self._config.data_path)
            candidates = await self._candidate_repo.get_all()

        texts = [
            f"{c.summary} {c.headline} {' '.join(s.name for s in c.skills)}"
            for c in candidates
        ]
        embeddings = await self._embedding_service.embed_texts(texts)
        emb_array = np.array(embeddings, dtype=np.float32)
        ids = [c.candidate_id for c in candidates]
        self._vector_index.add(emb_array, ids)
        self._vector_index.save()

        return {"indexed": len(ids), "dim": emb_array.shape[1] if len(emb_array) > 0 else 0}

    async def ingest_candidates(self, candidates: list[CandidateProfile]) -> dict:
        await self._candidate_repo.add_many(candidates)
        return {"ingested": len(candidates), "total": self._candidate_repo.size}
