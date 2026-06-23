"""
Learned Gate — A small 2-layer MLP that outputs a 3-way softmax weighting.

Why a learned gate instead of a fixed average?
  Different job families benefit from different expert emphasis. For a senior
  ML Engineer role, the semantic and career experts should dominate; for a
  high-volume sales role, behavioral engagement signals may be more predictive.
  The gate learns this mapping from a context vector:

    context = [role_family_embedding ; seniority_band_onehot ; pool_density_stat]

Architecture:
  Input: context_dim (default 16)
  Hidden: 32 → ReLU → dropout(0.1)
  Output: 3 → softmax

The gate ships with a FixedWeightGate fallback so the system works end-to-end
even before the gate is trained. Training scripts should save the state_dict
to a path that the serving code loads at startup.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from app.domain.models import JobPosting
from app.core.config import Settings


class GateMLP(nn.Module):
    def __init__(self, context_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.softmax(self.fc2(x), dim=-1)


class LearnedGate:
    def __init__(self, config: Settings | None = None, model_path: str | None = None):
        cfg = config or Settings()
        self._context_dim = cfg.gate_context_dim
        self._hidden_dim = cfg.gate_hidden_dim
        self._role_families = cfg.ROLE_FAMILIES
        self._seniority_bands = cfg.SENIORITY_BANDS

        self._model = GateMLP(self._context_dim, self._hidden_dim)
        self._model.eval()

        if model_path and Path(model_path).exists():
            self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self._trained = True
        else:
            self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def fuse(
        self,
        job: JobPosting,
        semantic_scores: list[float],
        career_scores: list[float],
        behavioral_scores: list[float],
        candidate_ids: list[str],
    ) -> list[tuple[str, float, list[float]]]:
        if not self._trained:
            raise RuntimeError("LearnedGate has not been trained. Use FixedWeightGate until training is ready.")

        ctx = self._build_context(job, semantic_scores)
        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            weights = self._model(ctx_t).squeeze(0).tolist()

        results: list[tuple[str, float, list[float]]] = []
        for i, cid in enumerate(candidate_ids):
            s_sem = semantic_scores[i] if i < len(semantic_scores) else 0.0
            s_car = career_scores[i] if i < len(career_scores) else 0.0
            s_beh = behavioral_scores[i] if i < len(behavioral_scores) else 0.0
            fused = weights[0] * s_sem + weights[1] * s_car + weights[2] * s_beh
            results.append((cid, fused, weights))
        return results

    def _build_context(self, job: JobPosting, scores: list[float]) -> list[float]:
        ctx: list[float] = [0.0] * self._context_dim
        # Role-family embedding (one-hot-like)
        for i, family in enumerate(self._role_families):
            if i < self._context_dim:
                ctx[i] = 1.0 if job.role_family == family else 0.0
        # Seniority band (one-hot, offset after families)
        offset = len(self._role_families)
        for i, band in enumerate(self._seniority_bands):
            if offset + i < self._context_dim:
                ctx[offset + i] = 1.0 if job.seniority_band == band else 0.0
        # Pool density stat — rough std of scores as a measure of spread
        if len(scores) > 1:
            pool_std = float(np.std(scores))
        else:
            pool_std = 0.0
        if offset + len(self._seniority_bands) < self._context_dim:
            ctx[offset + len(self._seniority_bands)] = min(pool_std, 1.0)
        return ctx
