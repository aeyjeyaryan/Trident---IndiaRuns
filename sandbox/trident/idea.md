# TRIDENT — Agent Context

## Project
Senior AI Engineer ranking for Redrob hackathon. 100K candidates → top 100 CSV.

## Architecture (5-stage pipeline)
1. **ANN Retrieval** — FAISS index built from MiniML-L6-v2 embeddings of short text (headline + 5 skills + title + company). Narrows 100K → 400.
2. **3 Expert Scorers** (parallel):
   - *Semantic*: cosine similarity of full-text embeddings vs JD
   - *Career*: role-sequence fit (title seniority, industry, tenure)
   - *Behavioral*: recency, response rate, recruiter saves, GitHub, notice period
3. **Gate Fusion** — fixed weights `[0.4 sem, 0.35 car, 0.25 beh]`
4. **Rerank + Diversity** — cross-encoder `ms-marco-MiniLM-L-6-v2` → MMR
5. **Explanation** — profile-specific NL reasoning referencing JD skills + Redrob signals

## Key Files
- `rank.py` — entry point: loads pre-built index, runs pipeline, writes CSV (≤5 min CPU)
- `build_index.py` — pre-computes embeddings + FAISS index (can exceed 5 min)
- `app/domain/pipeline.py` — orchestrator; batches embeddings in `_stage_1_retrieval`
- `app/domain/explanation/explainer.py` — generates JD-aware reasoning with profile facts
- `app/domain/experts/behavioral.py` — uses `last_active_date`, `recruiter_response_rate`, `saved_by_recruiters_30d`, etc.
- `app/domain/experts/career.py` — `CareerExpert._compute_role_family_score` and seniority inference
- `app/domain/experts/semantic.py` — `SemanticExpert.score()` with temperature=0.3
- `app/domain/retrieval/vector_index.py` — `FAISSIndex` with `IndexFlatL2`, L2→cosine conversion
- `app/domain/rerank/cross_encoder.py` — async wrapper for CrossEncoder
- `app/domain/rerank/mmr.py` — MMR with lambda=0.7
- `app/infra/embeddings.py` — `EmbeddingService` with batch `embed_texts`
- `app/infra/persistence.py` — `InMemoryCandidateRepository`
- `app/core/config.py` — all tunable hyperparameters

## Config (Settings in `app/core/config.py`)
- `retrieval_top_k=200` (overridden to 400 in rank.py)
- `rerank_k=50` (overridden to 100 in rank.py)
- `semantic_temperature=0.3`
- `mmr_lambda=0.7`
- `gate_fallback_weights=(0.4, 0.35, 0.25)`
- `index_path=data/faiss_index.bin`, `metadata_path=data/metadata.json`
- `gemini_api_key` — loaded from `TRIDENT_GEMINI_API_KEY` env var or `.env` file (not hardcoded)
- `use_llm_explanations=False` — LLM explainer available but opted out

## Data
- Dataset: `../dataset/candidates.jsonl` (100K lines)
- Candidate schema: `candidate_id` + `profile` dict + `skills` + `career_history` + `education` + `redrob_signals` (23 signals)
- Validation: `../dataset/validate_submission.py`
- Job: "Senior AI Engineer — Founding Team" at Series A startup

## Submission Format
`candidate_id,rank,score,reasoning` — exactly 100 rows, scores non-increasing, ties broken by candidate_id asc.

## Status
- Pipeline runs in ~17s on MPS, validates clean.
- Explainer produces JD-aware reasoning with 6+ Redrob signals (template-based, no LLM).
- 23/23 tests pass.
- FAISS index pre-built at `data/faiss_index.bin` (100K vectors, 384d).
- No honeypots detected in top 100.

## Secrets
- `gemini_api_key` was hardcoded in `app/core/config.py`, moved to `trident/.env` as `TRIDENT_GEMINI_API_KEY`.
- `.env` is in root `.gitignore` — never commit it.

## Gotchas
- Explainer needs `JobPosting` object (not just title string) for JD-aware reasoning.
- Score formatting: use raw float, NOT `:.4f` — avoids artificial ties.
- FAISS index must be pre-built via `build_index.py` before `rank.py`.
- Cross-encoder runs on rerank_k candidates only (not the full shortlist).
- MMR reorders by diversity but scores stay at fused values — re-sort by `-score, candidate_id` before CSV write.
- `faiss-cpu` in requirements.txt ensures CPU-only in Stage 3 sandbox.
- Template-based explanations are intentional — deterministic, fast, zero cost. LLM explainer (`_llm_explain`) exists but is disabled by default (`use_llm_explanations=False`).
