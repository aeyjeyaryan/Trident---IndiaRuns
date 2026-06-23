# TRIDENT — Candidate Ranking Pipeline

Top-100 ranking for the Redrob Intelligent Candidate Discovery & Ranking Challenge. Designed for the "Senior AI Engineer — Founding Team" JD at a Series A startup.

## How it works

TRIDENT uses a 5-stage pipeline to go from 100K candidates → ranked top 100:

```
candidates.jsonl
    │
    ▼
[1] ANN RETRIEVAL ─── FAISS index (MiniLM-L6-v2 embeddings on short text)
    │                   Narrow pool from 100K → 400 candidates
    ▼
[2] THREE EXPERT SCORERS ─── run in parallel on the shortlist:
    │   • Semantic  — cosine similarity of full-text embeddings vs job description
    │   • Career    — role-sequence fit (title seniority, industry, tenure patterns)
    │   • Behavioral — engagement signals (recency, response rate, recruiter saves, GitHub)
    ▼
[3] GATE FUSION ─── fixed weights [0.4 sem, 0.35 car, 0.25 beh] → fused score
    │
    ▼
[4] RERANK + DIVERSITY ─── cross-encoder (ms-marco-MiniLM) re-scores top 100,
    │                       then MMR diversifies based on embedding similarity
    ▼
[5] EXPLANATION ─── per-candidate reasoning with profile facts + JD context
    │                 + Redrob behavioral signals
    ▼
submission.csv
```

### Key design decisions

- **Two-phase retrieval**: FAISS ANN (fast, coarse) → cross-encoder (accurate, on shortlist) avoids scoring 100K candidates with the expensive transformer.
- **Three parallel experts**: Each captures a different match dimension. A candidate strong in only one dimension won't dominate — the gate requires consensus.
- **MMR diversity**: Prevents the top 100 from being 100 nearly-identical ML engineers by promoting embedding diversity.
- **JD-aware reasoning** (template-based): The explainer cross-references candidate skills against a curated JD key-skill list and flags concerns like consulting backgrounds, long notice periods, and inactivity. An LLM-based explainer (Gemini) is available in `Explainer` by passing an `llm_client` — set `use_llm_explanations=True` and provide a `gemini_api_key` via env var. The template version is preferred for determinism, speed, and zero API cost.
- **Pre-computed index**: Building the FAISS index (embedding 100K candidates) runs separately. The ranking step itself is fast (~10s on GPU, ~60s CPU).

## Usage

```bash
pip install -r requirements.txt

# Step 1: Pre-compute (embeds all 100K, saves to data/faiss_index.bin)
python build_index.py

# Step 2: Rank → CSV (must complete within 5 min CPU-only)
python rank.py --candidates ../dataset/candidates.jsonl --out submission.csv

# Validate
python ../dataset/validate_submission.py submission.csv
```
