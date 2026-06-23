from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIDENT_", env_file=".env", extra="ignore")

    # --- Retrieval ---
    retrieval_top_k: int = 200
    retrieval_ef_search: int = 40

    # --- Embedding model ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_temperature: float = 0.3

    # --- Behavioral expert ---
    behavioral_hl_acute_days: float = 10.0
    behavioral_hl_responsiveness_days: float = 45.0
    behavioral_hl_presence_days: float = 75.0
    behavioral_w_acute: float = 0.4
    behavioral_w_responsiveness: float = 0.35
    behavioral_w_presence: float = 0.25
    behavioral_neutral_prior: float = 0.5

    # --- Career expert ---
    career_short_history_threshold: int = 2
    career_neutral_prior: float = 0.5
    career_embed_dim: int = 64
    career_gru_hidden: int = 32

    # --- Rerank ---
    rerank_k: int = 50
    mmr_lambda: float = 0.7

    # --- Gate ---
    gate_context_dim: int = 16
    gate_hidden_dim: int = 32
    gate_fallback_weights: tuple[float, float, float] = (0.4, 0.35, 0.25)

    # --- API ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Data paths ---
    data_path: str = "/Users/aryanjha/Desktop/Indiaruns/dataset/candidates.jsonl"
    index_path: str = "data/faiss_index.bin"
    metadata_path: str = "data/metadata.json"

    # --- Gemini (optional, for LLM explanations) ---
    gemini_api_key: str = ""
    use_llm_explanations: bool = False

    # --- Fixed set of known role families for career expert ---
    ROLE_FAMILIES: ClassVar[list[str]] = [
        "engineering", "data_science", "product", "design", "marketing",
        "sales", "operations", "hr", "finance", "support",
    ]

    SENIORITY_BANDS: ClassVar[list[str]] = [
        "junior", "mid", "senior", "lead", "executive",
    ]
