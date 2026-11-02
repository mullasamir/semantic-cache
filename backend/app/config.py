"""Central configuration. Values come from environment / .env; a few are
tunable at runtime through POST /config (threshold, verify mode, TTL)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _e(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    database_url: str = _e("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/semantic_cache")

    vector_store: str = _e("VECTOR_STORE", "pgvector").lower()
    pinecone_api_key: str = _e("PINECONE_API_KEY", "")
    pinecone_index: str = _e("PINECONE_INDEX", "semantic-cache")
    pinecone_cloud: str = _e("PINECONE_CLOUD", "aws")
    pinecone_region: str = _e("PINECONE_REGION", "us-east-1")

    embedding_backend: str = _e("EMBEDDING_BACKEND", "sentence-transformers").lower()
    embedding_model: str = _e("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(_e("EMBEDDING_DIM", "384"))

    llm_provider: str = _e("LLM_PROVIDER", "mock").lower()
    llm_model: str = _e("LLM_MODEL", "")
    llm_max_tokens: int = int(_e("LLM_MAX_TOKENS", "512"))
    mock_latency_ms: float = float(_e("MOCK_LATENCY_MS", "800"))
    price_input_per_m: float = float(_e("PRICE_INPUT_PER_M", "1.00"))
    price_output_per_m: float = float(_e("PRICE_OUTPUT_PER_M", "5.00"))

    similarity_threshold: float = float(_e("SIMILARITY_THRESHOLD", "0.90"))
    ttl_seconds: int = int(_e("TTL_SECONDS", "86400"))  # 0 = never expire
    verify_mode: str = _e("VERIFY_MODE", "none").lower()  # none | crossencoder | llm
    verify_low: float = float(_e("VERIFY_LOW", "0.80"))
    verify_high: float = float(_e("VERIFY_HIGH", "0.93"))
    cross_encoder_model: str = _e("CROSS_ENCODER_MODEL", "cross-encoder/quora-distilroberta-base")
    cross_encoder_threshold: float = float(_e("CROSS_ENCODER_THRESHOLD", "0.5"))

    # Queries matching this regex skip the cache (answers that go stale quickly).
    time_sensitive_regex: str = _e(
        "TIME_SENSITIVE_REGEX",
        r"\b(today|tonight|right now|currently|latest|breaking|news|weather|score|stock price|exchange rate|this (week|month|year))\b",
    )
    cors_origins: str = _e("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080")

    def __post_init__(self):
        if not self.llm_model:
            self.llm_model = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}.get(self.llm_provider, "mock")


settings = Settings()
