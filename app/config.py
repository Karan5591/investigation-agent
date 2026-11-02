from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-compatible LLM endpoint (defaults to local Ollama)
    openai_api_key: str = ""
    openai_model: str = "qwen2.5:7b"
    openai_base_url: str = "http://localhost:11434/v1"

    # Elasticsearch
    es_enabled: bool = True
    es_url: str = "http://localhost:9200"
    es_index: str = "document_chunks"

    # SQLite
    sqlite_path: str = "data/app.db"

    # Embeddings + reranker
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    # OCR
    ocr_lang: str = "eng"
    ocr_min_confidence: float = 70.0

    # Agent guards
    agent_max_steps: int = 8
    agent_max_repeated_calls: int = 2
    llm_repair_retries: int = 1
    processing_stale_seconds: int = 300

    # Storage
    upload_dir: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
