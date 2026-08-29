"""Configuration loaded from environment / .env.

Keeps every secret out of the repo: keys come only from environment variables
or a git-ignored `.env` file (see `.env.example`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider selection: deepseek | qwen | vllm
    llm_provider: str = "deepseek"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "not-needed"

    # Phase 1 — RAG infrastructure
    # Remote inference service (bge-m3 + reranker) on the GPU host.
    inference_base_url: str = "http://127.0.0.1:8001"
    # Qdrant vector store.
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "papers"
    # Hybrid retrieval: number of candidates per branch before RRF + rerank.
    retrieval_top_k: int = 20
    retrieval_rerank_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
