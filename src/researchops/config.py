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

    # Phase 2 — labops MCP server: SSH to the GPU host for experiment orchestration.
    # Defaults point at the current AutoDL host; override via env for any other host.
    labops_host: str = "connect.westd.seetacloud.com"
    labops_port: int = 22050
    labops_user: str = "root"
    labops_key_path: str = "~/.ssh/id_rsa"
    # Remote working directory that jobs are sandboxed to.
    labops_workdir: str = "/root/autodl-tmp"
    # Per-command SSH timeout (seconds).
    labops_command_timeout: float = 30.0

    # Phase 2 — persistence: experiment/job/metric records (SQLAlchemy 2.0).
    # SQLite by default (zero infra, git-ignored). Swap to a Postgres DSN
    # (e.g. postgresql+asyncpg://user:pass@host/db) for production — no code change.
    db_url: str = "sqlite+aiosqlite:///.researchops/experiments.db"

    # Cost estimation & budget cap. Prices are USD per 1M tokens and feed the local
    # trace summary + Langfuse cost panel. Defaults are the current DeepSeek-chat list
    # prices; override per provider/deployment via env. ``agent_max_cost_usd`` is the
    # hard cap for one agent run (0 = unlimited); set it for public deployments.
    llm_input_price_per_1m: float = 0.14
    llm_output_price_per_1m: float = 0.28
    agent_max_cost_usd: float = 0.0

    # Phase 3 — observability: Langfuse (cloud) trace/cost/latency dashboards.
    # Empty keys => exporter disabled (the local `--trace` summary still works).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
