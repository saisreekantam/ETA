"""
Central settings -- replaces the hardcoded paths/URLs scattered through server/main.py,
agents/, and the frontend's API_BASE constant. Reads from environment variables / a
.env file so the same code runs locally (Postgres on 5433, see README) and in Docker
(Postgres on the default 5432 inside the compose network) without code changes.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://localhost:5433/industrial_safety"
    redis_url: str = "redis://localhost:6379/0"
    ollama_url: str = "http://localhost:11434"
    # Layers Ollama offloads to the GPU for the report LLM. Default 20 fits an 8GB
    # NVIDIA card (see orchestrator_node). Set high (e.g. 999) on Macs / large GPUs to
    # run the whole model on the accelerator, or 0 to let Ollama decide.
    ollama_num_gpu: int = 20
    api_key_required: bool = False  # flip true once keys are seeded (see db/seed.py)
    cors_origins: str = "*"


settings = Settings()
