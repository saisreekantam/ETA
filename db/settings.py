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
    # Chat assistant LLM (server/chat.py). Defaults to the same local Ollama + model as
    # the incident-report LLM; set both to point chat at a bigger remote model instead
    # (e.g. qwen2.5:14b on the GPU box, reached via an SSH tunnel: CHAT_OLLAMA_URL=
    # http://localhost:11435 after `ssh -f -N -L 11435:localhost:11434 <gpu-host>`).
    chat_ollama_url: str = ""  # empty = use ollama_url
    chat_model: str = ""  # empty = use the report model (orchestrator_node.OLLAMA_MODEL)
    api_key_required: bool = False  # flip true once keys are seeded (see db/seed.py)
    cors_origins: str = "*"
    # Optional outbound notification channel: alert/emergency escalations POST a JSON
    # payload here (Slack/Teams/incident webhook). Empty = disabled.
    alert_webhook_url: str = ""
    # Background thread that generates readings for source_type="simulated" sensors so
    # the Devices page demos with moving data. Disable for real deployments.
    iot_simulator: bool = True
    # Background watcher that evaluates ingested IoT readings against each device's own
    # trailing baseline and raises alerts on deviation (server/live_watch.py).
    device_watcher: bool = True


settings = Settings()
