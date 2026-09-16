"""
Centralized configuration for the project.

Why this exists: hardcoding API keys, URLs, or magic constants across
multiple files makes the project brittle and hard to reconfigure for
different environments (local dev, CI, deployed). Every other module
should import settings from here rather than reading os.environ directly.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a local .env file if present.
# In production (Render, HF Spaces), env vars are injected directly by the platform,
# so this call is a no-op there — safe either way.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object. frozen=True prevents accidental mutation
    at runtime, which matters once multiple modules (router, workers,
    API) all read from the same settings instance.
    """

    # --- Phase 2+: Groq (baseline model) ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    # --- Phase 4+: Redis / arq queue ---
    redis_url: str = os.getenv("REDIS_URL", "")

    # --- Phase 5+: Postgres ---
    database_url: str = os.getenv("DATABASE_URL", "")

    # --- General ---
    # Used later for validation retry logic (Phase 1+)
    schema_validation_max_retries: int = 1
    
    # --- Phase 6: Ollama / local fine-tuned model ---
    ollama_host: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )
    ollama_model_name: str = os.getenv(
        "OLLAMA_MODEL_NAME",
        "qwen-invoice-lora",
    )


# Single shared instance — import this, don't instantiate Settings() elsewhere.
settings = Settings()


