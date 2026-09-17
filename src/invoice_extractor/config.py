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
# In production (Render), environment variables are injected directly
# by the platform, so this is effectively a no-op there.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object.

    frozen=True prevents accidental mutation at runtime, which matters
    once multiple modules (router, workers, API) all read from the same
    settings instance.
    """

    # --- Groq ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_name: str = os.getenv(
        "GROQ_MODEL_NAME",
        "openai/gpt-oss-120b",
    )

    # --- Redis / arq ---
    redis_url: str = os.getenv("REDIS_URL", "")

    # --- PostgreSQL ---
    database_url: str = os.getenv("DATABASE_URL", "")

    # --- Local fine-tuned model ---
    ollama_host: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )
    ollama_model_name: str = os.getenv(
        "OLLAMA_MODEL_NAME",
        "qwen-invoice-lora",
    )

    # --- Deployment ---
    # "local" keeps the LoRA → Groq fallback architecture.
    # "groq" uses Groq directly for the public deployment.
    deployment_mode: str = os.getenv(
        "DEPLOYMENT_MODE",
        "local",
    )

    # --- Validation ---
    schema_validation_max_retries: int = 1


# Single shared instance.
# Import this instead of instantiating Settings() elsewhere.
settings = Settings()