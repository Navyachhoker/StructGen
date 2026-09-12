"""
Real model client for Groq's hosted Llama 3.3 70B.

Implements the same ModelClient interface as FakeModelClient (Phase 1) —
extraction.py, prompts.py, and the schema require zero changes to use
this instead. That interchangeability was the entire point of Phase 1's
abstraction.

This client also tracks latency and token usage per call, exposed via
get_last_usage(). This is optional on the base interface (defaults to
None) so other clients (fake, and Ollama in Phase 7) aren't forced to
implement something meaningless for them.
"""

import time
from typing import Optional

from groq import Groq

from invoice_extractor.clients.base import ModelClient
from invoice_extractor.config import settings

# Approximate Groq pricing for llama-3.3-70b-versatile, per million tokens.
# NOTE: verify current rates at https://groq.com/pricing before quoting
# these numbers in your Phase 8 write-up — pricing changes over time and
# this estimate should be labeled as such.
_PRICE_PER_M_INPUT_TOKENS = 0.59
_PRICE_PER_M_OUTPUT_TOKENS = 0.79


class GroqClient(ModelClient):
    """Calls Groq's chat completions API using the model configured in settings."""

    def __init__(self):
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = Groq(api_key=settings.groq_api_key)
        self._last_usage: Optional[dict] = None

    def extract_raw(self, prompt: str) -> str:
        start = time.perf_counter()

        response = self._client.chat.completions.create(
            model=settings.groq_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic output — we want consistent extraction, not creativity
        )

        latency_seconds = time.perf_counter() - start
        usage = response.usage  # token counts returned by the API

        input_cost = (usage.prompt_tokens / 1_000_000) * _PRICE_PER_M_INPUT_TOKENS
        output_cost = (usage.completion_tokens / 1_000_000) * _PRICE_PER_M_OUTPUT_TOKENS

        # Stored on the instance rather than returned directly, because
        # extract_raw's return type (str) is fixed by the ModelClient
        # interface — every client must return raw text the same way.
        self._last_usage = {
            "latency_seconds": round(latency_seconds, 3),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 6),
        }

        return response.choices[0].message.content

    def get_last_usage(self) -> Optional[dict]:
        """Returns metrics for the most recent extract_raw call, or None if
        no call has been made yet. Callers use this right after extract_raw."""
        return self._last_usage