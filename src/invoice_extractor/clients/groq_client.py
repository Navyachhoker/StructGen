"""
Real model client for Groq-hosted models.

Implements the same ModelClient interface as FakeModelClient (Phase 1) —
extraction.py, prompts.py, and the schema require zero changes to use
this instead. That interchangeability was the entire point of Phase 1's
abstraction.

This client also tracks latency and token usage per call, exposed via
get_last_usage(). This is optional on the base interface (defaults to
None) so other clients (fake, and Ollama in Phase 7) aren't forced to
implement something meaningless for them.

Model override: by default this client uses settings.groq_model_name
(the production baseline, e.g. openai/gpt-oss-120b). An explicit
model_name can be passed at construction to point this same client at a
different Groq model for a different purpose - e.g. using a
cheaper/higher-rate-limit model like openai/gpt-oss-20b for synthetic
data rendering (scripts/generate_synthetic_data.py), while keeping the
actual production baseline client unchanged. This does NOT change which
model the live /extract API uses - that still reads settings.groq_model_name
via the default (no-argument) constructor.
"""

import time
from typing import Optional

from groq import Groq

from invoice_extractor.clients.base import ModelClient
from invoice_extractor.config import settings

# Approximate Groq pricing per million tokens, by model. NOTE: verify
# current rates at https://groq.com/pricing before quoting these numbers
# in your Phase 8 write-up - pricing changes over time and these are
# estimates, not live-fetched figures.
_PRICING_PER_M_TOKENS = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    # Fallback used if the configured/override model isn't in this table -
    # keeps get_last_usage() working (with a clearly approximate estimate)
    # rather than raising, since cost tracking shouldn't block extraction.
    "_default": {"input": 0.15, "output": 0.60},
}

# Some Groq models (e.g. openai/gpt-oss-20b, a reasoning model) can spend
# part of their output budget on a hidden reasoning stream and return
# empty visible content if max_tokens is too low for a given prompt. This
# default is generous enough for both extraction and invoice-rendering
# prompts used in this project; extract_raw() also checks for and warns
# on empty responses rather than silently returning "".
_DEFAULT_MAX_TOKENS = 4096


class GroqClient(ModelClient):
    """Calls Groq's chat completions API using either the configured
    production model (default) or an explicit override model."""

    def __init__(self, model_name: Optional[str] = None):
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = Groq(api_key=settings.groq_api_key)
        # Falls back to the production baseline model if no override is given -
        # existing callers (GroqClient()) are completely unaffected.
        self._model_name = model_name or settings.groq_model_name
        self._last_usage: Optional[dict] = None

    def extract_raw(self, prompt: str) -> str:
        start = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic output — we want consistent extraction, not creativity
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

        latency_seconds = time.perf_counter() - start
        usage = response.usage  # token counts returned by the API
        content = response.choices[0].message.content

        if not content:
            # Surfaces the reasoning-model empty-content gotcha loudly
            # instead of silently returning "" and letting a downstream
            # JSON-parse failure obscure the real cause.
            raise ValueError(
                f"Groq model '{self._model_name}' returned empty content "
                f"for this prompt. If this is a reasoning model (e.g. "
                f"openai/gpt-oss-20b), it may have spent its token budget "
                f"on hidden reasoning - consider raising max_tokens."
            )

        pricing = _PRICING_PER_M_TOKENS.get(self._model_name, _PRICING_PER_M_TOKENS["_default"])
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]

        # Stored on the instance rather than returned directly, because
        # extract_raw's return type (str) is fixed by the ModelClient
        # interface — every client must return raw text the same way.
        self._last_usage = {
            "latency_seconds": round(latency_seconds, 3),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 6),
            "model_name": self._model_name,
        }

        return content

    def get_last_usage(self) -> Optional[dict]:
        """Returns metrics for the most recent extract_raw call, or None if
        no call has been made yet. Callers use this right after extract_raw."""
        return self._last_usage