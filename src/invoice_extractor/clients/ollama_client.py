"""
Ollama model client.

This client connects to a locally running Ollama server and sends
the extraction prompt to the fine-tuned Qwen model.

It implements the same ModelClient interface as GroqClient, so
extraction.py does not need to know which model is being used.
"""

import time
from typing import Optional

from ollama import Client

from invoice_extractor.clients.base import ModelClient
from invoice_extractor.config import settings


class OllamaClient(ModelClient):
    """Client for locally hosted Ollama models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self._model_name = model_name or settings.ollama_model_name
        self._host = host or settings.ollama_host

        self._client = Client(host=self._host)
        self._last_usage: Optional[dict] = None

    def extract_raw(self, prompt: str) -> str:
        """Send the extraction prompt to Ollama and return raw text."""

        start = time.perf_counter()

        response = self._client.chat(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        latency_seconds = time.perf_counter() - start

        content = response["message"]["content"]

        if not content:
            raise ValueError(
                f"Ollama model '{self._model_name}' returned empty content."
            )

        self._last_usage = {
            "latency_seconds": round(latency_seconds, 3),
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
            "estimated_cost_usd": 0.0,
            "model_name": self._model_name,
        }

        return content

    def get_last_usage(self) -> Optional[dict]:
        """Return metrics from the most recent model call."""

        return self._last_usage