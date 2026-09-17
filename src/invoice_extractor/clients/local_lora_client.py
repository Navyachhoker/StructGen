import time

import httpx

from invoice_extractor.clients.base import ModelClient


class LocalLoRAClient(ModelClient):
    """Client for the local Qwen + LoRA inference service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._last_usage: dict | None = None

    def extract_raw(self, prompt: str) -> str:
        start_time = time.perf_counter()

        response = httpx.post(
            f"{self.base_url}/extract",
            json={"invoice_text": self._extract_invoice_text(prompt)},
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        latency = time.perf_counter() - start_time

        self._last_usage = {
            "latency_seconds": latency,
            "estimated_cost_usd": 0.0,
            "model_name": "qwen-invoice-lora-v3",
        }

        return data["output"]

    def get_last_usage(self) -> dict | None:
        return self._last_usage

    @staticmethod
    def _extract_invoice_text(prompt: str) -> str:
        """Extract invoice text from the shared StructGen prompt."""

        marker = 'Invoice text:\n"""\n'

        if marker not in prompt:
            return prompt

        invoice_text = prompt.split(marker, 1)[1]

        if '"""\n\nJSON output:' in invoice_text:
            invoice_text = invoice_text.split(
                '"""\n\nJSON output:',
                1,
            )[0]

        return invoice_text.strip()