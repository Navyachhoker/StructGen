from dataclasses import dataclass

from invoice_extractor.clients.base import ModelClient


@dataclass
class RouterMetadata:
    """Metadata describing how the request was routed."""

    model_used: str
    fallback_used: bool


class ModelRouter(ModelClient):
    """
    Routes extraction requests between a primary model
    and a fallback model.
    """

    def __init__(
        self,
        primary: ModelClient,
        fallback: ModelClient,
        primary_name: str,
        fallback_name: str,
    ):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

        self._last_metadata: RouterMetadata | None = None

    def extract_raw(self, prompt: str) -> str:
        """
        Try the primary model first.

        If the primary model raises an exception,
        use the fallback model.
        """
        try:
            raw_output = self.primary.extract_raw(prompt)

            self._last_metadata = RouterMetadata(
                model_used=self.primary_name,
                fallback_used=False,
            )

            return raw_output

        except Exception:
            raw_output = self.fallback.extract_raw(prompt)

            self._last_metadata = RouterMetadata(
                model_used=self.fallback_name,
                fallback_used=True,
            )

            return raw_output

    def get_last_metadata(self) -> RouterMetadata | None:
        """Return metadata from the most recent routing attempt."""
        return self._last_metadata