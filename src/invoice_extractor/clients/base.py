"""
Abstract interface every model client must implement.

Why this exists: the extraction service (extraction.py) should never
know or care whether it's talking to a fake client, Groq, or Ollama.
As long as a client implements `extract_raw`, it's swappable — this is
what makes the baseline-vs-fine-tuned comparison possible without
duplicating pipeline logic per model.
"""

from abc import ABC, abstractmethod


class ModelClient(ABC):
    """Common interface for anything that can turn a prompt into raw text output."""

    @abstractmethod
    def extract_raw(self, prompt: str) -> str:
        """
        Send the prompt to the model and return its raw text response.

        Returns raw text (not parsed JSON) deliberately — parsing and
        validation are the extraction service's job, not the client's.
        This keeps clients dumb and swappable.
        """
        raise NotImplementedError