"""
Phase 2 verification for the real Groq client.

These tests hit a live API, so they're automatically skipped if
GROQ_API_KEY isn't set (e.g., in CI environments without secrets
configured). This keeps `pytest` runnable for anyone cloning the repo
without requiring API keys, while still giving you real verification
locally.
"""

import dataclasses

import pytest

from invoice_extractor import config
from invoice_extractor.config import settings
from invoice_extractor.clients import groq_client as groq_client_module
from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.extraction import extract_invoice
from invoice_extractor.data.sample_invoices import ALL_SAMPLES
requires_groq_key = pytest.mark.skipif(
    not settings.groq_api_key, reason="GROQ_API_KEY not set — skipping live API test"
)


@requires_groq_key
def test_groq_extracts_valid_json_for_clean_sample():
    """The clean, well-formatted sample invoice should extract successfully."""
    client = GroqClient()
    raw_text, _expected = ALL_SAMPLES[0]  # clean ACME invoice

    result = extract_invoice(raw_text, client)

    assert result.success is True
    assert result.invoice.vendor_name  # non-empty
    assert result.invoice.total_amount > 0


@requires_groq_key
def test_groq_client_reports_usage_after_call():
    """get_last_usage() should be populated after a real call, confirming
    the latency/cost tracking wiring actually works end-to-end."""
    client = GroqClient()
    raw_text, _expected = ALL_SAMPLES[0]

    extract_invoice(raw_text, client)
    usage = client.get_last_usage()

    assert usage is not None
    assert usage["latency_seconds"] > 0
    assert usage["input_tokens"] > 0
    assert usage["estimated_cost_usd"] >= 0


def test_groq_client_raises_without_api_key(monkeypatch):
    """Sanity check: instantiating without a key should fail loudly and
    early, not silently produce a client that fails on first use."""
    empty_settings = dataclasses.replace(settings, groq_api_key="")

    monkeypatch.setattr(config, "settings", empty_settings)
    monkeypatch.setattr(groq_client_module, "settings", empty_settings, raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient()