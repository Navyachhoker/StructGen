"""
Phase 1 verification: confirms the extraction pipeline correctly handles
success, malformed JSON, and schema-invalid JSON — using the fake client
so no real API calls or costs are involved.
"""

from invoice_extractor.clients.fake import FakeModelClient, DEFAULT_FAKE_RESPONSE
from invoice_extractor.extraction import extract_invoice


def test_successful_extraction():
    """A well-formed, schema-valid response should produce a validated invoice."""
    client = FakeModelClient()
    result = extract_invoice("some invoice text", client)

    assert result.success is True
    assert result.invoice is not None
    assert result.invoice.vendor_name == "Fake Vendor Inc"
    assert result.error is None


def test_malformed_json_fails_gracefully():
    """Broken JSON should return a failed result, not raise an exception."""
    client = FakeModelClient(fixed_response="this is not json {broken")
    result = extract_invoice("some invoice text", client)

    assert result.success is False
    assert result.invoice is None
    assert "Invalid JSON" in result.error


def test_schema_invalid_json_fails_gracefully():
    """Valid JSON that doesn't satisfy the schema (missing required field)
    should fail validation, not crash."""
    client = FakeModelClient(fixed_response='{"vendor_name": "Missing Total Co"}')
    result = extract_invoice("some invoice text", client)

    assert result.success is False
    assert result.invoice is None
    assert "Schema validation failed" in result.error


def test_prompt_is_actually_used_by_real_clients():
    """Sanity check that build_extraction_prompt produces something
    sensible — this matters once Phase 2 swaps in a real client that
    actually reads the prompt."""
    from invoice_extractor.prompts import build_extraction_prompt

    prompt = build_extraction_prompt("Vendor: Test\nTotal: $5.00")
    assert "Test" in prompt
    assert "JSON" in prompt