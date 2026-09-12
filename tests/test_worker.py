"""
Phase 4 verification for the background job function.

Tests extract_invoice_task directly with a manually built ctx dict
containing a FakeModelClient - this verifies the task's own logic
(calling extract_invoice, producing a JSON-safe result) without needing
a real Redis connection or a running arq worker process.
"""

import asyncio

from invoice_extractor.worker import extract_invoice_task
from invoice_extractor.clients.fake import FakeModelClient


def test_extract_invoice_task_returns_json_safe_dict():
    ctx = {"groq_client": FakeModelClient()}
    result = asyncio.run(extract_invoice_task(ctx, "some invoice text"))

    assert result["success"] is True
    assert result["invoice"]["vendor_name"] == "Fake Vendor Inc"
    # Decimal/date must serialize to plain strings - arq needs to store
    # this dict in Redis, which requires it to be JSON-safe.
    assert isinstance(result["invoice"]["total_amount"], str)
    assert isinstance(result["invoice"]["invoice_date"], str)


def test_extract_invoice_task_handles_failure():
    ctx = {"groq_client": FakeModelClient(fixed_response="not json")}
    result = asyncio.run(extract_invoice_task(ctx, "some invoice text"))

    assert result["success"] is False
    assert result["invoice"] is None
    assert "Invalid JSON" in result["error"]