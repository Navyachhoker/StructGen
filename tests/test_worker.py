"""
Phase 4 verification for the background job function.

Tests extract_invoice_task directly with a manually built ctx dict.
The worker receives a ModelRouter containing fake clients so the tests
verify the task logic without requiring Redis, the LoRA service, or Groq.
"""

import asyncio

from invoice_extractor.clients.fake import FakeModelClient
from invoice_extractor.clients.router import ModelRouter
from invoice_extractor.db.connection import get_pool
from invoice_extractor.worker import extract_invoice_task


async def _run_with_pool(
    ctx_extra: dict,
    raw_text: str,
) -> dict:
    """Run the task and database pool on the same event loop."""

    pool = await get_pool()

    ctx = {
        **ctx_extra,
        "db_pool": pool,
    }

    return await extract_invoice_task(ctx, raw_text)


def _build_router(
    primary: FakeModelClient,
    fallback: FakeModelClient | None = None,
) -> ModelRouter:
    """Build a router for worker tests."""

    return ModelRouter(
        primary=primary,
        fallback=fallback or FakeModelClient(),
        primary_name="qwen-invoice-lora-v3",
        fallback_name="openai/gpt-oss-120b",
    )


def test_extract_invoice_task_returns_json_safe_dict():
    router = _build_router(
        primary=FakeModelClient(),
    )

    result = asyncio.run(
        _run_with_pool(
            {
                "router": router,
                "job_id": "test-job-001",
            },
            "some invoice text",
        )
    )

    assert result["success"] is True
    assert result["invoice"]["vendor_name"] == "Fake Vendor Inc"

    # Decimal/date must serialize to plain strings.
    # arq needs the returned dict to be JSON-safe.
    assert isinstance(result["invoice"]["total_amount"], str)
    assert isinstance(result["invoice"]["invoice_date"], str)

    assert result["model_name"] == "qwen-invoice-lora-v3"
    assert result["fallback_used"] is False


def test_extract_invoice_task_handles_failure():
    router = _build_router(
        primary=FakeModelClient(fixed_response="not json"),
        fallback=FakeModelClient(fixed_response="not json"),
    )

    result = asyncio.run(
        _run_with_pool(
            {
                "router": router,
                "job_id": "test-job-002",
            },
            "some invoice text",
        )
    )

    assert result["success"] is False
    assert result["invoice"] is None
    assert "Invalid JSON" in result["error"]