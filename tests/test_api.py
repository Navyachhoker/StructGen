"""
Phase 3 verification for the FastAPI service.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from invoice_extractor.config import settings
from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.clients.fake import FakeModelClient
from invoice_extractor.clients.router import ModelRouter
from invoice_extractor.worker import extract_invoice_task
from invoice_extractor.db.connection import get_pool


requires_groq_key = pytest.mark.skipif(
    not settings.groq_api_key,
    reason="GROQ_API_KEY not set — skipping live API test",
)

requires_redis = pytest.mark.skipif(
    not settings.redis_url,
    reason="REDIS_URL not set - skipping live Redis test",
)


@pytest.fixture
def client():
    from invoice_extractor.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_rejects_empty_text(client):
    response = client.post(
        "/extract",
        json={"raw_text": "   "},
    )

    assert response.status_code == 400


async def _run_with_pool(
    ctx_extra: dict,
    raw_text: str,
) -> dict:
    pool = await get_pool()

    ctx = {
        **ctx_extra,
        "db_pool": pool,
    }

    return await extract_invoice_task(ctx, raw_text)


@requires_groq_key
def test_extract_invoice_task_produces_valid_result_via_live_groq():
    from invoice_extractor.data.sample_invoices import ALL_SAMPLES

    raw_text, _expected = ALL_SAMPLES[0]

    groq_client = GroqClient()

    router = ModelRouter(
        primary=groq_client,
        fallback=FakeModelClient(),
        primary_name="openai/gpt-oss-120b",
        fallback_name="test-fallback",
    )

    result = asyncio.run(
        _run_with_pool(
            {
                "router": router,
                "job_id": "test-job-live-001",
            },
            raw_text,
        )
    )

    assert result["success"] is True
    assert result["invoice"]["vendor_name"]
    assert result["latency_seconds"] > 0
    assert result["model_name"] == "openai/gpt-oss-120b"
    assert result["fallback_used"] is False


@requires_redis
def test_extract_enqueues_job_and_returns_job_id(client):
    response = client.post(
        "/extract",
        json={"raw_text": "ACME Corp\nTotal: $50.00"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "queued"
    assert body["job_id"]


@requires_redis
def test_job_status_not_found_for_unknown_id(client):
    response = client.get(
        "/jobs/nonexistent-job-id-12345"
    )

    assert response.status_code == 404