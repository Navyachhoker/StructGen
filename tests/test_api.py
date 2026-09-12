"""
Phase 3 verification for the FastAPI service.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from invoice_extractor.config import settings
from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.worker import extract_invoice_task

requires_groq_key = pytest.mark.skipif(
    not settings.groq_api_key, reason="GROQ_API_KEY not set — skipping live API test"
)
requires_redis = pytest.mark.skipif(
    not settings.redis_url, reason="REDIS_URL not set - skipping live Redis test"
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
    response = client.post("/extract", json={"raw_text": "   "})
    assert response.status_code == 400


@requires_groq_key
def test_extract_invoice_task_produces_valid_result_via_live_groq():
    """Tests the worker's job function directly against the real Groq
    API, bypassing arq's own event loop entirely — running arq's full
    worker synchronously inside a test causes event-loop conflicts with
    TestClient's own async context on Windows. /extract's enqueue
    behavior is verified separately in test_extract_enqueues_job_and_returns_job_id."""
    from invoice_extractor.data.sample_invoices import ALL_SAMPLES

    raw_text, _expected = ALL_SAMPLES[0]
    ctx = {"groq_client": GroqClient()}

    result = asyncio.run(extract_invoice_task(ctx, raw_text))

    assert result["success"] is True
    assert result["invoice"]["vendor_name"]
    assert result["latency_seconds"] > 0


@requires_redis
def test_extract_enqueues_job_and_returns_job_id(client):
    response = client.post("/extract", json={"raw_text": "ACME Corp\nTotal: $50.00"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]


@requires_redis
def test_job_status_not_found_for_unknown_id(client):
    response = client.get("/jobs/nonexistent-job-id-12345")
    assert response.status_code == 404