"""
Thin HTTP client wrapping the Invoice Extractor FastAPI service.

This module translates Streamlit UI actions into HTTP requests
against the FastAPI backend.

No extraction logic, schema validation, or model calls happen here.
"""

import os
import time
from typing import Optional

import requests


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

MAX_POLL_SECONDS = 120
POLL_INTERVAL_SECONDS = 1.5


def submit_extraction(raw_text: str) -> str:
    """Submit invoice text and return the queued job ID."""

    response = requests.post(
        f"{API_BASE_URL}/extract",
        json={"raw_text": raw_text},
        timeout=10,
    )

    response.raise_for_status()

    return response.json()["job_id"]


def poll_job(
    job_id: str,
    on_update: Optional[callable] = None,
) -> dict:
    """Poll the API until the extraction job completes or times out."""

    elapsed = 0.0

    while elapsed < MAX_POLL_SECONDS:
        response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}",
            timeout=10,
        )

        response.raise_for_status()

        body = response.json()

        if on_update is not None:
            on_update(body)

        if body["status"] == "complete":
            return body

        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    return {
        "job_id": job_id,
        "status": "timeout",
        "result": None,
    }


def get_stats() -> Optional[dict]:
    """Fetch aggregate extraction statistics."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/stats",
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def get_recent_invoices(
    limit: int = 20,
) -> Optional[list]:
    """Fetch recent extraction records."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/invoices",
            params={"limit": limit},
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None