"""
Thin HTTP client wrapping the Invoice Extractor FastAPI service.

This module's only job is translating Streamlit's needs (submit text,
wait for a result, show stats) into calls against endpoints that already
exist and are already tested (Phases 3-5). No extraction logic, no
schema validation, no model calls happen here — that would duplicate
work the API already does correctly.
"""

import os
import time
from typing import Optional

import requests

# In production this would point at your deployed Render URL instead of
# localhost — kept as a plain constant here since Streamlit's own secrets
# manager (Phase 6 deploy step) is the right place for that, not code.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# How long to keep polling before giving up — LLM calls are usually fast,
# but free-tier cold starts (Render spinning up) can take a while, and
# local CPU LoRA inference alone can take ~48 seconds. 120s gives that
# a comfortable margin instead of timing out mid-inference.
MAX_POLL_SECONDS = 120
POLL_INTERVAL_SECONDS = 1.5


def submit_extraction(raw_text: str) -> str:
    """Submits invoice text for extraction, returns the job_id."""
    response = requests.post(f"{API_BASE_URL}/extract", json={"raw_text": raw_text})
    response.raise_for_status()
    return response.json()["job_id"]


def poll_job(job_id: str, on_update: Optional[callable] = None) -> dict:
    """
    Polls GET /jobs/{job_id} until status is 'complete' or MAX_POLL_SECONDS
    elapses. Returns the final job status dict either way — the caller
    decides how to display a timeout vs. a real result.

    on_update, if given, is called with each raw poll response (e.g.
    {"status": "in_progress", ...}) as it arrives, so a caller like the
    Streamlit UI can show live status instead of an opaque spinner for
    the whole wait. Optional and defaults to None so existing callers
    (and the test suite) are unaffected.
    """
    elapsed = 0.0
    while elapsed < MAX_POLL_SECONDS:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        body = response.json()

        if on_update is not None:
            on_update(body)

        if body["status"] == "complete":
            return body

        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    return {"job_id": job_id, "status": "timeout", "result": None}


def get_stats() -> Optional[dict]:
    """Fetches aggregate stats. Returns None if the API is unreachable,
    so the UI can show a friendly message instead of crashing."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
    
    
    
def get_recent_invoices(limit: int = 20) -> Optional[list]:
    """Fetches recent extraction records for the dashboard table. Returns
    None if the API is unreachable, same pattern as get_stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/invoices", params={"limit": limit}, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None