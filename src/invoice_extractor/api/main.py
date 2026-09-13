"""
FastAPI application. /stats now reads from Postgres instead of the
deleted in-memory store (api/store.py is gone as of this phase).
"""

from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus
from fastapi import FastAPI, HTTPException

from invoice_extractor.api.models import (
    ExtractRequest,
    ExtractResponse,
    JobSubmitResponse,
    JobStatusResponse,
    StatsResponse,
)
from invoice_extractor.config import settings
from invoice_extractor.db.connection import get_pool, close_pool
from invoice_extractor.db.repository import get_stats

_redis_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_pool
    _redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await get_pool()  # establishes the Postgres pool for this process
    yield
    await _redis_pool.close()
    await close_pool()


app = FastAPI(
    title="Invoice Extractor API",
    description="Extracts structured JSON from invoice/receipt text using LLMs.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/extract", response_model=JobSubmitResponse)
async def extract(request: ExtractRequest):
    if not request.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")

    job = await _redis_pool.enqueue_job("extract_invoice_task", request.raw_text)
    return JobSubmitResponse(job_id=job.job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    """Unchanged from Phase 4 — still reads the immediate result from arq.
    The permanent Postgres record is now written independently by the
    worker, so this endpoint no longer needs to log anything itself."""
    job = Job(job_id, _redis_pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    if status != JobStatus.complete:
        return JobStatusResponse(job_id=job_id, status=status.value)

    job_result = await job.result_info()
    return JobStatusResponse(
        job_id=job_id,
        status="complete",
        result=ExtractResponse(**job_result.result),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Now sourced from Postgres — correct across restarts, multiple
    processes, and jobs nobody ever polled."""
    pool = await get_pool()
    return StatsResponse(**(await get_stats(pool)))