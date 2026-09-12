"""
FastAPI application - async version.

POST /extract now enqueues a job and returns immediately with a job ID.
GET /jobs/{job_id} polls for the result. The actual Groq call happens in
a separate arq worker process (worker.py), not in this API process.
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
from invoice_extractor.api.store import store, RequestLog
from invoice_extractor.config import settings

# Redis connection pool used to enqueue jobs and check their status.
# Created once at app startup via the lifespan context, not per-request.
_redis_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_pool
    _redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    yield
    await _redis_pool.close()


app = FastAPI(
    title="Invoice Extractor API",
    description="Extracts structured JSON from invoice/receipt text using LLMs.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/extract", response_model=JobSubmitResponse)
async def extract(request: ExtractRequest):
    """Enqueues an extraction job and returns instantly with a job ID.
    Poll GET /jobs/{job_id} for the actual result."""
    if not request.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")

    job = await _redis_pool.enqueue_job("extract_invoice_task", request.raw_text)
    return JobSubmitResponse(job_id=job.job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    """Checks a job's status. Keep polling until status is 'complete'."""
    job = Job(job_id, _redis_pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    if status != JobStatus.complete:
        return JobStatusResponse(job_id=job_id, status=status.value)

    job_result = await job.result_info()
    result_dict = job_result.result

    # See store.py: add_once prevents double-counting if this job gets polled
    # again after completion. NOTE: this only updates stats for jobs someone
    # actually polls to completion through this API process - Phase 5's
    # Postgres table (written by the worker directly) fixes that gap for good.
    store.add_once(
        job_id,
        RequestLog(
            success=result_dict["success"],
            latency_seconds=result_dict["latency_seconds"],
            estimated_cost_usd=result_dict["estimated_cost_usd"],
        ),
    )

    return JobStatusResponse(
        job_id=job_id,
        status="complete",
        result=ExtractResponse(**result_dict),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    """See the caveat in job_status() above regarding when this updates."""
    return StatsResponse(**store.compute_stats())