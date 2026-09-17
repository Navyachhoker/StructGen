"""
FastAPI application. Runs the arq worker as an in-process background task
(see lifespan) rather than a separate Render service, since Render's free
tier only has a free instance type for web services — background workers
start at $7/mo. Documented trade-off, not an oversight.
"""

import asyncio
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus
from arq.worker import Worker
from fastapi import FastAPI, HTTPException

from invoice_extractor.api.models import (
    ExtractRequest,
    ExtractResponse,
    JobSubmitResponse,
    JobStatusResponse,
    StatsResponse,
    InvoiceListItem,
)
from invoice_extractor.config import settings
from invoice_extractor.db.connection import get_pool, close_pool
from invoice_extractor.db.repository import get_stats, list_recent_extractions
from invoice_extractor.worker import (
    extract_invoice_task,
    startup as _worker_on_startup,
    shutdown as _worker_on_shutdown,
)

_redis_pool = None
_arq_worker = None
_arq_worker_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_pool, _arq_worker, _arq_worker_task

    _redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await get_pool()  # establishes the Postgres pool for this process

    _arq_worker = Worker(
        functions=[extract_invoice_task],
        redis_pool=_redis_pool,
        on_startup=_worker_on_startup,
        on_shutdown=_worker_on_shutdown,
        # Don't let arq install its own SIGINT/SIGTERM handlers — uvicorn
        # already owns process signal handling in this combined process.
        handle_signals=False,
    )
    _arq_worker_task = asyncio.create_task(_arq_worker.async_run())

    yield

    _arq_worker_task.cancel()
    try:
        await _arq_worker_task
    except asyncio.CancelledError:
        pass
    try:
        await _arq_worker.close()
    except AttributeError:
        # arq's Worker.close() sends itself SIGUSR1 internally, which
        # doesn't exist on Windows. Harmless locally — the worker task
        # is already cancelled above. Not an issue on Render's Linux env.
        pass
    await _redis_pool.aclose()
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
    job = Job(job_id, _redis_pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    if status != JobStatus.complete:
        return JobStatusResponse(job_id=job_id, status=status.value)

    job_result = await job.result_info()

    if not job_result.success:
        # The task itself raised (a bug in extract_invoice_task, not a
        # normal extraction failure — those are reported via
        # ExtractResponse.success=False instead). job_result.result is
        # the raised exception object here, not a result dict, so it
        # can't be **-unpacked into ExtractResponse.
        raise HTTPException(
            status_code=500,
            detail=f"Worker task raised an exception: {job_result.result!r}",
        )

    return JobStatusResponse(
        job_id=job_id,
        status="complete",
        result=ExtractResponse(**job_result.result),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    pool = await get_pool()
    return StatsResponse(**(await get_stats(pool)))


@app.get("/invoices", response_model=list[InvoiceListItem])
async def list_invoices(limit: int = 20):
    """Recent extraction records for the dashboard's invoice table."""
    pool = await get_pool()
    return await list_recent_extractions(pool, limit)