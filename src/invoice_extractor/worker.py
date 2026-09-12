"""
arq worker: runs invoice extraction as a background job instead of
blocking an HTTP request.

Why this exists: Phase 3's /extract endpoint blocked on the Groq call
for as long as the model took to respond. That's fine at low volume,
but ties up the web server's request thread — a real problem under load
or on a free-tier host prone to cold starts. arq moves the actual
extraction work into a separate process that pulls jobs off a
Redis-backed queue, so the API can respond instantly with a job ID.

Run this as a standalone process:
    arq invoice_extractor.worker.WorkerSettings
"""

from arq.connections import RedisSettings

from invoice_extractor.config import settings
from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.extraction import extract_invoice


async def startup(ctx: dict) -> None:
    """
    Runs once when the worker process starts. `ctx` is a plain dict arq
    shares across every job this worker processes — we stash the client
    here so it's created once per process, not once per job (avoids
    re-authenticating with Groq on every single extraction).
    """
    ctx["groq_client"] = GroqClient()


async def shutdown(ctx: dict) -> None:
    """Runs once when the worker process stops. Nothing to release yet —
    GroqClient doesn't hold a connection that needs explicit closing."""
    pass


async def extract_invoice_task(ctx: dict, raw_text: str) -> dict:
    """
    The actual background job. Whatever this returns is automatically
    stored by arq as the job's result, retrievable later via job.result().

    Returns a plain, JSON-safe dict rather than the ExtractionResult
    dataclass directly — arq needs to serialize the result to store it in
    Redis, and Pydantic's Decimal/date fields don't survive that without
    explicit conversion (hence model_dump(mode="json") below).
    """
    client = ctx["groq_client"]
    result = extract_invoice(raw_text, client)
    usage = client.get_last_usage() or {"latency_seconds": 0.0, "estimated_cost_usd": 0.0}

    return {
        "success": result.success,
        "invoice": result.invoice.model_dump(mode="json") if result.invoice else None,
        "error": result.error,
        "latency_seconds": usage["latency_seconds"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }


class WorkerSettings:
    """arq reads this class to configure and launch the worker process."""

    functions = [extract_invoice_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)