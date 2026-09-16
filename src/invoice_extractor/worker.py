"""
arq worker — now writes directly to Postgres after each job, instead of
relying on the API to log results (Phase 4's limitation).
"""

from arq.connections import RedisSettings

from invoice_extractor.config import settings
from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.extraction import extract_invoice
from invoice_extractor.db.connection import get_pool, close_pool
from invoice_extractor.db.repository import insert_extraction_record

# Identifies which model produced a given record — see schema.sql comment
# for why this column exists ahead of Phase 7's second model.
_MODEL_NAME = "openai/gpt-oss-120b"


async def startup(ctx: dict) -> None:
    ctx["groq_client"] = GroqClient()
    ctx["db_pool"] = await get_pool()


async def shutdown(ctx: dict) -> None:
    await close_pool()


async def extract_invoice_task(ctx: dict, raw_text: str) -> dict:
    client = ctx["groq_client"]
    result = extract_invoice(raw_text, client)
    usage = client.get_last_usage() or {"latency_seconds": 0.0, "estimated_cost_usd": 0.0}

    invoice_dict = result.invoice.model_dump(mode="json") if result.invoice else None

    result_payload = {
        "success": result.success,
        "invoice": invoice_dict,
        "error": result.error,
        "latency_seconds": usage["latency_seconds"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }

    # Write the permanent record here, in the worker, right after the
    # job completes — this is what makes /stats correct regardless of
    # whether any client ever polls this specific job to completion.
    job_id = ctx["job_id"]  # arq injects the current job's ID into ctx automatically
    await insert_extraction_record(
        pool=ctx["db_pool"],
        job_id=job_id,
        model_name=_MODEL_NAME,
        success=result.success,
        invoice_dict=invoice_dict,
        error=result.error,
        latency_seconds=usage["latency_seconds"],
        estimated_cost_usd=usage["estimated_cost_usd"],
    )

    return result_payload


class WorkerSettings:
    functions = [extract_invoice_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)