"""
arq worker for invoice extraction.

Primary model:
    Local Qwen2.5-1.5B + LoRA v3

Fallback model:
    Groq GPT-OSS-120B

Results are persisted to Postgres after each job.
"""

from arq.connections import RedisSettings

from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.clients.local_lora_client import LocalLoRAClient
from invoice_extractor.clients.router import ModelRouter
from invoice_extractor.config import settings
from invoice_extractor.db.connection import close_pool, get_pool
from invoice_extractor.db.repository import insert_extraction_record
from invoice_extractor.extraction import extract_invoice


_PRIMARY_MODEL_NAME = "qwen-invoice-lora-v3"
_FALLBACK_MODEL_NAME = "openai/gpt-oss-120b"


async def startup(ctx: dict) -> None:
    """Create model clients and database pool once per worker."""

    local_lora_client = LocalLoRAClient()

    groq_client = GroqClient()

    ctx["router"] = ModelRouter(
        primary=local_lora_client,
        fallback=groq_client,
        primary_name=_PRIMARY_MODEL_NAME,
        fallback_name=_FALLBACK_MODEL_NAME,
    )

    ctx["db_pool"] = await get_pool()


async def shutdown(ctx: dict) -> None:
    """Close shared resources when the worker shuts down."""

    await close_pool()


async def extract_invoice_task(ctx: dict, raw_text: str) -> dict:
    """Extract an invoice and persist the result."""

    router = ctx["router"]

    result = extract_invoice(raw_text, router)

    metadata = router.get_last_metadata()

    # Determine which underlying client produced the result.
    if metadata and metadata.fallback_used:
        client = router.fallback
    else:
        client = router.primary

    usage = client.get_last_usage() or {
        "latency_seconds": 0.0,
        "estimated_cost_usd": 0.0,
        "model_name": metadata.model_used if metadata else _PRIMARY_MODEL_NAME,
    }

    invoice_dict = (
        result.invoice.model_dump(mode="json")
        if result.invoice
        else None
    )

    result_payload = {
        "success": result.success,
        "invoice": invoice_dict,
        "error": result.error,
        "latency_seconds": usage["latency_seconds"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "model_name": metadata.model_used if metadata else _PRIMARY_MODEL_NAME,
        "fallback_used": metadata.fallback_used if metadata else False,
    }

    job_id = ctx["job_id"]

    await insert_extraction_record(
        pool=ctx["db_pool"],
        job_id=job_id,
        model_name=metadata.model_used if metadata else _PRIMARY_MODEL_NAME,
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