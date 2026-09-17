import asyncio

from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.clients.local_lora_client import LocalLoRAClient
from invoice_extractor.clients.router import ModelRouter
from invoice_extractor.db.connection import get_pool
from invoice_extractor.worker import extract_invoice_task


async def main() -> None:
    with open(
        "data/samples/invoice_001_clean.txt",
        "r",
        encoding="utf-8",
    ) as file:
        raw_text = file.read()

    # The repository sample file is currently empty, so use a real
    # invoice directly for this integration test.
    if not raw_text.strip():
        raw_text = """TechMart
Invoice Number: INV-1001
Invoice Date: 2026-09-16

Laptop Stand    1    45.02    45.02
Mousepad        3    26.32    78.96
USB Hub         2    6.28    12.56

Subtotal: 136.54
Tax: 10.92
Total: 147.46
Currency: USD
"""

    lora_client = LocalLoRAClient()
    groq_client = GroqClient()

    router = ModelRouter(
        primary=lora_client,
        fallback=groq_client,
        primary_name="qwen-invoice-lora-v3",
        fallback_name="openai/gpt-oss-120b",
    )

    pool = await get_pool()

    ctx = {
        "router": router,
        "db_pool": pool,
        "job_id": "manual-lora-worker-test-001",
    }

    print("Running worker with Local LoRA as primary...")
    print()

    result = await extract_invoice_task(
        ctx,
        raw_text,
    )

    print("=" * 60)
    print("WORKER RESULT")
    print("=" * 60)

    print(f"Success: {result['success']}")
    print(f"Model: {result['model_name']}")
    print(f"Fallback used: {result['fallback_used']}")
    print(f"Latency: {result['latency_seconds']:.2f}s")
    print(f"Cost: ${result['estimated_cost_usd']:.6f}")

    print()
    print("Invoice:")
    print(result["invoice"])

    if result["error"]:
        print()
        print("Error:")
        print(result["error"])

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())