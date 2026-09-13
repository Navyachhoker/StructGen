"""
One-time setup script: creates the extraction_requests table in Supabase.

Run this once after setting DATABASE_URL in .env. Safe to re-run —
schema.sql uses CREATE TABLE IF NOT EXISTS, so it won't error or wipe
existing data on subsequent runs.
"""

import asyncio
from pathlib import Path

import asyncpg

from invoice_extractor.config import settings


async def main():
    schema_path = Path(__file__).parent.parent / "src" / "invoice_extractor" / "db" / "schema.sql"
    schema_sql = schema_path.read_text()

    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(schema_sql)
        print("✅ extraction_requests table ready.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())