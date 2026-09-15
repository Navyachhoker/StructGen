"""
All SQL for extraction_requests lives here — the one place that knows
the table's column names. Nothing else in the codebase should write
raw SQL; if you need a new query, add a function here.
"""

import json
from typing import Optional

import asyncpg


async def insert_extraction_record(
    pool: asyncpg.Pool,
    job_id: str,
    model_name: str,
    success: bool,
    invoice_dict: Optional[dict],
    error: Optional[str],
    latency_seconds: float,
    estimated_cost_usd: float,
) -> None:
    """
    Inserts one completed extraction record. Called by the worker
    immediately after each job finishes — not by the API — so a record
    exists whether or not any client ever polls the job.

    ON CONFLICT guards against the (unlikely but possible) case of arq
    retrying a job after a transient failure and re-running the task —
    without this, a retried job would violate the UNIQUE constraint on job_id.
    """
    await pool.execute(
        """
        INSERT INTO extraction_requests
            (job_id, model_name, success, invoice_json, error, latency_seconds, estimated_cost_usd)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (job_id) DO NOTHING
        """,
        job_id,
        model_name,
        success,
        json.dumps(invoice_dict) if invoice_dict else None,
        error,
        latency_seconds,
        estimated_cost_usd,
    )


async def get_stats(pool: asyncpg.Pool) -> dict:
    """
    Computes the same aggregate shape the in-memory store produced in
    Phase 3/4, but now sourced from every extraction ever recorded,
    across all processes and restarts — not just this process's memory.
    """
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_requests,
            COUNT(*) FILTER (WHERE success) AS successful_requests,
            AVG(latency_seconds) AS average_latency_seconds,
            COALESCE(SUM(estimated_cost_usd), 0) AS total_estimated_cost_usd
        FROM extraction_requests
        """
    )

    total = row["total_requests"]
    successful = row["successful_requests"]

    return {
        "total_requests": total,
        "successful_requests": successful,
        "success_rate": round(successful / total, 4) if total > 0 else 0.0,
        "average_latency_seconds": round(row["average_latency_seconds"], 3) if row["average_latency_seconds"] else None,
        "total_estimated_cost_usd": round(float(row["total_estimated_cost_usd"]), 6),
    }
    
async def list_recent_extractions(pool: asyncpg.Pool, limit: int = 20) -> list[dict]:
    """
    Recent extraction records for the dashboard's invoice list. Pulls
    display fields out of invoice_json since the table stores the full
    extracted invoice as JSONB, not as separate columns (see schema.sql).
    Rows where extraction failed still show up, with these fields NULL —
    the dashboard needs to show failures alongside successes.
    """
    rows = await pool.fetch(
        """
        SELECT
            job_id,
            model_name,
            success,
            error,
            invoice_json ->> 'vendor_name'    AS vendor_name,
            invoice_json ->> 'invoice_number' AS invoice_number,
            invoice_json ->> 'invoice_date'   AS invoice_date,
            invoice_json ->> 'total_amount'   AS total_amount,
            latency_seconds,
            estimated_cost_usd,
            created_at
        FROM extraction_requests
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]