"""
Manages the asyncpg connection pool.

Same singleton-pool pattern used for Redis in api/main.py (Phase 4):
one pool created at process startup, shared across all requests/jobs
in that process, rather than opening a new connection per query.
"""

import asyncpg

from invoice_extractor.config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Lazily creates the pool on first use, reuses it after that.
    Both the API process and the worker process call this independently —
    each gets its own pool, since they're separate processes."""
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise ValueError("DATABASE_URL is not set. Copy .env.example to .env and add it.")
        _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    return _pool


async def close_pool() -> None:
    """Call on process shutdown to release connections cleanly."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None