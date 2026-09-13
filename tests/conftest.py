"""
Shared pytest fixtures. Autouse fixtures here apply to every test in
the suite automatically, without needing to be requested by name.
"""

import asyncio
import pytest

from invoice_extractor.db import connection


@pytest.fixture(autouse=True)
def reset_db_pool():
    """
    db.connection.get_pool() caches a singleton Postgres pool at module
    level - correct for a long-running app process, but broken across
    tests, since each test's asyncio.run() (or TestClient's lifespan)
    creates and destroys its own event loop. A pool's connections are
    bound to the event loop that created them, so reusing a cached pool
    across test-boundary event loops causes "Event loop is closed" and
    "another operation is in progress" errors.

    This fixture forces a fresh pool per test by resetting the module-level
    cache before each test runs, and closing/clearing it after, so no
    connection outlives the event loop it was born on.
    """
    connection._pool = None
    yield
    if connection._pool is not None:
        try:
            asyncio.run(connection._pool.close())
        except Exception:
            pass  # pool's loop may already be closed - nothing to clean up
        connection._pool = None