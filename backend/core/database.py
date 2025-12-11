import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL")

_connection_pool = None
_checkpointer = None


async def initialize_checkpointer():
    global _connection_pool, _checkpointer

    _connection_pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        max_size=20,
        min_size=2,
        open=False
    )

    await _connection_pool.open()

    _checkpointer = AsyncPostgresSaver(_connection_pool)
    await _checkpointer.setup()

    return _checkpointer


def get_checkpointer():
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call initialize_checkpointer() first.")
    return _checkpointer


async def cleanup_checkpointer():
    global _connection_pool, _checkpointer
    if _connection_pool is not None:
        await _connection_pool.close()
        _connection_pool = None
    _checkpointer = None
