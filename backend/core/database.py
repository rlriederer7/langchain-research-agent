import os
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

DATABASE_URL = os.getenv("DATABASE_URL")

_connection = None
_checkpointer = None


def initialize_checkpointer():
    global _connection, _checkpointer
    _connection = psycopg.connect(DATABASE_URL, autocommit=True)
    _checkpointer = PostgresSaver(_connection)
    _checkpointer.setup()

    return _checkpointer


def get_checkpointer():
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call initialize_checkpointer() first.")
    return _checkpointer


def cleanup_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        if hasattr(_checkpointer, 'close'):
            _checkpointer.close()
        _checkpointer = None
