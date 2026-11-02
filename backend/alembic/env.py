"""Alembic env: the connection is handed in by app.store.init() so one code path migrates sqlite/postgres alike."""
from alembic import context

from app.store import Base

target_metadata = Base.metadata
conn = context.config.attributes["connection"]
context.configure(connection=conn, target_metadata=target_metadata, compare_type=True)
with context.begin_transaction():
    context.run_migrations()
