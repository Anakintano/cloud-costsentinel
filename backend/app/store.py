"""Postgres (prod) / SQLite (tests). Durable: billing, series, anomalies, chat messages, summaries, facts."""
import datetime as dt
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

now = lambda: dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Billing(Base):
    __tablename__ = "billing"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    service: Mapped[str] = mapped_column(String(128))
    region: Mapped[str] = mapped_column(String(64))
    account: Mapped[str] = mapped_column(String(64))
    cost: Mapped[float] = mapped_column(Float)


class Series(Base):
    __tablename__ = "series"
    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[str] = mapped_column(String(128), index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    actual: Mapped[float] = mapped_column(Float)
    yhat: Mapped[float] = mapped_column(Float)
    lo: Mapped[float] = mapped_column(Float)
    hi: Mapped[float] = mapped_column(Float)
    z: Mapped[float] = mapped_column(Float)
    sigma: Mapped[float] = mapped_column(Float)
    evaluated: Mapped[bool] = mapped_column(Boolean, default=True)  # False = warm-up days (no blind forecast yet)


class Anomaly(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[str] = mapped_column(String(128), index=True)
    rule: Mapped[str] = mapped_column(String(32))
    start: Mapped[dt.date] = mapped_column(Date)
    end: Mapped[dt.date] = mapped_column(Date)
    peak_z: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(12), default="open")  # open | expected
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(12))  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)  # assistant: [{id,name,arguments}]
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    compacted: Mapped[bool] = mapped_column(Boolean, default=False)
    created: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)


class Summary(Base):
    __tablename__ = "summaries"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    compactions: Mapped[int] = mapped_column(Integer, default=0)


class Fact(Base):
    """Long-term memory. Written ONLY by the user (API), never by the agent: a poisoned tool output can't persist."""
    __tablename__ = "facts"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String(300))
    created: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)


SessionLocal = None


def migrate(eng):
    """Apply Alembic migrations on startup. On Postgres a transaction-scoped advisory lock serialises replicas that boot
    together (DDL is transactional there, so a loser just waits, then finds nothing to do)."""
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "alembic"))
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            conn.execute(text("select pg_advisory_xact_lock(727274)"))
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def init(url: str):
    global SessionLocal
    kw = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool} if url.startswith("sqlite") else {"pool_pre_ping": True}
    eng = create_engine(url, **kw)
    migrate(eng)
    SessionLocal = sessionmaker(eng, expire_on_commit=False)
    return eng


def get_db():
    with SessionLocal() as s:
        yield s
