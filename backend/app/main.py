"""CostSentinel API. Stateless: all state lives in Postgres (durable), Redis (rate counters), Vault (secrets)."""
import hmac
import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import openai
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import delete, text
from sqlalchemy.exc import OperationalError

from . import agent, memory, store
from .detect import detect
from .focus import clean, daily, normalize
from .metrics import DETECT_SECONDS
from .ratelimit import client_id, limit, limiter
from .secrets import SecretError, get_secret
from .sixsigma import capability
from .store import Anomaly, Billing, Fact, Message, Series, Setting, Summary, get_db
from .tools import Toolbox

MIN_DAYS = 84  # 42 warm-up + 28 to seed sigma + >=14 evaluated days
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", 50)) * 1024 * 1024
DAILY_TOKENS = int(os.environ.get("DAILY_LLM_TOKENS", 200_000))
SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@asynccontextmanager
async def lifespan(app):
    if store.SessionLocal is None:
        for attempt in range(20):  # DB may still be starting (k8s starts pods in parallel): retry ~1 min before giving up
            try:
                store.init(os.environ.get("DATABASE_URL", "sqlite:///./costsentinel.db"))
                break
            except OperationalError:
                if attempt == 19:
                    raise
                time.sleep(3)
    yield


app = FastAPI(title="CostSentinel", version="0.1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def auth(req: Request):
    """Enforced only when REQUIRE_API_KEY=1; the expected key lives in Vault (app_api_key)."""
    if os.environ.get("REQUIRE_API_KEY") != "1":
        return
    try:
        expected = get_secret("app_api_key")
    except SecretError:
        raise HTTPException(503, "auth backend unavailable") from None
    if not hmac.compare_digest(req.headers.get("x-api-key", ""), expected):
        raise HTTPException(401, "invalid or missing X-API-Key")


def get_llm():
    try:
        return agent.NvidiaLLM()
    except SecretError as e:
        raise HTTPException(503, f"LLM unavailable: {e}") from None


def toolbox(db) -> Toolbox:
    d = lambda q: pd.read_sql(q.statement, db.connection())
    s, b = d(db.query(Series)), d(db.query(Billing))
    for df in (s, b):
        df["date"] = pd.to_datetime(df["date"])
    an = [dict(id=a.id, scope=a.scope, rule=a.rule, start=a.start, end=a.end, peak_z=a.peak_z, severity=a.severity, status=a.status, explanation=a.explanation)
          for a in db.query(Anomaly).order_by(Anomaly.id)]
    cap = db.get(Setting, "capability")
    return Toolbox(s, b, an, cap.value if cap else None)


def need_data(db):
    if not db.query(Series).first():
        raise HTTPException(409, "no dataset yet: POST /upload first")


# ---------------------------------------------------------------- health
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz(db=Depends(get_db)):
    out = {"database": "ok"}
    try:
        db.execute(text("select 1"))
    except Exception:
        raise HTTPException(503, {"database": "down"}) from None
    lim = limiter()
    out["redis"] = "n/a (in-process limiter)" if not lim.r else ("ok" if _ping(lim.r) else "down (degraded to local limits)")
    try:
        get_secret("nvidia_api_key")
        out["vault"] = "ok"
    except SecretError as e:
        out["vault"] = f"unavailable: {e}"
    return out


def _ping(r):
    try:
        return r.ping()
    except Exception:
        return False


# ---------------------------------------------------------------- data
@app.post("/upload", dependencies=[Depends(auth), Depends(limit("upload", 6))])
def upload(file: UploadFile = File(...), budget: float | None = Form(None), db=Depends(get_db)):
    """Replaces the current dataset. Synchronous (a few seconds per ~10 series); move to a worker if datasets grow."""
    raw = file.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD // 2**20} MB")
    try:
        n = clean(normalize(pd.read_csv(io.BytesIO(raw), low_memory=False)))
        w = daily(n)
    except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise HTTPException(422, f"could not parse as billing CSV: {e}") from None
    if len(w) < MIN_DAYS:
        raise HTTPException(422, f"need at least {MIN_DAYS} days of history (6 weeks warm-up + 4 weeks to calibrate limits + evaluation), got {len(w)}")
    t0 = time.time()
    res = detect(w, budget)
    DETECT_SECONDS.observe(time.time() - t0)
    for model in (Billing, Series, Anomaly):
        db.execute(delete(model))
    g = n.groupby(["date", "service", "region", "account"], as_index=False).cost.sum()
    db.bulk_insert_mappings(Billing, [dict(date=r.date.date(), service=r.service, region=str(r.region), account=str(r.account), cost=float(r.cost)) for r in g.itertuples()])
    for scope, f in res["forecasts"].items():
        db.bulk_insert_mappings(Series, [dict(scope=scope, date=d.date(), actual=float(r.actual), yhat=float(r.yhat), lo=float(r.lo), hi=float(r.hi), z=float(r.z), sigma=float(r.sigma), evaluated=bool(r.evaluated)) for d, r in f.iterrows()])
    for a in res["anomalies"]:
        db.add(Anomaly(scope=a["scope"], rule=a["rule"], start=a["start"].date(), end=a["end"].date(), peak_z=a["peak_z"], severity=a["severity"]))
    cap = db.get(Setting, "capability")
    if cap:
        cap.value = res["capability"]
    else:
        db.add(Setting(key="capability", value=res["capability"]))
    db.commit()
    return {"days": len(w), "services": list(w.columns), "anomalies": len(res["anomalies"]), "capability": res["capability"],
            "detect_seconds": round(time.time() - t0, 1)}


@app.get("/anomalies", dependencies=[Depends(auth)])
def anomalies(scope: str | None = None, status: str | None = None, db=Depends(get_db)):
    q = db.query(Anomaly).order_by(Anomaly.start.desc())
    if scope:
        q = q.filter(Anomaly.scope == scope)
    if status:
        q = q.filter(Anomaly.status == status)
    return [dict(id=a.id, scope=a.scope, rule=a.rule, start=a.start, end=a.end, peak_z=a.peak_z, severity=a.severity, status=a.status,
                 explained=a.explanation is not None) for a in q]


@app.get("/series/{scope}", dependencies=[Depends(auth)])
def series(scope: str, db=Depends(get_db)):
    rows = db.query(Series).filter(Series.scope == scope).order_by(Series.date).all()
    if not rows:
        raise HTTPException(404, f"unknown scope '{scope}'")
    return [dict(date=r.date, actual=r.actual, expected=r.yhat, lo=r.lo, hi=r.hi, ucl=r.yhat + 3 * r.sigma, lcl=r.yhat - 3 * r.sigma, z=r.z, evaluated=r.evaluated) for r in rows]


@app.get("/summary", dependencies=[Depends(auth)])
def summary(db=Depends(get_db)):
    """Everything the UI header needs in one call."""
    scopes = [r[0] for r in db.query(Series.scope).distinct()]
    if not scopes:
        return {"has_data": False}
    tot = db.query(Series).filter(Series.scope == "TOTAL").order_by(Series.date).all()
    cap = db.get(Setting, "capability")
    sev = {s: db.query(Anomaly).filter(Anomaly.severity == s, Anomaly.status == "open").count() for s in ("high", "medium", "low")}
    return {"has_data": True, "scopes": sorted(scopes, key=lambda x: (x != "TOTAL", x)), "start": tot[0].date, "end": tot[-1].date, "days": len(tot),
            "capability": cap.value if cap else None, "open_by_severity": sev}


class BudgetIn(BaseModel):
    budget: float = Field(gt=0)


@app.put("/budget", dependencies=[Depends(auth)])
def set_budget(b: BudgetIn, db=Depends(get_db)):
    need_data(db)
    x = [r.actual for r in db.query(Series).filter(Series.scope == "TOTAL").order_by(Series.date.desc()).limit(28)]
    cap = {**capability(x, b.budget), "budget": b.budget}
    s = db.get(Setting, "capability")
    s.value = cap
    db.commit()
    return cap


# ---------------------------------------------------------------- agent
class ChatIn(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


def _run(db, req, sid, message, llm):
    need_data(db)
    cid = client_id(req)
    if limiter().tokens_used(cid) > DAILY_TOKENS:
        raise HTTPException(429, "daily LLM token budget exhausted", headers={"Retry-After": "3600"})
    try:
        out = agent.run_agent(db, toolbox(db), llm, sid, message)
    except openai.APIError as e:
        raise HTTPException(502, f"LLM provider error: {type(e).__name__}") from None
    except agent.AgentError as e:
        raise HTTPException(502, str(e)) from None
    limiter().add_tokens(cid, out["tokens_used"])
    return out


@app.post("/chat", dependencies=[Depends(auth), Depends(limit("llm", 10))])
def chat(body: ChatIn, req: Request, db=Depends(get_db), llm=Depends(get_llm)):
    sid = body.session_id or uuid.uuid4().hex
    if not SID.match(sid):
        raise HTTPException(422, "session_id must match [A-Za-z0-9_-]{8,64}")
    return {"session_id": sid, **_run(db, req, sid, body.message, llm)}


@app.post("/anomalies/{aid}/explain", dependencies=[Depends(auth), Depends(limit("llm", 10))])
def explain(aid: int, req: Request, refresh: bool = False, db=Depends(get_db), llm=Depends(get_llm)):
    a = db.get(Anomaly, aid)
    if not a:
        raise HTTPException(404, "no such anomaly")
    if a.explanation and not refresh:
        return {"id": aid, "explanation": a.explanation, "cached": True}
    q = (f"Investigate anomaly #{a.id}: scope={a.scope}, rule={a.rule}, alert window {a.start} to {a.end}, peak z-score {a.peak_z}. "
         "Determine whether spend really changed (compare against the weeks BEFORE the window), when it began, who drove it, "
         "and the likely root cause and next step. Write for a non-technical manager: no statistics terms (no z-score, changepoint, ramp-onset), "
         "at most 5 short bullets, and finish with a bullet starting 'Next step:'.")
    out = _run(db, req, f"anomaly-{aid}-{uuid.uuid4().hex[:6]}", q, llm)
    a.explanation = out["reply"]
    db.commit()
    return {"id": aid, "explanation": a.explanation, "cached": False, "trace": out["trace"], "tokens_used": out["tokens_used"],
            "ungrounded_figures": out["ungrounded"]}


class StatusIn(BaseModel):
    status: Literal["open", "expected"]
    note: str = Field("", max_length=250)


@app.post("/anomalies/{aid}/status", dependencies=[Depends(auth)])
def set_status(aid: int, body: StatusIn, db=Depends(get_db)):
    """User feedback. Marking 'expected' with a note becomes a pinned long-term fact the agent sees from then on."""
    a = db.get(Anomaly, aid)
    if not a:
        raise HTTPException(404, "no such anomaly")
    a.status = body.status
    if body.status == "expected" and body.note:
        db.add(Fact(text=f"{a.scope} {a.start}..{a.end} ({a.rule}) was expected: {body.note}"[:300]))
    db.commit()
    return {"id": aid, "status": a.status}


# ---------------------------------------------------------------- memory
class FactIn(BaseModel):
    text: str = Field(min_length=3, max_length=300)


@app.get("/memory/facts", dependencies=[Depends(auth)])
def facts(db=Depends(get_db)):
    return [dict(id=f.id, text=f.text, created=f.created) for f in db.query(Fact).order_by(Fact.id)]


@app.post("/memory/facts", dependencies=[Depends(auth), Depends(limit("write", 30))])
def add_fact(f: FactIn, db=Depends(get_db)):
    row = Fact(text=f.text)
    db.add(row)
    db.commit()
    return {"id": row.id}


@app.delete("/memory/facts/{fid}", dependencies=[Depends(auth)])
def del_fact(fid: int, db=Depends(get_db)):
    db.execute(delete(Fact).where(Fact.id == fid))
    db.commit()
    return {"deleted": fid}


@app.get("/sessions/{sid}", dependencies=[Depends(auth)])
def session(sid: str, db=Depends(get_db)):
    info = memory.session_info(db, sid)
    if not info["live_messages"] and not info["compacted_messages"]:
        raise HTTPException(404, "no such session")
    return info


@app.post("/sessions/{sid}/compact", dependencies=[Depends(auth), Depends(limit("llm", 10))])
def force_compact(sid: str, db=Depends(get_db), llm=Depends(get_llm)):
    return {"mode": memory.compact(db, sid, llm, force=True), **memory.session_info(db, sid)}


@app.delete("/sessions/{sid}", dependencies=[Depends(auth)])
def del_session(sid: str, db=Depends(get_db)):
    db.execute(delete(Message).where(Message.session_id == sid))
    db.execute(delete(Summary).where(Summary.session_id == sid))
    db.commit()
    return {"deleted": sid}


# The UI is static files served by the API itself (one container, same origin, no CORS). Mounted last so API routes win.
_ui = Path(__file__).parents[1] / "ui"
if _ui.is_dir():
    app.mount("/", StaticFiles(directory=_ui, html=True), name="ui")
