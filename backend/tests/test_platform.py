"""Rate limiting, secrets and the HTTP API. Redis/Vault tests run against the compose containers and skip if absent."""
import io

import httpx
import pytest
import redis
from fastapi.testclient import TestClient

from app import ratelimit, secrets, store
from app.main import app, get_llm
from tests.conftest import FakeLLM, call, say


def test_limiter_blocks_after_limit_and_isolates_clients():
    lim = ratelimit.Limiter()
    assert [lim.hit("a", "llm", 3)[0] for _ in range(5)] == [True, True, True, False, False]
    assert lim.hit("b", "llm", 3)[0] is True


def test_token_budget_accumulates():
    lim = ratelimit.Limiter()
    lim.add_tokens("a", 500)
    lim.add_tokens("a", 700)
    assert lim.tokens_used("a") == 1200 and lim.tokens_used("b") == 0


def _redis():
    try:
        r = redis.Redis(port=6379, socket_timeout=0.5)
        r.ping()
        return r
    except redis.RedisError:
        pytest.skip("redis container not running")


def test_limiter_with_real_redis_is_shared_across_instances():
    r = _redis()
    r.flushdb()
    a, b = ratelimit.Limiter("redis://127.0.0.1:6379/0"), ratelimit.Limiter("redis://127.0.0.1:6379/0")  # two "replicas"
    res = [a.hit("c", "llm", 4)[0], b.hit("c", "llm", 4)[0], a.hit("c", "llm", 4)[0], b.hit("c", "llm", 4)[0], a.hit("c", "llm", 4)[0]]
    assert res == [True, True, True, True, False]
    assert r.ttl(next(iter(r.scan_iter("rl:llm:c:*")))) > 0  # window keys expire (no unbounded growth)


def test_limiter_degrades_to_local_when_redis_dies():
    lim = ratelimit.Limiter("redis://127.0.0.1:1/0")  # nothing listens
    assert [lim.hit("a", "x", 2)[0] for _ in range(3)] == [True, True, False]


def test_secrets_fail_closed_without_vault(monkeypatch):
    secrets._cache.clear()
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("ALLOW_ENV_SECRETS", raising=False)
    with pytest.raises(secrets.SecretError):
        secrets.get_secret("nvidia_api_key")


def test_secrets_from_real_vault():
    secrets._cache.clear()
    try:
        httpx.get("http://127.0.0.1:8200/v1/sys/health", timeout=1)
    except httpx.HTTPError:
        pytest.skip("vault container not running")
    import os
    os.environ.update(VAULT_ADDR="http://127.0.0.1:8200", VAULT_TOKEN="dev-root-token")
    try:
        k = secrets.get_secret("nvidia_api_key")
    except secrets.SecretError:
        pytest.skip("vault not seeded (make seed)")
    finally:
        os.environ.pop("VAULT_ADDR"), os.environ.pop("VAULT_TOKEN")
    assert len(k) > 20  # a real value came back from Vault (no vendor-prefix assumption: CI seeds a placeholder)
    with pytest.raises(secrets.SecretError):
        os.environ.update(VAULT_ADDR="http://127.0.0.1:8200", VAULT_TOKEN="wrong-token")
        secrets._cache.clear()
        try:
            secrets.get_secret("nvidia_api_key")
        finally:
            os.environ.pop("VAULT_ADDR"), os.environ.pop("VAULT_TOKEN")


# ---------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    store.init("sqlite:///:memory:")
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def demo_csv(days=160, **kw):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from make_demo_csv import make
    return make(days=days, **kw).to_csv(index=False).encode()


def test_api_rejects_bad_uploads(client):
    assert client.post("/upload", files={"file": ("x.csv", b"a,b\n1,2\n")}).status_code == 422
    assert client.post("/upload", files={"file": ("x.csv", demo_csv(days=60))}).status_code == 422  # too short
    assert client.post("/upload", files={"file": ("x.csv", b"\xff\xfe\x00bad")}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code in (409, 503)  # no data / no LLM, never a 500


def test_full_flow_upload_detect_agent_memory(client):
    r = client.post("/upload", files={"file": ("demo.csv", demo_csv())}, data={"budget": "4000"})
    assert r.status_code == 200, r.text
    assert r.json()["anomalies"] >= 3 and r.json()["capability"]["budget"] == 4000
    an = client.get("/anomalies").json()
    spike = next(a for a in an if a["scope"] == "AmazonEC2" and a["rule"] == "nelson1_spike")
    ser = client.get("/series/TOTAL").json()
    assert len(ser) == 160 and ser[0]["ucl"] > ser[0]["expected"]
    assert client.get("/series/Nope").status_code == 404

    llm = FakeLLM([call("top_movers", {"date": str(spike["start"]), "dimension": "service"}), say("EC2 in us-east-1 quadrupled."),
                   say("second reply"), say("third")])
    app.dependency_overrides[get_llm] = lambda: llm
    e = client.post(f"/anomalies/{spike['id']}/explain")
    assert e.status_code == 200 and e.json()["trace"][0]["result"]["movers"][0]["name"] == "AmazonEC2"
    assert client.post(f"/anomalies/{spike['id']}/explain").json()["cached"] is True  # no second LLM spend

    # user feedback -> pinned fact -> visible to the agent on the next chat
    client.post(f"/anomalies/{spike['id']}/status", json={"status": "expected", "note": "load test by platform team"})
    c = client.post("/chat", json={"message": "was anything planned?", "session_id": "session-abc123"})
    assert c.status_code == 200 and "load test by platform team" in llm.calls[-1]["messages"][0]["content"]
    info = client.get("/sessions/session-abc123").json()
    assert info["live_messages"] == 2
    assert client.post("/chat", json={"message": "x", "session_id": "../etc"}).status_code == 422
    assert client.delete("/sessions/session-abc123").status_code == 200

    assert client.put("/budget", json={"budget": 10}).json()["cpk"] < 0  # budget far below spend -> negative capability


def test_chat_is_rate_limited_with_retry_after(client):
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    ratelimit._limiter = ratelimit.Limiter()
    codes = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(12)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_daily_token_budget_blocks_llm(client, monkeypatch):
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    ratelimit._limiter = ratelimit.Limiter()
    ratelimit._limiter.add_tokens("testclient", 10**9)
    assert client.post("/chat", json={"message": "hi"}).status_code == 429
    ratelimit._limiter = ratelimit.Limiter()


def test_api_key_enforced_when_required(client, monkeypatch):
    monkeypatch.setenv("REQUIRE_API_KEY", "1")
    monkeypatch.setenv("ALLOW_ENV_SECRETS", "1")
    monkeypatch.setenv("APP_API_KEY", "k-123")
    secrets._cache.clear()
    assert client.get("/anomalies").status_code == 401
    assert client.get("/anomalies", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/anomalies", headers={"x-api-key": "k-123"}).status_code == 200
    assert client.get("/healthz").status_code == 200  # probes stay open
    secrets._cache.clear()


def test_metrics_endpoint_exposes_custom_and_http_metrics(client):
    body = client.get("/metrics").text
    for m in ("costsentinel_agent_tool_calls_total", "costsentinel_rate_limited_total", "costsentinel_detect_seconds", "http_request_duration"):
        assert m in body, m


def test_empty_llm_output_is_a_502_never_a_cached_empty_explanation(client):
    empty = {"content": "", "tool_calls": [], "usage": {}, "finish": "length"}
    app.dependency_overrides[get_llm] = lambda: FakeLLM([empty, empty])
    ratelimit._limiter = ratelimit.Limiter()
    aid = client.get("/anomalies").json()[0]["id"]
    assert client.post(f"/anomalies/{aid}/explain?refresh=true").status_code == 502
    assert [a for a in client.get("/anomalies").json() if a["id"] == aid][0]["explained"] in (True, False)
    from app.store import Anomaly
    with store.SessionLocal() as s:
        assert not s.get(Anomaly, aid).explanation  # nothing empty was stored as an answer
