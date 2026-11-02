"""Fixed-window rate limits + a daily LLM-token budget. Redis when REDIS_URL is set (shared by all replicas),
else an in-process dict (single replica only). If Redis errors mid-flight we degrade to the local dict rather than
fail open: cost protection matters more than a perfectly shared counter."""
import os
import time

import redis
from fastapi import HTTPException, Request


class Limiter:
    def __init__(self, url=None):
        self.r = redis.Redis.from_url(url, socket_timeout=0.5) if url else None
        self.mem: dict = {}

    def _incr(self, key, n, ttl):
        if self.r:
            try:
                p = self.r.pipeline()
                p.incrby(key, n)
                p.expire(key, ttl, nx=True)
                return int(p.execute()[0])
            except redis.RedisError:
                pass  # degrade to local counter
        now = time.time()
        self.mem = {k: v for k, v in self.mem.items() if v[1] > now}  # purge expired
        c, exp = self.mem.get(key, (0, now + ttl))
        self.mem[key] = (c + n, exp)
        return c + n

    def hit(self, client, name, limit, window=60):
        """Returns (allowed, retry_after_seconds)."""
        w = int(time.time() // window)
        c = self._incr(f"rl:{name}:{client}:{w}", 1, window)
        return c <= limit, window - int(time.time() % window)

    def tokens_used(self, client):
        return self._incr(f"tok:{client}:{time.strftime('%Y%m%d')}", 0, 86400)

    def add_tokens(self, client, n):
        self._incr(f"tok:{client}:{time.strftime('%Y%m%d')}", n, 86400)


_limiter = None


def limiter() -> Limiter:
    global _limiter
    if _limiter is None:
        _limiter = Limiter(os.environ.get("REDIS_URL"))
    return _limiter


def client_id(req: Request) -> str:
    return req.headers.get("x-api-key", "")[:16] or (req.client.host if req.client else "unknown")


def limit(name: str, per_minute: int):
    """FastAPI dependency: 429 + Retry-After when the caller exceeds `per_minute` on route-class `name`."""
    def dep(req: Request):
        ok, retry = limiter().hit(client_id(req), name, per_minute)
        if not ok:
            from .metrics import RATE_LIMITED
            RATE_LIMITED.labels(name).inc()
            raise HTTPException(429, f"rate limit exceeded ({per_minute}/min for '{name}')", headers={"Retry-After": str(retry)})
    return dep
