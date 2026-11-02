"""Secrets come from Vault (KV v2). Token auth for dev, AppRole for prod. Cached 5 min so rotation propagates.
ALLOW_ENV_SECRETS=1 is a dev/test-only escape hatch; without Vault or it, we fail closed."""
import os
import time

import httpx

TTL = 300
_cache: dict = {}


class SecretError(RuntimeError):
    pass


def _token(addr):
    if t := os.environ.get("VAULT_TOKEN"):
        return t
    role, sec = os.environ.get("VAULT_ROLE_ID"), os.environ.get("VAULT_SECRET_ID")
    if not (role and sec):
        raise SecretError("Vault configured but no VAULT_TOKEN or VAULT_ROLE_ID/VAULT_SECRET_ID")
    r = httpx.post(f"{addr}/v1/auth/approle/login", json={"role_id": role, "secret_id": sec}, timeout=5)
    r.raise_for_status()
    return r.json()["auth"]["client_token"]


def get_secret(key: str, path: str = "costsentinel") -> str:
    hit = _cache.get(key)
    if hit and time.time() - hit[1] < TTL:
        return hit[0]
    addr = os.environ.get("VAULT_ADDR")
    try:
        if addr:
            r = httpx.get(f"{addr}/v1/secret/data/{path}", headers={"X-Vault-Token": _token(addr)}, timeout=5)
            r.raise_for_status()
            v = r.json()["data"]["data"].get(key)
        elif os.environ.get("ALLOW_ENV_SECRETS") == "1":
            v = os.environ.get(key.upper())
        else:
            raise SecretError("no VAULT_ADDR configured (set ALLOW_ENV_SECRETS=1 for dev only)")
    except httpx.HTTPError as e:
        raise SecretError(f"vault unreachable or denied: {type(e).__name__}") from None
    if not v:
        raise SecretError(f"secret '{key}' not found at secret/{path}")
    _cache[key] = (v, time.time())
    return v
