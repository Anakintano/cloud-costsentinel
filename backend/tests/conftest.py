import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
os.environ.pop("VAULT_ADDR", None)
os.environ.pop("REDIS_URL", None)
os.environ.pop("REQUIRE_API_KEY", None)

import pytest

from app import store


@pytest.fixture
def db():
    store.init("sqlite:///:memory:")
    with store.SessionLocal() as s:
        yield s


class FakeLLM:
    """Scripted LLM: pops responses; records every call so tests can assert on prompts."""
    def __init__(self, script=None, fail=False):
        self.script, self.calls, self.fail = list(script or []), [], fail

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=700):
        self.calls.append({"messages": messages, "tools": tools})
        if self.fail:
            raise RuntimeError("llm down")
        if self.script:
            return self.script.pop(0)
        return {"content": "ok", "tool_calls": [], "usage": {"prompt": 10, "completion": 5}}


def say(text):
    return {"content": text, "tool_calls": [], "usage": {"prompt": 10, "completion": 5}}


def call(name, args, cid="c1"):
    import json
    return {"content": "", "tool_calls": [{"id": cid, "name": name, "arguments": json.dumps(args)}], "usage": {"prompt": 10, "completion": 5}}
