from prometheus_client import Counter, Histogram

RATE_LIMITED = Counter("costsentinel_rate_limited_total", "Requests rejected by the rate limiter", ["route"])
TOOL_CALLS = Counter("costsentinel_agent_tool_calls_total", "Agent tool invocations", ["tool", "ok"])
LLM_TOKENS = Counter("costsentinel_llm_tokens_total", "LLM tokens consumed", ["kind"])
COMPACTIONS = Counter("costsentinel_memory_compactions_total", "Session memory compactions", ["mode"])
DETECT_SECONDS = Histogram("costsentinel_detect_seconds", "Forecast+detect job duration", buckets=(1, 3, 5, 10, 20, 45, 90, 180))
