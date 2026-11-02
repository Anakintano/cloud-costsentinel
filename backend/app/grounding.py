"""Deterministic check that the agent's final answer only states numbers/dates found in tool results or the question.
LLMs mis-add, mis-copy and typo (observed: '2103-11-02'); this catches it without trusting the model to self-police."""
import json
import re

_TYPO = {0x202F: " ", 0x00A0: " ", 0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2212: "-"}
normalize = lambda t: (t or "").translate(_TYPO)  # models emit U+2011 inside ISO dates and NNBSP inside "July 20"
_NUM = re.compile(r"(?<![\d-])\d[\d,]*\.?\d*")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def ungrounded(reply: str, trace: list, question: str = "") -> list[str]:
    blob = json.dumps(trace) + " " + question
    reply = normalize(reply)
    bad = [d for d in _DATE.findall(reply) if d not in blob and not _plausible_range_date(d, blob)]
    for x in _NUM.findall(_DATE.sub(" ", reply)):  # dates were judged above; don't re-report their digits
        v = x.replace(",", "").rstrip(".")
        digits = v.replace(".", "")
        if len(digits) < 3 or re.fullmatch(r"(19|20)\d\d", v) or not v:
            continue
        f = float(v)
        if not any(s in blob for s in {v, f"{f:.1f}", f"{f:.2f}", f"{round(f)}", f"{f:g}"}):
            bad.append(x)
    return list(dict.fromkeys(bad))


def _plausible_range_date(d: str, blob: str) -> bool:
    """A date inside a window the tools returned (e.g. a day mentioned from a get_series range) is fine if its month appears."""
    return d[:7] in blob and d[:4] in blob and d[:4] != "2103" and d[:7] + "-" in blob
