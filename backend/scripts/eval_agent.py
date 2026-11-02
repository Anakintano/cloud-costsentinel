"""Ground-truth evaluation of the LLM agent (real NVIDIA model, key fetched from Vault like production).
Scenarios inject a known fault into a real billing CSV; the agent investigates the detected alert and we check it
names the right service, dates the onset correctly, used tools, and did not state numbers absent from tool results.
Usage: VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=dev-root-token python scripts/eval_agent.py ../data/kaggle/cc/cloud_budget_2023_dataset.csv [trials]"""
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import pandas as pd

from app import store
from app.agent import NvidiaLLM, run_agent
from app.detect import detect
from app.focus import clean, daily, normalize
from app.grounding import ungrounded
from app.tools import Toolbox

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # models emit U+202F etc; cp1252 consoles crash on them
logging.disable(logging.INFO)
N = normalize(pd.read_csv(sys.argv[1]))
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
T0 = N.date.min()
day = lambda d: T0 + pd.Timedelta(days=d)
store.init("sqlite:///:memory:")
from app.grounding import normalize as nbsp  # NBSP/NNBSP inside "July 20", U+2011 inside ISO dates


def inject(n, svc, d0, d1, f):
    n = n.copy()
    for k, d in enumerate(range(d0, d1 + 1)):
        n.loc[(n.service == svc) & (n.date == day(d)), "cost"] *= f(k)
    return n


def setup(n):
    r = detect(daily(n))
    ser = pd.concat([f.assign(scope=sc).rename_axis("date").reset_index() for sc, f in r["forecasts"].items()], ignore_index=True)
    c = clean(n)
    bill = c.groupby(["date", "service", "region", "account"], as_index=False).cost.sum()
    an = [dict(id=i + 1, status="open", **a) for i, a in enumerate(r["anomalies"])]
    return Toolbox(ser, bill, an, r["capability"]), an


def pick(an, scopes, lo, hi):
    """The strongest alert in the scenario's window: rule-1 first, then largest |z|."""
    c = [a for a in an if a["scope"] in scopes and a["end"] >= day(lo) and a["start"] <= day(hi) and a["rule"].startswith("nelson")]
    return max(c, key=lambda a: (a["rule"] == "nelson1_spike", abs(a["peak_z"]))) if c else None


def question(a):
    return (f"Investigate anomaly #{a['id']}: scope={a['scope']}, rule={a['rule']}, alert window {a['start'].date()} to {a['end'].date()}, "
            f"peak z-score {a['peak_z']}. Determine whether spend really changed (compare against the weeks BEFORE the window), "
            "when it began, who drove it, and the likely root cause and next step.")


def dates_in(text):
    text = nbsp(text)
    out = [pd.Timestamp(m) for m in re.findall(r"\d{4}-\d{2}-\d{2}", text)]
    for mon, d in re.findall(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (\d{1,2})\b", text):
        try:
            out.append(pd.Timestamp(f"2023-{mon}-{d}"))
        except ValueError:
            pass
    return out


SCENARIOS = [
    dict(name="spike: Compute x4 on d200-201", n=lambda: inject(N, "Compute", 200, 201, lambda k: 4.0), svc="Compute", onset=200, tol=3,
         scopes=("Compute", "TOTAL"), window=(199, 215)),
    dict(name="leak: Storage +3%/day from d260", n=lambda: inject(N, "Storage", 260, 290, lambda j: 1.03 ** (j + 1)), svc="Storage", onset=260, tol=10,
         scopes=("Storage", "TOTAL"), window=(258, 300)),
    dict(name="REAL: organic Nov-2023 shift (+28%, no injected truth)", n=lambda: N, svc=None, onset=(pd.Timestamp("2023-11-01") - T0).days, tol=4,
         scopes=("TOTAL",), window=((pd.Timestamp("2023-10-29") - T0).days, (pd.Timestamp("2023-11-12") - T0).days)),
]

llm = NvidiaLLM()
rows = []
for sc in SCENARIOS:
    box, an = setup(sc["n"]())
    a = pick(an, sc["scopes"], *sc["window"])
    print(f"\n### {sc['name']}\nalert: {a and (a['scope'], a['rule'], str(a['start'].date()), str(a['end'].date()), a['peak_z'])}", flush=True)
    if not a:
        print("   NO ALERT RAISED -> detection miss, agent not run")
        rows.append((sc["name"], 0, TRIALS, None))
        continue
    ok_n = 0
    for t in range(TRIALS):
        with store.SessionLocal() as db:
            try:
                out = run_agent(db, box, llm, f"eval-{len(rows)}-{t}-xx", question(a))
            except Exception as e:
                print(f"   trial {t}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
        rep, tr = nbsp(out["reply"]), out["trace"]
        used = [x["tool"] for x in tr]
        svc_ok = sc["svc"] is None or sc["svc"].lower() in rep.lower()
        date_ok = any(abs((d - day(sc["onset"])).days) <= sc["tol"] for d in dates_in(rep))
        tools_ok = len(set(used)) >= 2
        bad = out['ungrounded']  # remaining after the agent's own repair round
        passed = svc_ok and date_ok and tools_ok
        ok_n += passed
        print(f"   trial {t}: {'PASS' if passed else 'FAIL'} service={svc_ok} onset_date={date_ok} tools={used} tokens={out['tokens_used']} repaired={out['repaired']} still_ungrounded={bad[:6]}")
        print("      reply:", rep.replace("\n", " ")[:420], flush=True)
    rows.append((sc["name"], ok_n, TRIALS, a["rule"]))
print("\n==== SUMMARY (pass = right service AND onset date within tolerance AND >=2 distinct tools)")
for n, k, t, r in rows:
    print(f"  {k}/{t}  {n}")
