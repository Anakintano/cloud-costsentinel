"""Evaluate detection on a long billing CSV. Walk-forward makes every day after warm-up out-of-sample, so one run
per scenario is a faithful 'daily monitor' replay.
Usage: python scripts/eval_dataset.py ../data/kaggle/cc/cloud_budget_2023_dataset.csv
Note: the dataset has no usable anomaly labels (is_anomaly is all 0), so recall is measured on injected faults and
false alarms on the un-modified data (which contains genuine organic level shifts: Jun-Jul and Nov-Dec)."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import pandas as pd

from app.detect import detect
from app.focus import daily, normalize

logging.disable(logging.INFO)
raw = pd.read_csv(sys.argv[1])
N = normalize(raw)
T0 = N.date.min()
day = lambda d: T0 + pd.Timedelta(days=d)


def inject(n, svc, d0, d1, f):
    """Multiply service `svc` rows on days d0..d1 by f(k), k = days since d0."""
    n = n.copy()
    for k, d in enumerate(range(d0, d1 + 1)):
        n.loc[(n.service == svc) & (n.date == day(d)), "cost"] *= f(k)
    return n


def run(n):
    t = time.time()
    r = detect(daily(n))
    return pd.DataFrame(r["anomalies"]), r, time.time() - t


def hits(a, scopes, lo, hi, rules=None):
    m = a.scope.isin(scopes) & (a.end >= day(lo)) & (a.start <= day(hi))
    return a[m & a.rule.isin(rules)] if rules else a[m]


a, r, s = run(N)
nser = len(r["forecasts"])
ev = sum(int(f.evaluated.sum()) for f in r["forecasts"].values())
print(f"== CLEAN (un-modified) full year: {nser} series, {ev} evaluated series-days, {s:.0f}s")
print(a.groupby("rule").size().to_dict())
n1 = int((a.rule == "nelson1_spike").sum())
print(f"-> rule-1 false-alarm rate {100 * n1 / ev:.2f}% ({n1}/{ev}); 3-sigma theory 0.27%")
tot = a[(a.scope == "TOTAL") & a.rule.str.startswith("nelson")]
print("TOTAL alerts (organic shifts expected around Jun-Jul and Nov-Dec 2023):")
for x in tot.sort_values("start").itertuples():
    print(f"   {x.rule:14s} {x.start.date()} -> {x.end.date()}  peak z {x.peak_z}")

print("\n== SENSITIVITY: single-service spike (Compute, 1 weekday at d200)")
for m in (1.5, 2.0, 3.0):
    a, r, s = run(inject(N, "Compute", 200, 200, lambda k: m))
    h = hits(a, ["Compute", "TOTAL"], 199, 201, ["nelson1_spike"])
    print(f"x{m}: detected={not h.empty}  " + ", ".join(f"{x.scope} z={x.peak_z}" for x in h.itertuples()))

print("\n== SENSITIVITY: two services at once (Database+Analytics, d300)")
for m in (1.5, 2.0, 2.5, 3.0):
    a, r, s = run(inject(inject(N, "Database", 300, 300, lambda k: m), "Analytics", 300, 300, lambda k: m))
    h = hits(a, ["MULTI", "Database", "Analytics", "TOTAL"], 299, 301)
    print(f"x{m}: detected={not h.empty}  " + ", ".join(f"{x.scope}:{x.rule} z={x.peak_z}" for x in h.itertuples()))

print("\n== SENSITIVITY: slow leak (Storage ramps r/day for 40 days from d260, then holds)")
for rate in (0.01, 0.02, 0.04):
    a, r, s = run(inject(N, "Storage", 260, 300, lambda j: (1 + rate) ** (min(j, 40) + 1)))
    h = hits(a, ["Storage"], 260, 301, ["nelson1_spike", "nelson2_shift", "nelson3_trend"])
    if h.empty:
        print(f"{rate:.0%}/day: NOT detected (cost reached x{(1 + rate) ** 41:.2f})")
    else:
        lag = (h.start.min() - day(260)).days
        print(f"{rate:.0%}/day: first alert at leak day +{lag} (cost then x{(1 + rate) ** (lag + 1):.2f}) via {h.sort_values('start').iloc[0].rule}")
