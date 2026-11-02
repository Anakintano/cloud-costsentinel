import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import pandas as pd
from make_demo_csv import TRUTH, make

from app.detect import detect
from app.focus import daily, normalize

START = pd.Timestamp("2026-01-01")
day = lambda d: START + pd.Timedelta(days=d)


def run(**kw):
    return detect(daily(normalize(make(**kw))))


def test_injected_anomalies_found():
    r = run()
    a = pd.DataFrame(r["anomalies"])
    ec2 = a[(a.scope == "AmazonEC2") & (a.rule == "nelson1_spike")]
    assert ((ec2.start <= day(100)) & (ec2.end >= day(100))).any(), "EC2 spike missed"
    s3 = a[(a.scope == "AmazonS3") & a.rule.isin(["nelson2_shift", "nelson3_trend", "nelson1_spike"]) & (a.end >= day(TRUTH["leak_start"]))]
    assert not s3.empty, "S3 leak missed"
    assert s3.start.min() <= day(TRUTH["leak_start"] + 21), "S3 leak caught too late"
    blip = a[(a.start <= day(TRUTH["multi_day"])) & (a.end >= day(TRUTH["multi_day"]))]
    assert not blip.empty, "multi-service day not flagged by any chart"


def test_clean_series_is_quiet():
    a = pd.DataFrame(run(inject=False)["anomalies"])
    n1 = a[a.rule == "nelson1_spike"]
    # ~11 series x ~90 evaluated days at the 3-sigma rate of 0.27% -> Poisson mean ~2.7; 7 is its 99th percentile
    assert len(n1) <= 7, n1
    assert len(a[a.rule.str.startswith("nelson")]) <= 12, a


def test_warmup_days_are_never_alerted():
    r = run()
    for a in r["anomalies"]:
        assert a["start"] >= day(70)  # 42 warm-up + 28 sigma-seeding block
