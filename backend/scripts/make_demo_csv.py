"""Synthetic FOCUS billing CSV with known anomalies (also the test fixture).
Truth: EC2 spike on day 100-101, S3 slow leak from day 120, multi-service blip on day 135 (all after the 70-day warm-up)."""
import sys

import numpy as np
import pandas as pd

TRUTH = {"spike_days": [100, 101], "leak_start": 120, "multi_day": 135}


def make(days=160, seed=7, inject=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    base = {"AmazonEC2": 500, "AmazonS3": 120, "AmazonRDS": 250, "AWSLambda": 40, "AmazonCloudFront": 60}
    rows = []
    for d in range(days):
        date = start + pd.Timedelta(days=d)
        weekly = 0.8 if date.dayofweek >= 5 else 1.0
        for svc, b in base.items():
            for region in ("us-east-1", "eu-west-1"):
                share = 0.65 if region == "us-east-1" else 0.35
                cost = b * share * weekly * (1 + 0.002 * d) * rng.normal(1, 0.02)
                if inject:
                    if svc == "AmazonEC2" and d in TRUTH["spike_days"] and region == "us-east-1":
                        cost *= 4
                    if svc == "AmazonS3" and d >= TRUTH["leak_start"]:
                        cost *= 1 + 0.02 * (d - TRUTH["leak_start"])
                    if d == TRUTH["multi_day"] and svc in ("AWSLambda", "AmazonCloudFront"):
                        cost *= 2.2
                rows.append((date.isoformat(), round(cost, 4), svc, region, "111122223333"))
    return pd.DataFrame(rows, columns=["ChargePeriodStart", "BilledCost", "ServiceName", "RegionName", "SubAccountId"])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_focus.csv"
    make().to_csv(out, index=False)
    print("wrote", out)
