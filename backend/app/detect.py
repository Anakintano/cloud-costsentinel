"""Walk-forward forecast -> out-of-sample residual z-scores -> Nelson rules (+ advisory Isolation Forest) -> anomalies."""
import numpy as np
import pandas as pd

from .forecast import MIN_TRAIN, STEP, walk_forward
from .iforest import multi_service_days
from .sixsigma import capability, nelson

RULE = {1: "nelson1_spike", 2: "nelson2_shift", 3: "nelson3_trend"}
TOP_N = 10


def _runs(idx, dates):
    """Collapse sorted index hits into contiguous (start, end) date runs."""
    runs = []
    for i in sorted(idx):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(dates[a], dates[b]) for a, b in runs]


def detect_series(scope: str, s: pd.Series):
    """Returns (anomalies, forecast_df, z).
    Forecast-error chart on RELATIVE residuals (error / expected spend). center = 0 (the forecast IS the expectation; the
    mean error flipped sign between fits in testing). sigma for block k = RMSE of all earlier blind blocks, separately for
    the early and late half of the block; so limits are always out-of-sample. Block 0 only seeds sigma.
    Floors: expected spend >= 1% of mean spend, sigma >= 1% relative, so a flat/zero series can't alert on rounding."""
    f = walk_forward(s)
    blk = f.block.values
    floor = 0.01 * abs(s.mean()) + 1e-6
    scale = np.maximum(f.yhat.values, floor)
    rel = (f.actual.values - f.yhat.values) / scale  # billing noise is multiplicative: judge error as a share of expected spend
    late = (np.arange(len(s)) - MIN_TRAIN) % STEP >= STEP // 2  # forecast error grows with horizon: calibrate halves separately
    sig_rel = np.full(len(s), np.nan)
    for k in range(1, blk.max() + 1):
        for h in (False, True):
            prev = rel[(blk >= 0) & (blk < k) & (late == h)]
            sig_rel[(blk == k) & (late == h)] = max(float(np.sqrt((prev ** 2).mean())), 0.01)
    z = rel / sig_rel
    sigma = sig_rel * scale  # absolute sigma, for UCL/LCL = yhat +/- 3 sigma
    ok = ~np.isnan(z)
    out = []
    if ok.any():
        idx = np.where(ok)[0]
        for rule, hits in nelson(z[ok], 0.0, 1.0).items():
            for a, b in _runs([int(idx[h]) for h in hits], s.index):
                seg = pd.Series(z, index=s.index)[a:b]
                peak = seg.abs().max()
                # shift/trend rules fire ~2x the iid rate on real data (forecast bias, autocorrelation): only trust them
                # as urgent when the magnitude is also large
                sev = "high" if peak >= 5 else "medium" if (rule == 1 or peak >= 3) else "low"
                out.append(dict(scope=scope, rule=RULE[rule], start=a, end=b, peak_z=round(float(seg.loc[seg.abs().idxmax()]), 2), severity=sev))
    first = int(np.argmax(ok)) if ok.any() else 0
    sigma[:first] = sigma[first] if ok.any() else floor  # warm-up days: show first band, but they are not evaluated
    f = f.assign(z=np.where(ok, z, 0.0), sigma=sigma, evaluated=ok)
    return out, f, pd.Series(np.where(ok, z, 0.0), index=s.index)


def detect(daily: pd.DataFrame, budget: float | None = None):
    """daily: date x service cost. Returns dict(anomalies, capability, forecasts)."""
    total = daily.sum(axis=1)
    top = daily.sum().nlargest(TOP_N).index
    anomalies, zs, fcs = [], {}, {}
    for scope, s in [("TOTAL", total), *[(c, daily[c]) for c in top]]:
        a, f, z = detect_series(scope, s)
        anomalies += a
        fcs[scope] = f
        if scope != "TOTAL":
            zs[scope] = z[f.evaluated]
    Z = pd.DataFrame(zs)
    for d in multi_service_days(Z):
        row = Z.loc[d]
        anomalies.append(dict(scope="MULTI", rule="iforest_multi_service", start=d, end=d,
                              peak_z=round(float(row.loc[row.abs().idxmax()]), 2), severity="low"))  # advisory: unproven beyond per-series charts
    budget = budget or 1.1 * total.iloc[: len(total) - STEP].mean()
    cap = capability(total.iloc[-STEP:].values, budget)
    return dict(anomalies=anomalies, capability={**cap, "budget": budget}, forecasts=fcs)
