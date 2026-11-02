"""Walk-forward Prophet forecast. Every day after MIN_TRAIN is predicted by a model that never saw it:
refit every STEP days on all earlier data, predict the next STEP days blind. Residuals are therefore genuinely
out-of-sample for the whole history (in-sample residuals are 1.2-1.6x too tight, and a one-shot holdout lets any shift
older than the holdout get absorbed into the trend, which is how a +28% two-month shift went unflagged)."""
import logging

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

STEP = 28       # 4 weeks: aligned to the weekly cycle
MIN_TRAIN = 42  # 6 weeks before the first blind forecast


def _fit(df):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, interval_width=0.99,
                changepoint_prior_scale=0.05, seasonality_mode="multiplicative")  # weekly dip scales with spend level
    m.fit(df)
    return m


def _robust_fit(train: pd.DataFrame):
    """Fit, drop training outliers (>4 robust sigma), refit so a past spike doesn't bend the baseline."""
    m = _fit(train)
    res = train.y - m.predict(train[["ds"]]).yhat
    mad = (res - res.median()).abs().median() * 1.4826 or 1e-9
    keep = (res - res.median()).abs() < 4 * mad
    return m if keep.all() else _fit(train.assign(y=train.y.where(keep)))


def walk_forward(s: pd.Series) -> pd.DataFrame:
    """s: daily cost indexed by date. Returns [actual, yhat, lo, hi, block]; block = -1 for the in-sample warm-up days,
    else 0,1,2.. (block 0 only seeds sigma; alerts start at block 1)."""
    n, ds, y = len(s), s.index, s.values
    yhat, lo, hi, block = (np.full(n, np.nan) for _ in range(4))
    block[:] = -1
    for k, start in enumerate(range(MIN_TRAIN, n, STEP)):
        stop = min(start + STEP, n)
        m = _robust_fit(pd.DataFrame({"ds": ds[:start], "y": y[:start]}))
        a = 0 if k == 0 else start  # first fit also supplies the in-sample warm-up fit
        p = m.predict(pd.DataFrame({"ds": ds[a:stop]}))
        yhat[a:stop], lo[a:stop], hi[a:stop] = p.yhat.values, p.yhat_lower.values, p.yhat_upper.values
        block[start:stop] = k
    return pd.DataFrame({"actual": y, "yhat": yhat, "lo": lo, "hi": hi, "block": block.astype(int)}, index=s.index)
