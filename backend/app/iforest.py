"""Isolation Forest over the date x service standardized-residual matrix."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def multi_service_days(z: pd.DataFrame) -> list:
    """z: date x service z-scores. A day is flagged only if the forest calls it an outlier
    AND >=2 services are individually unusual (|z|>2): that's what the per-series charts can't see."""
    if z.shape[1] < 2 or len(z) < 20:
        return []
    pred = IsolationForest(n_estimators=200, contamination=0.03, random_state=0).fit_predict(z.values)
    multi = (z.abs() > 2).sum(axis=1).values >= 2
    return list(z.index[(pred == -1) & multi])
