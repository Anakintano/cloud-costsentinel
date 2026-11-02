"""Read-only statistics tools the agent can call. Every number the agent states should come from here.
Security: tools return numbers plus labels that pass `label()` (billing text is attacker-controlled: tags, resource names).
No tool writes anything or executes anything."""
import datetime as dt
import re

import numpy as np
import pandas as pd
from scipy import stats

MAX_ROWS = 31
_SAFE = re.compile(r"^[\w .\-/:()]{1,64}$")


def label(s) -> str:
    """Billing labels can carry prompt-injection text; allow only short plain identifiers."""
    s = str(s)
    return s if _SAFE.match(s) else "[unsafe-label]"


def _d(s: str) -> pd.Timestamp:
    return pd.Timestamp(dt.date.fromisoformat(s[:10]))


class Toolbox:
    def __init__(self, series: pd.DataFrame, billing: pd.DataFrame, anomalies: list[dict], capability: dict | None = None):
        self.series, self.billing, self.anomalies, self.capability = series, billing, anomalies, capability or {}
        self.scopes = {s.lower(): s for s in series.scope.unique()} if len(series) else {}

    # ---- helpers
    def _scope(self, scope: str) -> str:
        s = self.scopes.get(str(scope).lower())
        if not s:
            raise ValueError(f"unknown scope '{label(scope)}'. valid: {sorted(self.scopes.values())}")
        return s

    def _rows(self, scope, start=None, end=None, evaluated=False) -> pd.DataFrame:
        df = self.series[self.series.scope == self._scope(scope)]
        if evaluated and "evaluated" in df:  # z is only meaningful on days that had a blind forecast
            df = df[df.evaluated]
        if start:
            df = df[df.date >= _d(start)]
        if end:
            df = df[df.date <= _d(end)]
        return df.sort_values("date")

    # ---- tools
    def get_series(self, scope, start=None, end=None):
        df = self._rows(scope, start, end)
        note = f"showing last {MAX_ROWS} of {len(df)} days" if len(df) > MAX_ROWS else None
        df = df.tail(MAX_ROWS)
        return {"scope": self._scope(scope), "note": note, "columns": ["date", "actual", "expected", "z"], "rows": [
            [r.date.date().isoformat(), round(r.actual, 1), round(r.yhat, 1), round(r.z, 1)] for r in df.itertuples()]}

    def chart_stats(self, scope):
        df = self._rows(scope)
        last = df.iloc[-1]
        out = {"scope": self._scope(scope), "sigma": round(float(last.sigma), 2),
               "latest_day": last.date.date().isoformat(), "latest_expected": round(float(last.yhat), 2),
               "latest_ucl": round(float(last.yhat + 3 * last.sigma), 2), "latest_lcl": round(float(last.yhat - 3 * last.sigma), 2),
               "days_beyond_3sigma_last_30": int((df.z.tail(30).abs() > 3).sum())}
        if out["scope"] == "TOTAL" and self.capability:
            out["budget_per_day"] = round(self.capability["budget"], 2)
            out["cpk_vs_budget"] = round(self.capability["cpk"], 2)
            out["cpk_meaning"] = ">1.33 comfortable, 1.0-1.33 marginal, <1.0 spend regularly exceeds budget"
        return out

    def top_movers(self, date, dimension="service", scope=None, n=5):
        """Who changed on `date` vs the same weekday over the previous 4 weeks (so weekly seasonality isn't flagged)."""
        if dimension not in ("service", "region", "account"):
            raise ValueError("dimension must be service|region|account")
        b, d0 = self.billing, _d(date)
        if scope and str(scope).upper() != "TOTAL":
            b = b[b.service == self._scope(scope)]
        base_days = [d0 - pd.Timedelta(days=7 * k) for k in range(1, 5)]
        base_days = [d for d in base_days if d >= b.date.min()]
        if not base_days:
            raise ValueError("not enough history before that date to form a baseline")
        today = b[b.date == d0].groupby(dimension).cost.sum()
        base = b[b.date.isin(base_days)].groupby([dimension, "date"]).cost.sum().unstack(fill_value=0).reindex(columns=base_days, fill_value=0).mean(axis=1)
        m = pd.DataFrame({"today": today, "baseline": base}).fillna(0.0)
        m["delta"] = m.today - m.baseline
        tot = m.delta.sum()
        m = m.reindex(m.delta.abs().sort_values(ascending=False).index).head(int(min(n, 10)))
        return {"date": d0.date().isoformat(), "dimension": dimension, "baseline": f"mean of same weekday, previous {len(base_days)} weeks",
                "total_delta": round(float(tot), 2), "movers": [
                    {"name": label(i), "today": round(r.today, 2), "baseline": round(r.baseline, 2), "delta": round(r.delta, 2),
                     "share_of_total_delta": round(float(r.delta / tot), 2) if abs(tot) > 1e-9 else None}
                    for i, r in m.iterrows()]}

    def compare_periods(self, scope, a_start, a_end, b_start, b_end):
        a = self._rows(scope, a_start, a_end).actual.values
        b = self._rows(scope, b_start, b_end).actual.values
        if len(a) < 3 or len(b) < 1:
            raise ValueError(f"baseline period A needs >=3 days and period B >=1 day (got {len(a)} and {len(b)})")
        if len(b) < 3:  # typical spike alert is 1-2 days: a t-test is meaningless, so compare the points to the baseline distribution
            sd = a.std(ddof=1)
            return {"scope": self._scope(scope), "mode": "short_window_vs_baseline", "n_a": len(a), "n_b": len(b),
                    "mean_a": round(a.mean(), 2), "baseline_sd": round(float(sd), 2), "mean_b": round(b.mean(), 2),
                    "pct_change": round(100 * (b.mean() / a.mean() - 1), 1) if a.mean() else None,
                    "z_vs_baseline": round(float((b.mean() - a.mean()) / sd), 2) if sd else None,
                    "caveat": "period B has <3 days so no significance test; z_vs_baseline compares its mean to the raw baseline spread (includes weekday swings)"}
        t = stats.ttest_ind(b, a, equal_var=False)
        u = stats.mannwhitneyu(b, a, alternative="two-sided")
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return {"scope": self._scope(scope), "n_a": len(a), "n_b": len(b), "mean_a": round(a.mean(), 2), "mean_b": round(b.mean(), 2),
                "pct_change": round(100 * (b.mean() / a.mean() - 1), 1) if a.mean() else None,
                "welch_t_p": float(f"{t.pvalue:.3g}"), "mann_whitney_p": float(f"{u.pvalue:.3g}"),
                "cohens_d": round(float((b.mean() - a.mean()) / pooled), 2) if pooled else None,
                "caveat": "daily cost is autocorrelated, so p-values are optimistic; weigh effect size (cohens_d, pct_change) more"}

    def changepoint(self, scope, start=None, end=None):
        """CUSUM on weekly-seasonality-free residual z-scores: where did the level change?"""
        df = self._rows(scope, start, end, evaluated=True)
        if len(df) < 10:
            raise ValueError("need >=10 days")
        z = df.z.values
        cs = np.cumsum(z - z.mean())
        k = int(np.argmax(np.abs(cs)))
        if k >= len(z) - 2:
            return {"scope": self._scope(scope), "changepoint": None, "note": "no interior level change found"}
        before, after = df.actual.values[: k + 1], df.actual.values[k + 1:]
        return {"scope": self._scope(scope), "first_day_of_new_level": df.date.iloc[k + 1].date().isoformat(),
                "mean_before": round(before.mean(), 2), "mean_after": round(after.mean(), 2),
                "pct_change": round(100 * (after.mean() / before.mean() - 1), 1) if before.mean() else None,
                "z_mean_before": round(float(z[: k + 1].mean()), 2), "z_mean_after": round(float(z[k + 1:].mean()), 2)}

    def ramp_onset(self, scope, start=None, end=None):
        """Dates a GRADUAL ramp (leak): fit 'flat, then linear rise from day t0' to weekday-normalised spend and pick the best t0.
        Use a window starting ~8 weeks before the alert; the alert itself fires late, after the ramp is already large."""
        df = self._rows(scope, start, end)
        if len(df) < 21:
            raise ValueError("need >=21 days")
        y, dow = df.actual.values.astype(float), df.date.dt.dayofweek.values
        prof = np.array([y[dow == k].mean() if (dow == k).any() else y.mean() for k in range(7)])
        yn = y / (prof / prof.mean())[dow]
        t = np.arange(len(yn))
        sse_flat = float(((yn - yn.mean()) ** 2).sum()) or 1e-9
        best = None
        for t0 in range(7, len(yn) - 5):
            X = np.column_stack([np.ones(len(t)), np.maximum(0, t - t0)])
            beta = np.linalg.lstsq(X, yn, rcond=None)[0]
            sse = float(((yn - X @ beta) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, t0, beta)
        sse, t0, beta = best
        step = min(float(((yn - np.where(t >= s0, yn[s0:].mean(), yn[:s0].mean())) ** 2).sum()) for s0 in range(7, len(yn) - 5))
        shape = "gradual ramp" if sse < 0.8 * step else "abrupt step"  # a steep hinge fits any step, so compare against a pure step
        return {"scope": self._scope(scope), "shape_that_fits_best": shape, "ramp_improvement_vs_flat": round(1 - sse / sse_flat, 2),
                "step_improvement_vs_flat": round(1 - step / sse_flat, 2), "ramp_start": df.date.iloc[t0].date().isoformat(),
                "level_before_ramp": round(float(beta[0]), 1), "slope_pct_of_base_per_day": round(100 * float(beta[1]) / beta[0], 2) if beta[0] else None,
                "level_at_window_end": round(float(beta[0] + beta[1] * (len(yn) - 1 - t0)), 1),
                "fit_improvement_vs_flat": round(1 - sse / sse_flat, 2),
                "note": "Trust shape_that_fits_best. 'abrupt step' = sudden level change (deploy, new workload); 'gradual ramp' = leak/creep. improvement near 0 = neither"}

    def correlate(self, scope_a, scope_b, start=None, end=None):
        a, b = (self._rows(x, start, end, evaluated=True).set_index("date").z for x in (scope_a, scope_b))
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(j) < 8:
            raise ValueError("need >=8 overlapping days")
        return {"a": self._scope(scope_a), "b": self._scope(scope_b), "n": len(j),
                "pearson_r": round(float(stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])[0]), 2),
                "spearman_r": round(float(stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]), 2),
                "note": "correlation of deseasonalised residuals; high r suggests a shared driver, not proof"}

    def list_anomalies(self, scope=None, limit=15):
        a = [x for x in self.anomalies if not scope or x["scope"].lower() == str(scope).lower()]
        return {"total": len(a), "anomalies": [{**{k: x[k] for k in ("id", "scope", "rule", "peak_z", "severity", "status") if k in x},
                               "start": str(x["start"])[:10], "end": str(x["end"])[:10],
                               **({"earlier_analyst_note": x["explanation"][:600]} if x.get("explanation") else {})}
                              for x in a[: int(min(limit, 30))]]}

    def period_report(self, start, end):
        """One-call manager summary of a date range: spend, change vs the equal-length period before, budget, alerts, who drove it."""
        s, e = _d(start), _d(end)
        if e < s:
            raise ValueError("end is before start")
        n = (e - s).days + 1
        ps, pe = s - pd.Timedelta(days=n), s - pd.Timedelta(days=1)
        tot = self.series[self.series.scope == "TOTAL"]
        cur = tot[(tot.date >= s) & (tot.date <= e)]
        if cur.empty:
            raise ValueError(f"no data in that period; data covers {tot.date.min().date()} to {tot.date.max().date()}")
        prev = tot[(tot.date >= ps) & (tot.date <= pe)]
        b = self.billing
        spend = b[(b.date >= s) & (b.date <= e)].groupby("service").cost.sum()
        before = b[(b.date >= ps) & (b.date <= pe)].groupby("service").cost.sum()
        m = pd.DataFrame({"spend": spend, "previous": before}).fillna(0.0)
        m["change"] = m.spend - m.previous
        have_prev = len(prev) == n
        out = {"period": f"{s.date()} to {e.date()}", "days_with_data": len(cur), "days_in_period": n,
               "total_spend": round(float(cur.actual.sum()), 2), "avg_per_day": round(float(cur.actual.mean()), 2),
               "previous_period": f"{ps.date()} to {pe.date()}" if have_prev else None,
               "previous_total_spend": round(float(prev.actual.sum()), 2) if have_prev else None,
               "pct_change_vs_previous": round(100 * (cur.actual.sum() / prev.actual.sum() - 1), 1) if have_prev and prev.actual.sum() else None,
               "top_services_by_spend": [{"name": label(i), "spend": round(float(v), 2)} for i, v in m.spend.nlargest(5).items()]}
        if have_prev:
            out["biggest_service_changes"] = [{"name": label(i), "change": round(float(r.change), 2), "previous": round(float(r.previous), 2), "spend": round(float(r.spend), 2)}
                                              for i, r in m.reindex(m.change.abs().sort_values(ascending=False).index).head(5).iterrows()]
        ev = cur[cur.evaluated] if "evaluated" in cur else cur
        if len(ev):
            w = ev.loc[(ev.actual - ev.yhat).abs().idxmax()]
            out["day_furthest_from_expected"] = {"date": w.date.date().isoformat(), "actual": round(float(w.actual), 2), "expected": round(float(w.yhat), 2)}
        if self.capability:
            out["budget_per_day"] = round(self.capability["budget"], 2)
            out["days_over_budget"] = int((cur.actual > self.capability["budget"]).sum())
        hits = [x for x in self.anomalies if pd.Timestamp(x["start"]) <= e and pd.Timestamp(x["end"]) >= s]
        out["alerts_in_period"] = len(hits)
        out["alerts"] = [{**{k: x[k] for k in ("id", "scope", "severity", "status") if k in x}, "kind": x["rule"], "start": str(x["start"])[:10], "end": str(x["end"])[:10]}
                         for x in sorted(hits, key=lambda x: (x.get("severity") != "high", x.get("severity") != "medium", -abs(x.get("peak_z", 0))))[:10]]
        return out

    def card(self) -> str:
        """Compact description of the loaded data, put in the system prompt so the agent starts knowing what exists."""
        t = self.series[self.series.scope == "TOTAL"] if len(self.series) else self.series
        if t.empty:
            return ""
        d0, d1 = t.date.min().date(), t.date.max().date()
        sev = {}
        for x in self.anomalies:
            sev[x.get("severity", "?")] = sev.get(x.get("severity", "?"), 0) + 1
        noted = [x["id"] for x in self.anomalies if x.get("explanation")]
        services = sorted(s for s in self.scopes.values() if s != "TOTAL")
        c = [f"Loaded data: daily cloud spend, {d0} to {d1} ({len(t)} days). Treat {d1} as 'now'; a month named without a year means the latest such month in the data (say which year you used).",
             "Scopes: TOTAL plus services " + ", ".join(label(s) for s in services[:20]) + "."]
        if self.capability:
            c.append(f"Budget: ${self.capability['budget']:.2f} per day.")
        c.append(f"Alerts: {len(self.anomalies)} (" + ", ".join(f"{v} {k}" for k, v in sev.items()) + ")." if self.anomalies else "Alerts: none.")
        if noted:
            c.append(f"Earlier analyst notes exist for alert ids {noted[:15]}; list_anomalies shows them.")
        return "\n".join(c)

    # ---- dispatch
    def call(self, name: str, args: dict):
        if name not in SCHEMAS:
            return {"error": f"unknown tool '{name}'"}
        spec = SCHEMAS[name]["function"]["parameters"]
        missing = [k for k in spec.get("required", []) if k not in args]
        extra = [k for k in args if k not in spec["properties"]]
        if missing or extra:
            return {"error": f"bad arguments: missing={missing} unexpected={extra}"}
        try:
            return getattr(self, name)(**args)
        except (ValueError, KeyError, TypeError) as e:
            return {"error": str(e)[:300]}


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S, DATE = {"type": "string"}, {"type": "string", "description": "ISO date YYYY-MM-DD"}
SCOPE = {"type": "string", "description": "TOTAL or a service name"}
SCHEMAS = {t["function"]["name"]: t for t in [
    _fn("get_series", "Daily actual, forecast-expected cost and z-score for a scope (last 31 days of the range).",
        {"scope": SCOPE, "start": DATE, "end": DATE}, ["scope"]),
    _fn("chart_stats", "Control-chart limits (sigma, UCL/LCL) at the latest day and, for TOTAL, Cpk vs budget.", {"scope": SCOPE}, ["scope"]),
    _fn("top_movers", "Which services/regions/accounts moved most on a date vs the same weekday over the prior 4 weeks.",
        {"date": DATE, "dimension": {"type": "string", "enum": ["service", "region", "account"]}, "scope": SCOPE, "n": {"type": "integer"}}, ["date"]),
    _fn("compare_periods", "Welch t-test, Mann-Whitney and effect size comparing mean daily cost in period A vs B.",
        {"scope": SCOPE, "a_start": DATE, "a_end": DATE, "b_start": DATE, "b_end": DATE}, ["scope", "a_start", "a_end", "b_start", "b_end"]),
    _fn("changepoint", "CUSUM changepoint: the first day of a new cost level within a window.", {"scope": SCOPE, "start": DATE, "end": DATE}, ["scope"]),
    _fn("ramp_onset", "Date a gradual cost ramp/leak: fits flat-then-linear-rise to weekday-normalised spend and returns whether a gradual ramp or an abrupt step fits better, when it started, and its slope. Use a window starting ~8 weeks before the alert.",
        {"scope": SCOPE, "start": DATE, "end": DATE}, ["scope"]),
    _fn("correlate", "Correlation between two scopes' deseasonalised residuals (shared driver check).",
        {"scope_a": SCOPE, "scope_b": SCOPE, "start": DATE, "end": DATE}, ["scope_a", "scope_b"]),
    _fn("list_anomalies", "Detected anomalies (with ids, dates, severity, and any earlier analyst note), optionally for one scope.", {"scope": SCOPE, "limit": {"type": "integer"}}),
    _fn("period_report", "START HERE for broad questions (a month's summary, 'what went wrong', 'how are we doing'): total spend, change vs the previous equal-length period, budget, alerts, top services and biggest service changes for a date range.",
        {"start": DATE, "end": DATE}, ["start", "end"]),
]}
TOOL_LIST = list(SCHEMAS.values())
