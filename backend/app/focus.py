"""Normalize a billing CSV to FOCUS columns and roll up to daily cost."""
import pandas as pd

# canonical FOCUS name -> accepted aliases (lowercased). AWS CUR / GCP export / Azure.
ALIASES = {
    "ChargePeriodStart": ["chargeperiodstart", "lineitem/usagestartdate", "usage_start_time", "date", "usagedate"],
    "BilledCost": ["billedcost", "lineitem/unblendedcost", "cost", "net_cost", "costinbillingcurrency"],
    "ServiceName": ["servicename", "lineitem/productcode", "service.description", "metercategory", "service"],
    "RegionName": ["regionname", "product/region", "location.region", "resourcelocation", "region"],
    "SubAccountId": ["subaccountid", "lineitem/usageaccountid", "project.id", "subscriptionid", "account_id"],
    "ChargeCategory": ["chargecategory", "lineitem/linetype"],
}
REQUIRED = ["ChargePeriodStart", "BilledCost", "ServiceName"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows: date, service, region, account, cost. Raises ValueError on missing required columns."""
    lower = {c.lower(): c for c in df.columns}
    out = {}
    for canon, aliases in ALIASES.items():
        src = next((lower[a] for a in aliases if a in lower), None)
        if src is None and canon in REQUIRED:
            raise ValueError(f"missing required column {canon} (accepted: {aliases})")
        out[canon] = df[src] if src else "unknown"
    n = pd.DataFrame(out)
    n["date"] = pd.to_datetime(n.pop("ChargePeriodStart"), utc=True).dt.tz_localize(None).dt.normalize()
    n["cost"] = pd.to_numeric(n.pop("BilledCost"), errors="raise")
    return n.rename(columns={"ServiceName": "service", "RegionName": "region", "SubAccountId": "account",
                             "ChargeCategory": "category"})


def clean(n: pd.DataFrame) -> pd.DataFrame:
    """Gross spend only: credits/refunds dropped (they mask real spikes). Sparse leading/trailing dates
    (< 10% of the median daily row count, e.g. stray back-dated rows in an export) are trimmed.
    ponytail: row count as the coverage proxy assumes steady infra; swap for a per-source expected-rows check if needed."""
    if "category" in n:
        n = n[~n.category.astype(str).str.lower().isin(["credit", "refund"])]
    rows = n.groupby("date").size()
    dense = rows[rows >= 0.1 * rows.median()].index
    return n[(n.date >= dense.min()) & (n.date <= dense.max())]


def daily(n: pd.DataFrame, by="service") -> pd.DataFrame:
    """Wide date x <by> cost matrix over clean(n); interior gaps filled with 0."""
    w = clean(n).pivot_table(index="date", columns=by, values="cost", aggfunc="sum").fillna(0.0)
    return w.reindex(pd.date_range(w.index.min(), w.index.max()), fill_value=0.0)
