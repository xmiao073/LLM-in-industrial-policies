#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute extended quantitative metrics for all portfolios.

Inputs (discovered by scanning):
- daily_ret_allpct.pkl.gz files under daily/ and rolling/ result folders

Outputs (to --out):
- all_metrics.csv       Overall metrics for every strategy (full sample)
- yearly_metrics.csv    Yearly metrics for every strategy (per calendar year)
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd


def infer_from_path(p: Path) -> dict:
    """Infer holding/period/return/sign/weight/frequency from directory tree.

    Supported layouts:
    - New layout (daily):   daily/<ret>/<sign>/<wt>/<freq_tag>/
    - Old layout (daily):   daily/<freq_tag>/<ret>/<sign>/<wt>/
    - New layout (rolling): rolling/p<period>/<ret>/<sign>/<wt>/<freq_tag>/
    - Old layout (rolling): rolling/p<period>/<freq_tag>/<ret>/<sign>/<wt>/
    """
    parts = p.parts
    d: dict = {}

    if "daily" in parts:
        d["holding"], d["period"] = "daily", 1
        base = parts.index("daily") + 1
        rem = parts[base:]
        if len(rem) >= 4 and (rem[0] == "static" or rem[0].startswith("dynamic")):
            d["frequency"], d["return_type"], d["sign"], d["weighting"] = rem[:4]
        else:
            d["return_type"], d["sign"], d["weighting"] = rem[:3]
            d["frequency"] = rem[3] if len(rem) >= 4 else "unknown"
    else:
        d["holding"] = "rolling"
        ridx = parts.index("rolling")
        d["period"] = int(parts[ridx + 1].lstrip("p"))
        base = ridx + 2
        rem = parts[base:]
        if len(rem) >= 4 and (rem[0] == "static" or rem[0].startswith("dynamic")):
            d["frequency"], d["return_type"], d["sign"], d["weighting"] = rem[:4]
        else:
            d["return_type"], d["sign"], d["weighting"] = rem[:3]
            d["frequency"] = rem[3] if len(rem) >= 4 else "unknown"
    return d


def extract_lag_from_base(base: str) -> int | None:
    m = re.search(r"_lag(\d+)", base)
    return int(m.group(1)) if m else None


def compute_drawdown_metrics(ret: pd.Series) -> Tuple[float, float, pd.Timestamp | None, pd.Timestamp | None]:
    """Return (max_drawdown, calmar_like_denominator, peak_date, trough_date).
    max_drawdown is negative (e.g., -0.35 for -35%).
    """
    if ret.empty:
        return 0.0, np.nan, None, None
    wealth = (1.0 + ret).cumprod()
    rolling_max = wealth.cummax()
    drawdown = wealth / rolling_max - 1.0
    mdd = float(drawdown.min())
    trough_idx = drawdown.idxmin() if not drawdown.empty else None
    peak_idx = (wealth.loc[:trough_idx].idxmax() if trough_idx in wealth.index else None)
    return mdd, abs(mdd), peak_idx, trough_idx


def compute_extended_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute extended metrics given a DataFrame with columns ['date','ret'] sorted by date."""
    s = df["ret"].dropna().astype(float)
    out: Dict[str, Any] = {}
    if s.empty:
        return {"num_days": 0}

    out["num_days"] = int(s.size)
    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if s.size > 1 else 0.0
    out["avg_daily"] = mu
    out["std_daily"] = sigma
    # Annualized
    out["ann_ret"] = (1.0 + mu) ** 252 - 1.0
    out["ann_vol"] = sigma * np.sqrt(252.0)
    out["sharpe"] = (out["ann_ret"] / out["ann_vol"]) if out["ann_vol"] > 0 else np.nan

    # Sortino (downside deviation)
    downside = s[s < 0.0]
    dd = float(downside.std(ddof=1)) if downside.size > 1 else (float(downside.std(ddof=0)) if downside.size > 0 else 0.0)
    ann_down = dd * np.sqrt(252.0)
    out["downside_dev_ann"] = ann_down
    out["sortino"] = (out["ann_ret"] / ann_down) if ann_down > 0 else np.nan

    # Drawdown metrics
    mdd, mdd_den, peak_dt, trough_dt = compute_drawdown_metrics(s)
    out["max_drawdown"] = mdd
    out["mdd_peak_date"] = peak_dt
    out["mdd_trough_date"] = trough_dt
    out["calmar"] = (out["ann_ret"] / mdd_den) if mdd_den > 0 else np.nan

    # Distributional
    out["skew"] = float(s.skew())
    out["kurt_excess"] = float(s.kurt())
    out["hit_ratio"] = float((s > 0).mean())
    gains = s[s > 0]
    losses = s[s < 0]
    out["avg_gain"] = float(gains.mean()) if not gains.empty else np.nan
    out["avg_loss"] = float(losses.mean()) if not losses.empty else np.nan
    out["gain_loss_ratio"] = (abs(out["avg_gain"] / out["avg_loss"]) if (out.get("avg_gain") and out.get("avg_loss")) else np.nan)

    # Tail risk (historical VaR/ES)
    for q in (0.95, 0.99):
        alpha = 1.0 - q
        var_q = float(np.nanquantile(s.values, 1.0 - q))  # e.g., 5% quantile for q=0.95
        tail = s[s <= var_q]
        es_q = float(tail.mean()) if not tail.empty else np.nan
        out[f"VaR_{int(q*100)}"] = var_q
        out[f"ES_{int(q*100)}"] = es_q

    # Tail ratio (95th / |5th|)
    p95 = float(np.nanquantile(s.values, 0.95))
    p05 = float(np.nanquantile(s.values, 0.05))
    out["tail_ratio_95_05"] = (p95 / abs(p05)) if p05 != 0 else np.nan

    # Best/worst day
    out["best_day"] = float(s.max())
    out["worst_day"] = float(s.min())
    return out


def build_portfolio_label(meta: dict, lag: int, pct: float) -> str:
    return (f"{meta['holding']}_{int(meta['period'])}_lag{int(lag)}_"
            f"{meta['weighting']}_p{int(pct*100)}_{meta['frequency']}_{meta['return_type']}")


def compute_turnover_series(df: pd.DataFrame) -> pd.Series:
    """Compute daily turnover series from columns longset(list[str]) and weight_vec(list[float]).
    turnover_t = 0.5 * sum_i |w_t(i) - w_{t-1}(i)|
    First day is set to NaN (no prior holdings).
    """
    if ("longset" not in df.columns) or ("weight_vec" not in df.columns):
        return pd.Series(index=df["date"], dtype=float)
    prev: dict[str, float] = {}
    vals: list[float] = []
    for names, weights in zip(df["longset"], df["weight_vec"]):
        cur = {str(s): float(w) for s, w in zip(names, weights)} if isinstance(names, list) else {}
        keys = set(prev.keys()) | set(cur.keys())
        if prev:
            turn = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
            vals.append(turn)
        else:
            vals.append(np.nan)
        prev = cur
    return pd.Series(vals, index=df["date"], name="turnover")


def main() -> None:
    pa = argparse.ArgumentParser("Compute extended portfolio metrics")
    pa.add_argument("--root", default="/hpc2hdd/home/jliu043/policy/portfolio", help="portfolio root dir")
    pa.add_argument("--out",  default="/hpc2hdd/home/jliu043/policy/portfolio/portfolio_report", help="output dir")
    # Optional filters to limit scope
    pa.add_argument("--filter-holding", choices=["daily", "rolling"], nargs="+", default=None)
    pa.add_argument("--filter-periods", type=int, nargs="+", default=None, help="Only for rolling; e.g., 3 5 10")
    pa.add_argument("--filter-return-types", nargs="+", default=None, help="open_open close_close open_close close_open")
    pa.add_argument("--filter-weightings", nargs="+", default=None, help="equal mv")
    pa.add_argument("--filter-signs", nargs="+", default=None, help="pos neg")
    pa.add_argument("--filter-frequencies", nargs="+", default=None, help="e.g., dynamic_12m dynamic_24m (exact match)")
    pa.add_argument("--filter-lags", type=int, nargs="+", default=None, help="e.g., 1 3 5 10")
    pa.add_argument("--filter-percents", type=float, nargs="+", default=None, help="e.g., 0.1 0.2 0.5 1.0")
    # Append mode
    pa.add_argument("--append", action="store_true", default=True, help="Append to existing CSVs (default: True)")
    pa.add_argument("--rewrite", action="store_true", help="If set, ignore --append and rewrite outputs")
    args = pa.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_records: List[dict] = []
    yearly_records: List[dict] = []

    # Pre-build filter sets
    fh = set(args.filter_holding) if args.filter_holding else None
    fp = set(args.filter_periods) if args.filter_periods else None
    fr = set(args.filter_return_types) if args.filter_return_types else None
    fw = set(args.filter_weightings) if args.filter_weightings else None
    fs = set(args.filter_signs) if args.filter_signs else None
    ff = set(args.filter_frequencies) if args.filter_frequencies else None
    flagr = set(args.filter_lags) if args.filter_lags else None
    fperc = set(args.filter_percents) if args.filter_percents else None

    for pkl_path in root.rglob("daily_ret_allpct.pkl.gz"):
        meta = infer_from_path(pkl_path.parent)
        # Directory-level filters
        if fh and meta.get("holding") not in fh:
            continue
        if fp and meta.get("holding") == "rolling" and int(meta.get("period", -1)) not in fp:
            continue
        if fr and meta.get("return_type") not in fr:
            continue
        if fw and meta.get("weighting") not in fw:
            continue
        if fs and meta.get("sign") not in fs:
            continue
        if ff and meta.get("frequency") not in ff:
            continue

        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # data: { base_label_without_percent : { pct -> DataFrame(date, ret, ... ) } }
        for base, inner in data.items():
            lag = extract_lag_from_base(base) or -1
            if flagr and int(lag) not in flagr:
                continue
            for pct, tbl in inner.items():
                if fperc and float(pct) not in fperc:
                    continue
                df = tbl.copy()
                if "date" not in df.columns or "ret" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"])  # ensure datetime
                df = df.sort_values("date")

                # Overall metrics + turnover
                tser = compute_turnover_series(df)
                dfm = df[["date", "ret"]].copy()
                dfm["turnover"] = tser.values
                metrics = compute_extended_metrics(dfm[["date", "ret"]])
                # Turnover aggregates
                metrics["avg_turnover_daily"] = float(np.nanmean(tser.values)) if len(tser) else np.nan
                metrics["turnover_annualized"] = metrics["avg_turnover_daily"] * 252 if metrics.get("avg_turnover_daily") is not None else np.nan
                rec = {
                    "holding": meta["holding"],
                    "period":  int(meta["period"]),
                    "return_type": meta["return_type"],
                    "sign": meta["sign"],
                    "weighting": meta["weighting"],
                    "frequency": meta["frequency"],
                    "lag": int(lag),
                    "percent": float(pct),
                    "portfolio": build_portfolio_label(meta, int(lag), float(pct)),
                }
                rec.update(metrics)
                overall_records.append(rec)

                # Yearly metrics
                df["year"] = df["date"].dt.year
                for yr, dfy in df.groupby("year"):
                    ty = compute_turnover_series(dfy)
                    dfym = dfy[["date", "ret"]].copy()
                    dfym["turnover"] = ty.values
                    ymetrics = compute_extended_metrics(dfym[["date", "ret"]])
                    ymetrics["avg_turnover_daily"] = float(np.nanmean(ty.values)) if len(ty) else np.nan
                    ymetrics["turnover_annualized"] = ymetrics["avg_turnover_daily"] * 252 if ymetrics.get("avg_turnover_daily") is not None else np.nan
                    yrec = rec.copy()
                    yrec["year"] = int(yr)
                    yrec.update(ymetrics)
                    yearly_records.append(yrec)

    all_path = out_dir / "all_metrics.csv"
    yr_path = out_dir / "yearly_metrics.csv"

    new_all = pd.DataFrame(overall_records) if overall_records else pd.DataFrame()
    new_yr  = pd.DataFrame(yearly_records) if yearly_records else pd.DataFrame()

    if args.rewrite:
        new_all.to_csv(all_path, index=False)
        new_yr.to_csv(yr_path, index=False)
    else:
        # Append with de-duplication on keys
        if all_path.exists():
            old_all = pd.read_csv(all_path)
            combined = pd.concat([old_all, new_all], ignore_index=True) if not new_all.empty else old_all
            if "portfolio" in combined.columns:
                combined = combined.drop_duplicates(subset=["portfolio"], keep="last")
            combined.to_csv(all_path, index=False)
        else:
            new_all.to_csv(all_path, index=False)

        if yr_path.exists():
            old_yr = pd.read_csv(yr_path)
            combined_y = pd.concat([old_yr, new_yr], ignore_index=True) if not new_yr.empty else old_yr
            if set(["portfolio", "year"]).issubset(combined_y.columns):
                combined_y = combined_y.drop_duplicates(subset=["portfolio", "year"], keep="last")
            combined_y.to_csv(yr_path, index=False)
        else:
            new_yr.to_csv(yr_path, index=False)

    print("Saved:", all_path.resolve())
    print("Saved:", yr_path.resolve())


if __name__ == "__main__":
    main()


