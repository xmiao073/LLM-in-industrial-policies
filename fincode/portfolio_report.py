#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
Aggregate portfolio results (daily + rolling) and output two tables only:
1) all_portfolios.csv   Overall metrics for all strategies
2) all_yearly.csv       Yearly metrics for all strategies

Portfolio naming must include 7 parameters (order):
holding, period, lag, weight, percent, frequency, return type
"""
# %%
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

# === Load defaults from config.py (reads .env automatically) ===
from config import (
    DATA_DIR as DATA_DIR_DEFAULT,
    PORTFOLIO_DIR as OUT_DIR_DEFAULT,
    REGRESSION_DIR as REG_DIR_DEFAULT,
    debug_print,
)

# 给本脚本的默认根目录与输出目录
DEFAULT_ROOT = Path(OUT_DIR_DEFAULT)                 # 默认到 .env 里的 PORTFOLIO_DIR
DEFAULT_OUT  = Path(OUT_DIR_DEFAULT) / "portfolio_report"

# %%
def _extract_lag(text: str) -> int | None:
    m = re.search(r"_lag(\d+)", text)
    return int(m.group(1)) if m else None

def parse_strategy(row: pd.Series) -> pd.Series:
    """extract lag & percent from strategy string"""
    lag = re.search(r"_lag(\d+)", row["strategy"])
    pct = re.search(r"_p(\d+)",   row["strategy"])
    row["lag"] = int(lag.group(1)) if lag else None
    row["percent"] = int(pct.group(1))/100 if pct else None
    return row

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
        # Decide whether first token is freq_tag or return_type
        if len(rem) >= 4 and (rem[0] == "static" or rem[0].startswith("dynamic")):
            # Old layout: freq, ret, sign, wt
            d["frequency"], d["return_type"], d["sign"], d["weighting"] = rem[:4]
        else:
            # New layout: ret, sign, wt, freq
            d["return_type"], d["sign"], d["weighting"] = rem[:3]
            d["frequency"] = rem[3] if len(rem) >= 4 else "unknown"
    else:
        # rolling
        d["holding"] = "rolling"
        ridx = parts.index("rolling")
        d["period"] = int(parts[ridx + 1].lstrip("p"))
        base = ridx + 2
        rem = parts[base:]
        if len(rem) >= 4 and (rem[0] == "static" or rem[0].startswith("dynamic")):
            # Old layout: freq, ret, sign, wt
            d["frequency"], d["return_type"], d["sign"], d["weighting"] = rem[:4]
        else:
            # New layout: ret, sign, wt, freq
            d["return_type"], d["sign"], d["weighting"] = rem[:3]
            d["frequency"] = rem[3] if len(rem) >= 4 else "unknown"
    return d
# %%
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default=str(DEFAULT_ROOT), help="portfolio root dir")
    pa.add_argument("--out",  default=str(DEFAULT_OUT),  help="output dir")
    args = pa.parse_args()

    root, out = Path(args.root), Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rec_total, rec_year = [], []

    for dirpath in root.rglob("*"):
        if dirpath.name=="overall_summary.csv":
            meta = infer_from_path(dirpath.parent)
            df = pd.read_csv(dirpath).apply(parse_strategy, axis=1)
            for k,v in meta.items(): df[k]=v
            rec_total.append(df)

        elif dirpath.name=="yearly_performance.csv":
            meta = infer_from_path(dirpath.parent)
            dfy = pd.read_csv(dirpath)
            dfy["lag"] = dfy["base"].apply(_extract_lag)
            for k,v in meta.items(): dfy[k]=v
            rec_year.append(dfy)

    if rec_total:
        total = pd.concat(rec_total, ignore_index=True)
    else:
        total = pd.DataFrame()
    if rec_year:
        yearly = pd.concat(rec_year , ignore_index=True)
    else:
        yearly = pd.DataFrame()

    # Build unified portfolio name including 7 parameters
    # format: holding_period_lag{lag}_weight_p{percent*100}_frequency_returnType
    if not total.empty:
        total["portfolio"] = (
            total.apply(lambda r: f"{r['holding']}_{int(r['period'])}_lag{int(r['lag'])}_"
                                  f"{r['weighting']}_p{int(r['percent']*100)}_"
                                  f"{r['frequency']}_{r['return_type']}", axis=1)
        )

    # yearly: rename top_pct -> percent, then build portfolio the same way
    if not yearly.empty:
        if "top_pct" in yearly.columns and "percent" not in yearly.columns:
            yearly = yearly.rename(columns={"top_pct": "percent"})
        yearly["portfolio"] = (
            yearly.apply(lambda r: f"{r['holding']}_{int(r['period'])}_lag{int(r['lag'])}_"
                                     f"{r['weighting']}_p{int(r['percent']*100)}_"
                                     f"{r['frequency']}_{r['return_type']}", axis=1)
        )

    total.to_csv(out/"all_portfolios.csv", index=False)
    yearly.to_csv(out/"all_yearly.csv" , index=False)

    # Only two outputs as requested

    print("Outputs saved in", out.resolve())

if __name__ == "__main__":
    main()
