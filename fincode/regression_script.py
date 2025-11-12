#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/**
 * Unified industry-level OLS regressions.
 * --------------------------------------
 * Model:   return = α + β · weighted_strength
 *
 * CLI example:
 *   python regression_script.py --frequency dynamic --price open --lags 1 3 5 10 --periods 1 3 5
 *
 * The script can also be imported:
 *   import regression_script as reg
 *   reg.run_regression(frequency="static", price="close")
 *
 * @author  jliu
 * @version 1.0
 */
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

# === Load defaults from config.py (reads .env automatically) ===
from config import (
    DATA_DIR as DATA_DIR_DEFAULT,
    REGRESSION_DIR as OUT_DIR_DEFAULT,
    YEAR_START as YEAR_START_DEFAULT,
    WINSOR_P as WINSOR_P_DEFAULT,
    debug_print,
)

# ========== Default parameters ==========
# Defaults resolved via config.py (overridable by CLI)
DATA_DIR   = DATA_DIR_DEFAULT
OUT_DIR    = OUT_DIR_DEFAULT
WINSOR_P   = WINSOR_P_DEFAULT
YEAR_START = YEAR_START_DEFAULT

# Keep script-local constants here
MIN_N_OBS  = 30
# ========================================


# ---------- Utilities ----------
def _compute_returns(ohlcv: pd.DataFrame, price_col: str,
                     periods: Sequence[int]) -> pd.DataFrame:
    """
    /** Compute daily and multi-day (cumulative) returns.
     *  @param {DataFrame} ohlcv       Source price table.
     *  @param {str}       price_col   "Opnprc" | "Clsprc".
     *  @param {List[int]} periods     e.g. [1,3,5].
     *  @return {DataFrame}            ohlcv with extra return columns.
     */
    """
    if 1 in periods:
        col_name = f"{price_col.lower()}_ret"
        ohlcv[col_name] = ohlcv.groupby("Stkcd")[price_col].pct_change()

    for p in periods:
        if p == 1:
            continue
        col_name = f"cum_{price_col.lower()}_ret_{p}d"
        ohlcv[col_name] = (
            ohlcv.groupby("Stkcd")[price_col]
                 .transform(lambda s: s.shift(-p) / s - 1)
        )
    return ohlcv


def _build_lag_sample(df_pol: pd.DataFrame,
                      lag: int,
                      y_col: str,
                      exposure: pd.DataFrame,
                      ohlcv: pd.DataFrame,
                      trading_dates: np.ndarray,
                      winsor_p: float) -> pd.DataFrame:
    """
    /** Merge policy, exposure and return data; create lagged sample.
     *  @param {DataFrame} df_pol        Policy table.
     *  @param {int}       lag           Lag in trading days.
     *  @param {str}       y_col         Return column to be explained.
     *  @return {DataFrame}              Cleaned & merged sample.
     */
    """
    idx = np.searchsorted(trading_dates, df_pol["date_p"].values, side="left")
    eff_idx = idx + lag

    eff_date = np.full(len(df_pol), np.datetime64("NaT"), dtype="datetime64[ns]")
    mask = eff_idx < len(trading_dates)
    eff_date[mask] = trading_dates[eff_idx[mask]]

    df_pol = df_pol.copy()
    df_pol["effective_date"] = pd.to_datetime(eff_date)

    merged = (
        df_pol.merge(exposure, how="inner", on=["city_code", "category_code"])
              .assign(weighted_strength=lambda d: d["ind_policy_strength"] * d["rev_pct"])
              .merge(ohlcv, how="left",
                     left_on=["Stkcd", "effective_date"],
                     right_on=["Stkcd", "Trddt"])
              .drop(columns=["Trddt"])
    )

    lo, hi = merged["weighted_strength"].quantile([winsor_p, 1 - winsor_p])
    merged["weighted_strength"] = merged["weighted_strength"].clip(lo, hi)

    return merged.dropna(subset=["weighted_strength", y_col])


def _run_industry_ols(df: pd.DataFrame, y_col: str, lag_label) -> pd.DataFrame:
    """
    /** Run cross-sectional OLS for each industry.
     *  @return {DataFrame} β, t, p, r², n_obs, lag
     */
    """
    out = []
    for ind, sub in df.groupby("category_code"):
        if len(sub) < MIN_N_OBS:
            continue
        X = sm.add_constant(sub["weighted_strength"])
        model = sm.OLS(sub[y_col], X).fit()
        out.append({
            "category_code": ind,
            "beta":  model.params["weighted_strength"],
            "t":     model.tvalues["weighted_strength"],
            "p":     model.pvalues["weighted_strength"],
            "r2":    model.rsquared,
            "n_obs": int(model.nobs),
            "lag":   lag_label
        })
    return pd.DataFrame(out)


def _run_yearly_industry_ols(df: pd.DataFrame, y_col: str, lag_label) -> pd.DataFrame:
    """
    /** Further split by year and run OLS.
     *  @return {DataFrame}  Same as `_run_industry_ols` plus `year`.
     */
    """
    yearly_out = []
    for yr, sub in df.groupby(df["effective_date"].dt.year):
        res = _run_industry_ols(sub, y_col, lag_label)
        if not res.empty:
            res["year"] = yr
            yearly_out.append(res)
    return pd.concat(yearly_out, ignore_index=True) if yearly_out else pd.DataFrame()


def _assign_window_id(dates: pd.Series, window_months: int, anchor_month: int = 1) -> pd.Series:
    """
    /** Assign a non-overlapping time-window id by months.
     *  Windows start at `anchor_month` of each year (default: January).
     *  Example: window_months=3 => Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec.
     */
    """
    # Normalize to month integers since year 0 and bucket by window size
    months_since_anchor = dates.dt.year * 12 + (dates.dt.month - anchor_month)
    return (months_since_anchor // window_months).astype("Int64")


def _run_windowed_industry_ols(df: pd.DataFrame,
                               y_col: str,
                               lag_label,
                               window_months: int) -> pd.DataFrame:
    """
    /** Split sample by fixed-length month windows and run OLS within each window.
     *  Adds metadata columns: window_months, window_start, window_end.
     */
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df[df["effective_date"].notna()]
    if df.empty:
        return pd.DataFrame()

    df["_window_id"] = _assign_window_id(df["effective_date"], window_months, anchor_month=1)

    windowed_out: list[pd.DataFrame] = []
    for wid, sub in df.groupby("_window_id"):
        if sub.empty:
            continue
        res = _run_industry_ols(sub, y_col, lag_label)
        if res.empty:
            continue
        # Compute window start/end aligned to the bucket defined by _assign_window_id
        any_date = sub["effective_date"].iloc[0]
        months_since_anchor = any_date.year * 12 + (any_date.month - 1)  # anchor at January
        bucket_start_msa = (months_since_anchor // window_months) * window_months
        start_year = bucket_start_msa // 12
        start_month_num = (bucket_start_msa % 12) + 1
        start_month = pd.Timestamp(year=int(start_year), month=int(start_month_num), day=1)
        # Window end is the last day of the end month in this bucket
        end_month_last_day = start_month + pd.offsets.MonthEnd(window_months)
        res["window_months"] = int(window_months)
        res["window_start"] = start_month
        res["window_end"] = end_month_last_day
        windowed_out.append(res)

    return pd.concat(windowed_out, ignore_index=True) if windowed_out else pd.DataFrame()


def _run_rolling_window_industry_ols(df: pd.DataFrame,
                                     y_col: str,
                                     lag_label,
                                     window_months: int,
                                     step_months: int) -> pd.DataFrame:
    """
    /** Rolling month-windows with configurable step.
     *  Example (24m, step=12): 2014-01~2015-12, 2015-01~2016-12, ...
     *  Adds: window_months, window_step_months, window_start, window_end.
     */
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df[df["effective_date"].notna()]
    if df.empty:
        return pd.DataFrame()

    # Build month indices
    eff_month = df["effective_date"].dt.to_period("M").dt.to_timestamp()
    df["_month_start"] = eff_month
    df["_month_id"] = df["_month_start"].dt.year * 12 + (df["_month_start"].dt.month - 1)

    # Anchor at YEAR_START-01
    anchor_start = pd.Timestamp(year=YEAR_START, month=1, day=1)
    anchor_id = anchor_start.year * 12 + (anchor_start.month - 1)

    min_id = int(max(anchor_id, int(df["_month_id"].min())))
    max_id = int(df["_month_id"].max())

    windowed_out: list[pd.DataFrame] = []
    for start_id in range(min_id, max_id - window_months + 2, step_months):
        end_id = start_id + window_months - 1
        if end_id > max_id:
            break

        # Convert ids to timestamps
        start_year = start_id // 12
        start_month = (start_id % 12) + 1
        start_ts = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
        end_ts = start_ts + pd.offsets.MonthEnd(window_months)

        sub = df[(df["_month_id"] >= start_id) & (df["_month_id"] <= end_id)]
        if sub.empty:
            continue
        res = _run_industry_ols(sub, y_col, lag_label)
        if res.empty:
            continue
        res["window_months"] = int(window_months)
        res["window_step_months"] = int(step_months)
        res["window_start"] = start_ts
        res["window_end"] = end_ts
        windowed_out.append(res)

    return pd.concat(windowed_out, ignore_index=True) if windowed_out else pd.DataFrame()


# ---------- Core runner ----------
def run_regression(frequency: str = "static",
                   price: str = "open",
                   lags: Sequence[int] | None = None,
                   periods: Sequence[int] | None = None,
                   dynamic_windows: Sequence[int] | None = None,
                   data_dir: str = DATA_DIR,
                   out_dir: str = OUT_DIR,
                   winsor_p: float = WINSOR_P) -> None:
    """
    /** Entry point callable by other scripts.
     *  @param {str} frequency  "static" | "dynamic"
     *  @param {str} price      "open" | "close"
     *  @param {List[int]} lags      List of lags (e.g. [1,3,5,10])
     *  @param {List[int]} periods   Holding periods (e.g. [1,3,5,10])
     */
    """
    lags = list(lags or [1, 3, 5, 10])
    periods = list(periods or [1])

    price_col = "Opnprc" if price == "open" else "Clsprc"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- 0) Load data ----
    policy   = pd.read_csv(f"{data_dir}/policy.csv",   parse_dates=["date_p"])
    exposure = pd.read_csv(f"{data_dir}/exposure.csv")
    usecols  = ["Stkcd", "Trddt", price_col]
    ohlcv    = (
        pd.read_csv(f"{data_dir}/ohlcv.csv", usecols=usecols, parse_dates=["Trddt"])
          .assign(Stkcd=lambda d: d["Stkcd"].astype("int32"))
          .sort_values(["Stkcd", "Trddt"])
    )

    if price_col not in ohlcv.columns:
        raise KeyError(f"{price_col} not found in ohlcv.csv.")

    # Compute returns
    ohlcv = _compute_returns(ohlcv, price_col, periods)

    # Restrict policy to YEAR_START+
    policy = policy.loc[policy["date_p"].dt.year >= YEAR_START]

    trading_dates = np.sort(ohlcv["Trddt"].unique())
    print(f"[INFO] Data loaded – policy:{policy.shape}, exposure:{exposure.shape}, ohlcv:{ohlcv.shape}")

    # ---- 1) Main loops ----
    for period in periods:
        y_col = (f"{price_col.lower()}_ret"
                 if period == 1
                 else f"cum_{price_col.lower()}_ret_{period}d")

        # Decide dynamic window list: if not provided, fall back to yearly behavior
        windows_to_run: List[int | None]
        if frequency == "dynamic" and dynamic_windows:
            windows_to_run = list(dynamic_windows)  # type: ignore[assignment]
        else:
            windows_to_run = [None]

        for win in windows_to_run:
            all_res = []

            for lag in lags:
                sample = _build_lag_sample(policy, lag, y_col,
                                           exposure, ohlcv, trading_dates, winsor_p)

                if frequency == "static":
                    res = _run_industry_ols(sample, y_col, lag_label=lag)
                else:
                    if win is None:
                        res = _run_yearly_industry_ols(sample, y_col, lag_label=lag)
                    else:
                        wm = int(win)
                        # Step rule: if wm >= 12 → step 12 months; else step = wm
                        step_m = 12 if wm >= 12 else wm
                        res = _run_rolling_window_industry_ols(sample, y_col, lag_label=lag,
                                                               window_months=wm, step_months=step_m)

                if not res.empty:
                    all_res.append(res)

            if not all_res:
                tag = f"dynamic_{win}m" if (frequency == "dynamic" and win is not None) else ("dynamic" if frequency == "dynamic" else "static")
                print(f"[WARN] No valid results for period={period} ({tag}).")
                continue

            df_out = pd.concat(all_res, ignore_index=True)
            if frequency == "static":
                freq_tag = "static"
                csv_name = f"industry_regressions_{price}_period{period}_{freq_tag}.csv"
            else:
                if win is None:
                    freq_tag = "dynamic"
                    csv_name = f"industry_regressions_{price}_period{period}_{freq_tag}.csv"
                else:
                    freq_tag = f"dynamic_{int(win)}m"
                    csv_name = f"industry_regressions_{price}_period{period}_{freq_tag}.csv"

            out_path = Path(out_dir) / csv_name
            df_out.to_csv(out_path, index=False)
            print(f"[INFO] Results saved → {out_path.resolve()}")


# ---------- CLI ----------
def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified Industry Regression Script")

    parser.add_argument("--frequency", choices=["static", "dynamic"],
                        default="static", help="static: all years together; dynamic: yearly regression")

    parser.add_argument("--price", choices=["open", "close"],
                        default="open", help="Return based on open or close price")

    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="Lag days (e.g. --lags 1 3 5)")

    parser.add_argument("--periods", nargs="+", type=int, default=[1],
                        help="Holding periods in days (e.g. --periods 1 3 5)")

    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Directory containing policy, exposure, ohlcv csv (default: {DATA_DIR})")

    parser.add_argument("--out-dir", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")

    parser.add_argument("--dynamic-windows", nargs="+", type=int, default=None,
                        help="For frequency=dynamic, month window sizes to group by (e.g. --dynamic-windows 1 3 6 9 12 24 36). If omitted, groups by calendar year.")

    return parser.parse_args()


def main() -> None:
    """CLI wrapper."""
    args = _parse_cli()
    run_regression(frequency=args.frequency,
                   price=args.price,
                   lags=args.lags,
                   periods=args.periods,
                   dynamic_windows=args.dynamic_windows,
                   data_dir=args.data_dir,
                   out_dir=args.out_dir)


if __name__ == "__main__":
    main() 
