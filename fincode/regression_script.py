#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified industry-level OLS regressions (config-enabled & robust).

Model:
    return = α + β · weighted_strength

Examples:
    # 静态（最小可复现）
    python fincode/regression_script.py --frequency static --price close --lags 1 --periods 1

    # 动态（滚动月窗）
    python fincode/regression_script.py --frequency dynamic --price open --lags 1 3 5 --periods 1 3 \
        --dynamic-windows 1 3 6 9 12 24 36

Notes:
    - 默认路径与参数均来自 .env / config.py，可用 CLI 覆盖
    - frequency=dynamic 且未给 --dynamic-windows 时，按“按年份分组”回归
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
# --- column normalization helpers ---
def _normalize_policy_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}

    # date
    if "date_p" not in df.columns:
        for cand in ["date_p", "date", "policy_date"]:
            if cand in df.columns:
                rename_map[cand] = "date_p"; break

    # industry code
    if "category_code" not in df.columns:
        for cand in ["category_code", "Indcd", "industry_code", "ind_code"]:
            if cand in df.columns:
                rename_map[cand] = "category_code"; break

    # policy strength
    if "ind_policy_strength" not in df.columns:
        for cand in ["ind_policy_strength", "policy_strength", "strength"]:
            if cand in df.columns:
                rename_map[cand] = "ind_policy_strength"; break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _normalize_exposure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}

    # stock code
    if "Stkcd" not in df.columns:
        for cand in ["Stkcd", "stkcd", "stock_code", "ticker"]:
            if cand in df.columns:
                rename_map[cand] = "Stkcd"; break

    # industry code
    if "category_code" not in df.columns:
        for cand in ["category_code", "Indcd", "industry_code", "ind_code"]:
            if cand in df.columns:
                rename_map[cand] = "category_code"; break

    # weight / revenue %
    if "rev_pct" not in df.columns:
        for cand in ["rev_pct", "weight", "w", "exposure_weight"]:
            if cand in df.columns:
                rename_map[cand] = "rev_pct"; break

    # city code（有就用，没有就以后用“仅行业”合并）
    if "city_code" not in df.columns:
        for cand in ["city_code", "CityCode"]:
            if cand in df.columns:
                rename_map[cand] = "city_code"; break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# ---------- Defaults (overridable by CLI) ----------
DATA_DIR   = DATA_DIR_DEFAULT
OUT_DIR    = OUT_DIR_DEFAULT
WINSOR_P   = WINSOR_P_DEFAULT
YEAR_START = YEAR_START_DEFAULT
MIN_N_OBS  = 30  # 每个行业最小样本数


# ===================== Utilities =====================
def _require_columns(df: pd.DataFrame, required: set[str], name: str):
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{name} 缺少列: {sorted(missing)}\n"
                         f"请对照 data_sample/*.csv 的列名或调整本脚本的列名映射。")


def _compute_returns(ohlcv: pd.DataFrame, price_col: str,
                     periods: Sequence[int]) -> pd.DataFrame:
    """计算 1日与多日累计收益（向前视窗）。"""
    if 1 in periods:
        col_name = f"{price_col.lower()}_ret"
        ohlcv[col_name] = ohlcv.groupby("Stkcd")[price_col].pct_change()

    for p in periods:
        if p == 1:
            continue
        col_name = f"cum_{price_col.lower()}_ret_{p}d"
        # 向前 p 日累计收益（右对齐到窗口起点）
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
    """合并政策 × 曝光 × 收益，并将政策日期映射到滞后 lag 的交易日。"""
    # 将政策日期映射到第 lag 个不早于政策日的交易日
    idx = np.searchsorted(trading_dates, df_pol["date_p"].values, side="left")
    eff_idx = idx + lag
    eff_date = np.full(len(df_pol), np.datetime64("NaT"), dtype="datetime64[ns]")
    mask = eff_idx < len(trading_dates)
    eff_date[mask] = trading_dates[eff_idx[mask]]

    df_pol = df_pol.copy()
    df_pol["effective_date"] = pd.to_datetime(eff_date)

    # 注意：这里按 (city_code, category_code) 与曝光表合并；
    # 如果你的 exposure 没有 city_code，可改为仅用 category_code 合并。
# 优先 city+industry；若 exposure 没有 city_code，则仅按 industry 合并
keys = ["category_code"]
if "city_code" in df_pol.columns and "city_code" in exposure.columns:
    keys = ["city_code", "category_code"]

merged = (
    df_pol.merge(exposure, how="inner", on=keys)
          .assign(weighted_strength=lambda d: d["ind_policy_strength"] * d["rev_pct"])
          .merge(ohlcv, how="left",
                 left_on=["Stkcd", "effective_date"],
                 right_on=["Stkcd", "Trddt"])
          .drop(columns=["Trddt"])
)


    # Winsorize 以稳健处理极端值
    lo, hi = merged["weighted_strength"].quantile([winsor_p, 1 - winsor_p])
    merged["weighted_strength"] = merged["weighted_strength"].clip(lo, hi)

    return merged.dropna(subset=["weighted_strength", y_col])


def _run_industry_ols(df: pd.DataFrame, y_col: str, lag_label) -> pd.DataFrame:
    """行业截面回归：返回 beta, t, p, r2, n_obs, lag。"""
    out = []
    for ind, sub in df.groupby("category_code"):
        if len(sub) < MIN_N_OBS:
            continue
        X = sm.add_constant(sub["weighted_strength"])
        model = sm.OLS(sub[y_col], X).fit()
        out.append({
            "category_code": ind,
            "beta":  model.params.get("weighted_strength", np.nan),
            "t":     model.tvalues.get("weighted_strength", np.nan),
            "p":     model.pvalues.get("weighted_strength", np.nan),
            "r2":    model.rsquared,
            "n_obs": int(model.nobs),
            "lag":   lag_label,
        })
    return pd.DataFrame(out)


def _run_yearly_industry_ols(df: pd.DataFrame, y_col: str, lag_label) -> pd.DataFrame:
    """按年份切分样本做行业回归。"""
    yearly_out = []
    df = df[df["effective_date"].notna()]
    for yr, sub in df.groupby(df["effective_date"].dt.year):
        res = _run_industry_ols(sub, y_col, lag_label)
        if not res.empty:
            res["year"] = yr
            yearly_out.append(res)
    return pd.concat(yearly_out, ignore_index=True) if yearly_out else pd.DataFrame()


def _assign_window_id(dates: pd.Series, window_months: int, anchor_month: int = 1) -> pd.Series:
    """固长月窗分桶（与 anchor_month 对齐，默认1月）。"""
    months_since_anchor = dates.dt.year * 12 + (dates.dt.month - anchor_month)
    return (months_since_anchor // window_months).astype("Int64")


def _run_windowed_industry_ols(df: pd.DataFrame,
                               y_col: str,
                               lag_label,
                               window_months: int) -> pd.DataFrame:
    """固定长度月窗（不重叠）做行业回归，附带窗口起止时间。"""
    if df.empty:
        return pd.DataFrame()
    df = df[df["effective_date"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["_wid"] = _assign_window_id(df["effective_date"], window_months, anchor_month=1)
    outs = []
    for wid, sub in df.groupby("_wid"):
        if sub.empty:
            continue
        res = _run_industry_ols(sub, y_col, lag_label)
        if res.empty:
            continue
        any_date = sub["effective_date"].iloc[0]
        months_since_anchor = any_date.year * 12 + (any_date.month - 1)
        bucket_start_msa = (months_since_anchor // window_months) * window_months
        start_year = bucket_start_msa // 12
        start_month_num = (bucket_start_msa % 12) + 1
        start_month = pd.Timestamp(year=int(start_year), month=int(start_month_num), day=1)
        end_month_last_day = start_month + pd.offsets.MonthEnd(window_months)
        res["window_months"] = int(window_months)
        res["window_start"] = start_month
        res["window_end"] = end_month_last_day
        outs.append(res)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def _run_rolling_window_industry_ols(df: pd.DataFrame,
                                     y_col: str,
                                     lag_label,
                                     window_months: int,
                                     step_months: int) -> pd.DataFrame:
    """滚动月窗（自定义步长）做行业回归。"""
    if df.empty:
        return pd.DataFrame()
    df = df[df["effective_date"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    eff_month = df["effective_date"].dt.to_period("M").dt.to_timestamp()
    df["_month_start"] = eff_month
    df["_mid"] = df["_month_start"].dt.year * 12 + (df["_month_start"].dt.month - 1)

    anchor_start = pd.Timestamp(year=YEAR_START, month=1, day=1)
    anchor_id = anchor_start.year * 12 + (anchor_start.month - 1)
    min_id = int(max(anchor_id, int(df["_mid"].min())))
    max_id = int(df["_mid"].max())

    outs = []
    for start_id in range(min_id, max_id - window_months + 2, step_months):
        end_id = start_id + window_months - 1
        if end_id > max_id:
            break
        start_year = start_id // 12
        start_month = (start_id % 12) + 1
        start_ts = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
        end_ts = start_ts + pd.offsets.MonthEnd(window_months)
        sub = df[(df["_mid"] >= start_id) & (df["_mid"] <= end_id)]
        if sub.empty:
            continue
        res = _run_industry_ols(sub, y_col, lag_label)
        if res.empty:
            continue
        res["window_months"] = int(window_months)
        res["window_step_months"] = int(step_months)
        res["window_start"] = start_ts
        res["window_end"] = end_ts
        outs.append(res)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# ===================== Core runner =====================
def run_regression(frequency: str = "static",
                   price: str = "open",
                   lags: Sequence[int] | None = None,
                   periods: Sequence[int] | None = None,
                   dynamic_windows: Sequence[int] | None = None,
                   data_dir: str = DATA_DIR,
                   out_dir: str = OUT_DIR,
                   winsor_p: float = WINSOR_P) -> None:
    """主流程，可供 import 调用。"""
    lags = list(lags or [1, 3, 5, 10])
    periods = list(periods or [1])

    price_col = "Opnprc" if price == "open" else "Clsprc"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
policy   = pd.read_csv(f"{data_dir}/policy.csv")
exposure = pd.read_csv(f"{data_dir}/exposure.csv")
ohlcv    = pd.read_csv(f"{data_dir}/ohlcv.csv", parse_dates=["Trddt"])

# normalize columns
policy   = _normalize_policy_cols(policy)
exposure = _normalize_exposure_cols(exposure)

# ---- Column validations ----
_require_columns(ohlcv, {"Stkcd", "Trddt", "Opnprc", "Clsprc"}, "ohlcv.csv")
_require_columns(policy, {"date_p", "category_code", "ind_policy_strength"}, "policy.csv")
# 对 exposure：允许没有 city_code（合并时退化为仅按行业合并）
_require_columns(exposure, {"category_code", "Stkcd", "rev_pct"}, "exposure.csv")

# ohlcv 只保留必要列并规范类型
ohlcv = (ohlcv[["Stkcd", "Trddt", "Opnprc", "Clsprc"]]
         .assign(Stkcd=lambda d: d["Stkcd"].astype("int64"))
         .sort_values(["Stkcd", "Trddt"]))
exposure = exposure.assign(Stkcd=lambda d: d["Stkcd"].astype("int64"))

# policy 日期解析 & YEAR_START 过滤
if not np.issubdtype(policy["date_p"].dtype, np.datetime64):
    policy["date_p"] = pd.to_datetime(policy["date_p"])
policy = policy.loc[policy["date_p"].dt.year >= YEAR_START]


    # 类型 & 排序
    ohlcv = (ohlcv[["Stkcd", "Trddt", "Opnprc", "Clsprc"]]
             .assign(Stkcd=lambda d: d["Stkcd"].astype("int64"))
             .sort_values(["Stkcd", "Trddt"]))
    exposure = exposure.assign(Stkcd=lambda d: d["Stkcd"].astype("int64"))

    # Compute returns
    ohlcv = _compute_returns(ohlcv, price_col, periods)

    # Restrict policy to YEAR_START+
    policy = policy.loc[policy["date_p"].dt.year >= YEAR_START]

    trading_dates = np.sort(ohlcv["Trddt"].unique())
    print(f"[INFO] Data loaded – policy:{policy.shape}, exposure:{exposure.shape}, ohlcv:{ohlcv.shape}")

    # ---- Main loops ----
    for period in periods:
        y_col = (f"{price_col.lower()}_ret"
                 if period == 1
                 else f"cum_{price_col.lower()}_ret_{period}d")

        # 动态窗口列表：未提供则按“按年”动态
        windows_to_run: List[int | None]
        if frequency == "dynamic" and dynamic_windows:
            windows_to_run = list(dynamic_windows)  # 滚动月窗
        else:
            windows_to_run = [None]  # 按年

        for win in windows_to_run:
            all_res = []

            for lag in lags:
                sample = _build_lag_sample(policy, lag, y_col,
                                           exposure, ohlcv, trading_dates, winsor_p)

                if frequency == "static":
                    res = _run_industry_ols(sample, y_col, lag_label=lag)

                else:
                    if win is None:
                        # 动态：按年
                        res = _run_yearly_industry_ols(sample, y_col, lag_label=lag)
                    else:
                        # 动态：滚动月窗
                        wm = int(win)
                        step_m = 12 if wm >= 12 else wm
                        res = _run_rolling_window_industry_ols(
                            sample, y_col, lag_label=lag,
                            window_months=wm, step_months=step_m
                        )

                if not res.empty:
                    all_res.append(res)

            # 保存
            if not all_res:
                tag = (f"dynamic_{win}m" if (frequency == "dynamic" and win is not None)
                       else ("dynamic" if frequency == "dynamic" else "static"))
                print(f"[WARN] No valid results for period={period} ({tag}).")
                continue

            df_out = pd.concat(all_res, ignore_index=True)
            if frequency == "static":
                freq_tag = "static"
            else:
                freq_tag = (f"dynamic_{int(win)}m" if win is not None else "dynamic")

            csv_name = f"industry_regressions_{price}_period{period}_{freq_tag}.csv"
            out_path = Path(out_dir) / csv_name
            df_out.to_csv(out_path, index=False)
            print(f"[INFO] Results saved → {out_path.resolve()}")


# ===================== CLI =====================
def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments (dash style, with underscore aliases)."""
    parser = argparse.ArgumentParser(description="Unified Industry Regression Script")

    parser.add_argument("--frequency", choices=["static", "dynamic"],
                        default="static", help="static: 全样本; dynamic: 按年/按窗口分组")

    parser.add_argument("--price", choices=["open", "close"],
                        default="open", help="收益基于 open 或 close 价计算")

    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="滞后交易日，如 --lags 1 3 5")

    parser.add_argument("--periods", nargs="+", type=int, default=[1],
                        help="持有期（天），如 --periods 1 3 5")

    # 路径（短横线为主），并兼容旧下划线名
    parser.add_argument("--data-dir", dest="data_dir", default=DATA_DIR,
                        help=f"包含 policy/exposure/ohlcv 的目录（默认: {DATA_DIR})")
    parser.add_argument("--data_dir", dest="data_dir")  # 兼容

    parser.add_argument("--out-dir", dest="out_dir", default=OUT_DIR,
                        help=f"输出目录（默认: {OUT_DIR})")
    parser.add_argument("--out_dir", dest="out_dir")  # 兼容

    parser.add_argument("--dynamic-windows", dest="dynamic_windows", nargs="+", type=int, default=None,
                        help="当 frequency=dynamic 时可指定月窗（如 1 3 6 9 12 24 36）；不指定则按年份分组。")
    parser.add_argument("--dynamic_windows", dest="dynamic_windows", nargs="+", type=int)  # 兼容

    return parser.parse_args()


def main() -> None:
    """CLI wrapper."""
    debug_print()  # 打印当前 .env / config 生效配置，便于排错
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
