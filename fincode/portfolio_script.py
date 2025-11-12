#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Policy-based portfolio back-test script (config-enabled & robust CLI).

Changes vs your original:
- Defaults now come from config.py / .env (no hardcoded absolute paths)
- CLI unified to dash-style flags (and keeps underscore aliases for compatibility)
- Clear column validation for ohlcv/policy/exposure
- Friendly error if frequency=dynamic without --dynamic-windows
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from collections import deque
from pathlib import Path
from typing import List, Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Read defaults from config.py / .env =====
from config import (
    DATA_DIR as DATA_DIR_DEFAULT,
    PORTFOLIO_DIR as OUT_DIR_DEFAULT,
    REGRESSION_DIR as REG_DIR_DEFAULT,
    debug_print,
)

# === Default hyper-params (can still be overridden by CLI) ===
ALPHA_DEF  = 0.05                         # default significance threshold
PCTS_DEF   = [i / 10 for i in range(1, 11)]  # 10%,20%,...,100%

# ---------- Helper: trade-date shift ----------
def _shift_trade_date(trade_days: pd.Index, date: "pd.Timestamp", lag: int):
    """Return the lag-th trading day ≥ date; NaT if out of range."""
    pos = trade_days.searchsorted(pd.to_datetime(date), side="left")
    if (pos == len(trade_days)) or ((pos + lag) >= len(trade_days)):
        return pd.NaT
    return trade_days[pos + lag]

def _map_effective_dates(trade_days_np: np.ndarray,
                         policy_dates: np.ndarray,
                         lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mapping from policy dates to effective trade dates.
    Returns a tuple (mask, eff_dates) where mask indicates valid rows.
    """
    idx = np.searchsorted(trade_days_np, policy_dates, side="left")
    eff_idx = idx + lag
    mask = eff_idx < len(trade_days_np)
    eff_dates = np.empty(mask.sum(), dtype=trade_days_np.dtype)
    if mask.any():
        eff_dates[:] = trade_days_np[eff_idx[mask]]
    return mask, eff_dates

# ---------- Helper: month-window id ----------
def _assign_window_id_series(dates: pd.Series, window_months: int, anchor_month: int = 1) -> pd.Series:
    """Assign non-overlapping month-window ids (aligned to anchor_month=1 by default)."""
    months_since_anchor = dates.dt.year * 12 + (dates.dt.month - anchor_month)
    return (months_since_anchor // window_months).astype("Int64")

# ---------- Helper: portfolio return ----------
def _basket_return(basket: pd.DataFrame, ret_today: "pd.Series") -> float:
    """Compute today's weighted basket return."""
    if basket.empty:
        return 0.0
    idx_ret = ret_today.reindex(basket["stkcd"]).values
    w = basket["weight"].values
    mask = ~np.isnan(idx_ret)
    if not mask.any():
        return 0.0
    return float(np.dot(w[mask], idx_ret[mask]))

# ---------- Rolling portfolio engine ----------
def rolling_engine_capital(dates: Sequence[pd.Timestamp],
                           signal_iter: Sequence[pd.DataFrame],
                           ret_lookup: Dict[pd.Timestamp, pd.Series],
                           period: int) -> List[float]:
    """Finite-capital rolling engine:
    - Capital split into `period` slices.
    - Each day settle slice i%period, then reuse it to open today's basket.
    - Portfolio ret = (Total_after − Total_before) / Total_before.
    """
    caps = np.full(period, 1.0 / period, dtype=float)
    ret_queue: deque = deque([0.0] * period, maxlen=period)
    daily_ret: List[float] = []

    for i, (dt, basket) in enumerate(zip(dates, signal_iter)):
        idx = i % period
        cap_start = caps[idx]

        # 1) settle the position opened period days ago
        ret_old = ret_queue[0]
        cap_after = cap_start * (1.0 + ret_old)
        caps[idx] = cap_after

        total_before = caps.sum() - cap_after + cap_start
        total_after  = caps.sum()
        port_ret = (total_after - total_before) / total_before if total_before != 0 else 0.0
        daily_ret.append(port_ret)

        # 2) enqueue today's new position (expected return)
        new_ret = _basket_return(basket, ret_lookup.get(dt, pd.Series(dtype=float)))
        ret_queue.append(new_ret)

    return daily_ret

# ---------- CLI ----------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Policy portfolio back-test (enhanced)")

    # ---- regression locating ----
    p.add_argument("--reg-path", dest="reg_path", default=None,
                   help="Full path to regression CSV; if omitted, script builds the path from price/period/frequency.")
    p.add_argument("--reg_path", dest="reg_path")  # underscore alias (compat)
    p.add_argument("--reg-dir", dest="reg_dir", default=REG_DIR_DEFAULT,
                   help=f"Directory where regression CSVs are stored (default: {REG_DIR_DEFAULT})")
    p.add_argument("--reg_dir", dest="reg_dir")  # underscore alias (compat)

    # ---- frequency / dynamic windows ----
    p.add_argument("--frequency", choices=["static", "dynamic"], default="static",
                   help="If 'dynamic', eligibility is based on prior-year significant β or rolling windows.")
    p.add_argument("--dynamic-windows", dest="dynamic_windows", nargs="+", type=int, default=None,
                   help="If frequency=dynamic, specify month windows (e.g. 1 3 6 9 12 24 36). Required for dynamic.")
    p.add_argument("--dynamic_windows", dest="dynamic_windows", nargs="+", type=int)  # underscore alias (compat)

    # ---- beta sign / alpha / top-percent ----
    p.add_argument("--beta-sign", dest="beta_sign", choices=["pos", "neg"], default="pos",
                   help="Trade sign of β")
    p.add_argument("--beta_sign", dest="beta_sign")  # alias
    p.add_argument("--alpha", type=float, default=ALPHA_DEF, help="Significance threshold on β")
    p.add_argument("--percentages", nargs="+", type=float, default=PCTS_DEF,
                   help="Top-percent cut list (0<p<=1); default 0.1 0.2 ... 1.0")

    # ---- weighting ----
    p.add_argument("--weighting", choices=["equal", "mv"], default="equal",
                   help="equal=equal weight; mv=market-cap weight (requires Dsmvosd in ohlcv.csv)")

    # ---- lag / period / holding ----
    p.add_argument("--lags", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--period", type=int, default=1, help="Holding period (days); period>1 triggers rolling by default")
    p.add_argument("--holding", choices=["daily", "rolling", "auto"], default="auto",
                   help="'auto' → daily if period=1 else rolling")

    # ---- return type & price ----
    p.add_argument("--return-type", dest="return_type",
                   choices=["open_open", "close_close", "open_close", "close_open"],
                   default="open_open")
    p.add_argument("--return_type", dest="return_type")  # alias

    # ---- IO paths ----
    p.add_argument("--data-dir", dest="data_dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--data_dir", dest="data_dir")  # alias
    p.add_argument("--out-dir", dest="out_dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--out_dir", dest="out_dir")  # alias

    return p.parse_args()

# ---------- Validators ----------
def _require_columns(df: pd.DataFrame, required: set[str], name: str):
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{name} 缺少列: {sorted(missing)}\n"
                         f"请对照 data_sample/*.csv 的列名或你的真实数据列名进行适配。")

# ---------- Main ----------
def main():
    debug_print()  # show paths from .env/config.py (helpful for debugging)
    args = _parse_cli()

    base_out_dir = Path(args.out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # price inferred from return_type prefix
    price = "open" if args.return_type.startswith("open") else "close"

    # frequency=dynamic must provide dynamic windows unless explicit reg_path supplied
    if args.frequency == "dynamic" and not args.reg_path and not args.dynamic_windows:
        raise SystemExit("frequency=dynamic 需要提供 --dynamic-windows，例如：--dynamic-windows 1 3 6 9 12 24 36")

    # If user specifies an explicit regression path, run a single job
    if args.reg_path:
        reg_path = Path(args.reg_path)
        freq_tag = "dynamic" if args.frequency == "dynamic" else "static"
        out_dir = (base_out_dir / freq_tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_portfolio_for_reg(args, reg_path, out_dir, freq_tag)
        print("[INFO] All outputs saved in:", out_dir)
        return

    # Build window list for dynamic mode
    if args.frequency == "dynamic":
        windows_to_run = list(args.dynamic_windows)  # already validated above
    else:
        windows_to_run = [None]

    for win in windows_to_run:
        if args.frequency == "static":
            freq_tag = "static"
            fname = f"industry_regressions_{price}_period{args.period}_static.csv"
        else:
            freq_tag = f"dynamic_{int(win)}m"
            fname = f"industry_regressions_{price}_period{args.period}_{freq_tag}.csv"

        reg_path = Path(args.reg_dir) / fname
        if not reg_path.exists():
            raise FileNotFoundError(
                f"未找到回归文件: {reg_path}\n"
                f"请检查 --reg-dir / --period / --frequency / --dynamic-windows / --return-type 是否一致。"
            )

        out_dir = (base_out_dir / freq_tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_portfolio_for_reg(args, reg_path, out_dir, freq_tag)

    print("[INFO] All outputs saved in:", base_out_dir)

def _run_portfolio_for_reg(args: argparse.Namespace,
                           reg_path: Path,
                           out_dir: Path,
                           freq_tag: str) -> None:
    # Load regression table
    reg = pd.read_csv(reg_path)
    if "lag_days" in reg.columns and "lag" not in reg.columns:
        reg = reg.rename(columns={"lag_days": "lag"})

    is_dynamic = (freq_tag.startswith("dynamic"))
    is_windowed = ("window_months" in reg.columns)

    # β filtering
    if args.beta_sign == "pos":
        reg_sig = reg[(reg["p"] < args.alpha) & (reg["beta"] > 0)]
    else:
        reg_sig = reg[(reg["p"] < args.alpha) & (reg["beta"] < 0)]

    # Pre-compute eligible industry sets by lag
    sig_ind_lag = {}
    if is_dynamic and is_windowed:
        reg_sig = reg_sig.copy()
        reg_sig["window_end"] = pd.to_datetime(reg_sig["window_end"]).dt.normalize()
        for lg in args.lags:
            sub = reg_sig[reg_sig["lag"] == lg]
            end_to_set = {end_ts: set(tbl["category_code"].unique())
                          for end_ts, tbl in sub.groupby("window_end")}
            sorted_ends = sorted(end_to_set.keys())
            # wm/step_m are optional meta; safe defaults if missing
            wm = int(sub["window_months"].iloc[0]) if "window_months" in sub.columns and not sub.empty else 12
            step_m = int(sub["window_step_months"].iloc[0]) if "window_step_months" in sub.columns and not sub.empty else (12 if wm >= 12 else wm)
            sig_ind_lag[lg] = (end_to_set, sorted_ends, wm, step_m)
    else:
        for lg in args.lags:
            sub = reg_sig[reg_sig["lag"] == lg]
            if is_dynamic and ("year" in reg.columns):
                sig_ind_lag[lg] = sub  # keep table with years
            else:
                sig_ind_lag[lg] = set(sub["category_code"])  # static-like

    # -------- Load base data --------
    policy   = pd.read_csv(f"{args.data_dir}/policy.csv", parse_dates=["date_p"])
    exposure = pd.read_csv(f"{args.data_dir}/exposure.csv")
    ohlcv    = (
        pd.read_csv(f"{args.data_dir}/ohlcv.csv", parse_dates=["Trddt"])
          .rename(columns={"Stkcd": "stkcd", "Trddt": "trade_dt"})
          .sort_values(["stkcd", "trade_dt"])
    )

    # ---- Validate required columns (strict & friendly) ----
    _require_columns(ohlcv, {"stkcd", "trade_dt", "Opnprc", "Clsprc"}, "ohlcv.csv")
    # mv weighting requires Dsmvosd
    if args.weighting == "mv" and "Dsmvosd" not in ohlcv.columns:
        raise SystemExit("当 --weighting mv 时，ohlcv.csv 需要包含列 Dsmvosd（流通市值或等价权重基数）。")

    # the script expects policy × exposure keys below; if你的数据列不同，请在这里适配
    # Required for merging policy & exposure:
    #   policy  : date_p, category_code, ind_policy_strength
    #   exposure: category_code, stkcd, rev_pct (or weight column you use)
    required_policy = {"date_p", "category_code", "ind_policy_strength"}
    required_expo   = {"category_code", "stkcd", "rev_pct"}
    # 如果你的数据列名不同，请改成你自己的列名（或在这里增加兼容映射）
    _require_columns(policy, required_policy, "policy.csv")
    _require_columns(exposure, required_expo, "exposure.csv")

    # ---- return series ----
    ohlcv["open_open"]   = ohlcv.groupby("stkcd")["Opnprc"].pct_change()
    ohlcv["close_close"] = ohlcv.groupby("stkcd")["Clsprc"].pct_change()
    ohlcv["open_close"]  = (ohlcv["Clsprc"] - ohlcv["Opnprc"]) / ohlcv["Opnprc"]
    ohlcv["close_open"]  = (
        ohlcv["Opnprc"] - ohlcv.groupby("stkcd")["Clsprc"].shift()
    ) / ohlcv.groupby("stkcd")["Clsprc"].shift()

    ret_col = args.return_type
    if ret_col not in ohlcv.columns:
        raise KeyError(f"{ret_col} not prepared in ohlcv.csv")

    # Forward cumulative return for rolling holding when period>1
    if args.period > 1:
        ret_fwd_col = f"{ret_col}_fwd{args.period}"
        ohlcv[ret_fwd_col] = (
            ohlcv.groupby("stkcd")[ret_col]
                 .transform(
                     lambda s: (1 + s)
                     .rolling(window=args.period, min_periods=args.period)
                     .apply(np.prod, raw=True)
                     .shift(-args.period + 1) - 1
                 )
        )
    else:
        ret_fwd_col = ret_col

    ret_use_col = ret_fwd_col if (args.period > 1 and (args.holding in {"rolling", "auto"})) else ret_col

    if args.weighting == "mv":
        ohlcv["mv_weight"] = (
            ohlcv.groupby("trade_dt")["Dsmvosd"].transform(lambda x: x / x.sum())
        )
    else:
        ohlcv["mv_weight"] = np.nan

    ohlcv_small = (
        ohlcv[["stkcd", "trade_dt", "mv_weight", ret_use_col]]
          .rename(columns={ret_use_col: "ret"})
          .dropna(subset=["ret"])
    )

    # Policy × exposure → weighted_strength
    policy_exp = (
        policy.merge(exposure, on=["category_code"], how="left")
              .rename(columns={"Stkcd": "stkcd"})
    )
    policy_exp["weighted_strength"] = (
        policy_exp["ind_policy_strength"] * policy_exp["rev_pct"]
    )

    trade_days = pd.Index(pd.to_datetime(ohlcv_small["trade_dt"].unique())).sort_values()
    trade_days_np = trade_days.values

    # Policy events after each lag
    pol_rows = []
    base_cols = ["stkcd", "category_code", "weighted_strength", "date_p"]
    for lg in args.lags:
        tmp = policy_exp[base_cols]
        mask, eff_dates = _map_effective_dates(trade_days_np, tmp["date_p"].values.astype("datetime64[ns]"), lg)
        if not mask.any():
            continue
        tmp2 = tmp.loc[mask].copy()
        tmp2["trade_dt"] = pd.to_datetime(eff_dates)
        tmp2["lag"] = lg
        pol_rows.append(tmp2)

    policy_events = (
        pd.concat(pol_rows, ignore_index=True)
        if pol_rows else
        pd.DataFrame(columns=["stkcd","category_code","weighted_strength","date_p","trade_dt","lag"])
    )

    policy_events = (
        policy_events.merge(ohlcv_small, on=["stkcd", "trade_dt"], how="left")
                     .dropna(subset=["ret", "weighted_strength"])
    )

    # 时间过滤（按需修改）
    policy_events = policy_events[(policy_events["trade_dt"].dt.year >= 2014) &
                                  (policy_events["trade_dt"].dt.year <= 2024)]

    # -------- Portfolio construction --------
    daily_ret_records = []
    holding_method = ("daily" if args.period == 1 and args.holding == "auto"
                      else ("rolling" if args.holding == "auto" else args.holding))

    # cache for eligibility in dynamic-year mode
    reg_sig = None
    # rebuild reg_sig in case we filtered earlier
    # (we only need it for dynamic-year branch below)
    # NOTE: in this function, `reg` & `reg_sig` are local; they came from caller scope logically
    # but we reconstructed filtering above. For simplicity, re-prepare here:
    # (already prepared in outer scope; we keep local references if needed)

    for lg in args.lags:
        # eligibility set by mode
        if is_dynamic and ("year" in locals().get('reg', pd.DataFrame()).columns):
            elig_cache_year: dict[int, set] = {}
        elif is_dynamic and locals().get('is_windowed', False):
            # was prepared above: sig_ind_lag[lg] = (end_to_set, sorted_ends, wm, step_m)
            end_to_set, sorted_ends, wm, step_m = locals()['sig_ind_lag'][lg]
        else:
            elig_set = locals()['sig_ind_lag'][lg] if 'sig_ind_lag' in locals() else set()

        ev_lag = policy_events[policy_events["lag"] == lg].copy()

        grouped_obj = ev_lag.groupby("trade_dt", sort=True)
        dates, signals = [], []
        for dt, df_day in grouped_obj:
            df_day = df_day[df_day["weighted_strength"] > 0]

            if is_dynamic and ('reg' in locals() and "year" in reg.columns):
                prev_year = dt.year - 1
                if prev_year not in elig_cache_year:
                    sub = reg[(reg["p"] < args.alpha) & (reg["beta"] > 0 if args.beta_sign=="pos" else reg["beta"] < 0)]
                    sub = sub[(sub["lag"] == lg) & (sub["year"] == prev_year)]
                    elig_cache_year[prev_year] = set(sub["category_code"])
                inds_today = elig_cache_year[prev_year]
                if len(inds_today) == 0:
                    continue
            elif is_dynamic and locals().get('is_windowed', False):
                # choose the latest window whose end_date <= (dt - 1 day)
                from bisect import bisect_right
                cutoff = (pd.Timestamp(dt).normalize() - pd.Timedelta(days=1))
                pos = bisect_right(sorted_ends, cutoff)
                if pos == 0:
                    inds_today = set()
                else:
                    chosen_end = sorted_ends[pos - 1]
                    inds_today = end_to_set.get(chosen_end, set())
                if len(inds_today) == 0:
                    continue
            else:
                inds_today = elig_set if isinstance(elig_set, set) else set(elig_set)

            df_day = df_day[df_day["category_code"].isin(inds_today)] if inds_today else df_day
            if df_day.empty:
                dates.append(dt)
                empty_sig = pd.DataFrame({
                    "stkcd":  pd.Series(dtype=str),
                    "weight": pd.Series(dtype=float),
                    "pct":    pd.Series(dtype=float),
                })
                signals.append(empty_sig)
                continue

            df_day = df_day.sort_values("weighted_strength", ascending=False)
            n_total = len(df_day)
            pct_records = []
            for pct in args.percentages:
                sel_n = max(1, int(np.floor(n_total * pct)))
                top_rows = df_day.head(sel_n)
                if args.weighting == "equal":
                    wt = np.repeat(1 / sel_n, sel_n)
                else:
                    mv = top_rows["mv_weight"].values
                    wt = mv / mv.sum() if mv.sum() != 0 else np.repeat(1 / sel_n, sel_n)
                basket = top_rows[["stkcd"]].copy()
                basket["weight"] = wt
                basket["pct"] = pct
                pct_records.append(basket)

            dates.append(dt)
            signals.append(pd.concat(pct_records, ignore_index=True))

        # make today's return lookup
        ret_lookup = {}
        for dt in dates:
            df_dt = grouped_obj.get_group(dt)
            ret_lookup[dt] = df_dt.drop_duplicates("stkcd").set_index("stkcd")["ret"]

        for pct in args.percentages:
            sig_iter = [df_sig.loc[df_sig["pct"] == pct, ["stkcd", "weight"]] for df_sig in signals]

            if holding_method == "daily":
                daily_rets, daily_ns, daily_ls, daily_ws = [], [], [], []
                for dt, basket in zip(dates, sig_iter):
                    daily_rets.append(_basket_return(basket, ret_lookup[dt]))
                    daily_ns.append(basket["stkcd"].nunique())
                    daily_ls.append(basket["stkcd"].tolist())
                    daily_ws.append(basket["weight"].tolist())
                series_ret = pd.Series(daily_rets, index=dates, name="ret")
                series_n   = pd.Series(daily_ns,   index=dates, name="n")
                series_ls  = pd.Series(daily_ls,  index=dates, name="longset")
                series_w   = pd.Series(daily_ws,  index=dates, name="weight_vec")
            else:
                series_ret_vals = rolling_engine_capital(dates, sig_iter, ret_lookup, args.period)
                series_ret = pd.Series(series_ret_vals, index=dates, name="ret")
                series_n  = pd.Series([b["stkcd"].nunique() for b in sig_iter], index=dates, name="n")
                series_ls = pd.Series([b["stkcd"].tolist() for b in sig_iter], index=dates, name="longset")
                series_w  = pd.Series([b["weight"].tolist() for b in sig_iter], index=dates, name="weight_vec")

            lbl = (
                f"{holding_method}_{int(args.period)}_lag{lg}_"
                f"{args.weighting}_p{int(pct*100)}_"
                f"{freq_tag}_{args.return_type}"
            )
            out = pd.DataFrame({
                "date":       series_ret.index,
                "ret":        series_ret.values,
                "n":          series_n.values,
                "longset":    series_ls.values,
                "weight_vec": series_w.values,
                "strategy":   lbl
            })
            daily_ret_records.append(out)

    # ---- collect & persist outputs ----
    out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir  # guard
    daily_df = pd.concat(daily_ret_records, ignore_index=True)

    daily_df["pct"]  = daily_df["strategy"].str.extract(r'_p(\d+)').astype(int) / 100
    daily_df["base"] = daily_df["strategy"].str.replace(r'_p\d+', '', regex=True)

    daily_ret_all: Dict[str, Dict[float, pd.DataFrame]] = {}
    for base, df_base in daily_df.groupby("base"):
        inner: Dict[float, pd.DataFrame] = {}
        for pct, tbl in df_base.groupby("pct"):
            tbl = tbl.sort_values("date").copy()
            tbl["cum_ret"] = (1 + tbl["ret"]).cumprod()
            inner[pct] = tbl[["date", "ret", "cum_ret", "n", "longset", "weight_vec"]]
        daily_ret_all[base] = inner

    pkl_path = out_dir / "daily_ret_allpct.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(daily_ret_all, f)
    print("[INFO] Saved daily returns →", pkl_path)

    # yearly performance
    yearly_records = []
    for base, inner in daily_ret_all.items():
        for pct, tbl in inner.items():
            tbl = tbl.copy()
            tbl["year"] = pd.to_datetime(tbl["date"]).dt.year
            perf = tbl.groupby("year").agg(
                avg_daily=("ret", "mean"),
                std_daily=("ret", "std"),
                days=("ret", "size"),
            )
            n_mean = (daily_df[(daily_df["base"] == base) & (daily_df["pct"]  == pct)]
                      .assign(year=pd.to_datetime(daily_df["date"]).dt.year)
                      .groupby("year")["n"].mean())
            perf["avg_n"] = n_mean
            perf["ann_ret"] = (1 + perf["avg_daily"]) ** 252 - 1
            perf["ann_vol"] = perf["std_daily"] * np.sqrt(252)
            perf["sharpe"]  = perf["ann_ret"] / perf["ann_vol"]
            perf["top_pct"] = pct
            perf["base"]    = base
            yearly_records.append(perf.reset_index())

    yearly_perf = pd.concat(yearly_records, ignore_index=True)
    yearly_perf.to_csv(out_dir / "yearly_performance.csv", index=False)

    overall_tbl = []
    for strat, tbl in daily_df.groupby("strategy"):
        ann_ret = (1 + tbl["ret"].mean()) ** 252 - 1
        ann_vol = tbl["ret"].std() * np.sqrt(252)
        avg_n   = tbl["n"].mean()
        sharpe  = ann_ret / ann_vol if ann_vol != 0 else np.nan
        overall_tbl.append([strat, ann_ret, ann_vol, sharpe, avg_n])

    pd.DataFrame(overall_tbl,
                 columns=["strategy", "ann_ret", "ann_vol", "sharpe", "avg_n"]
                 ).to_csv(out_dir / "overall_summary.csv", index=False)

    # quick cumulative return plot
    plt.figure(figsize=(10, 5))
    for strat, tbl in daily_df.groupby("strategy"):
        tbl = tbl.sort_values("date")
        plt.plot(pd.to_datetime(tbl["date"]), (1 + tbl["ret"]).cumprod(), label=strat)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "cum_ret_all.png", dpi=150)
    plt.close()

    print("[INFO] All outputs saved in:", out_dir)

if __name__ == "__main__":
    main()

