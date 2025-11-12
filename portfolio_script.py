#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/**
 * Policy-based portfolio back-test script (enhanced).
 * --------------------------------------------------
 * New features:
 *   1) Sign-based β filtering and arbitrary top-percent cut.
 *   2) Equal-weight or market-cap weight.
 *   3) Holding mechanism:
 *        • daily   : re-balance every trading day (period=1 only)
 *        • rolling : capital split into `period` slices, re-cycled via deque.
 *   4) Full CLI parameterisation (lag/period/frequency/price etc. consistent with regression).
 *
 * Example
 *   python portfolio_script.py \
 *          --reg_path "/path/to/industry_regressions_open_period1_static.csv" \
 *          --beta_sign pos \
 *          --percentages 0.1 0.2 0.3 0.4 0.5 1.0 \
 *          --weighting equal --period 1 \
 *          --return_type open_open --lags 1 3 5 --alpha 0.05
 *
 * @author  jliu
 * @version 2.0
 */
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from collections import deque
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# seaborn not required

# ========== Global defaults ==========
DATA_DIR   = "/hpc2hdd/home/jliu043/policy/data"
OUT_DIR    = "/hpc2hdd/home/jliu043/policy/portfolio"
ALPHA_DEF  = 0.05                         # default significance threshold
PCTS_DEF   = [i / 10 for i in range(1, 11)]  # 10%,20%,...,100%
# =====================================
REG_DIR   = "/hpc2hdd/home/jliu043/policy/regression"    # default regression dir


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
def _basket_return(basket: pd.DataFrame,
                   ret_today: "pd.Series") -> float:
    """
    /** Compute today’s return of a static basket.
     *  @param {DataFrame} basket      stkcd, weight
     *  @param {Series}    ret_today   Index=stkcd, value=ret
     *  @return {float}                Weighted basket return
     */
    """
    if basket.empty:
        return 0.0
    # Direct indexing; drop stocks without return today
    idx_ret = ret_today.reindex(basket["stkcd"]).values
    w = basket["weight"].values
    mask = ~np.isnan(idx_ret)
    if not mask.any():
        return 0.0
    return float(np.dot(w[mask], idx_ret[mask]))


# ---------- Rolling portfolio engine ----------
def rolling_engine_capital(dates: Sequence[pd.Timestamp],
                           signal_iter: Sequence[pd.DataFrame],
                           ret_lookup: dict[pd.Timestamp, pd.Series],
                           period: int) -> List[float]:
    """Finite-capital rolling engine mimicking ipynb logic.

    • Initial capital = 1, equally split into `period` sub-accounts.
    • At each day `t`, sub-account idx = t % period is settled, its
      capital grows by its realised return, then reused to build the
      new basket at t.
    • Portfolio return = (Total_after − Total_before) / Total_before.

    Returns a python list of daily portfolio returns (len = len(dates)).
    """
    caps = np.full(period, 1.0 / period, dtype=float)   # capital of each slice
    # queue holds realised returns of open baskets; length = period
    from collections import deque
    ret_queue: deque = deque([0.0] * period, maxlen=period)

    daily_ret: List[float] = []

    for i, (dt, basket) in enumerate(zip(dates, signal_iter)):
        idx = i % period                      # which slice rotates today
        cap_start = caps[idx]

        # 1) settle old position opened period days ago
        ret_old = ret_queue[0]
        cap_after = cap_start * (1.0 + ret_old)
        caps[idx] = cap_after                 # capital after settlement

        total_before = caps.sum() - cap_after + cap_start
        total_after  = caps.sum()
        port_ret = (total_after - total_before) / total_before
        daily_ret.append(port_ret)

        # 2) compute today new basket's expected return and enqueue
        new_ret = _basket_return(basket, ret_lookup[dt])
        ret_queue.append(new_ret)

    return daily_ret


# ---------- CLI ----------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Policy portfolio back-test (enhanced)")

    # ---------- Regression locating ----------
    p.add_argument("--reg_path", default=None,
                   help="Full path to regression CSV. "
                        "If omitted, script builds it from price/period/frequency.")
    p.add_argument("--reg_dir", default=REG_DIR,
                   help=f"Directory that stores regression CSVs (default: {REG_DIR})")

    # --- Regression related ---
    p.add_argument("--frequency", choices=["static", "dynamic"], default="static",
                   help="If 'dynamic', industry eligibility is based on prior-year significant β")

    # --- Beta sign & portfolio rules ---
    p.add_argument("--beta_sign", choices=["pos", "neg"], default="pos",
                   help="Trade sign of β")
    p.add_argument("--alpha", type=float, default=ALPHA_DEF,
                   help="Significance threshold on β")
    p.add_argument("--percentages", nargs="+", type=float, default=PCTS_DEF,
                   help="Top-percent cut list (0<p<=1); default 0.1 0.2 ... 1.0")

    # --- Market / equal weight ---
    p.add_argument("--weighting", choices=["equal", "mv"], default="equal")

    # --- Lag / period etc. ---
    p.add_argument("--lags", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--period", type=int, default=1,
                   help="Holding period (days); period>1 triggers rolling by default")
    p.add_argument("--holding", choices=["daily", "rolling"], default="auto",
                   help="'auto' → daily if period=1 else rolling")

    # --- Return type ---
    p.add_argument("--return_type", choices=["open_open", "close_close",
                                             "open_close", "close_open"],
                   default="open_open")

    # --- Misc paths ---
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--out_dir",  default=OUT_DIR)

    # --- Dynamic windows (for dynamic frequency) ---
    p.add_argument("--dynamic-windows", nargs="+", type=int, default=None,
                   help="If provided and frequency=dynamic, run also for each month-window (e.g. 1 3 6 9 12 24 36)")

    return p.parse_args()


# ---------- Main ----------
def main():
    args = _parse_cli()
    base_out_dir = Path(args.out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # price inferred from return_type prefix
    price = "open" if args.return_type.startswith("open") else "close"

    # If user specifies an explicit regression path, run a single job
    if args.reg_path:
        reg_path = Path(args.reg_path)
        freq_tag = "dynamic" if args.frequency == "dynamic" else "static"
        out_dir = (base_out_dir / freq_tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_portfolio_for_reg(args, reg_path, out_dir, freq_tag)
        return

    # Build window list for dynamic mode
    if args.frequency == "dynamic":
        if args.dynamic_windows:
            windows_to_run = list(args.dynamic_windows)
        else:
            raise ValueError("frequency=dynamic requires --dynamic-windows (e.g. --dynamic-windows 1 3 6 9 12 24 36)")
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
                f"Regression file not found: {reg_path}\n"
                f"(Hint: check --reg_dir / --period / --frequency / dynamic-windows / return_type)"
            )

        out_dir = (base_out_dir / freq_tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_portfolio_for_reg(args, reg_path, out_dir, freq_tag)


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
        wm = int(reg["window_months"].iloc[0])
        step_m = int(reg["window_step_months"].iloc[0]) if "window_step_months" in reg.columns else (12 if wm >= 12 else wm)
        # Build mapping by window_end (no look-ahead): for each lag, map end_date -> set(category_code)
        reg_sig = reg_sig.copy()
        reg_sig["window_end"] = pd.to_datetime(reg_sig["window_end"]).dt.normalize()
        for lg in args.lags:
            sub = reg_sig[reg_sig["lag"] == lg]
            end_to_set = {end_ts: set(tbl["category_code"].unique())
                          for end_ts, tbl in sub.groupby("window_end")}
            sorted_ends = sorted(end_to_set.keys())
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

    # ---- return series ----
    ohlcv["open_open"]   = ohlcv.groupby("stkcd")["Opnprc"].pct_change()
    ohlcv["close_close"] = ohlcv.groupby("stkcd")["Clsprc"].pct_change()
    ohlcv["open_close"]  = (ohlcv["Clsprc"] - ohlcv["Opnprc"]) / ohlcv["Opnprc"]
    ohlcv["close_open"]  = (
        ohlcv["Opnprc"] - ohlcv.groupby("stkcd")["Clsprc"].shift()
    ) / ohlcv.groupby("stkcd")["Clsprc"].shift()

    ret_col = args.return_type
    if ret_col not in ohlcv.columns:
        raise KeyError(f"{ret_col} not prepared in OHLCV")

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

    ret_use_col = ret_fwd_col if (args.period > 1 and (args.holding == "rolling" or args.holding == "auto")) else ret_col

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
        policy.merge(exposure, on=["city_code", "category_code"], how="left")
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

    policy_events = policy_events[(policy_events["trade_dt"].dt.year >= 2014) &
                                  (policy_events["trade_dt"].dt.year <= 2024)]

    # -------- Portfolio construction --------
    daily_ret_records = []
    holding_method = ("daily" if args.period == 1 and args.holding == "auto"
                      else ("rolling" if args.holding == "auto" else args.holding))

    for lg in args.lags:
        if is_dynamic and ("year" in reg.columns):
            elig_cache_year: dict[int, set] = {}
        elif is_dynamic and is_windowed:
            end_to_set, sorted_ends, wm, step_m = sig_ind_lag[lg]
        else:
            elig_set = sig_ind_lag[lg]

        ev_lag = policy_events[policy_events["lag"] == lg].copy()

        grouped_obj = ev_lag.groupby("trade_dt", sort=True)
        dates, signals = [], []
        for dt, df_day in grouped_obj:
            df_day = df_day[df_day["weighted_strength"] > 0]

            if is_dynamic and ("year" in reg.columns):
                prev_year = dt.year - 1
                if prev_year not in elig_cache_year:
                    sub = reg_sig[(reg_sig["lag"] == lg) & (reg_sig["year"] == prev_year)]
                    elig_cache_year[prev_year] = set(sub["category_code"])
                inds_today = elig_cache_year[prev_year]
                if len(inds_today) == 0:
                    continue
            elif is_dynamic and is_windowed:
                # choose the latest window whose end_date <= (dt - 1 day)
                # binary search on sorted_ends
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
                inds_today = elig_set if isinstance(elig_set, set) else elig_set

            df_day = df_day[df_day["category_code"].isin(inds_today)]
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
                    wt = mv / mv.sum()
                basket = top_rows[["stkcd"]].copy()
                basket["weight"] = wt
                basket["pct"] = pct
                pct_records.append(basket)

            dates.append(dt)
            signals.append(pd.concat(pct_records, ignore_index=True))

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
                series_ret = rolling_engine_capital(dates, sig_iter, ret_lookup, args.period)
                series_n  = pd.Series([b["stkcd"].nunique() for b in sig_iter], index=dates, name="n")
                series_ls = pd.Series([b["stkcd"].tolist() for b in sig_iter], index=dates, name="longset")
                series_w  = pd.Series([b["weight"].tolist() for b in sig_iter], index=dates, name="weight_vec")
                series_ret = pd.Series(series_ret, index=dates, name="ret")

            lbl = (
                f"{holding_method}_{int(args.period)}_lag{lg}_"
                f"{args.weighting}_p{int(pct*100)}_"
                f"{freq_tag}_{args.return_type}"
            )
            daily_ret_records.append(
                pd.DataFrame({"date":     series_ret.index,
                              "ret":      series_ret.values,
                              "n":        series_n.values,
                              "longset":  series_ls.values,
                              "weight_vec": series_w.values,
                              "strategy": lbl})
            )

    daily_df = pd.concat(daily_ret_records, ignore_index=True)

    daily_df["pct"]  = daily_df["strategy"].str.extract(r'_p(\d+)').astype(int) / 100
    daily_df["base"] = daily_df["strategy"].str.replace(r'_p\d+', '', regex=True)

    daily_ret_all = {}
    for base, df_base in daily_df.groupby("base"):
        inner = {}
        for pct, tbl in df_base.groupby("pct"):
            tbl = tbl.sort_values("date").copy()
            tbl["cum_ret"] = (1 + tbl["ret"]).cumprod()
            inner[pct] = tbl[["date", "ret", "cum_ret", "n", "longset", "weight_vec"]]
        daily_ret_all[base] = inner

    pkl_path = out_dir / "daily_ret_allpct.pkl.gz"
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(daily_ret_all, f)
    print("[INFO] Saved daily returns →", pkl_path)

    yearly_records = []
    for base, inner in daily_ret_all.items():
        for pct, tbl in inner.items():
            tbl["year"] = tbl["date"].dt.year
            perf = tbl.groupby("year").agg(
                avg_daily=("ret", "mean"),
                std_daily=("ret", "std"),
                days=("ret", "size"),
            )
            n_mean = (daily_df[(daily_df["base"] == base) & (daily_df["pct"]  == pct)]
                      .groupby(daily_df["date"].dt.year)["n"].mean())
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
        overall_tbl.append([strat, ann_ret, ann_vol, ann_ret / ann_vol, avg_n])

    pd.DataFrame(overall_tbl,
                 columns=["strategy", "ann_ret", "ann_vol", "sharpe", "avg_n"]
                 ).to_csv(out_dir / "overall_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    for strat, tbl in daily_df.groupby("strategy"):
        tbl = tbl.sort_values("date")
        plt.plot(tbl["date"], (1 + tbl["ret"]).cumprod(), label=strat)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "cum_ret_all.png", dpi=150)
    plt.close()

    print("[INFO] All outputs saved in:", out_dir)


if __name__ == "__main__":
    main() 