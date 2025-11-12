#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline vs Benchmark Regression Analysis Script

This script performs regression analysis where:
- Y (dependent variable): True baseline strategy returns (lag3, mv, 70%, dynamic_1m)
- X (independent variables): Four benchmark strategy returns (1/N, CSI300, Momentum, Reversal)

The regression helps understand how much of the baseline strategy's performance
can be explained by the benchmark strategies.

Author: jliu
Date: 2025-09-23
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set matplotlib backend
plt.style.use("default")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Default paths
DATA_DIR = "/hpc2hdd/home/jliu043/policy/data"
OUT_DIR = "/hpc2hdd/home/jliu043/policy/portfolio/baseline/analysis"
YEAR_START = 2014
YEAR_END = 2024

# Debug collector
DEBUG_LINES: List[str] = []

def _debug(msg: str) -> None:
    """Print and collect debug messages."""
    try:
        DEBUG_LINES.append(str(msg))
        print(msg)
    except Exception:
        pass


def load_strategy_returns(pkl_path: str, strategy_key: str, percentage: float = 1.0) -> pd.Series:
    """Load strategy returns from pkl file."""
    try:
        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        if strategy_key in data:
            bundle = data[strategy_key]
            if percentage in bundle:
                df = bundle[percentage]
            elif 1.0 in bundle:
                df = bundle[1.0]
            else:
                # Use first available percentage
                first_pct = list(bundle.keys())[0]
                df = bundle[first_pct]
                _debug(f"Using percentage {first_pct} for {strategy_key}")
            
            series = pd.Series(
                df["ret"].values, 
                index=pd.to_datetime(df["date"])
            ).sort_index()
            
            return series
        else:
            _debug(f"Strategy key '{strategy_key}' not found in {pkl_path}")
            return pd.Series(dtype=float)
            
    except Exception as e:
        _debug(f"Failed to load {pkl_path}: {e}")
        return pd.Series(dtype=float)


def load_all_strategies() -> Dict[str, pd.Series]:
    """Load all strategy returns for regression analysis."""
    
    # Define strategy configurations
    strategies = {
        "True_Baseline": {
            "path": "/hpc2hdd/home/jliu043/policy/portfolio/daily/close_close/pos/mv/dynamic_1m/daily_ret_allpct.pkl.gz",
            "key": "daily_1_lag3_mv_dynamic_1m_close_close",
            "percentage": 0.7  # 70%选股
        },
        "CSI300_Index": {
            "path": "/hpc2hdd/home/jliu043/policy/portfolio/baseline/daily/close_close/pos/index/index_csi300/daily_ret_allpct.pkl.gz",
            "key": None,  # Will be auto-detected
            "percentage": 1.0
        },
        "1N_Strategy": {
            "path": "/hpc2hdd/home/jliu043/policy/portfolio/baseline/daily/close_close/pos/equal/baseline_1n/daily_ret_allpct.pkl.gz",
            "key": None,  # Will be auto-detected
            "percentage": 1.0
        },
        "Momentum_2_12": {
            "path": "/hpc2hdd/home/jliu043/policy/portfolio/baseline/daily/close_close/pos/equal/momentum_LD20/daily_ret_allpct.pkl.gz",
            "key": None,  # Will be auto-detected
            "percentage": 1.0
        },
        "Reversal_1m": {
            "path": "/hpc2hdd/home/jliu043/policy/portfolio/baseline/daily/close_close/neg/equal/reversal_L20/daily_ret_allpct.pkl.gz",
            "key": None,  # Will be auto-detected
            "percentage": 1.0
        }
    }
    
    loaded_strategies = {}
    
    for strategy_name, config in strategies.items():
        pkl_path = config["path"]
        
        if not Path(pkl_path).exists():
            _debug(f"File not found: {pkl_path}")
            continue
        
        # Auto-detect key if not specified
        if config["key"] is None:
            try:
                with gzip.open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                
                # Find appropriate key based on strategy name
                key_candidates = []
                for key in data.keys():
                    if strategy_name == "1N_Strategy" and "baseline_1n" in key.lower():
                        key_candidates.append(key)
                    elif strategy_name == "CSI300_Index" and "index_csi300" in key.lower():
                        key_candidates.append(key)
                    elif strategy_name == "Momentum_2_12" and "momentum" in key.lower():
                        key_candidates.append(key)
                    elif strategy_name == "Reversal_1m" and "reversal" in key.lower():
                        key_candidates.append(key)
                
                if key_candidates:
                    config["key"] = key_candidates[0]
                    _debug(f"Auto-detected key for {strategy_name}: {config['key']}")
                else:
                    _debug(f"No suitable key found for {strategy_name}")
                    continue
                    
            except Exception as e:
                _debug(f"Failed to auto-detect key for {strategy_name}: {e}")
                continue
        
        # Load the strategy returns
        series = load_strategy_returns(pkl_path, config["key"], config["percentage"])
        
        if not series.empty:
            # Filter to 2014-2024 period
            date_start = pd.Timestamp(YEAR_START, 1, 1)
            date_end = pd.Timestamp(YEAR_END, 12, 31)
            series = series[(series.index >= date_start) & (series.index <= date_end)]
            
            if not series.empty:
                loaded_strategies[strategy_name] = series
                _debug(f"Loaded {strategy_name}: {len(series)} observations, "
                       f"range=[{series.index.min().date()} .. {series.index.max().date()}]")
            else:
                _debug(f"{strategy_name}: No data in 2014-2024 period")
        else:
            _debug(f"{strategy_name}: Failed to load returns")
    
    return loaded_strategies


def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """Calculate comprehensive performance metrics for a return series."""
    
    if returns.empty:
        return {}
    
    # Basic statistics
    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Calmar ratio (annual return / max drawdown)
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Additional metrics
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "observations": len(returns)
    }


def perform_benchmark_comparison(strategies: Dict[str, pd.Series], out_dir: Path) -> Dict:
    """Perform comprehensive benchmark comparison analysis."""
    
    if "True_Baseline" not in strategies:
        _debug("Error: True_Baseline not found in loaded strategies")
        return {}
    
    baseline = strategies["True_Baseline"]
    benchmark_names = ["CSI300_Index", "1N_Strategy", "Momentum_2_12", "Reversal_1m"]
    
    # Check which benchmarks are available
    available_benchmarks = [name for name in benchmark_names if name in strategies]
    
    if not available_benchmarks:
        _debug("Error: No benchmark strategies available for comparison")
        return {}
    
    _debug(f"Available benchmarks for comparison: {available_benchmarks}")
    
    # Calculate performance metrics for baseline
    baseline_metrics = calculate_performance_metrics(baseline)
    _debug(f"Baseline performance metrics calculated")
    
    # Calculate performance metrics for each benchmark
    comparison_results = {}
    
    for benchmark_name in available_benchmarks:
        _debug(f"Calculating performance metrics for {benchmark_name}")
        
        benchmark_series = strategies[benchmark_name]
        benchmark_metrics = calculate_performance_metrics(benchmark_series)
        
        # Calculate spread (baseline - benchmark)
        common_dates = baseline.index.intersection(benchmark_series.index).sort_values()
        if len(common_dates) > 0:
            baseline_aligned = baseline.loc[common_dates]
            benchmark_aligned = benchmark_series.loc[common_dates]
            spread = baseline_aligned - benchmark_aligned
            spread_metrics = calculate_performance_metrics(spread)
        else:
            spread_metrics = {}
        
        comparison_results[benchmark_name] = {
            "baseline_metrics": baseline_metrics,
            "benchmark_metrics": benchmark_metrics,
            "spread_metrics": spread_metrics,
            "common_observations": len(common_dates)
        }
        
        _debug(f"  {benchmark_name}: {len(common_dates)} common observations")
    
    # Save comparison results
    save_benchmark_comparison_results(comparison_results, out_dir)
    
    return comparison_results


def save_benchmark_comparison_results(comparison_results: Dict, out_dir: Path):
    """Save benchmark comparison results to files."""
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive comparison table
    comparison_data = []
    
    for benchmark_name, results in comparison_results.items():
        baseline_metrics = results["baseline_metrics"]
        benchmark_metrics = results["benchmark_metrics"]
        spread_metrics = results["spread_metrics"]
        
        comparison_data.append({
            "Strategy": "True_Baseline",
            "Benchmark": benchmark_name,
            "Observations": results["common_observations"],
            "Ann_Return": f"{baseline_metrics['ann_return']*100:.2f}%",
            "Ann_Vol": f"{baseline_metrics['ann_vol']*100:.2f}%",
            "Sharpe": f"{baseline_metrics['sharpe']:.4f}",
            "Calmar": f"{baseline_metrics['calmar']:.4f}",
            "Max_Drawdown": f"{baseline_metrics['max_drawdown']*100:.2f}%",
            "Skewness": f"{baseline_metrics['skewness']:.4f}",
            "Kurtosis": f"{baseline_metrics['kurtosis']:.4f}"
        })
        
        comparison_data.append({
            "Strategy": benchmark_name,
            "Benchmark": benchmark_name,
            "Observations": results["common_observations"],
            "Ann_Return": f"{benchmark_metrics['ann_return']*100:.2f}%",
            "Ann_Vol": f"{benchmark_metrics['ann_vol']*100:.2f}%",
            "Sharpe": f"{benchmark_metrics['sharpe']:.4f}",
            "Calmar": f"{benchmark_metrics['calmar']:.4f}",
            "Max_Drawdown": f"{benchmark_metrics['max_drawdown']*100:.2f}%",
            "Skewness": f"{benchmark_metrics['skewness']:.4f}",
            "Kurtosis": f"{benchmark_metrics['kurtosis']:.4f}"
        })
        
        if spread_metrics:
            comparison_data.append({
                "Strategy": f"Spread (Baseline - {benchmark_name})",
                "Benchmark": benchmark_name,
                "Observations": results["common_observations"],
                "Ann_Return": f"{spread_metrics['ann_return']*100:.2f}%",
                "Ann_Vol": f"{spread_metrics['ann_vol']*100:.2f}%",
                "Sharpe": f"{spread_metrics['sharpe']:.4f}",
                "Calmar": f"{spread_metrics['calmar']:.4f}",
                "Max_Drawdown": f"{spread_metrics['max_drawdown']*100:.2f}%",
                "Skewness": f"{spread_metrics['skewness']:.4f}",
                "Kurtosis": f"{spread_metrics['kurtosis']:.4f}"
            })
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(out_dir / "benchmark_performance_comparison.csv", index=False)
    
    # Create detailed comparison report
    with open(out_dir / "benchmark_comparison_report.txt", "w") as f:
        f.write("Benchmark Performance Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time Period: {YEAR_START}-{YEAR_END}\n")
        f.write(f"Number of Benchmarks: {len(comparison_results)}\n\n")
        
        f.write("Performance Metrics Summary:\n")
        f.write("-" * 40 + "\n\n")
        
        for benchmark_name, results in comparison_results.items():
            baseline_metrics = results["baseline_metrics"]
            benchmark_metrics = results["benchmark_metrics"]
            spread_metrics = results["spread_metrics"]
            
            f.write(f"{benchmark_name} vs True_Baseline:\n")
            f.write(f"  Common Observations: {results['common_observations']}\n\n")
            
            f.write(f"  True_Baseline:\n")
            f.write(f"    Annual Return: {baseline_metrics['ann_return']*100:.2f}%\n")
            f.write(f"    Annual Volatility: {baseline_metrics['ann_vol']*100:.2f}%\n")
            f.write(f"    Sharpe Ratio: {baseline_metrics['sharpe']:.4f}\n")
            f.write(f"    Calmar Ratio: {baseline_metrics['calmar']:.4f}\n")
            f.write(f"    Max Drawdown: {baseline_metrics['max_drawdown']*100:.2f}%\n\n")
            
            f.write(f"  {benchmark_name}:\n")
            f.write(f"    Annual Return: {benchmark_metrics['ann_return']*100:.2f}%\n")
            f.write(f"    Annual Volatility: {benchmark_metrics['ann_vol']*100:.2f}%\n")
            f.write(f"    Sharpe Ratio: {benchmark_metrics['sharpe']:.4f}\n")
            f.write(f"    Calmar Ratio: {benchmark_metrics['calmar']:.4f}\n")
            f.write(f"    Max Drawdown: {benchmark_metrics['max_drawdown']*100:.2f}%\n\n")
            
            if spread_metrics:
                f.write(f"  Spread (Baseline - {benchmark_name}):\n")
                f.write(f"    Annual Return: {spread_metrics['ann_return']*100:.2f}%\n")
                f.write(f"    Annual Volatility: {spread_metrics['ann_vol']*100:.2f}%\n")
                f.write(f"    Sharpe Ratio: {spread_metrics['sharpe']:.4f}\n")
                f.write(f"    Calmar Ratio: {spread_metrics['calmar']:.4f}\n")
                f.write(f"    Max Drawdown: {spread_metrics['max_drawdown']*100:.2f}%\n\n")
            
            f.write("-" * 40 + "\n\n")
    
    _debug(f"Benchmark comparison results saved to {out_dir}")


def perform_single_benchmark_regression(baseline: pd.Series, benchmark_series: pd.Series, 
                                       benchmark_name: str) -> Dict:
    """Perform regression analysis: Y=baseline, X=single benchmark."""
    
    # Find common dates
    common_dates = baseline.index.intersection(benchmark_series.index).sort_values()
    
    if len(common_dates) < 100:  # Minimum observations
        _debug(f"Error: Only {len(common_dates)} common dates found for {benchmark_name}, need at least 100")
        return {}
    
    # Prepare data
    y = baseline.loc[common_dates].dropna()
    x = benchmark_series.loc[common_dates].loc[y.index]
    
    _debug(f"  After alignment - y: {len(y)}, x: {len(x)}")
    _debug(f"  y stats - mean: {y.mean():.6f}, std: {y.std():.6f}, min: {y.min():.6f}, max: {y.max():.6f}")
    _debug(f"  x stats - mean: {x.mean():.6f}, std: {x.std():.6f}, min: {x.min():.6f}, max: {x.max():.6f}")
    
    if len(y) < 100:
        _debug(f"Error: Only {len(y)} valid observations for {benchmark_name}")
        return {}
    
    # Check for constant series (no variation)
    if x.std() == 0:
        _debug(f"Error: {benchmark_name} has no variation (std=0)")
        return {}
    
    if y.std() == 0:
        _debug(f"Error: Baseline has no variation (std=0)")
        return {}
    
    # Add constant term
    X = sm.add_constant(x)
    
    _debug(f"  Data shapes - y: {len(y)}, x: {len(x)}, X: {X.shape}")
    
    try:
        model = sm.OLS(y, X).fit()
        _debug(f"  Model fitted successfully - params: {list(model.params.index)}")
        
        # Extract results
        results = {
            "benchmark_name": benchmark_name,
            "model": model,
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "baseline_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "ann_return": float((1 + y.mean()) ** 252 - 1),
                "ann_vol": float(y.std() * np.sqrt(252)),
                "sharpe": float((1 + y.mean()) ** 252 - 1) / (y.std() * np.sqrt(252))
            },
            "benchmark_stats": {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "ann_return": float((1 + x.mean()) ** 252 - 1),
                "ann_vol": float(x.std() * np.sqrt(252)),
                "sharpe": float((1 + x.mean()) ** 252 - 1) / (x.std() * np.sqrt(252))
            },
            "coefficients": {
                "constant": {
                    "coefficient": float(model.params.iloc[0]),
                    "ann_coefficient": float((1 + model.params.iloc[0]) ** 252 - 1),  # Annualized alpha
                    "std_error": float(model.bse.iloc[0]),
                    "ann_std_error": float(model.bse.iloc[0] * np.sqrt(252)),  # Annualized std error
                    "t_statistic": float(model.tvalues.iloc[0]),
                    "p_value": float(model.pvalues.iloc[0]),
                    "confidence_interval": model.conf_int().iloc[0].tolist(),
                    "ann_confidence_interval": [(1 + ci) ** 252 - 1 for ci in model.conf_int().iloc[0].tolist()]  # Annualized CI
                },
                "benchmark": {
                    "coefficient": float(model.params.iloc[1]),
                    "std_error": float(model.bse.iloc[1]),
                    "t_statistic": float(model.tvalues.iloc[1]),
                    "p_value": float(model.pvalues.iloc[1]),
                    "confidence_interval": model.conf_int().iloc[1].tolist()
                }
            }
        }
        
        return results
        
    except Exception as e:
        _debug(f"Regression failed for {benchmark_name}: {e}")
        _debug(f"  Exception type: {type(e).__name__}")
        import traceback
        _debug(f"  Traceback: {traceback.format_exc()}")
        return {}




def perform_regression_analysis(strategies: Dict[str, pd.Series], out_dir: Path) -> Dict:
    """Perform regression analysis: Y=baseline, X=each benchmark separately."""
    
    if "True_Baseline" not in strategies:
        _debug("Error: True_Baseline not found in loaded strategies")
        return {}
    
    baseline = strategies["True_Baseline"]
    benchmark_names = ["CSI300_Index", "1N_Strategy", "Momentum_2_12", "Reversal_1m"]
    
    # Check which benchmarks are available
    available_benchmarks = [name for name in benchmark_names if name in strategies]
    
    if not available_benchmarks:
        _debug("Error: No benchmark strategies available for regression")
        return {}
    
    _debug(f"Available benchmarks for regression: {available_benchmarks}")
    
    # Perform regression for each benchmark separately
    all_results = {}
    
    for benchmark_name in available_benchmarks:
        _debug(f"Performing regression: True_Baseline vs {benchmark_name}")
        
        benchmark_series = strategies[benchmark_name]
        results = perform_single_benchmark_regression(baseline, benchmark_series, benchmark_name)
        
        if results:
            all_results[benchmark_name] = results
            _debug(f"  R-squared: {results['r_squared']:.4f}")
            _debug(f"  Coefficient: {results['coefficients']['benchmark']['coefficient']:.6f}")
            _debug(f"  T-statistic: {results['coefficients']['benchmark']['t_statistic']:.4f}")
            _debug(f"  P-value: {results['coefficients']['benchmark']['p_value']:.4f}")
        else:
            _debug(f"  Failed to perform regression for {benchmark_name}")
        
        _debug("")
    
    if all_results:
        # Save all results
        save_all_regression_results(all_results, out_dir)
        
        # Create combined summary
        create_combined_summary(all_results, out_dir)
        
        return all_results
    else:
        _debug("Error: No successful regressions performed")
        return {}


def save_all_regression_results(all_results: Dict, out_dir: Path):
    """Save all regression results to files."""
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for individual regression results
    individual_dir = out_dir / "individual_regressions"
    individual_dir.mkdir(exist_ok=True)
    
    # Save individual regression results
    for benchmark_name, results in all_results.items():
        save_single_regression_results(results, individual_dir, benchmark_name)
    
    _debug(f"All regression results saved to {individual_dir}")


def save_single_regression_results(results: Dict, out_dir: Path, benchmark_name: str):
    """Save single regression results to files."""
    
    # Save regression summary for this benchmark
    summary_data = []
    
    # Model statistics
    summary_data.append({
        "Variable": "Model Statistics",
        "Coefficient": "",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    summary_data.append({
        "Variable": "R-squared",
        "Coefficient": f"{results['r_squared']:.4f}",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    summary_data.append({
        "Variable": "Adjusted R-squared",
        "Coefficient": f"{results['adj_r_squared']:.4f}",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    summary_data.append({
        "Variable": "F-statistic",
        "Coefficient": f"{results['f_statistic']:.4f}",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": f"{results['f_pvalue']:.4f}",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    summary_data.append({
        "Variable": "Observations",
        "Coefficient": f"{results['n_obs']}",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    # Coefficients
    summary_data.append({
        "Variable": "",
        "Coefficient": "",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    summary_data.append({
        "Variable": "Coefficients",
        "Coefficient": "",
        "Std_Error": "",
        "T_Statistic": "",
        "P_Value": "",
        "CI_Lower": "",
        "CI_Upper": ""
    })
    
    for var_name, coef_data in results["coefficients"].items():
        summary_data.append({
            "Variable": var_name,
            "Coefficient": f"{coef_data['coefficient']:.6f}",
            "Std_Error": f"{coef_data['std_error']:.6f}",
            "T_Statistic": f"{coef_data['t_statistic']:.4f}",
            "P_Value": f"{coef_data['p_value']:.4f}",
            "CI_Lower": f"{coef_data['confidence_interval'][0]:.6f}",
            "CI_Upper": f"{coef_data['confidence_interval'][1]:.6f}"
        })
        
        # Add annualized values for constant (alpha)
        if var_name == "constant":
            summary_data.append({
                "Variable": f"{var_name}_annualized",
                "Coefficient": f"{coef_data['ann_coefficient']:.6f}",
                "Std_Error": f"{coef_data['ann_std_error']:.6f}",
                "T_Statistic": f"{coef_data['t_statistic']:.4f}",
                "P_Value": f"{coef_data['p_value']:.4f}",
                "CI_Lower": f"{coef_data['ann_confidence_interval'][0]:.6f}",
                "CI_Upper": f"{coef_data['ann_confidence_interval'][1]:.6f}"
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir / f"baseline_vs_{benchmark_name.lower()}_regression_summary.csv", index=False)
    
    # Save detailed model output
    with open(out_dir / f"baseline_vs_{benchmark_name.lower()}_regression_detailed.txt", "w") as f:
        f.write(f"Baseline vs {benchmark_name} Regression Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dependent Variable: True_Baseline (lag3, mv, 70%, dynamic_1m)\n")
        f.write(f"Independent Variable: {benchmark_name}\n")
        f.write(f"Time Period: {YEAR_START}-{YEAR_END}\n")
        f.write(f"Observations: {results['n_obs']}\n\n")
        
        f.write("Baseline Strategy Statistics:\n")
        f.write(f"  Annual Return: {results['baseline_stats']['ann_return']*100:.2f}%\n")
        f.write(f"  Annual Volatility: {results['baseline_stats']['ann_vol']*100:.2f}%\n")
        f.write(f"  Sharpe Ratio: {results['baseline_stats']['sharpe']:.4f}\n\n")
        
        f.write(f"{benchmark_name} Statistics:\n")
        f.write(f"  Annual Return: {results['benchmark_stats']['ann_return']*100:.2f}%\n")
        f.write(f"  Annual Volatility: {results['benchmark_stats']['ann_vol']*100:.2f}%\n")
        f.write(f"  Sharpe Ratio: {results['benchmark_stats']['sharpe']:.4f}\n\n")
        
        f.write("Regression Results:\n")
        f.write(str(results['model'].summary()))
        f.write("\n\n")
        
        # Add annualized alpha information
        const_data = results['coefficients']['constant']
        f.write("Annualized Alpha Information:\n")
        f.write(f"  Daily Alpha: {const_data['coefficient']:.6f}\n")
        f.write(f"  Annualized Alpha: {const_data['ann_coefficient']:.6f} ({const_data['ann_coefficient']*100:.2f}%)\n")
        f.write(f"  Annualized Std Error: {const_data['ann_std_error']:.6f}\n")
        f.write(f"  Annualized 95% CI: [{const_data['ann_confidence_interval'][0]:.6f}, {const_data['ann_confidence_interval'][1]:.6f}]\n")
        f.write(f"  T-statistic: {const_data['t_statistic']:.4f}\n")
        f.write(f"  P-value: {const_data['p_value']:.4f}\n")


def create_combined_summary(all_results: Dict, out_dir: Path):
    """Create a combined summary of all regression results."""
    
    # Create combined summary table
    summary_data = []
    
    for benchmark_name, results in all_results.items():
        coef_data = results['coefficients']['benchmark']
        const_data = results['coefficients']['constant']
        
        summary_data.append({
            "Benchmark": benchmark_name,
            "R_Squared": f"{results['r_squared']:.4f}",
            "Adj_R_Squared": f"{results['adj_r_squared']:.4f}",
            "F_Statistic": f"{results['f_statistic']:.4f}",
            "F_P_Value": f"{results['f_pvalue']:.4f}",
            "Observations": results['n_obs'],
            "Constant_Alpha_Daily": f"{const_data['coefficient']:.6f}",
            "Constant_Alpha_Annual": f"{const_data['ann_coefficient']:.6f}",
            "Constant_Alpha_Annual_Pct": f"{const_data['ann_coefficient']*100:.2f}%",
            "Constant_Std_Error": f"{const_data['std_error']:.6f}",
            "Constant_Ann_Std_Error": f"{const_data['ann_std_error']:.6f}",
            "Constant_T_Stat": f"{const_data['t_statistic']:.4f}",
            "Constant_P_Value": f"{const_data['p_value']:.4f}",
            "Beta_Coefficient": f"{coef_data['coefficient']:.6f}",
            "Beta_Std_Error": f"{coef_data['std_error']:.6f}",
            "Beta_T_Statistic": f"{coef_data['t_statistic']:.4f}",
            "Beta_P_Value": f"{coef_data['p_value']:.4f}",
            "Beta_CI_Lower": f"{coef_data['confidence_interval'][0]:.6f}",
            "Beta_CI_Upper": f"{coef_data['confidence_interval'][1]:.6f}",
            "Baseline_Ann_Ret": f"{results['baseline_stats']['ann_return']*100:.2f}%",
            "Baseline_Ann_Vol": f"{results['baseline_stats']['ann_vol']*100:.2f}%",
            "Baseline_Sharpe": f"{results['baseline_stats']['sharpe']:.4f}",
            "Benchmark_Ann_Ret": f"{results['benchmark_stats']['ann_return']*100:.2f}%",
            "Benchmark_Ann_Vol": f"{results['benchmark_stats']['ann_vol']*100:.2f}%",
            "Benchmark_Sharpe": f"{results['benchmark_stats']['sharpe']:.4f}"
        })
    
    # Save combined summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir / "baseline_vs_all_benchmarks_summary.csv", index=False)
    
    # Create detailed combined report
    with open(out_dir / "baseline_vs_all_benchmarks_detailed.txt", "w") as f:
        f.write("Baseline vs All Benchmarks Regression Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time Period: {YEAR_START}-{YEAR_END}\n")
        f.write(f"Number of Benchmarks: {len(all_results)}\n\n")
        
        f.write("Individual Regression Results:\n")
        f.write("-" * 40 + "\n\n")
        
        for benchmark_name, results in all_results.items():
            const_data = results['coefficients']['constant']
            coef_data = results['coefficients']['benchmark']
            
            f.write(f"{benchmark_name}:\n")
            f.write(f"  Model: True_Baseline = α + β×{benchmark_name} + ε\n")
            f.write(f"  R-squared: {results['r_squared']:.4f}\n")
            f.write(f"  Constant (α) - Daily: {const_data['coefficient']:.6f}\n")
            f.write(f"  Constant (α) - Annualized: {const_data['ann_coefficient']:.6f} ({const_data['ann_coefficient']*100:.2f}%)\n")
            f.write(f"    T-statistic: {const_data['t_statistic']:.4f}\n")
            f.write(f"    P-value: {const_data['p_value']:.4f}\n")
            f.write(f"  Coefficient (β): {coef_data['coefficient']:.6f}\n")
            f.write(f"    T-statistic: {coef_data['t_statistic']:.4f}\n")
            f.write(f"    P-value: {coef_data['p_value']:.4f}\n")
            f.write(f"  Observations: {results['n_obs']}\n")
            f.write(f"  Baseline Sharpe: {results['baseline_stats']['sharpe']:.4f}\n")
            f.write(f"  {benchmark_name} Sharpe: {results['benchmark_stats']['sharpe']:.4f}\n\n")
        
        # Find best and worst performing regressions
        r_squared_values = {name: results['r_squared'] for name, results in all_results.items()}
        best_r2 = max(r_squared_values, key=r_squared_values.get)
        worst_r2 = min(r_squared_values, key=r_squared_values.get)
        
        f.write("Summary Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Highest R-squared: {best_r2} ({r_squared_values[best_r2]:.4f})\n")
        f.write(f"Lowest R-squared: {worst_r2} ({r_squared_values[worst_r2]:.4f})\n")
        f.write(f"Average R-squared: {np.mean(list(r_squared_values.values())):.4f}\n")
        
        # Count significant coefficients
        significant_count = sum(1 for results in all_results.values() 
                              if results['coefficients']['benchmark']['p_value'] < 0.05)
        f.write(f"Significant coefficients (p<0.05): {significant_count}/{len(all_results)}\n")
    
    _debug(f"Combined summary saved to {out_dir}")




def create_regression_plots(all_results: Dict, out_dir: Path):
    """Create plots for all regression analyses."""
    
    if not all_results:
        return
    
    # Create subdirectory for plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Combined R-squared comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    benchmark_names = list(all_results.keys())
    r_squared_values = [all_results[name]['r_squared'] for name in benchmark_names]
    coefficients = [all_results[name]['coefficients']['benchmark']['coefficient'] for name in benchmark_names]
    p_values = [all_results[name]['coefficients']['benchmark']['p_value'] for name in benchmark_names]
    
    # R-squared comparison
    bars1 = ax1.bar(benchmark_names, r_squared_values, color='skyblue', alpha=0.7)
    ax1.set_ylabel('R-squared')
    ax1.set_title('R-squared Comparison Across Benchmarks')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add R-squared values on bars
    for bar, r2 in zip(bars1, r_squared_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.4f}', ha='center', va='bottom')
    
    # Coefficients comparison
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars2 = ax2.bar(benchmark_names, coefficients, color=colors, alpha=0.7)
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Regression Coefficients (Red = Significant, p<0.05)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add coefficient values on bars
    for bar, coef in zip(bars2, coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # P-values comparison
    bars3 = ax3.bar(benchmark_names, p_values, color='orange', alpha=0.7)
    ax3.set_ylabel('P-value')
    ax3.set_title('P-values Comparison (Horizontal line at 0.05)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='p=0.05')
    ax3.legend()
    
    # Add p-values on bars
    for bar, p_val in zip(bars3, p_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{p_val:.4f}', ha='center', va='bottom')
    
    # Sharpe ratios comparison
    baseline_sharpes = [all_results[name]['baseline_stats']['sharpe'] for name in benchmark_names]
    benchmark_sharpes = [all_results[name]['benchmark_stats']['sharpe'] for name in benchmark_names]
    
    x = np.arange(len(benchmark_names))
    width = 0.35
    
    ax4.bar(x - width/2, baseline_sharpes, width, label='Baseline', alpha=0.7)
    ax4.bar(x + width/2, benchmark_sharpes, width, label='Benchmark', alpha=0.7)
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Sharpe Ratios Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(benchmark_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "baseline_vs_all_benchmarks_comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual regression plots for each benchmark
    for benchmark_name, results in all_results.items():
        create_individual_regression_plot(results, benchmark_name, plots_dir)
    
    _debug(f"All regression plots saved to {plots_dir}")


def create_individual_regression_plot(results: Dict, benchmark_name: str, out_dir: Path):
    """Create individual regression plot for a single benchmark."""
    
    model = results["model"]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs Predicted
    y_actual = model.model.endog
    y_pred = model.fittedvalues
    
    ax1.scatter(y_actual, y_pred, alpha=0.5, s=1)
    ax1.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Baseline Returns')
    ax1.set_ylabel('Predicted Baseline Returns')
    ax1.set_title(f'Actual vs Predicted ({benchmark_name})')
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = model.resid
    ax2.scatter(y_pred, residuals, alpha=0.5, s=1)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Returns')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Residuals vs Predicted ({benchmark_name})')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot of Residuals ({benchmark_name})')
    ax3.grid(True, alpha=0.3)
    
    # Time series of actual vs predicted
    dates = pd.date_range(start='2014-01-01', end='2024-12-31', freq='D')
    # Create a simple time series plot (assuming daily data)
    ax4.plot(range(len(y_actual)), y_actual, label='Actual', alpha=0.7, linewidth=1)
    ax4.plot(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Returns')
    ax4.set_title(f'Time Series: Actual vs Predicted ({benchmark_name})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"baseline_vs_{benchmark_name.lower()}_regression_plots.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    global YEAR_START, YEAR_END
    
    parser = argparse.ArgumentParser(description="Baseline vs Benchmark Regression Analysis")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory path")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output directory path")
    parser.add_argument("--year_start", type=int, default=YEAR_START, help="Start year")
    parser.add_argument("--year_end", type=int, default=YEAR_END, help="End year")
    
    args = parser.parse_args()
    
    # Update global variables
    YEAR_START = args.year_start
    YEAR_END = args.year_end
    
    out_dir = Path(args.out_dir).resolve()
    
    _debug("=" * 60)
    _debug("Baseline vs Benchmark Analysis")
    _debug("=" * 60)
    _debug(f"Time Period: {YEAR_START}-{YEAR_END}")
    _debug(f"Output Directory: {out_dir}")
    _debug("")
    
    # Load all strategies
    _debug("Loading strategy returns...")
    strategies = load_all_strategies()
    
    if not strategies:
        _debug("Error: No strategies loaded successfully")
        return
    
    _debug(f"Successfully loaded {len(strategies)} strategies")
    _debug("")
    
    # Perform benchmark comparison analysis
    _debug("=== BENCHMARK COMPARISON ANALYSIS ===")
    _debug("Performing benchmark performance comparison...")
    _debug("Metrics: Annual Return, Volatility, Sharpe, Calmar, Max Drawdown, Spread")
    _debug("")
    
    comparison_results = perform_benchmark_comparison(strategies, out_dir)
    
    if comparison_results:
        _debug("Benchmark Comparison Results:")
        for benchmark_name, results in comparison_results.items():
            baseline_metrics = results["baseline_metrics"]
            benchmark_metrics = results["benchmark_metrics"]
            spread_metrics = results["spread_metrics"]
            
            _debug(f"  {benchmark_name} vs True_Baseline:")
            _debug(f"    Common Observations: {results['common_observations']}")
            _debug(f"    Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
            _debug(f"    {benchmark_name} Sharpe: {benchmark_metrics['sharpe']:.4f}")
            if spread_metrics:
                _debug(f"    Spread Sharpe: {spread_metrics['sharpe']:.4f}")
    else:
        _debug("Error: Benchmark comparison analysis failed")
    
    # Perform full sample regression analysis
    _debug("")
    _debug("=== REGRESSION ANALYSIS ===")
    _debug("Performing regression analysis...")
    _debug("Model: True_Baseline = α + β×Benchmark_Strategy + ε (separate regressions)")
    _debug("Focus: Alpha and Beta coefficients and significance")
    _debug("")
    
    results = perform_regression_analysis(strategies, out_dir)
    
    if results:
        _debug("Regression Analysis Results:")
        for benchmark_name, benchmark_results in results.items():
            const_data = benchmark_results['coefficients']['constant']
            coef_data = benchmark_results['coefficients']['benchmark']
            
            _debug(f"  {benchmark_name}:")
            _debug(f"    R-squared: {benchmark_results['r_squared']:.4f}")
            _debug(f"    Alpha (Daily): {const_data['coefficient']:.6f} (p={const_data['p_value']:.4f})")
            _debug(f"    Alpha (Annual): {const_data['ann_coefficient']:.6f} ({const_data['ann_coefficient']*100:.2f}%) (p={const_data['p_value']:.4f})")
            _debug(f"    Beta: {coef_data['coefficient']:.6f} (p={coef_data['p_value']:.4f})")
        
        # Create plots
        _debug("")
        _debug("Creating regression plots...")
        create_regression_plots(results, out_dir)
        
    else:
        _debug("Error: Regression analysis failed")
    
    _debug("")
    _debug("Analysis completed successfully!")
    _debug(f"Results saved to: {out_dir}")
    
    # Save comprehensive debug log
    debug_path = out_dir / "baseline_benchmark_analysis.log"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("Baseline vs Benchmark Analysis - Comprehensive Log\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Time Period: {YEAR_START}-{YEAR_END}\n")
        f.write(f"Output Directory: {out_dir}\n\n")
        
        for line in DEBUG_LINES:
            f.write(str(line) + "\n")


if __name__ == "__main__":
    main()
