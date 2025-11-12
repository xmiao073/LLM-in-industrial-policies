#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness Analysis and Transaction Cost Testing for Portfolio Strategies

This script performs comprehensive robustness analysis including:
1. Different percentage cutoffs analysis
2. Different weighting schemes comparison
3. Different return type comparison
4. Transaction cost sensitivity analysis

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import pickle
import argparse
import sys
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use("classic")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

def setup_ax(ax: plt.Axes) -> None:
    """Finance-journal style axes: white background, no grid, thin spines.
    """
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)

class RobustnessAnalyzer:
    """Main class for robustness analysis"""

    def __init__(self, base_dir: str = "/hpc2hdd/home/jliu043/policy/portfolio"):
        """Initialize analyzer with base directory"""
        self.BASE_DIR = Path(base_dir)

        # Define baseline strategies (keep Daily only)
        self.BASELINES = [
            {
                "name": "Baseline (Daily)",
                "holding": "daily",
                "period": 1,
                "lag": 3,
                "weighting": "mv",
                "return_type": "close_close",
                "frequency": "dynamic_1m",
                "pct": 0.7
            }
        ]

        # Transaction costs to test (A-share market levels)
        self.TRANSACTION_COSTS = [0.0, 0.0001, 0.0002]  # 0%, 0.01%, 0.02%

        # Output directory
        self.output_dir = self.BASE_DIR.parent / "robustness_analysis_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_strategy_name(self, strategy_name: str) -> Dict:
        """Parse strategy name to extract parameters"""
        parts = strategy_name.split("_")

        # Extract basic information
        holding_method = parts[0]  # rolling or daily
        period = int(parts[1]) if holding_method == "rolling" else 1

        # Extract la
        lag = None
        for part in parts:
            if part.startswith("lag"):
                lag = int(part[3:])  # Remove "lag" prefix
                break

        # Extract weighting
        weighting = None
        for part in parts:
            if part in ["equal", "mv"]:
                weighting = part
                break

        # Extract percent
        pct = None
        for part in parts:
            if part.startswith("p"):
                pct = int(part[1:]) / 100  # Remove "p" prefix and convert to decimal
                break

        # Extract frequency
        frequency = None
        for i, part in enumerate(parts):
            if part == "dynamic" or part == "static":
                if i + 1 < len(parts):
                    frequency = f"{part}_{parts[i+1]}"
                break

        # Extract return_type (last two parts)
        return_type = "_".join(parts[-2:]) if len(parts) >= 2 else None

        return {
            "holding_method": holding_method,
            "period": period,
            "lag": lag,
            "weighting": weighting,
            "pct": pct,
            "frequency": frequency,
            "return_type": return_type,
            "strategy": strategy_name
        }

    def build_strategy_path(self, config: Dict) -> Path:
        """Build strategy path"""
        if config["holding"] == "rolling":
            path = (self.BASE_DIR / "rolling" / f"p{config['period']}" /
                   config["return_type"] / "pos" / config["weighting"] / config["frequency"])
        else:
            path = (self.BASE_DIR / "daily" / config["return_type"] /
                   "pos" / config["weighting"] / config["frequency"])
        return path

    def load_performance_data(self, config: Dict) -> Optional[pd.DataFrame]:
        """Load performance data from CSV"""
        path = self.build_strategy_path(config)
        csv_file = path / "overall_summary.csv"

        if not csv_file.exists():
            print(f"Warning: CSV file not found: {csv_file}")
            return None

        try:
            df = pd.read_csv(csv_file)

            # Parse each strategy name
            parsed_data = []
            for _, row in df.iterrows():
                parsed = self.parse_strategy_name(row["strategy"])
                parsed["ann_ret"] = row["ann_ret"]
                parsed["ann_vol"] = row["ann_vol"]
                parsed["sharpe"] = row["sharpe"]
                parsed["avg_n"] = row["avg_n"]
                parsed_data.append(parsed)

            result_df = pd.DataFrame(parsed_data)

            # Filter data matching current config
            mask = (
                (result_df["holding_method"] == config["holding"]) &
                (result_df["period"] == config["period"]) &
                (result_df["weighting"] == config["weighting"]) &
                (result_df["frequency"] == config["frequency"]) &
                (result_df["return_type"] == config["return_type"])
            )

            return result_df[mask].copy()

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def load_strategy_returns(self, config: Dict, target_pct: float) -> Optional[pd.DataFrame]:
        """Load strategy returns from PKL file"""
        path = self.build_strategy_path(config)
        pkl_file = path / "daily_ret_allpct.pkl.gz"

        if not pkl_file.exists():
            print(f"Warning: PKL file not found: {pkl_file}")
            return None

        try:
            with gzip.open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # Build strategy base key
            base_key = (f"{config['holding']}_{config['period']}_lag{config['lag']}_"
                       f"{config['weighting']}_{config['frequency']}_{config['return_type']}")

            if base_key not in data:
                print(f"Warning: Strategy {base_key} not found in data")
                return None

            pct_data = data[base_key]

            # Find closest percentage
            available_pcts = list(pct_data.keys())
            if not available_pcts:
                return None

            closest_pct = min(available_pcts, key=lambda x: abs(x - target_pct))

            if abs(closest_pct - target_pct) > 0.01:  # 1% tolerance
                print(f"Using closest pct {closest_pct:.2f} instead of {target_pct:.2f}")

            df = pct_data[closest_pct].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            return df

        except Exception as e:
            print(f"Error loading PKL file: {e}")
            return None

    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.dropna().empty:
            return {
                "ann_ret": 0.0,
                "ann_vol": 0.0,
                "sharpe": 0.0,
                "max_dd": 0.0
            }

        returns = returns.dropna()

        # Annualized return
        ann_ret = (1 + returns.mean()) ** 252 - 1

        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        # Maximum drawdown
        wealth = (1 + returns).cumprod()
        rollmax = wealth.cummax()
        dd = wealth / rollmax - 1.0
        max_dd = dd.min()

        return {
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd
        }

    def run_robustness_analysis(self):
        """Run transaction cost sensitivity analysis"""
        print("="*80)
        print("TRANSACTION COST SENSITIVITY ANALYSIS")
        print("="*80)

        all_results = {}

        for baseline in self.BASELINES:
            print(f"\n{'='*60}")
            print(f"Analyzing: {baseline['name']}")
            print(f"{'='*60}")

            baseline_results = {}

            # Transaction cost analysis
            print("\nTransaction Cost Sensitivity Analysis")
            print("-" * 40)
            cost_results = self.analyze_transaction_costs(baseline)
            if cost_results:
                baseline_results["transaction_costs"] = cost_results

            if baseline_results:
                all_results[baseline["name"]] = baseline_results

        # Generate separate plots for each baseline
        print(f"\n{'='*60}")
        print("GENERATING SEPARATE CUMULATIVE RETURNS PLOTS")
        print(f"{'='*60}")

        plot_paths = self.plot_separate_cumulative_returns(all_results)

        # Generate summary report
        print(f"\n{'='*60}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*60}")

        self.generate_transaction_cost_summary(all_results)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED!")
        print(f"All results saved to: {self.output_dir}")
        print(f"{'='*80}")





    def analyze_transaction_costs(self, config: Dict) -> Optional[Dict]:
        """Analyze transaction costs"""
        # Load original data
        original_df = self.load_strategy_returns(config, config["pct"])
        if original_df is None:
            return None

        # Filter out any data beyond 2024 to ensure no future data
        original_df = original_df[original_df["date"].dt.year <= 2024].copy()

        if original_df.empty:
            print(f"Warning: No data available after filtering for dates <= 2024")
            return None

        print(f"  Using data from {original_df['date'].min().strftime('%Y-%m-%d')} to {original_df['date'].max().strftime('%Y-%m-%d')}")

        results = {}

        for cost_rate in self.TRANSACTION_COSTS:
            if cost_rate == 0.0:
                # No transaction costs
                df = original_df.copy()
                df["ret_after_cost"] = df["ret"]
            else:
                # Calculate transaction costs
                df = self.calculate_transaction_costs(original_df, cost_rate)

            # Calculate metrics
            metrics = self.calculate_performance_metrics(df["ret_after_cost"])
            metrics["cost_rate"] = cost_rate

            results[cost_rate] = {
                "metrics": metrics,
                "cumulative_returns": df["cum_ret_after_cost"].values if "cum_ret_after_cost" in df.columns else (1 + df["ret_after_cost"]).cumprod().values,
                "dates": df["date"].values
            }

        return results

    def calculate_transaction_costs(self, returns_df: pd.DataFrame, cost_rate: float) -> pd.DataFrame:
        """Calculate returns after transaction costs"""
        df = returns_df.copy()

        # Calculate turnover (change in number of stocks)
        df["n_prev"] = df["n"].shift(1).fillna(0)
        df["turnover"] = np.abs(df["n"] - df["n_prev"]) / df["n"].replace(0, np.inf)
        df["turnover"] = df["turnover"].fillna(0)

        # Calculate transaction costs
        df["cost"] = df["turnover"] * cost_rate

        # Adjusted returns
        df["ret_after_cost"] = df["ret"] - df["cost"]

        # Cumulative returns
        df["cum_ret_after_cost"] = (1 + df["ret_after_cost"]).cumprod()

        return df



    def plot_cumulative_returns(self, config: Dict, results: Dict, ax: plt.Axes, baseline_num: int):
        """Plot cumulative returns for a single baseline"""
        if not results:
            return

        # Journal-friendly color palette (three costs only)
        colors = {0.0: "black", 0.0001: "blue", 0.0002: "red"}
        linestyles = {0.0: "-", 0.0001: "--", 0.0002: ":"}
        # Minimal legend labels
        labels = {0.0: "0%", 0.0001: "0.01%", 0.0002: "0.02%"}

        # Plot cumulative returns (using actual dates)
        for cost_rate, result in results.items():
            if "dates" in result and result["dates"] is not None:
                dates = pd.to_datetime(result["dates"])

                # Debug: print date range to ensure no future dates
                if cost_rate == 0.0:  # Only print once
                    min_date = dates.min()
                    max_date = dates.max()
                    print(f"  Plotting date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

                ax.plot(dates, result["cumulative_returns"], color=colors[cost_rate],
                        linewidth=1.4, ls=linestyles[cost_rate], label=labels[cost_rate])
            else:
                # Fallback to synthetic dates if no actual dates
                dates = pd.date_range(start='2020-01-01', periods=len(result["cumulative_returns"]), freq='D')
                ax.plot(dates, result["cumulative_returns"], color=colors[cost_rate],
                        linewidth=1.4, ls=linestyles[cost_rate], label=labels[cost_rate])

        # Use y-axis label to distinguish baselines
        ax.set_ylabel("Cumulative return", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        # Legend below x-axis in a boxed frame
        setup_ax(ax)
        handles, lab = ax.get_legend_handles_labels()
        leg = ax.legend(handles, lab, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                        ncol=3, frameon=True, framealpha=1.0, edgecolor="black",
                        fontsize=8, handlelength=2.0, labelspacing=0.4, columnspacing=1.0)
        plt.gcf().subplots_adjust(bottom=0.18)

    def plot_individual_cumulative_returns(self, config: Dict, results: Dict, baseline_num: int):
        """Plot cumulative returns for a single baseline as individual plot"""
        if not results:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
        fig.patch.set_facecolor("white")
        setup_ax(ax)

        # Plot cumulative returns
        self.plot_cumulative_returns(config, results, ax, baseline_num)

        plt.tight_layout()

        # Save individual plot with clean filename
        safe_name = f"Baseline_{baseline_num}_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')}"
        plot_path = self.output_dir / f"cumulative_returns_{safe_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved individual plot: {plot_path}")

        return plot_path

    def plot_separate_cumulative_returns(self, all_results: Dict):
        """Plot separate cumulative returns for both baselines"""
        if not all_results:
            return

        # Get baseline configurations
        baseline_configs = {}
        for baseline in self.BASELINES:
            baseline_configs[baseline["name"]] = baseline

        plot_paths = []

        # Plot each baseline separately
        baseline_names = list(all_results.keys())
        for i, baseline_name in enumerate(baseline_names, 1):
            if "transaction_costs" in all_results[baseline_name]:
                config = baseline_configs.get(baseline_name, {"name": baseline_name})
                plot_path = self.plot_individual_cumulative_returns(
                    config,
                    all_results[baseline_name]["transaction_costs"],
                    i
                )
                if plot_path:
                    plot_paths.append(plot_path)

        return plot_paths

    def generate_transaction_cost_table(self, config: Dict, results: Dict):
        """Generate transaction cost analysis table"""
        if not results:
            return

        # Create table data
        table_data = []
        for cost_rate, result in results.items():
            metrics = result["metrics"]
            # Format cost rate display
            if cost_rate == 0.0:
                cost_display = "0%"
            elif cost_rate == 0.0001:
                cost_display = "0.01%"
            elif cost_rate == 0.0002:
                cost_display = "0.02%"
            elif cost_rate == 0.0005:
                cost_display = "0.05%"

            row = {
                "Transaction Cost": cost_display,
                "Annual Return": metrics["ann_ret"],
                "Annual Volatility": metrics["ann_vol"],
                "Sharpe Ratio": metrics["sharpe"],
                "Maximum Drawdown": metrics["max_dd"],
                "Turnover Impact": self.calculate_turnover_impact(result) if cost_rate > 0 else 0.0
            }
            table_data.append(row)

        df_table = pd.DataFrame(table_data)

        # Calculate sensitivity metrics
        if len(df_table) >= 2:
            base_row = df_table.iloc[0]  # 0% cost baseline

            print(f"\n{config['name']} Transaction Cost Sensitivity Analysis:")
            print("=" * 60)

            for i in range(1, len(df_table)):
                cost_row = df_table.iloc[i]
                cost_rate = self.TRANSACTION_COSTS[i]

                print(f"\nTransaction Cost: {cost_row['Transaction Cost']}")
                print(f"  Annual Return: {cost_row['Annual Return']:.2%} (vs {base_row['Annual Return']:.2%} baseline)")
                print(f"  Return Impact: {(cost_row['Annual Return'] - base_row['Annual Return']):.2%}")
                print(f"  Sharpe Ratio: {cost_row['Sharpe Ratio']:.3f} (vs {base_row['Sharpe Ratio']:.3f} baseline)")
                print(f"  Sharpe Impact: {cost_row['Sharpe Ratio'] - base_row['Sharpe Ratio']:.3f}")
                print(f"  Max Drawdown: {cost_row['Maximum Drawdown']:.2%} (vs {base_row['Maximum Drawdown']:.2%})")
                print(f"  Turnover Impact: {cost_row['Turnover Impact']:.4f}")

                # Calculate cost sensitivity
                if base_row['Annual Return'] != 0:
                    sensitivity = abs(cost_row['Annual Return'] - base_row['Annual Return']) / cost_rate
                    print(f"  Return Sensitivity: {sensitivity:.2f} (return impact per 1% cost increase)")

        # Save table to CSV
        table_path = self.output_dir / f"transaction_cost_table_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        df_table.to_csv(table_path, index=False)
        print(f"\n  Saved transaction cost table: {table_path}")

    def calculate_turnover_impact(self, result: Dict) -> float:
        """Calculate average turnover impact from the result data"""
        # This is a placeholder - in real implementation, you'd extract turnover from the data
        # For now, return a dummy value based on the metrics
        return abs(result["metrics"]["ann_ret"]) * 0.01  # Estimated 1% of return magnitude

    def generate_transaction_cost_summary(self, all_results: Dict):
        """Generate transaction cost sensitivity summary"""
        if not all_results:
            return

        # Create summary CSV
        summary_data = []

        for strategy_name, results in all_results.items():
            row = {"Strategy": strategy_name}

            # Transaction cost sensitivity
            if "transaction_costs" in results:
                costs = results["transaction_costs"]

                # Calculate sensitivity metrics
                if len(costs) >= 2:
                    base_metrics = costs[0.0]["metrics"]  # 0% cost baseline

                    for cost_rate in [0.0001, 0.0002]:  # Test different cost levels
                        if cost_rate in costs:
                            cost_metrics = costs[cost_rate]["metrics"]

                            # Calculate impacts
                            ret_impact = cost_metrics["ann_ret"] - base_metrics["ann_ret"]
                            sharpe_impact = cost_metrics["sharpe"] - base_metrics["sharpe"]
                            maxdd_impact = cost_metrics["max_dd"] - base_metrics["max_dd"]

                            # Calculate sensitivity (impact per unit cost)
                            ret_sensitivity = abs(ret_impact) / cost_rate if cost_rate > 0 else 0

                            row[f"Cost_{cost_rate:.1%}_Return_Impact"] = ret_impact
                            row[f"Cost_{cost_rate:.1%}_Sharpe_Impact"] = sharpe_impact
                            row[f"Cost_{cost_rate:.1%}_MaxDD_Impact"] = maxdd_impact
                            row[f"Cost_{cost_rate:.1%}_Sensitivity"] = ret_sensitivity

                    # Overall cost sensitivity (using highest tested cost level 0.02%)
                    if 0.0002 in costs:
                        high_cost_metrics = costs[0.0002]["metrics"]
                        overall_sensitivity = abs(high_cost_metrics["ann_ret"] - base_metrics["ann_ret"]) / 0.0002
                        row["Overall_Cost_Sensitivity"] = overall_sensitivity

                        if abs(high_cost_metrics["ann_ret"]) > 0:
                            cost_threshold = abs(high_cost_metrics["ann_ret"]) / overall_sensitivity
                            row["Cost_Tolerance_Threshold"] = cost_threshold

            summary_data.append(row)

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            csv_path = self.output_dir / "transaction_cost_summary.csv"
            df_summary.to_csv(csv_path, index=False)
            print(f"  Saved transaction cost summary: {csv_path}")

            # Print summary to console
            print("\nTRANSACTION COST SENSITIVITY SUMMARY:")
            print("-" * 60)
            for _, row in df_summary.iterrows():
                print(f"\n{row['Strategy']}:")

                # Print cost impacts for each level
                for cost_rate in [0.0001, 0.0002]:
                    cost_key = ".1%"
                    if cost_rate == 0.0002:
                        cost_key = ".2%"

                    ret_impact_key = f"Cost_{cost_rate:.1%}_Return_Impact"
                    sensitivity_key = f"Cost_{cost_rate:.1%}_Sensitivity"

                    if ret_impact_key in row and not pd.isna(row[ret_impact_key]):
                        print(f"  {cost_key} Cost Impact:")
                        print(f"    Return Impact: {row[ret_impact_key]:.2%}")
                        if sensitivity_key in row and not pd.isna(row[sensitivity_key]):
                            print(f"    Sensitivity: {row[sensitivity_key]:.2f}")

                if 'Overall_Cost_Sensitivity' in row and not pd.isna(row['Overall_Cost_Sensitivity']):
                    print(f"  Overall Sensitivity: {row['Overall_Cost_Sensitivity']:.2f}")

                if 'Cost_Tolerance_Threshold' in row and not pd.isna(row['Cost_Tolerance_Threshold']):
                    print(f"  Cost Tolerance Threshold: {row['Cost_Tolerance_Threshold']:.4%}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Robustness Analysis and Transaction Cost Testing")
    parser.add_argument("--base-dir", default="/hpc2hdd/home/jliu043/policy/portfolio",
                       help="Base directory for portfolio data")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: base_dir/analysis_output)")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = RobustnessAnalyzer(base_dir=args.base_dir)

    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer.run_robustness_analysis()


if __name__ == "__main__":
    main()
