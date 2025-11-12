# 🧠 LLM in Industrial Policies

> Exploring how Large Language Models (LLMs) and industrial policy signals interact to generate cross-industry insights and portfolio effects.
> ![CI](https://github.com/xmiao073/LLM-in-industrial-policies/actions/workflows/ci.yml/badge.svg)

---

## 📦 Project Structure

LLM-in-industrial-policies/
├── fincode/
│ ├── regression_script.py # Policy impact regression analysis
│ ├── portfolio_script.py # Portfolio construction based on regression results
│ ├── portfolio_report.py # Aggregates portfolio results into summary tables
│ └── ...
├── data_sample/ # Minimal reproducible dataset
│ ├── ohlcv.csv
│ ├── exposure.csv
│ ├── policy.csv
│ └── README.md
├── config.py # Loads .env configuration
├── .env # Default directories and global settings
├── requirements.txt
└── README.md

---

## ⚙️ Environment Setup

```bash
git clone https://github.com/xmiao073/LLM-in-industrial-policies.git
cd LLM-in-industrial-policies

python -m venv .venv
source .venv/bin/activate    # (Windows 用 .venv\Scripts\activate)

pip install -r requirements.txt
.env Example
ini
复制代码
DATA_DIR=./data_sample
REGRESSION_DIR=./regression
PORTFOLIO_DIR=./portfolio
WINSOR_P=0.01
MIN_N_OBS=30
YEAR_START=2014
🚀 Quickstart
1️⃣ Regression (政策回归分析)
运行回归，评估行业收益对政策暴露的敏感度：
python fincode/regression_script.py \
  --frequency static \
  --price close \
  --lags 1 \
  --periods 1
输出示例：
regression/
└── industry_regressions_close_period1_static.csv
2️⃣ Portfolio Construction (组合构建)
基于回归结果构建投资组合：
python fincode/portfolio_script.py \
  --frequency static \
  --return_type close_close \
  --beta_sign pos \
  --alpha 0.05 \
  --weighting equal \
  --lags 1 \
  --period 1
输出示例：
portfolio/
└── daily/close_close/pos/equal/static/overall_summary.csv
3️⃣ Portfolio Aggregation (结果汇总)
聚合所有组合结果：
python fincode/portfolio_report.py
输出：
portfolio/portfolio_report/
├── all_portfolios.csv
└── all_yearly.csv
📊 Example Output Snapshot
Portfolio	Mean Return	Sharpe	Win Rate	Period
rolling_p3_lag1_equal_p70_static_close_close	0.018	0.85	67%	2017–2024
daily_1_lag1_equal_p50_dynamic_close_close	0.012	0.73	61%	2017–2024

(示例数据，由 portfolio_report.py 汇总生成)

🧩 Data Description
详见 data_sample/README.md。

文件名	说明
ohlcv.csv	股票每日行情数据（开盘价、收盘价等）
exposure.csv	股票-行业映射及权重
policy.csv	行业层面的政策强度序列

🧠 Methodology Overview
Regression Stage — 估计行业收益对政策暴露的敏感度

Portfolio Stage — 基于回归信号构建投资组合

Reporting Stage — 聚合结果、输出总表，用于策略比较

🧩 Future Work
将 LLM 输出嵌入到政策文本特征中

引入动态回归（rolling windows）

可视化行业级因子载荷和组合表现

👥 Contributors
Name	Role	Contact
xmiao073	Lead Developer	—
ChatGPT (Assistant)	Project Advisor	—

🏆 Designed for reproducibility, interpretability, and transparent benchmarking of LLM–policy interactions.
