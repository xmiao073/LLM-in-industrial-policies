# LLM in Industrial Policies

> Exploring how Large Language Models (LLMs) and industrial policy signals interact to generate cross-industry insights and portfolio effects.

![CI](https://github.com/xmiao073/LLM-in-industrial-policies/actions/workflows/ci.yml/badge.svg)
---

## Project Structure

LLM-in-industrial-policies/
├── fincode/
│ ├── regression_script.py # Policy impact regression analysis
│ ├── portfolio_script.py # Portfolio construction based on regression results
│ ├── portfolio_report.py # Aggregates portfolio results into summary tables
│ └── ...
├── data_sample/ # Minimal reproducible dataset
│ ├── ohlcv.csv # Stock daily data (open, close, etc.)
│ ├── exposure.csv # Stock-industry mapping and weights
│ ├── policy.csv # Industry-level policy intensity sequence
│ └── README.md
├── config.py # Loads .env configuration
├── .env # Default directories and global settings
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions
---

## Environment Setup

1. **Clone the repository**
    git clone https://github.com/xmiao073/LLM-in-industrial-policies.git
    cd LLM-in-industrial-policies

2. **Create and activate a virtual environment**
    python -m venv .venv
    source .venv/bin/activate   # For Windows: .venv\Scripts\activate

3. **Install the required dependencies**
    pip install -r requirements.txt

4. **Set up your .env configuration**
    DATA_DIR=./data_sample
    REGRESSION_DIR=./regression
    PORTFOLIO_DIR=./portfolio
    WINSOR_P=0.01
    MIN_N_OBS=30
    YEAR_START=2014

## Quickstart
1. **Regression — Policy Impact Analysis**
Run the regression to assess industry returns’ sensitivity to policy exposure:

python fincode/regression_script.py \
  --frequency static \
  --price close \
  --lags 1 \
  --periods 1
  
Output：
regression/
└── industry_regressions_close_period1_static.csv

2. **Portfolio Construction — Based on Regression Results**
Build a portfolio using regression outcomes:
python fincode/portfolio_script.py \
  --frequency static \
  --return_type close_close \
  --beta_sign pos \
  --alpha 0.05 \
  --weighting equal \
  --lags 1 \
  --period 1
   
Output：
portfolio/
└── daily/close_close/pos/equal/static/overall_summary.csv

3. **Portfolio Aggregation — Summary Tables**
Aggregate all portfolio results:
python fincode/portfolio_report.py

Output：
portfolio/portfolio_report/
├── all_portfolios.csv
└── all_yearly.csv

## Data Description
  File	                        Description
ohlcv.csv	       Stock daily trading data (open, close, etc.)
exposure.csv	      Stock–industry mapping and weights
policy.csv	      Industry-level policy strength sequences

For detailed schema, see data_sample/README.md.

## Methodology Overview
**Stage 1 — Regression**
Estimate industry returns’ sensitivity to policy exposure.

**Stage 2 — Portfolio Construction**
Build portfolios based on regression coefficients (β significance and sign).

**Stage 3 — Reporting**
Aggregate results and output summary tables for strategy comparison.

## Future Work
Integrate LLM-based embeddings for policy text features

Introduce dynamic regression using rolling windows

Visualize industry-level factor loadings and portfolio performance

## Contributors
Xixi Miao, Hanzhi Xiao, Jiaxin Liu, Wenxuan Lyu, Zeyu Ma — Lead Developer

## One-Click Pipeline (Optional)
Run the entire workflow in one line:
bash run_pipeline.sh
# Example for dynamic mode:
FREQ=dynamic DYN_WINDOWS="1 3 6 9 12 24 36" bash run_pipeline.sh
