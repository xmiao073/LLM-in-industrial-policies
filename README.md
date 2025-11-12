# LLM-in-industrial-policies
> 用 LLM + 行业 OLS 回归，度量“产业政策强度 → 行业收益”的关系，并据此构建投资组合与做基准对比。支持 **dynamic/static** 两种回归频率、**滞后/持有期** 参数网格，以及 **HPC/Slurm** 并行。

## 🚀 快速开始（最小可用）
```bash
# 1) 安装依赖（临时示例，后续会提供 requirements.txt）
pip install pandas numpy statsmodels matplotlib

# 2) 跑一个动态回归示例（close 价、period=1、lags=1 3 5 10）
python scripts/regression_script.py \
  --frequency dynamic \
  --price close \
  --lags 1 3 5 10 \
  --periods 1

# 3) 基于回归结果构建日频组合（动态频率时需指定月窗）
python scripts/portfolio_script.py \
  --frequency dynamic \
  --return_type close_close \
  --beta_sign pos --alpha 0.05 \
  --weighting mv \
  --lags 1 3 5 10 \
  --period 1 \
  --dynamic-windows 1 3 6 9 12 24 36
