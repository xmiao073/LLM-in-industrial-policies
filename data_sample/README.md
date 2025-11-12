# 🧾 data_sample

本目录包含最小可运行样例数据，用于演示和测试项目流程。
这些数据不代表真实市场数据，仅用于功能验证。

## 文件说明

| 文件名 | 说明 |
|--------|------|
| **ohlcv.csv** | 股票每日行情数据（开盘价、收盘价）。列：`Stkcd`, `Trddt`, `Opnprc`, `Clsprc`。|
| **exposure.csv** | 股票所属行业及权重关系。列：`date`, `Stkcd`, `Indcd`, `weight`。|
| **policy.csv** | 行业对应的政策强度时间序列。列：`date_p`, `Indcd`, `policy_strength`。|

## 使用方式

运行回归示例命令（默认读取本目录）：
```bash
python fincode/regression_script.py --frequency static --price close --lags 1 --periods 1
