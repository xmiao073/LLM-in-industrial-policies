# config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# 允许项目根目录或任意位置放 .env
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

# 根路径（可被环境变量覆盖）
BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()

# 数据与输出（默认指向项目内的相对路径；生产/HPC 可用 .env 改）
DATA_DIR        = Path(os.getenv("DATA_DIR", BASE_DIR / "data_sample")).as_posix()
REGRESSION_DIR  = Path(os.getenv("REGRESSION_DIR", BASE_DIR / "regression")).as_posix()
PORTFOLIO_DIR   = Path(os.getenv("PORTFOLIO_DIR", BASE_DIR / "portfolio")).as_posix()
LOG_DIR         = Path(os.getenv("LOG_DIR", BASE_DIR / "logs")).as_posix()

# 业务参数（按需覆盖）
YEAR_START  = int(os.getenv("YEAR_START", "2014"))    # 最早纳入的政策年份（regression 用到）
WINSOR_P    = float(os.getenv("WINSOR_P", "0.01"))    # 去极值分位（regression）

# 方便打印检查
def debug_print():
    print(f"[CFG] BASE_DIR={BASE_DIR}")
    print(f"[CFG] DATA_DIR={DATA_DIR}")
    print(f"[CFG] REGRESSION_DIR={REGRESSION_DIR}")
    print(f"[CFG] PORTFOLIO_DIR={PORTFOLIO_DIR}")
    print(f"[CFG] LOG_DIR={LOG_DIR}")
    print(f"[CFG] YEAR_START={YEAR_START} WINSOR_P={WINSOR_P}")
