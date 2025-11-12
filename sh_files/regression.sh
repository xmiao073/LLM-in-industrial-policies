#!/bin/bash
#SBATCH --job-name=regression_dynamic     # 作业名（仅 dynamic）
#SBATCH --partition=i64m512u              # 分区
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11                      # 12 个子任务：2(价格) × 6(窗口组)
#SBATCH --output=/hpc2hdd/home/jliu043/policy/logs/regression_%A_%a.out
#SBATCH --error=/hpc2hdd/home/jliu043/policy/logs/regression_%A_%a.err

# ---------- Environment ----------
source ~/.bashrc
conda activate cnn_env

cd /hpc2hdd/home/jliu043/policy/scripts

# ---------- Parameters ----------
# 价格：2 种（用 2 个 array 任务区分）
PRICES=( open close )

# 滞后与持有期：在同一任务内一次性跑全组合
LAGS=( 1 3 5 10 )
PERIODS=( 1 3 5 10 )

# dynamic 月窗分组（每个子任务只跑一个组，提高并行度到 ~12 个）
# 6 组： [1], [3], [6], [9], [12], [24 36]
WGROUPS=( "1" "3" "6" "9" "12" "24 36" )

# ---------- Current task ----------
IDX=${SLURM_ARRAY_TASK_ID}
NG=${#WGROUPS[@]}
P_IDX=$(( IDX / NG ))
G_IDX=$(( IDX % NG ))
PRICE=${PRICES[$P_IDX]}
WINDOWS_STR=${WGROUPS[$G_IDX]}

echo "[INFO] idx=$IDX  price=$PRICE  group_idx=$G_IDX  windows=[$WINDOWS_STR]  periods=${PERIODS[*]}  lags=${LAGS[*]}"

# ---------- Run regression ----------
# dynamic（按月窗滚动窗口；不再单独跑“年度 dynamic”）
python regression_script.py \
        --price "$PRICE" \
        --frequency dynamic \
        --lags ${LAGS[@]} \
        --periods ${PERIODS[@]} \
        --dynamic-windows ${WINDOWS_STR}