#!/bin/bash
#SBATCH --job-name=port_daily_dynamic
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11                     # 12 组任务：2(ret) × 2(weight) × 3(window groups)
#SBATCH --output=/hpc2hdd/home/jliu043/policy/logs/daily_%A_%a.out
#SBATCH --error=/hpc2hdd/home/jliu043/policy/logs/daily_%A_%a.err

source ~/.bashrc
conda activate cnn_env
cd /hpc2hdd/home/jliu043/policy/scripts

# ---------- Parameter grid (daily only; dynamic) ----------
RET_ARR=(open_open close_close)
SIGN_ARR=(pos)                 # 如需做空策略，改为：(pos neg)
WT_ARR=(equal mv)
# 将 7 个窗切分为 3 组以形成 12 个任务：2(ret)×2(weight)×3(groups)
WGROUPS=("1 3" "6 9 12" "24 36")

idx=$SLURM_ARRAY_TASK_ID

# index mapping for 2(ret) × 1(sign) × 2(weight) × 3(groups) = 12 combos
nR=${#RET_ARR[@]}
nS=${#SIGN_ARR[@]}
nW=${#WT_ARR[@]}
nG=${#WGROUPS[@]}

ret=${RET_ARR[$(( idx / (nW * nG) ))]}
sub=$(( idx % (nW * nG) ))
wt=${WT_ARR[$(( sub / nG ))]}
gidx=$(( sub % nG ))
sign=${SIGN_ARR[0]}
WINDOWS_STR=${WGROUPS[$gidx]}

# ---------- Output directory: daily/<ret>/<sign>/<weight> ----------
# 备注：脚本会在该目录下自动创建 dynamic / dynamic_{Xm} 子目录
BASE_DIR="/hpc2hdd/home/jliu043/policy/portfolio/daily"
OUT_DIR="${BASE_DIR}/${ret}/${sign}/${wt}"
mkdir -p "$OUT_DIR"

echo "[INFO] idx=$idx → dynamic / $ret / $sign / $wt / daily / windows=[$WINDOWS_STR]"
echo "[INFO] outputs → $OUT_DIR"

python portfolio_script.py \
        --frequency   dynamic \
        --return_type "$ret" \
        --beta_sign   "$sign" \
        --weighting   "$wt" \
        --holding     daily \
        --period      1 \
        --lags        1 3 5 10 \
        --dynamic-windows ${WINDOWS_STR} \
        --out_dir     "$OUT_DIR" 