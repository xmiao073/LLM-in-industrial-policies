#!/bin/bash
#SBATCH --job-name=compare_baselines
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/hpc2hdd/home/jliu043/policy/logs/compare_%A_%a.out
#SBATCH --error=/hpc2hdd/home/jliu043/policy/logs/compare_%A_%a.err

set -euo pipefail

mkdir -p /hpc2hdd/home/jliu043/policy/logs
source ~/.bashrc
conda activate cnn_env
cd /hpc2hdd/home/jliu043/policy/scripts

# ---------------- Portfolio selections to compare ----------------
# (1) rolling=10, lag=1, equal,  close, dynamic=1m, p=40%
# (2) daily=1,   lag=3, mv,     close, dynamic=1m, p=70%
PHOLD=( rolling  daily )
PPER=(   10       1    )
PLAG=(   1        3    )
PWEI=(   equal    mv   )
PRET=(   close_close close_close )
PFREQ=(  dynamic_1m   dynamic_1m )
PPCT=(   0.4      0.7  )
PSIGN=(  pos      pos  )

# ---------------- Baseline strategies ----------------
STRATS=( one_over_n index momentum reversal )

IDX=${SLURM_ARRAY_TASK_ID}
NS=${#STRATS[@]}
NP=${#PHOLD[@]}

PIDX=$(( IDX / NS ))
SIDX=$(( IDX % NS ))

if [ $PIDX -ge $NP ]; then
  echo "[WARN] PIDX=$PIDX exceeds portfolios ($NP). Exiting."
  exit 0
fi

PH=${PHOLD[$PIDX]}
PP=${PPER[$PIDX]}
PL=${PLAG[$PIDX]}
PW=${PWEI[$PIDX]}
PR=${PRET[$PIDX]}
PF=${PFREQ[$PIDX]}
PC=${PPCT[$PIDX]}
PG=${PSIGN[$PIDX]}

STRAT=${STRATS[$SIDX]}

# 基准权重：除指数外，默认与被对比组合的权重一致（BWEI=$PW）；指数不依赖该参数
BWEI=$PW

echo "[INFO] portfolio #$PIDX: $PH p=$PP lag=$PL $PW $PR $PF pct=$PC; baseline=$STRAT (weight=$BWEI)"

BASE_CMD=( python /hpc2hdd/home/jliu043/policy/scripts/baseline_strategies.py \
  --strategies $STRAT \
  --holding $PH --period $PP \
  --weighting $BWEI \
  --return_type $PR \
  --compare \
  --compare-root /hpc2hdd/home/jliu043/policy/portfolio \
  --compare-holding $PH \
  --compare-period $PP \
  --compare-weighting $PW \
  --compare-return-type $PR \
  --compare-sign $PG \
  --compare-frequency $PF \
  --compare-lag $PL \
  --compare-percent $PC )

# 动量/反转使用与组合对应的持有与建仓参数：
# - 动量：2-12 个月窗口（排除最近 1 个月）
# - 反转：1 个月窗口（可在此处调整 --rev-months）
if [ "$STRAT" == "momentum" ]; then
  BASE_CMD+=( --mom-range 2 12 --percentages $PC )
elif [ "$STRAT" == "reversal" ]; then
  BASE_CMD+=( --rev-months 1 --percentages $PC )
fi

echo "[CMD] ${BASE_CMD[@]}"
"${BASE_CMD[@]}"


