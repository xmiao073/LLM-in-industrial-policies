#!/usr/bin/env bash
set -euo pipefail

# ===============================
# One-click pipeline runner
# - Reads defaults from .env via config.py
# - Uses data_sample + static mode by default (minimal, robust)
# ===============================

# ---- User-tunable params (defaults safe for first run) ----
FREQ="${FREQ:-static}"                 # static | dynamic
PRICE="${PRICE:-close}"                # open | close
LAGS="${LAGS:-1}"                      # e.g. "1" or "1 3 5 10"
PERIODS="${PERIODS:-1}"                # e.g. "1" or "1 3 5"
RETURN_TYPE="${RETURN_TYPE:-close_close}" # open_open | close_close | ...
BETA_SIGN="${BETA_SIGN:-pos}"          # pos | neg
ALPHA="${ALPHA:-0.05}"                 # significance level
WEIGHTING="${WEIGHTING:-equal}"        # equal | mv
DYN_WINDOWS="${DYN_WINDOWS:-}"         # e.g. "1 3 6 9 12 24 36" (only for dynamic)

# ---- Derived dirs (resolved in Python via config.py/.env) ----
# We still allow overriding via env if you want:
DATA_DIR="${DATA_DIR:-./data_sample}"
REG_DIR="${REG_DIR:-./regression}"
PORT_DIR="${PORT_DIR:-./portfolio}"
REPORT_DIR="${REPORT_DIR:-$PORT_DIR/portfolio_report}"

mkdir -p "$REG_DIR" "$PORT_DIR" "$REPORT_DIR"

echo "[1/3] Run regression..."
python fincode/regression_script.py \
  --frequency "$FREQ" \
  --price "$PRICE" \
  --lags $LAGS \
  --periods $PERIODS \
  --data-dir "$DATA_DIR" \
  --out-dir "$REG_DIR" \
  || { echo "Regression failed"; exit 1; }

echo "[2/3] Build portfolios..."
PORT_ARGS=(
  --frequency "$FREQ"
  --return_type "$RETURN_TYPE"
  --beta_sign "$BETA_SIGN"
  --alpha "$ALPHA"
  --weighting "$WEIGHTING"
  --lags $LAGS
  --period $PERIODS
  --data_dir "$DATA_DIR"
  --reg_dir "$REG_DIR"
  --out_dir "$PORT_DIR"
)

# dynamic mode requires explicit windows
if [[ "$FREQ" == "dynamic" ]]; then
  if [[ -z "$DYN_WINDOWS" ]]; then
    echo "ERROR: dynamic mode requires DYN_WINDOWS (e.g. \"1 3 6 9 12 24 36\")"
    exit 1
  fi
  PORT_ARGS+=( --dynamic-windows $DYN_WINDOWS )
fi

python fincode/portfolio_script.py "${PORT_ARGS[@]}" \
  || { echo "Portfolio step failed"; exit 1; }

echo "[3/3] Aggregate results..."
python fincode/portfolio_report.py \
  --root "$PORT_DIR" \
  --out  "$REPORT_DIR" \
  || { echo "Report aggregation failed"; exit 1; }

echo
echo "✅ Pipeline completed."
echo "   Regression out : $REG_DIR"
echo "   Portfolio out  : $PORT_DIR"
echo "   Report out     : $REPORT_DIR"
echo
echo "Tips:"
echo " - To use dynamic mode, run with: FREQ=dynamic DYN_WINDOWS=\"1 3 6 9 12 24 36\" ./run_pipeline.sh"
echo " - You can override any var above via env, e.g.: LAGS=\"1 3 5\" PERIODS=3 ./run_pipeline.sh"
