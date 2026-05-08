#!/bin/bash
# Run a single (mode, direction, model, window) experiment with logging.
# Usage: run_one.sh <mode> <direction> <model> <window>
# Honors $CUDA_VISIBLE_DEVICES from the caller (set by the session script).

set +e   # tolerate failures so subsequent experiments still run

mode="$1"
direction="$2"
model="$3"
window="$4"

REPO=/home/donghang/nyc_congestion_pricing
LOG_DIR=/home/donghang/nyc_congestion_pricing/logs/overnight
STATUS_FILE=$LOG_DIR/STATUS.txt
mkdir -p "$LOG_DIR"

key="${mode}_${direction}_${model}_${window}"
log="$LOG_DIR/${key}.log"

extra_args=""
if [ "$direction" != "all" ]; then
    extra_args="--direction $direction"
fi
# DeepAR + Optuna trains a full model per trial × 30 trials × 120 epochs
# → ~3-4 hours per experiment. Disable so we finish overnight.
if [ "$model" = "deepar" ]; then
    extra_args="$extra_args --no-optuna"
fi

t0=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$t0] BEGIN $key on CUDA=${CUDA_VISIBLE_DEVICES:-cpu}" | tee -a "$log"

cd "$REPO"
/home/donghang/anaconda3/envs/nyc_cp/bin/python -m scripts.train_forecast \
    --mode "$mode" --model "$model" --window "$window" $extra_args \
    >> "$log" 2>&1
rc=$?

t1=$(date '+%Y-%m-%d %H:%M:%S')
status=$([ $rc -eq 0 ] && echo "OK " || echo "FAIL")
echo "[$t1] END   $key rc=$rc ($status)" | tee -a "$log"

# Append to global status file (atomic-ish: one line at a time)
printf "%-4s  %s -> %s  %-50s  rc=%d  CUDA=%s\n" "$status" "$t0" "$t1" "$key" "$rc" "${CUDA_VISIBLE_DEVICES:-cpu}" >> "$STATUS_FILE"
