#!/usr/bin/env bash
# Overnight pipeline: train chronos+timesfm on replica, calibrate (qrcal +
# intercept+pooled), compute effects on test, generate val diagnostics.
# 12 experiments: 6 model variants × 2 directions (O, D).
#
# Replica is weekly (W-SAT). Val: 2024-09-28 → 2024-12-28 (14 weeks).
# Test: 2025-01-04 → 2025-04-26 (17 weeks).
#
# WARNING: val (Oct-Dec) and test (Jan-Apr) span DIFFERENT seasons. This is
# the same val/test month mismatch we hit with citibike — month dummies in
# the calibration features will be 0 for test months (Feb/Mar/Apr never seen
# in val), so they fall back to the January reference. Per-unit intercept
# (the level shift) is robust to this; pooled QR's seasonal coefficients
# may not be. Treat replica calibration results with caution.

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/replica
mkdir -p "$LOG"
PY=/home/donghang/anaconda3/envs/nyc_cp/bin/python
cd /home/donghang/nyc_congestion_pricing
export CUDA_VISIBLE_DEVICES=4   # L40S, ~45 GB free at start of run
START_TS=$(date +%s)

run() {
    local name=$1; shift
    local logfile="$LOG/${name}.log"
    echo "[$(date '+%H:%M:%S')] START  $name" | tee -a "$LOG/_master.log"
    if "$@" > "$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE   $name" | tee -a "$LOG/_master.log"
    else
        echo "[$(date '+%H:%M:%S')] FAIL   $name (see $logfile)" | tee -a "$LOG/_master.log"
    fi
}

# ============================================================================
# Phase 1: Train chronos + timesfm raw forecasts (8 runs)
# Each: prediction_length = val/test horizon; one-shot zero-shot inference.
# ============================================================================
echo "=== PHASE 1: train raw forecasts (chronos + timesfm × O,D × val,test) ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    for dir in O D; do
        for win in val test; do
            run "train_${base}_${dir}_${win}" \
                $PY -m scripts.train_forecast \
                    --mode replica --model $base --window $win --direction $dir
        done
    done
done

# ============================================================================
# Phase 2: Calibration — 8 runs (2 bases × 2 dirs × 2 cal schemes)
# --insample-val produces both val (re-applied to itself) and test outputs.
# ============================================================================
echo "=== PHASE 2: calibrate (global qrcal + intercept+pooled) ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    for dir in O D; do
        run "cal_${base}_${dir}_qrcal" \
            $PY -m scripts.calibrate_forecast \
                --mode replica --base-model $base --direction $dir --insample-val
        run "cal_${base}_${dir}_intercept" \
            $PY -m scripts.calibrate_forecast \
                --mode replica --base-model $base --direction $dir \
                --per-unit-intercept --insample-val
    done
done

# ============================================================================
# Phase 3: Compute test ATT for all 12 (model, direction) combos
# ============================================================================
echo "=== PHASE 3: compute_effects test ATT ===" | tee -a "$LOG/_master.log"
for model in chronos chronos_qrcal chronos_qrcal_intercept timesfm timesfm_qrcal timesfm_qrcal_intercept; do
    for dir in O D; do
        run "eff_${model}_${dir}" \
            $PY -m scripts.compute_effects \
                --mode replica --model $model --window test --direction $dir
    done
done

# ============================================================================
# Phase 4: Compare models (val + test summaries with all 13 forecasters)
# ============================================================================
echo "=== PHASE 4: compare_models ===" | tee -a "$LOG/_master.log"
# Note: arima / prophet / pcn replica triplets were trained on the OLD val
# window (Oct–Dec 2024) and would produce NaN rows under the new val
# (Jan–Apr 2024). Restrict to the 6 model variants we actually re-trained.
ALL="chronos chronos_qrcal chronos_qrcal_intercept timesfm timesfm_qrcal timesfm_qrcal_intercept"
for win in val test; do
    for dir in O D; do
        run "compare_replica_${win}_${dir}" \
            $PY -m scripts.compare_models \
                --mode replica --window $win --direction $dir --models $ALL
    done
done

# ============================================================================
# Phase 5: Residual diagnostic (val) — bias/coverage tables + figures
# ============================================================================
echo "=== PHASE 5: residual diagnostics ===" | tee -a "$LOG/_master.log"
run "diagnostic_replica" \
    $PY notebooks/_replica_val_residual_diagnostic.py

# ============================================================================
# DONE
# ============================================================================
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "" | tee -a "$LOG/_master.log"
echo "===================================================" | tee -a "$LOG/_master.log"
echo "ALL PHASES DONE in $((ELAPSED / 60)) min  ($(date))" | tee -a "$LOG/_master.log"
echo "===================================================" | tee -a "$LOG/_master.log"
echo "Master log: $LOG/_master.log" | tee -a "$LOG/_master.log"
echo "Per-step logs: $LOG/*.log" | tee -a "$LOG/_master.log"
echo "Outputs: /public_dataset/donghang/nyc_congestion_pricing/output_new/replica/" | tee -a "$LOG/_master.log"
echo "Diagnostic figures: /home/donghang/nyc_congestion_pricing/outputs/figures/replica_val_residual/" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
