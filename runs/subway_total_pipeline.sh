#!/usr/bin/env bash
# Full pipeline for subway/total = O+D ridership per (station, day).
# Mirrors the structure of the standard subway O,D pipeline but with the
# new aggregated direction. After this finishes, the headline 6 models
# (chronos × {raw,qrcal,intercept} + timesfm × {raw,qrcal,intercept}) plus
# the {qrcal_oos, qrcal_intercept_insample} sibling variants exist for
# subway/total.
#
# Stage layout:
#   A. Train raw forecasters (chronos+timesfm × val+test)
#   B. Calibrate (qrcal IS + qrcal OOS + intercept IS + intercept OOS) × 2 base
#   C. Compute effects (test) for the 6+4 headline variants
#   D. Geospatial analysis (test) for the headline 6 — produces significance
#      maps, CRZ summaries, trends_by_crz, regression
#   E. Re-mirror to repo

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/subway_total
mkdir -p "$LOG"
PY=/home/donghang/anaconda3/envs/nyc_cp/bin/python
cd /home/donghang/nyc_congestion_pricing
export CUDA_VISIBLE_DEVICES=4

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

# ----- A: train raw forecasts -----
echo "=== STAGE A: train raw chronos + timesfm ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    for win in val test; do
        run "train_${base}_${win}" \
            $PY -m scripts.train_forecast --mode subway --model $base --window $win --direction total
    done
done

# ----- B: calibration variants -----
echo "=== STAGE B: calibration ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    # qrcal IS (default --insample-val: writes IS val + test)
    run "cal_${base}_qrcal" \
        $PY -m scripts.calibrate_forecast --mode subway --base-model $base --direction total --insample-val
    # qrcal OOS (--val-kfold 5 with explicit suffix)
    run "cal_${base}_qrcal_oos" \
        $PY -m scripts.calibrate_forecast --mode subway --base-model $base --direction total \
            --val-kfold 5 --suffix qrcal_oos
    # intercept OOS (default suffix qrcal_intercept, --val-kfold 5)
    run "cal_${base}_qrcal_intercept" \
        $PY -m scripts.calibrate_forecast --mode subway --base-model $base --direction total \
            --per-unit-intercept --val-kfold 5
    # intercept IS
    run "cal_${base}_qrcal_intercept_insample" \
        $PY -m scripts.calibrate_forecast --mode subway --base-model $base --direction total \
            --per-unit-intercept --insample-val --suffix qrcal_intercept_insample
done

# ----- C: compute_effects on test for the 10 model variants -----
echo "=== STAGE C: compute_effects ===" | tee -a "$LOG/_master.log"
for model in chronos chronos_qrcal chronos_qrcal_oos chronos_qrcal_intercept chronos_qrcal_intercept_insample \
             timesfm timesfm_qrcal timesfm_qrcal_oos timesfm_qrcal_intercept timesfm_qrcal_intercept_insample; do
    run "eff_${model}" \
        $PY -m scripts.compute_effects --mode subway --model $model --window test --direction total
done

# ----- D: geospatial for the 6 headline -----
echo "=== STAGE D: geospatial_analysis ===" | tee -a "$LOG/_master.log"
for model in chronos chronos_qrcal chronos_qrcal_intercept timesfm timesfm_qrcal timesfm_qrcal_intercept; do
    run "geo_${model}" \
        $PY -m scripts.geospatial_analysis --mode subway --model $model --window test --direction total
done

# ----- E: compare_models tables (val + test, IS + OOS) -----
echo "=== STAGE E: compare_models ===" | tee -a "$LOG/_master.log"
RAW7="arima bsts prophet nhits tft chronos timesfm"
IS_MODELS="$RAW7 chronos_qrcal chronos_qrcal_intercept_insample timesfm_qrcal timesfm_qrcal_intercept_insample"
OOS_MODELS="$RAW7 chronos_qrcal_oos chronos_qrcal_intercept timesfm_qrcal_oos timesfm_qrcal_intercept"
for win in val test; do
    run "compare_IS_${win}" $PY -m scripts.compare_models --mode subway --window $win --direction total --models $IS_MODELS --suffix IS
    run "compare_OOS_${win}" $PY -m scripts.compare_models --mode subway --window $win --direction total --models $OOS_MODELS --suffix OOS
done

# ----- F: residual diagnostic notebook + mirror -----
echo "=== STAGE F: residual + mirror ===" | tee -a "$LOG/_master.log"
run "residual_subway_total" env MODE=subway DIRECTION=total $PY notebooks/06_residual_diagnostic.py
run "mirror_outputs" $PY runs/mirror_outputs.py
run "regression_unified" $PY runs/build_regression_table.py

echo "" | tee -a "$LOG/_master.log"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
