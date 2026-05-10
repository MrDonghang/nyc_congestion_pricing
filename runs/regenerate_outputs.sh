#!/usr/bin/env bash
# One-shot regeneration of all standardised figures and tables.
#
# Phases:
#   1. compare_models — val + test cross-model panel metrics for all modes
#                       (full 13-model list for bus/subway, 6 headline only for replica)
#   2. geospatial_analysis — for the 6 headline models on bus + subway + replica:
#      writes tract_effects.geojson, significance_map.png, crz_summary.csv,
#      spatial_regression.txt, ml_models.csv per (mode, model[, dir]) into
#      ``<output_root>/<mode>/<model>/[<dir>/]causal/``.
#   3. notebooks/06_residual_diagnostic.py — per-mode residual diagnostic
#      figures + tables under ``outputs/figures/<mode>/residual/``.
#   4. Mirror key figures + summaries into the repo for easy review.
#
# Forecast triplets (output_new/<mode>/<model>/...) are NOT regenerated —
# this script only re-derives downstream tables / figures from existing
# triplets + effects.

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/regen
mkdir -p "$LOG"
PY=/home/donghang/anaconda3/envs/nyc_cp/bin/python
cd /home/donghang/nyc_congestion_pricing
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

# 11 baseline + cal model variants per the project headline list.
# IS variants:  *_qrcal (insample)        + *_qrcal_intercept_insample
# OOS variants: *_qrcal_oos               + *_qrcal_intercept (kfold)
# Raw 7 models (arima/bsts/prophet/nhits/tft/chronos/timesfm) appear in both.
RAW7="arima bsts prophet nhits tft chronos timesfm"
IS_MODELS="$RAW7 chronos_qrcal chronos_qrcal_intercept_insample timesfm_qrcal timesfm_qrcal_intercept_insample"
OOS_MODELS="$RAW7 chronos_qrcal_oos chronos_qrcal_intercept timesfm_qrcal_oos timesfm_qrcal_intercept"
HEADLINE="chronos chronos_qrcal chronos_qrcal_intercept timesfm timesfm_qrcal timesfm_qrcal_intercept"

# ============================================================================
# Phase 1: compare_models — IS table (val + test) and OOS table (val + test)
# Test rows are the same forecast triplets in both views (test is always
# applied via full-val fit); only val rows differ between IS and OOS.
# ============================================================================
# Citibike: cal not viable (data starts 2024-01, val/test season mismatch
# unavoidable). Only RAW chronos + timesfm appear in IS/OOS tables — they
# look identical there (no cal applied) but we still report so the val/test
# accuracy of the foundation models is documented.
CITIBIKE_RAW="chronos timesfm"

echo "=== PHASE 1a: compare_models IS ===" | tee -a "$LOG/_master.log"
for win in val test; do
    run "compare_bus_IS_${win}" \
        $PY -m scripts.compare_models --mode bus --window $win --models $IS_MODELS --suffix IS
    for d in O D; do
        run "compare_subway_IS_${win}_${d}" \
            $PY -m scripts.compare_models --mode subway --window $win --direction $d --models $IS_MODELS --suffix IS
        run "compare_replica_IS_${win}_${d}" \
            $PY -m scripts.compare_models --mode replica --window $win --direction $d --models $IS_MODELS --suffix IS
        run "compare_citibike_IS_${win}_${d}" \
            $PY -m scripts.compare_models --mode citibike --window $win --direction $d --models $CITIBIKE_RAW --suffix IS
    done
done

echo "=== PHASE 1b: compare_models OOS ===" | tee -a "$LOG/_master.log"
for win in val test; do
    run "compare_bus_OOS_${win}" \
        $PY -m scripts.compare_models --mode bus --window $win --models $OOS_MODELS --suffix OOS
    for d in O D; do
        run "compare_subway_OOS_${win}_${d}" \
            $PY -m scripts.compare_models --mode subway --window $win --direction $d --models $OOS_MODELS --suffix OOS
        run "compare_replica_OOS_${win}_${d}" \
            $PY -m scripts.compare_models --mode replica --window $win --direction $d --models $OOS_MODELS --suffix OOS
        run "compare_citibike_OOS_${win}_${d}" \
            $PY -m scripts.compare_models --mode citibike --window $win --direction $d --models $CITIBIKE_RAW --suffix OOS
    done
done

# ============================================================================
# Phase 2: geospatial_analysis (headline 6 models, all modes)
# ============================================================================
echo "=== PHASE 2: geospatial_analysis ===" | tee -a "$LOG/_master.log"
for m in $HEADLINE; do
    run "geo_bus_${m}" $PY -m scripts.geospatial_analysis --mode bus --model $m --window test
    for d in O D; do
        run "geo_subway_${m}_${d}" $PY -m scripts.geospatial_analysis --mode subway --model $m --window test --direction $d
        run "geo_replica_${m}_${d}" $PY -m scripts.geospatial_analysis --mode replica --model $m --window test --direction $d
    done
done
# Citibike: only chronos / timesfm raw (no cal — see config caveat)
for m in chronos timesfm; do
    for d in O D; do
        run "geo_citibike_${m}_${d}" $PY -m scripts.geospatial_analysis --mode citibike --model $m --window test --direction $d
    done
done

# ============================================================================
# Phase 3: residual diagnostic notebook (per mode)
# ============================================================================
echo "=== PHASE 3: residual diagnostic ===" | tee -a "$LOG/_master.log"
for mode in bus subway replica citibike; do
    run "residual_${mode}" env MODE=$mode $PY notebooks/06_residual_diagnostic.py
done

# ============================================================================
# Phase 4: build unified regression table (parses all spatial_regression.txt)
# ============================================================================
echo "=== PHASE 4: regression unified table ===" | tee -a "$LOG/_master.log"
run "regression_table" $PY runs/build_regression_table.py

# ============================================================================
# Phase 5: mirror key figures + tables into the repo
# ============================================================================
echo "=== PHASE 5: mirror to repo ===" | tee -a "$LOG/_master.log"
run "mirror_outputs" $PY runs/mirror_outputs.py

# ============================================================================
# DONE
# ============================================================================
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "" | tee -a "$LOG/_master.log"
echo "===================================================" | tee -a "$LOG/_master.log"
echo "ALL DONE in $((ELAPSED / 60)) min  ($(date))" | tee -a "$LOG/_master.log"
echo "===================================================" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
