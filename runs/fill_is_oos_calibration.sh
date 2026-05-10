#!/usr/bin/env bash
# Fill in the missing IS/OOS calibration variants so we can produce two
# parallel validation-performance tables:
#   IS table  uses: chronos_qrcal, chronos_qrcal_intercept_insample, ...
#   OOS table uses: chronos_qrcal_oos, chronos_qrcal_intercept, ...
#
# This script ADDS the following 14 model directories (none destroy existing):
#   chronos_qrcal_oos                       on bus + subway O,D + replica O,D (5 panels)
#   timesfm_qrcal_oos                       same (5 panels)
#   chronos_qrcal_intercept_insample        on replica O,D (existing for bus + subway)
#   timesfm_qrcal_intercept_insample        on replica O,D
#
# Test triplets are also produced (full-val fit applied to test) so downstream
# compute_effects / geospatial_analysis work on the new model names too.

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/cal_fill
mkdir -p "$LOG"
PY=/home/donghang/anaconda3/envs/nyc_cp/bin/python
cd /home/donghang/nyc_congestion_pricing

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

# Phase A: chronos_qrcal_oos and timesfm_qrcal_oos (--val-kfold 5, default qrcal cal)
echo "=== PHASE A: qrcal OOS (k-fold) ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    run "${base}_qrcal_oos_bus" \
        $PY -m scripts.calibrate_forecast --mode bus --base-model $base \
            --val-kfold 5 --suffix qrcal_oos
    for d in O D; do
        run "${base}_qrcal_oos_subway_${d}" \
            $PY -m scripts.calibrate_forecast --mode subway --base-model $base --direction $d \
                --val-kfold 5 --suffix qrcal_oos
        run "${base}_qrcal_oos_replica_${d}" \
            $PY -m scripts.calibrate_forecast --mode replica --base-model $base --direction $d \
                --val-kfold 5 --suffix qrcal_oos
    done
done

# Phase B: chronos_qrcal_intercept_insample and timesfm_qrcal_intercept_insample on replica
# (already exist for bus + subway from earlier work)
echo "=== PHASE B: intercept IS for replica ===" | tee -a "$LOG/_master.log"
for base in chronos timesfm; do
    for d in O D; do
        run "${base}_intercept_is_replica_${d}" \
            $PY -m scripts.calibrate_forecast --mode replica --base-model $base --direction $d \
                --per-unit-intercept --insample-val --suffix qrcal_intercept_insample
    done
done

# Phase C: compute test effects + geospatial for the 4 new model names
# (so they appear in att_unified / crz_unified / spatial / regression tables)
echo "=== PHASE C: effects + geospatial for new variants ===" | tee -a "$LOG/_master.log"
for model in chronos_qrcal_oos timesfm_qrcal_oos chronos_qrcal_intercept_insample timesfm_qrcal_intercept_insample; do
    run "eff_${model}_bus" \
        $PY -m scripts.compute_effects --mode bus --model $model --window test
    run "geo_${model}_bus" \
        $PY -m scripts.geospatial_analysis --mode bus --model $model --window test
    for d in O D; do
        run "eff_${model}_subway_${d}" \
            $PY -m scripts.compute_effects --mode subway --model $model --window test --direction $d
        run "geo_${model}_subway_${d}" \
            $PY -m scripts.geospatial_analysis --mode subway --model $model --window test --direction $d
        run "eff_${model}_replica_${d}" \
            $PY -m scripts.compute_effects --mode replica --model $model --window test --direction $d
        run "geo_${model}_replica_${d}" \
            $PY -m scripts.geospatial_analysis --mode replica --model $model --window test --direction $d
    done
done

echo "" | tee -a "$LOG/_master.log"
echo "=== ALL CAL FILL DONE ($(date)) ===" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
