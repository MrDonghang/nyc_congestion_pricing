#!/usr/bin/env bash
# Rerun bus geospatial_analysis for the 6 headline models so that the new
# 2-class CRZ labelling ("Inside CRZ" / "Outside CRZ", merging the old
# fully_inside + partially_inside into Inside) propagates into:
#   - crz_summary.csv
#   - significance_map.png  (CRZ polygon + 2-class legend)
#   - trends_by_crz.png     (now 2 lines instead of 3)
# Then re-mirror so the repo's outputs/figures/bus/ + outputs/_summary/
# unified tables pick up the new labels.

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/bus_2class
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

for model in chronos chronos_qrcal chronos_qrcal_intercept timesfm timesfm_qrcal timesfm_qrcal_intercept; do
    run "geo_bus_${model}" \
        $PY -m scripts.geospatial_analysis --mode bus --model $model --window test
done

run "regression_unified" $PY runs/build_regression_table.py
run "mirror_outputs" $PY runs/mirror_outputs.py

echo "" | tee -a "$LOG/_master.log"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
