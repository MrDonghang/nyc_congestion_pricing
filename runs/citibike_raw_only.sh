#!/usr/bin/env bash
# Citibike: only raw chronos + timesfm (calibration not viable due to data
# coverage — see configs/modes/citibike.yaml caveat). compute_effects +
# geospatial_analysis for both bases × {O, D} = 8 jobs (CPU-only, fast).

set -uo pipefail
LOG=/home/donghang/nyc_congestion_pricing/logs/overnight/citibike_raw
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

for model in chronos timesfm; do
    for d in O D; do
        run "eff_${model}_${d}" \
            $PY -m scripts.compute_effects --mode citibike --model $model --window test --direction $d
        run "geo_${model}_${d}" \
            $PY -m scripts.geospatial_analysis --mode citibike --model $model --window test --direction $d
    done
done

echo "" | tee -a "$LOG/_master.log"
echo "=== ALL CITIBIKE RAW DONE ($(date)) ===" | tee -a "$LOG/_master.log"
touch "$LOG/_DONE"
