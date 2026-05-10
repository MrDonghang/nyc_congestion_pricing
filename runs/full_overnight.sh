#!/usr/bin/env bash
# Chain: 14 cal-fill jobs + downstream effects/geospatial → full regen.
# After this finishes, you have:
#   - IS and OOS calibration variants for both qrcal and intercept (4 new model dirs)
#   - effects + geospatial outputs for those new variants
#   - compare_<mode>_<window>_{IS,OOS}.csv per panel
#   - regression_unified.csv
#   - att_unified.csv with daily + cumulative + CI
#   - all standardised figures regenerated

set -uo pipefail
MASTER=/home/donghang/nyc_congestion_pricing/logs/overnight/full_chain.log
mkdir -p "$(dirname "$MASTER")"

echo "=== STAGE 1: cal-fill (14 cal jobs + downstream) ===" | tee "$MASTER"
bash /home/donghang/nyc_congestion_pricing/runs/fill_is_oos_calibration.sh
echo "=== STAGE 1 DONE ($(date)) ===" | tee -a "$MASTER"

echo "=== STAGE 2: regen (compare IS/OOS, geospatial, residual notebook, regression unified, mirror) ===" | tee -a "$MASTER"
# Wipe stale repo outputs so the regen is clean
rm -rf /home/donghang/nyc_congestion_pricing/outputs/figures /home/donghang/nyc_congestion_pricing/outputs/_summary
mkdir -p /home/donghang/nyc_congestion_pricing/outputs/_summary
bash /home/donghang/nyc_congestion_pricing/runs/regenerate_outputs.sh
echo "=== STAGE 2 DONE ($(date)) ===" | tee -a "$MASTER"

touch /home/donghang/nyc_congestion_pricing/logs/overnight/_FULL_DONE
echo "FULL DONE at $(date)" | tee -a "$MASTER"
