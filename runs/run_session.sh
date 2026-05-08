#!/bin/bash
# Run all experiments listed in a text file (one per line: mode direction model window).
# Usage: run_session.sh <list_file> [<gpu_id>]

list_file="$1"
gpu_id="$2"

if [ -n "$gpu_id" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
fi

REPO=/home/donghang/nyc_congestion_pricing
session_name=$(basename "$list_file" .txt)
echo "===== SESSION $session_name BEGIN: $(date) (CUDA=${CUDA_VISIBLE_DEVICES:-cpu}) ====="

while IFS= read -r line; do
    # skip blank / comment lines
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^# ]] && continue
    bash "$REPO/runs/run_one.sh" $line
done < "$list_file"

echo "===== SESSION $session_name DONE:  $(date) ====="
