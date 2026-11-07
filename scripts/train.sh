#!/bin/bash
# Upstream G1 reference launcher (scripts/presets.sh, PRESET[main]).
# Dex entry: uv run dex-mimic train --config configs/dex_blink_recipe.json ...
# See README.md and deployment/README_recipe_20261107.md for Dex data and gates.
# Usage: bash scripts/train.sh <log_dir> [extra train.py flags, e.g. --seed 1]
# About 7.8 min of training on an RTX 4090 for 3500 iterations. Python: $PYTHON, default "uv run python".
set -eu
cd "$(dirname "$0")/.."
source scripts/presets.sh
LOG_DIR=${1:?usage: bash scripts/train.sh <log_dir> [train.py flags]}; shift
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # the reference runs used 4 CPU threads
PYTHONPATH=src ${PYTHON:-uv run python} -u -m blinkmimic.train --log_dir "$LOG_DIR" ${PRESET[main]} "$@" 2>&1 \
  | tee "$LOG_DIR/stdout.log"
