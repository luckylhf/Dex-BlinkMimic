#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DEX_MIMIC_ROOT="$PWD"
export DEX_ASSETS="$PWD/assets/dex_evt"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MUJOCO_GL=egl
run_dir=${1:?usage: train_package_config.sh runs/run_name}
mkdir -p "$run_dir"
.bootstrap/bin/uv pip freeze --python .venv/bin/python > "$run_dir/environment_freeze.txt"
cp configs/dex_blink_safe.json "$run_dir/package_config.json"
# Training hyperparameters come entirely from the package config and parser defaults.
# Only input/output/control-file paths are supplied here.
exec .venv/bin/python -u -m dex_mimic.cli train \
  --config configs/dex_blink_safe.json \
  --motion_dir data/lafan1/motions_all77_exploratory \
  --log_dir "$run_dir" \
  --stop-file "$run_dir/STOP"
