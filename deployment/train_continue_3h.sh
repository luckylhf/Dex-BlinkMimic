#!/usr/bin/env bash
set -euo pipefail
cd /home/longhongfeng/Desktop/Dex-BlinkMimic
export DEX_MIMIC_ROOT="$PWD" DEX_ASSETS="$PWD/assets/dex_evt" PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
run_dir=${1:?run directory}
resume=${2:?full training checkpoint}
max_hours=${3:?cumulative wall-hour fallback limit}
mkdir -p "$run_dir"
.bootstrap/bin/uv pip freeze --python .venv/bin/python > "$run_dir/environment_freeze.txt"
cp configs/dex_blink_safe.json "$run_dir/package_config.json"
exec .venv/bin/python -u -m dex_mimic.cli train \
  --config configs/dex_blink_safe.json \
  --motion_dir data/lafan1/motions_all77_exploratory \
  --log_dir "$run_dir" --resume "$resume" \
  --iterations 1000000 --max_hours "$max_hours" --stop-file "$run_dir/STOP"
