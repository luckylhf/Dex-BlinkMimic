#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DEX_MIMIC_ROOT="$PWD"
export DEX_ASSETS="$PWD/assets/dex_evt"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MUJOCO_GL=egl
run_dir=${1:?usage: train_4096.sh runs/run_name}
motion_dir=${2:-data/lafan1/motions_all77_exploratory}
mkdir -p "$run_dir"
.bootstrap/bin/uv pip freeze --python .venv/bin/python > "$run_dir/environment_freeze.txt"
exec .venv/bin/python -u -m dex_mimic.cli train \
  --config configs/dex_blink_safe.json \
  --motion_dir "$motion_dir" \
  --log_dir "$run_dir" --num_envs 4096 \
  --buffer_steps 256 --pi_batch 1024 --wm_batch 1024 \
  --wm_hidden 1024,1024,1024 --pi_opt muon --wm_opt muon \
  --fused_adam --no_sync --fused_rot \
  --ckpt_every 100 --ckpt_minutes 3 --log_every 10 \
  --stop-file "$run_dir/STOP" --iterations 3500
