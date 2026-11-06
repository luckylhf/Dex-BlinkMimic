#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DEX_MIMIC_ROOT="$PWD" DEX_ASSETS="$PWD/assets/dex_evt" PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export MUJOCO_GL=egl
checkpoint=${1:?Usage: bash deployment/evaluate_three_test_pkls.sh CHECKPOINT NEW_STUDY_DIR}
root=${2:?Provide a new study directory; historical results are preserved}
checkpoint=$(.venv/bin/python -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$checkpoint")
root=$(.venv/bin/python -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$root")
.venv/bin/python -u deployment/prepare_three_test_pkls.py "$root"
.venv/bin/python -u -m dex_mimic.evaluation.evaluate \
  --checkpoint "$checkpoint" --suite "$root/suite.json" --out "$root/ema" \
  --history "$root/history" --actor ema
