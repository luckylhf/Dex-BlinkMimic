#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="$PWD/evaluations/latest_test_20261106_1430"
render_pid=$(cat "$root/render.pid")
while kill -0 "$render_pid" 2>/dev/null; do sleep 2; done
nice -n 10 .venv/bin/python -u deployment/finalize_three_test_evaluation.py "$root"
for motion in gbtc dance_boy dance_girl; do
  case "$motion" in gbtc) seconds=5;; dance_boy) seconds=10;; dance_girl) seconds=20;; esac
  ffmpeg -v error -threads 2 -ss "$seconds" \
    -i "$root/videos/${motion}_checkpoint_001899_reference_ema_live.mp4" \
    -frames:v 1 "$root/videos/qa_encoded_${motion}.png"
done
echo FINAL_VALIDATION_COMPLETE
