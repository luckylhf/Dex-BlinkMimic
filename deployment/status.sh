#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir=$(cat deployment/current_run.txt)
echo "Current run: $run_dir"
ps -p "$(cat "$run_dir/train.pid")" -o pid,etime,%cpu,rss,args || true
tail -n 15 "$run_dir/stdout.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
