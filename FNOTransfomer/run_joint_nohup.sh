#!/usr/bin/env bash
# Usage: bash run_joint_nohup.sh cartesian|taoca|both
set -euo pipefail
cd "$(dirname "$0")"
route=${1:-cartesian}
case "$route" in cartesian|taoca|both) ;; *) echo 'Use cartesian, taoca, or both'; exit 2 ;; esac
mkdir -p outputs
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
python_bin=$(python -c 'import sys; print(sys.executable)')
"$python_bin" scripts/check_cuda.py --devices 1
pidfile="outputs/joint_${route}_background.pid"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "Existing job is active: $(cat "$pidfile")"; exit 1
fi
stamp=$(date +%Y%m%d_%H%M%S)
log="outputs/joint_${route}_${stamp}.log"
nohup bash scripts/run_joint_worker.sh "$python_bin" "$route" "${BATCH_SIZE:-2}" "${WORKERS:-0}" "${EPOCHS:-100}" > "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "PID=$pid"
echo "LOG=$log"
echo "View: tail -f $log"
