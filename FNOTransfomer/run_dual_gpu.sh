#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
export BATCH_SIZE=${BATCH_SIZE:-4}
export WORKERS=${WORKERS:-2}
export EPOCHS=${EPOCHS:-100}
python scripts/check_cuda.py --devices 2
python -c 'import json; from pathlib import Path; m=json.loads(Path("datasets/fenli_paired/manifest.json").read_text()); assert sum(map(len,m["cases"].values()))==500; a=json.loads(Path("datasets/fenli_paired/audit.json").read_text()); assert a["summary"]["frames"]==500'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MKL_THREADING_LAYER=GNU
# Real-data forward/backward, validation, inverse mapping before full training.
CUDA_VISIBLE_DEVICES=0 python -u train_compare.py --route cartesian --device cuda --batch-size "$BATCH_SIZE" --workers "$WORKERS" --smoke --output outputs/server_smoke_cartesian > outputs/server_smoke_cartesian.log 2>&1
CUDA_VISIBLE_DEVICES=1 python -u train_compare.py --route taoca --device cuda --batch-size "$BATCH_SIZE" --workers "$WORKERS" --smoke --output outputs/server_smoke_taoca > outputs/server_smoke_taoca.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u train_compare.py --route cartesian --device cuda --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --workers "$WORKERS" > outputs/cartesian.log 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 python -u train_compare.py --route taoca --device cuda --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --workers "$WORKERS" > outputs/taoca.log 2>&1 &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
if [ "$status" -ne 0 ]; then
  echo 'Training failed; inspect outputs/cartesian.log and outputs/taoca.log.'
  exit 1
fi
python scripts/compare_runs.py
