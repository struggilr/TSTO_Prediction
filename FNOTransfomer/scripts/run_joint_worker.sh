#!/usr/bin/env bash
set -euo pipefail
python_bin=$1
route=$2
batch=$3
workers=$4
epochs=$5
run_one() {
  "$python_bin" -u train_joint.py --route "$1" --device cuda --batch-size "$batch" --workers "$workers" --epochs "$epochs"
}
if [ "$route" = both ]; then
  run_one cartesian
  run_one taoca
  "$python_bin" scripts/compare_joint.py
else
  run_one "$route"
fi
