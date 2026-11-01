#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p outputs
python scripts/check_cuda.py --devices 2
python scripts/check_joint_contract.py
batch=${BATCH_SIZE:-2}
workers=${WORKERS:-0}
epochs=${EPOCHS:-100}
CUDA_VISIBLE_DEVICES=0 python -u train_joint.py --route cartesian --batch-size "$batch" --workers "$workers" --smoke > outputs/joint_cartesian_smoke.log 2>&1
CUDA_VISIBLE_DEVICES=1 python -u train_joint.py --route taoca --batch-size "$batch" --workers "$workers" --smoke > outputs/joint_taoca_smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u train_joint.py --route cartesian --batch-size "$batch" --workers "$workers" --epochs "$epochs" > outputs/joint_cartesian.log 2>&1 &
pa=$!
CUDA_VISIBLE_DEVICES=1 python -u train_joint.py --route taoca --batch-size "$batch" --workers "$workers" --epochs "$epochs" > outputs/joint_taoca.log 2>&1 &
pb=$!
status=0
wait "$pa" || status=1
wait "$pb" || status=1
exit "$status"
