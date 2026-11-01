#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_physical_operators_v7.py --split all
python scripts/check_physical_operator_v7.py
python scripts/audit_physical_operators_v7.py
python -u training/train_taoca_v7.py --epochs 60
python scripts/compare_physical_v7.py
python scripts/analyze_physical_v7.py
