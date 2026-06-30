#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"

mkdir -p "$RESULTS_DIR"

python src/train_cifar10_resnet.py \
  --model resnet18 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULTS_DIR/resnet18_e${EPOCHS}"

python src/train_cifar10_resnet.py \
  --model plain18 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULTS_DIR/plain18_e${EPOCHS}"

python src/compare_runs.py \
  --runs "$RESULTS_DIR/resnet18_e${EPOCHS}" "$RESULTS_DIR/plain18_e${EPOCHS}" \
  --output-dir "$RESULTS_DIR"

echo "Done. See $RESULTS_DIR/ablation_summary.csv and $RESULTS_DIR/ablation_comparison.png"
