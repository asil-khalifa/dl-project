#!/usr/bin/env bash
# Full experiment: data -> fine-tune t5-small -> evaluate 4 models -> plots.
# Env overrides: TRAIN_SIZE, VAL_SIZE, EPOCHS, PYTHON
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
TRAIN=${TRAIN_SIZE:-8000}; VAL=${VAL_SIZE:-300}; EPOCHS=${EPOCHS:-1}

$PY scripts/prepare_data.py --train_size "$TRAIN" --val_size "$VAL"
$PY scripts/train.py --epochs "$EPOCHS"
$PY scripts/evaluate.py --model google-t5/t5-small          --name t5-small-zero-shot --limit "$VAL"
$PY scripts/evaluate.py --model models/t5-small-studyqg     --name t5-small-ours      --limit "$VAL"
$PY scripts/evaluate.py --model lmqg/t5-small-squad-qg-ae   --name t5-small-lmqg      --limit "$VAL"
$PY scripts/evaluate.py --model lmqg/t5-base-squad-qg-ae    --name t5-base-lmqg       --limit "$VAL"
$PY scripts/plot_results.py
echo "done: see results/"
