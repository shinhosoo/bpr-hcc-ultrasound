#!/usr/bin/env bash
# MedViT 학습만.
# Usage: bash tools/train_medvit.sh <train_dir> <val_dir> <out_dir>
set -e
TRAIN="${1:?}"; VAL="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
EPOCHS="${EPOCHS:-100}"; PATIENCE="${PATIENCE:-20}"; WARMUP="${WARMUP:-5}"
SEED="${SEED:-42}"; BS="${BS:-32}"; MODEL="${MODEL:-MedViT_small}"; NBC="${NBC:-2}"
# 샘플러 선택 — BALANCED=1 이면 BalancedBatchSampler, 아니면 WeightedRandomSampler 기본
SAMPLER_FLAG="--weighted-sampler"
if [ "${BALANCED:-0}" = "1" ]; then SAMPLER_FLAG="--balanced-sampler"; fi

mkdir -p "$OUT"
TRAIN_ABS="$( cd "$TRAIN" && pwd )"
VAL_ABS="$( cd "$VAL" && pwd )"
OUT_ABS="$( cd "$OUT" && pwd )"
cd "$ROOT/models/MedViT-main/CustomDataset"
python3 main.py --data-set image_folder \
    --data-path "$TRAIN_ABS" --eval-data-path "$VAL_ABS" \
    --nb-classes "$NBC" --model "$MODEL" \
    --batch-size "$BS" --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --seed "$SEED" \
    --pretrained $SAMPLER_FLAG --early-stop-patience "$PATIENCE" \
    --output-dir "$OUT_ABS" \
    --save-predictions "$OUT_ABS/predictions_val.npz"
echo "[train_medvit] best ckpt: $OUT_ABS/checkpoint_best.pth"
