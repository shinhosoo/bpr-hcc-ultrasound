#!/usr/bin/env bash
# MedViTV2 학습.
# Usage: bash tools/train_medvitv2.sh <train_dir> <val_dir> <out_dir>
set -e
TRAIN="${1:?}"; VAL="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
EPOCHS="${EPOCHS:-100}"; PATIENCE="${PATIENCE:-20}"
SEED="${SEED:-42}"; BS="${BS:-32}"
MODEL="${MODEL:-MedViT_small}"  # MedViTV2 의 small/base/large/tiny
LR="${LR:-0.0001}"
PRETRAINED="${PRETRAINED:-True}"

mkdir -p "$OUT"
TRAIN_ABS="$( cd "$TRAIN" && pwd )"
VAL_ABS="$( cd "$VAL" && pwd )"
OUT_ABS="$( cd "$OUT" && pwd )"

cd "$ROOT/models/MedViTV2-main"
python3 main.py \
    --dataset lesion_binary \
    --train-path "$TRAIN_ABS" \
    --val-path   "$VAL_ABS" \
    --model_name "$MODEL" \
    --batch_size "$BS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --pretrained "$PRETRAINED" \
    --output-dir "$OUT_ABS" \
    $( [ "${BALANCED:-0}" = "1" ] && echo "--balanced-sampler" || echo "--weighted-sampler" ) \
    --early-stop-patience "$PATIENCE" \
    --seed "$SEED" \
    --save-predictions "$OUT_ABS/predictions_val.npz"
echo "[train_medvitv2] best ckpt: $OUT_ABS/${MODEL}_lesion_binary_best.pth"
