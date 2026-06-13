#!/usr/bin/env bash
# 저장된 MedViT ckpt 로 test set 추론만 수행하고 predictions npz를 생성.
# Usage:
#   bash tools/test_medvit.sh <ckpt> <test_imagefolder> <out_npz>
# Example:
#   bash tools/test_medvit.sh \
#       results/medvit_baseline/checkpoint_best.pth \
#       data/3way/imagefolder/test \
#       results/medvit_baseline/predictions_test.npz
set -e
CKPT="${1:?usage: bash test_medvit.sh <ckpt> <test_imagefolder> <out_npz>}"
TEST_DIR="${2:?usage: bash test_medvit.sh <ckpt> <test_imagefolder> <out_npz>}"
OUT="${3:?usage: bash test_medvit.sh <ckpt> <test_imagefolder> <out_npz>}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

MODEL="${MODEL:-MedViT_small}"
NBC="${NBC:-2}"
SEED="${SEED:-42}"
BS="${BS:-32}"
mkdir -p "$(dirname "$OUT")"
ABS_OUT="$( cd "$(dirname "$OUT")" && pwd )/$(basename "$OUT")"

cd "$ROOT/models/MedViT-main/CustomDataset"
python3 main.py \
    --eval --resume "$CKPT" \
    --data-set image_folder \
    --data-path      "$TEST_DIR" \
    --eval-data-path "$TEST_DIR" \
    --nb-classes "$NBC" --model "$MODEL" \
    --batch-size "$BS" --seed "$SEED" \
    --save-predictions "$ABS_OUT"
echo "[test_medvit] wrote: $ABS_OUT"
