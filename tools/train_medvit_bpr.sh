#!/usr/bin/env bash
# MedViT + BPR (1-stage, 학습 중 직접 주입).
# Usage: bash tools/train_medvit_bpr.sh <train_dir> <val_dir> <out_dir>
set -e
TRAIN="${1:?}"; VAL="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
EPOCHS="${EPOCHS:-100}"; PATIENCE="${PATIENCE:-20}"
SEED="${SEED:-42}"; BS="${BS:-32}"; MODEL="${MODEL:-MedViT_small}"; NBC="${NBC:-2}"
export BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
export BPR_ADV="${BPR_ADV:-0}"
export BPR_NUM_CLASSES="$NBC"
MIXUP="${MEDVIT_BPR_MIXUP:-0}"
CUTMIX="${MEDVIT_BPR_CUTMIX:-0}"
SMOOTHING="${MEDVIT_BPR_SMOOTHING:-0}"

mkdir -p "$OUT"
TRAIN_ABS="$( cd "$TRAIN" && pwd )"
VAL_ABS="$( cd "$VAL" && pwd )"
OUT_ABS="$( cd "$OUT" && pwd )"

cd "$ROOT/models/MedViT-main/CustomDataset"
PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
python3 "$ROOT/tools/bpr/run_medvit_bpr.py" \
    --data-set image_folder \
    --data-path "$TRAIN_ABS" --eval-data-path "$VAL_ABS" \
    --nb-classes "$NBC" --model "$MODEL" \
    --batch-size "$BS" --epochs "$EPOCHS" --seed "$SEED" \
    --pretrained $( [ "${BALANCED:-0}" = "1" ] && echo "--balanced-sampler" || echo "--weighted-sampler" ) --early-stop-patience "$PATIENCE" \
    --mixup "$MIXUP" --cutmix "$CUTMIX" --smoothing "$SMOOTHING" \
    --output-dir "$OUT_ABS" \
    --save-predictions "$OUT_ABS/predictions_val.npz"
echo "[train_medvit_bpr] BPR_LAMBDA=$BPR_LAMBDA  mixup=$MIXUP cutmix=$CUTMIX smoothing=$SMOOTHING  ckpt: $OUT_ABS/checkpoint_best.pth"
