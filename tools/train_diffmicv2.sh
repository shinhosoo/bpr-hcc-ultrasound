#!/usr/bin/env bash
# DiffMICv2 학습만.
# Usage: bash tools/train_diffmicv2.sh <train_pkl> <val_pkl> <out_dir>
set -e
TR="${1:?}"; VA="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
PATIENCE="${PATIENCE:-20}"
export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMICV2_BALANCED_SAMPLER=1
    export DIFFMICV2_WEIGHTED_SAMPLER=0
else
    export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"
fi

mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

CFG="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml" --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$ROOT/models/DiffMICv2-main"
# 이전 학습 logs가 version_N 누적되지 않도록 — train.sh dispatcher는 CLEAN=1로 호출
if [ "${CLEAN:-0}" = "1" ] && [ -d ./logs ]; then rm -rf ./logs; fi
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
python3 diffuser_trainer.py --config "$CFG" --early-stop-patience "$PATIENCE"
# Lightning은 ./logs 아래에 저장하므로 결과 폴더로 복사
[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[train_diffmicv2] ckpts under: ./logs/<runname>/version_X/checkpoints/  (또는 $OUT_ABS/lightning_logs/)"
