#!/usr/bin/env bash
# DiffMICv2 EfficientSAM backbone 학습.
# Usage: bash tools/train_diffmicv2_sam.sh <train_pkl> <val_pkl> <out_dir>
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

# EfficientSAM 패키지가 import 가능하도록 PYTHONPATH 에 추가
if [ ! -d "$ROOT/EfficientSAM" ]; then
    echo "[train_diffmicv2_sam] ERROR: EfficientSAM repo 없음."
    echo "먼저 실행: bash tools/install_efficientsam.sh"
    exit 1
fi
export PYTHONPATH="$ROOT:$PYTHONPATH"

mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

# DiffMICv2-sam 의 SAM config 를 베이스로 traindata/testdata 만 갱신
CFG="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/DiffMICv2-sam/configs/lesion_binary.yml" --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$ROOT/models/DiffMICv2-main"
if [ "${CLEAN:-0}" = "1" ] && [ -d ./logs ]; then rm -rf ./logs; fi
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
python3 diffuser_trainer.py --config "$CFG" --early-stop-patience "$PATIENCE"
[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[train_diffmicv2_sam] ckpts under: $OUT_ABS/lightning_logs/"
