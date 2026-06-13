#!/usr/bin/env bash
# DiffMICv2 변형 C (1-stage: BPR on encoder_x via forward hook, DCG 무변경).
# Usage:
#   bash tools/train_diffmicv2_bpr_C.sh <train_pkl> <val_pkl> <out_dir>
set -e
TR="${1:?usage: bash tools/train_diffmicv2_bpr_C.sh <train_pkl> <val_pkl> <out_dir>}"
VA="${2:?}"; OUT="${3:?}"

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
VARIANT="$ROOT/DiffMICv2-bpr-C"

PATIENCE="${PATIENCE:-20}"
export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMICV2_BALANCED_SAMPLER=1
    export DIFFMICV2_WEIGHTED_SAMPLER=0
else
    export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"
fi
export BPR_LAMBDA="${BPR_LAMBDA:-0.05}"
export BPR_PROTOTYPE="${BPR_PROTOTYPE:-geomedian}"
export BPR_USE_ADV="${BPR_USE_ADV:-1}"
export BPR_SINKHORN_EPS="${BPR_SINKHORN_EPS:-0.1}"

mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

CFG="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in  "$VARIANT/configs/lesion_binary.yml" \
    --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$VARIANT"
[ "${CLEAN:-1}" = "1" ] && [ -d ./logs ] && rm -rf ./logs
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
python3 diffuser_trainer.py --config "$CFG" --early-stop-patience "$PATIENCE"

[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[bpr-C] BPR_LAMBDA=$BPR_LAMBDA  done — ckpt under $OUT_ABS/lightning_logs/"
