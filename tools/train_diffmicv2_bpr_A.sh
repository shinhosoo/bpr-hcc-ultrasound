#!/usr/bin/env bash
# DiffMICv2 변형 A (2-stage: BPR-DCG pretrain + 표준 diffuser).
# Usage:
#   bash tools/train_diffmicv2_bpr_A.sh <train_pkl> <val_pkl> <out_dir>
# 기존 baseline (tools/train_diffmicv2.sh) 과 같은 인자 시그니처 — train.sh
# 와 호환되도록 설계.
set -e
TR="${1:?usage: bash tools/train_diffmicv2_bpr_A.sh <train_pkl> <val_pkl> <out_dir>}"
VA="${2:?}"; OUT="${3:?}"

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
VARIANT="$ROOT/DiffMICv2-bpr-A"

# ── 환경변수 (덮어쓰기 가능) ───────────────────────────────────────────────
PATIENCE="${PATIENCE:-20}"
export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMICV2_BALANCED_SAMPLER=1
    export DIFFMICV2_WEIGHTED_SAMPLER=0
else
    export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"
fi
S1_EPOCHS="${S1_EPOCHS:-100}"        # Stage 1 max epoch
S1_PATIENCE="${S1_PATIENCE:-20}"      # Stage 1 early-stop patience
BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
BPR_PROTOTYPE="${BPR_PROTOTYPE:-geomedian}"
BPR_USE_ADV="${BPR_USE_ADV:-1}"
BPR_SINKHORN_EPS="${BPR_SINKHORN_EPS:-0.1}"

mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

# ── 0. config 한 번 갈아끼우기 (train/val pkl 주입) ────────────────────────
CFG="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in  "$VARIANT/configs/lesion_binary.yml" \
    --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

# ── 1. Stage 1: DCG 사전학습 (BPR + CE) ────────────────────────────────────
DCG_OUT="$OUT_ABS/pretrained_dcg.pth"
echo "[bpr-A] Stage 1: DCG pretrain → $DCG_OUT"
cd "$VARIANT"
python3 pretraining/train_dcg_bpr.py \
    --config "$CFG" \
    --out    "$DCG_OUT" \
    --epochs "$S1_EPOCHS" \
    --patience "$S1_PATIENCE" \
    --bpr-lambda    "$BPR_LAMBDA" \
    --bpr-prototype "$BPR_PROTOTYPE" \
    --sinkhorn-eps  "$BPR_SINKHORN_EPS" \
    $( [ "$BPR_USE_ADV" = "1" ] || echo "--no-bpr-adv" )

# ── 2. aux_ckpt_path 를 Stage 1 결과로 주입 ────────────────────────────────
sed -i.bak "s|aux_ckpt_path:.*|aux_ckpt_path: $DCG_OUT|" "$CFG"
echo "[bpr-A] aux_ckpt_path → $DCG_OUT"

# ── 3. Stage 2: v2 표준 diffuser (DCG 동결) ────────────────────────────────
echo "[bpr-A] Stage 2: diffuser training"
[ "${CLEAN:-1}" = "1" ] && [ -d ./logs ] && rm -rf ./logs
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
python3 diffuser_trainer.py --config "$CFG" --early-stop-patience "$PATIENCE"

# Lightning logs 를 결과 폴더로 복사 (test.sh 가 찾는 곳)
[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[bpr-A] done — ckpt under $OUT_ABS/lightning_logs/"
