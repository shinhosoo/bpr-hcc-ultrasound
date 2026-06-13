#!/usr/bin/env bash
# DiffMICv2 + BPR — DiffMIC-v1 식 2-stage.
#   Stage 1: DCG(aux_model) 를 CE + lambda*BPR 로 단독 표현학습 -> dcg_bpr.pth
#   Stage 2: 그 DCG 를 aux_ckpt_path 로 로드->freeze, diffusion 만 baseline 학습.
# Usage: bash tools/train_diffmicv2_2stage.sh <train_pkl> <val_pkl> <out_dir>
# (train_diffmicv2_bpr.sh 가 BPR_TWO_STAGE=1 일 때 이 스크립트로 위임)
set -e
TR="${1:?}"; VA="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

PATIENCE="${PATIENCE:-20}"
STAGE1_EPOCHS="${BPR_STAGE1_EPOCHS:-50}"     # Stage1 DCG 표현학습 epoch (기본 50 = 전체 절반)

# Stage1 BPR 설정 — 미설정 시 default (run_diffmicv2_dcg_pretrain.py 가 env 로 읽음)
export BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
export BPR_ADV="${BPR_ADV:-0}"
export BPR_NUM_CLASSES="${BPR_NUM_CLASSES:-2}"
export BPR_PROTO="${BPR_PROTO:-mean}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-batch}"
export BPR_WARMUP_EPOCHS="${BPR_WARMUP_EPOCHS:-0}"

mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"
DCG_CKPT="$OUT_ABS/dcg_bpr.pth"

echo "============================================================"
echo " [diffmicv2 2-stage]  Stage1=$STAGE1_EPOCHS ep  PATIENCE=$PATIENCE"
echo "   BPR: lambda=$BPR_LAMBDA adv=$BPR_ADV proto=$BPR_PROTO/$BPR_PROTO_SCOPE warmup=$BPR_WARMUP_EPOCHS"
echo "============================================================"

# ===== Stage 1: DCG BPR pretrain =====
CFG_S1="$OUT_ABS/dcg_pretrain_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml" --out "$CFG_S1" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$ROOT/models/DiffMICv2-main"
echo "[diffmicv2 2-stage] === Stage 1: DCG (CE + BPR) pretrain ==="
PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
python3 "$ROOT/tools/bpr/run_diffmicv2_dcg_pretrain.py" \
    --config "$CFG_S1" --out "$DCG_CKPT" \
    --epochs "$STAGE1_EPOCHS" --early-stop-patience "$PATIENCE"

if [ ! -f "$DCG_CKPT" ]; then
    echo "[diffmicv2 2-stage] ERROR: Stage1 DCG ckpt not produced: $DCG_CKPT"; exit 1
fi

# ===== Stage 2: diffusion 학습 (frozen, BPR-pretrained DCG 로드) =====
CFG_S2="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml" --out "$CFG_S2" \
    --traindata "$TR_ABS" --testdata "$VA_ABS" --aux_ckpt_path "$DCG_CKPT"

echo "[diffmicv2 2-stage] === Stage 2: diffusion train (DCG frozen, loaded from Stage1) ==="
if [ "${CLEAN:-1}" = "1" ] && [ -d ./logs ]; then rm -rf ./logs; fi
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
PYTHONPATH="$PWD:$PYTHONPATH" \
python3 diffuser_trainer.py --config "$CFG_S2" --early-stop-patience "$PATIENCE"

[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[diffmicv2 2-stage] DONE  DCG=$DCG_CKPT  diffusion ckpts: $OUT_ABS/lightning_logs/"
