#!/usr/bin/env bash
# MedViTV2 + BPR (joint or 2-stage). MedViT v1 의 BPR wrapper 와 동일 인터페이스.
# Usage: bash tools/train_medvitv2_bpr.sh <train_dir> <val_dir> <out_dir>
set -e
TRAIN="${1:?}"; VAL="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

EPOCHS="${EPOCHS:-100}"; PATIENCE="${PATIENCE:-20}"
SEED="${SEED:-42}"; BS="${BS:-32}"
MODEL="${MODEL:-MedViT_small}"
LR="${LR:-0.0001}"
PRETRAINED="${PRETRAINED:-True}"

# Best ckpt 선택 기준 — auc (default) | f1 | acc
# baseline (train_medvitv2.sh) 와 같은 값으로 비교해야 fair.
export BEST_BY="${BEST_BY:-auc}"
echo "[train_medvitv2_bpr] BEST_BY=$BEST_BY  (best ckpt selection criterion)"

# BPR 환경변수 — 미설정 시 default
export BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
export BPR_ADV="${BPR_ADV:-0}"
export BPR_NUM_CLASSES="${BPR_NUM_CLASSES:-2}"
export BPR_HOOK="${BPR_HOOK:-global}"           # global | aux | dual_gl
export BPR_BN_DIM="${BPR_BN_DIM:-512}"

# dual_gl 전용 옵션 (BPR_HOOK=dual_gl 일 때만 의미 있음)
export BPR_LOCAL_STAGE="${BPR_LOCAL_STAGE:-2}"    # 0~3 (depths 의 stage 인덱스)
export BPR_LOCAL_POOL="${BPR_LOCAL_POOL:-mean}"   # mean | parallel
export BPR_DUAL_GLOBAL_W="${BPR_DUAL_GLOBAL_W:-1.0}"
export BPR_DUAL_LOCAL_W="${BPR_DUAL_LOCAL_W:-0.5}"
export BPR_PROTO="${BPR_PROTO:-mean}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-batch}"
# v1 (MedViT) 와 통일 — 첫 5 epoch CE-only
export BPR_WARMUP_EPOCHS="${BPR_WARMUP_EPOCHS:-5}"

# v1 호환: global prototype refresh (BPR_PROTO_SCOPE=global 일 때만 의미 있음)
export BPR_PROTO_REFRESH="${BPR_PROTO_REFRESH:-1}"   # N epoch 마다 전체 데이터셋 refresh
export BPR_PROTO_BS="${BPR_PROTO_BS:-64}"            # refresh 시 batch size
export BPR_PROTO_EMA="${BPR_PROTO_EMA:-0.0}"         # 0 = 매번 새로, >0 = EMA 보간

# projection head + faithful 모드
export BPR_USE_PROJ="${BPR_USE_PROJ:-0}"
export BPR_PROJ_DIM="${BPR_PROJ_DIM:-128}"
export BPR_PROJ_HIDDEN="${BPR_PROJ_HIDDEN:-512}"
export BPR_FAITHFUL="${BPR_FAITHFUL:-0}"

# SupCon hybrid — BPR 와 같은 projected z 위에 추가 contrastive loss
# 0 이면 비활성. 권장 시작값: λ=0.1~0.3, τ=0.1
export BPR_SUPCON_LAMBDA="${BPR_SUPCON_LAMBDA:-0.0}"
export BPR_SUPCON_TEMP="${BPR_SUPCON_TEMP:-0.1}"

# 2-stage
export BPR_TWO_STAGE="${BPR_TWO_STAGE:-0}"
export BPR_STAGE1_EPOCHS="${BPR_STAGE1_EPOCHS:--1}"

mkdir -p "$OUT"
TRAIN_ABS="$( cd "$TRAIN" && pwd )"
VAL_ABS="$( cd "$VAL" && pwd )"
OUT_ABS="$( cd "$OUT" && pwd )"

# 샘플러 결정
SAMPLER_FLAG="--weighted-sampler"
if [ "${BALANCED:-0}" = "1" ]; then SAMPLER_FLAG="--balanced-sampler"; fi

echo "[train_medvitv2_bpr] BPR_HOOK=$BPR_HOOK  BPR_LAMBDA=$BPR_LAMBDA  BPR_ADV=$BPR_ADV"
echo "[train_medvitv2_bpr] PROTO=$BPR_PROTO/$BPR_PROTO_SCOPE  WARMUP=$BPR_WARMUP_EPOCHS"
if [ "$BPR_PROTO_SCOPE" = "global" ]; then
    echo "[train_medvitv2_bpr] global-refresh: every ${BPR_PROTO_REFRESH}ep  bs=${BPR_PROTO_BS}  ema=${BPR_PROTO_EMA}"
fi
if [ "$BPR_USE_PROJ" = "1" ]; then
    echo "[train_medvitv2_bpr] projection head: $BPR_PROJ_HIDDEN → $BPR_PROJ_DIM"
fi
if [ "$BPR_HOOK" = "dual_gl" ]; then
    echo "[train_medvitv2_bpr] dual_gl: local_stage=$BPR_LOCAL_STAGE  pool=$BPR_LOCAL_POOL  w_g=$BPR_DUAL_GLOBAL_W  w_l=$BPR_DUAL_LOCAL_W"
fi
# SupCon — λ>0 일 때만 로그
if [ "$(echo "$BPR_SUPCON_LAMBDA > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "[train_medvitv2_bpr] BPR+SupCon hybrid: SUPCON_LAMBDA=$BPR_SUPCON_LAMBDA  TEMP=$BPR_SUPCON_TEMP"
fi
if [ "$BPR_TWO_STAGE" = "1" ]; then
    echo "[train_medvitv2_bpr] [2-stage] ENABLED  STAGE1_EPOCHS=$BPR_STAGE1_EPOCHS (-1 = epochs/2)"
fi

cd "$ROOT/models/MedViTV2-main"
PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
python3 "$ROOT/tools/bpr/run_medvitv2_bpr.py" \
    --dataset lesion_binary \
    --train-path "$TRAIN_ABS" \
    --val-path   "$VAL_ABS" \
    --model_name "$MODEL" \
    --batch_size "$BS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --pretrained "$PRETRAINED" \
    --output-dir "$OUT_ABS" \
    $SAMPLER_FLAG \
    --early-stop-patience "$PATIENCE" \
    --seed "$SEED" \
    --save-predictions "$OUT_ABS/predictions_val.npz"
echo "[train_medvitv2_bpr] BPR_LAMBDA=$BPR_LAMBDA  ckpt: $OUT_ABS/${MODEL}_lesion_binary_best.pth"
