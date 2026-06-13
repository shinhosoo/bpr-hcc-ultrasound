#!/usr/bin/env bash
# DiffMICv2 + BPR (1-stage, Lightning training_step 에 BPR loss 주입).
# Usage: bash tools/train_diffmicv2_bpr.sh <train_pkl> <val_pkl> <out_dir>
set -e
TR="${1:?}"; VA="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
PATIENCE="${PATIENCE:-20}"

# === stage 2 자동탐색에서 사용하므로 OUT_ABS 를 먼저 만든다 ===
mkdir -p "$OUT"
OUT_ABS="$( cd "$OUT" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMICV2_BALANCED_SAMPLER=1
    export DIFFMICV2_WEIGHTED_SAMPLER=0
else
    export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"
fi
export BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
export BPR_NUM_CLASSES="${BPR_NUM_CLASSES:-2}"
# BPR hook 위치 + 시점 게이트
export BPR_HOOK="${BPR_HOOK:-attn}"                 # attn | prelin4 | enc512 | enc512_local | xweight | xweight_bn | xweight_aux | dual_gl
# xweight_bn / xweight_aux 공통
export BPR_BN_DIM="${BPR_BN_DIM:-512}"              # bottleneck 중간 차원
# xweight_bn 전용
export BPR_BN_SKIP="${BPR_BN_SKIP:-0}"              # 0 = pure / 1 = residual
# dual_gl / enc512_local 의 K=6 crop pooling 방식
export BPR_LOCAL_POOL="${BPR_LOCAL_POOL:-mean}"     # mean | parallel
export BPR_T_MAX="${BPR_T_MAX:-1.0}"                # 0<x≤1, 이미지별 평균 timestep fraction 게이트 (prelin4 에만 의미)
export BPR_WARMUP_EPOCHS="${BPR_WARMUP_EPOCHS:-0}"  # 첫 N epoch BPR 비활성
# DCG (aux_model) 선택적 unfreeze
export DCG_UNFREEZE="${DCG_UNFREEZE:-0}"      # 0 | attn | local | all
export DCG_LR_SCALE="${DCG_LR_SCALE:-0.1}"    # main LR 대비 DCG LR 배수
export DCG_WARMUP="${DCG_WARMUP:-0}"          # 첫 N epoch 동안은 frozen 유지
# Stage (1=joint / 2=post-hoc refinement)
export BPR_STAGE="${BPR_STAGE:-1}"
if [ "$BPR_STAGE" = "2" ]; then
    export BPR_STAGE2_DIFF_W="${BPR_STAGE2_DIFF_W:-0.0}"
    export BPR_STAGE2_LR_SCALE="${BPR_STAGE2_LR_SCALE:-0.1}"
    if [ -z "${BPR_STAGE2_CKPT:-}" ]; then
        STAGE1_TAG="${BPR_STAGE2_FROM:-baseline}"
        STAGE1_DIR="$(dirname "$(dirname "$OUT_ABS")")/$STAGE1_TAG/diffmicv2"
        FOUND="$(ls -t "$STAGE1_DIR"/lightning_logs/*/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)"
        if [ -n "$FOUND" ]; then
            export BPR_STAGE2_CKPT="$FOUND"
        else
            echo "[train_diffmicv2_bpr] [stage2] ERROR: stage1 ckpt not found"
            echo "[train_diffmicv2_bpr] [stage2]   searched: $STAGE1_DIR/lightning_logs/*/version_*/checkpoints/*.ckpt"
            echo "[train_diffmicv2_bpr] [stage2]   먼저 'bash train.sh <opt> diffmicv2 $STAGE1_TAG' 로 stage1 학습 또는 BPR_STAGE2_CKPT 직접 지정"
            exit 1
        fi
    fi
    echo "[train_diffmicv2_bpr] [stage2] FROM=${BPR_STAGE2_FROM:-baseline}  CKPT=$BPR_STAGE2_CKPT"
    echo "[train_diffmicv2_bpr] [stage2] DIFF_W=$BPR_STAGE2_DIFF_W  LR_SCALE=$BPR_STAGE2_LR_SCALE"
fi
echo "[train_diffmicv2_bpr] BPR_HOOK=$BPR_HOOK  BPR_LAMBDA=$BPR_LAMBDA  BPR_MODE=${BPR_MODE:-joint}  BPR_PROTO=${BPR_PROTO:-mean}  BPR_PROTO_SCOPE=${BPR_PROTO_SCOPE:-batch}"
echo "[train_diffmicv2_bpr] BPR_T_MAX=$BPR_T_MAX  BPR_WARMUP_EPOCHS=$BPR_WARMUP_EPOCHS  BPR_STAGE=$BPR_STAGE"
if [ "$BPR_HOOK" = "xweight_bn" ] || [ "$BPR_HOOK" = "xweight_aux" ]; then
    echo "[train_diffmicv2_bpr] BPR_BN_DIM=$BPR_BN_DIM  BPR_BN_SKIP=$BPR_BN_SKIP (xweight_bn 전용)"
fi
if [ "$BPR_HOOK" = "dual_gl" ] || [ "$BPR_HOOK" = "enc512_local" ]; then
    echo "[train_diffmicv2_bpr] BPR_LOCAL_POOL=$BPR_LOCAL_POOL"
fi
echo "[train_diffmicv2_bpr] DCG_UNFREEZE=$DCG_UNFREEZE  DCG_LR_SCALE=$DCG_LR_SCALE  DCG_WARMUP=$DCG_WARMUP"

CFG="$OUT_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml" --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$ROOT/models/DiffMICv2-main"
if [ "${CLEAN:-1}" = "1" ] && [ -d ./logs ]; then rm -rf ./logs; fi
DIFFMICV2_PRED_PATH="$OUT_ABS/predictions_val.npz" \
PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
python3 "$ROOT/tools/bpr/run_diffmicv2_bpr.py" \
    --config "$CFG" --early-stop-patience "$PATIENCE"

[ -d ./logs ] && cp -r ./logs "$OUT_ABS/lightning_logs" 2>/dev/null || true
echo "[train_diffmicv2_bpr] BPR_LAMBDA=$BPR_LAMBDA  ckpts: $OUT_ABS/lightning_logs/"
