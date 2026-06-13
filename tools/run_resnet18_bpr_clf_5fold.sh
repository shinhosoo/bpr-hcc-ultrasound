#!/usr/bin/env bash
# (B) 같은 백본(ResNet-18) BPR 분류기 — diffusion 없음. DiffMICv2 와 공정 비교.
#
#   DiffMICv2 = ResNet-18 encoder + diffusion.  이 실험 = 동일 ResNet-18 DCG 분류기 + BPR (CE+BPR),
#   diffusion 단계 제거. 둘이 같은 백본이라 "BPR 분류기 vs diffusion" 을 정당하게 비교.
#
#   각 fold:
#     1) run_diffmicv2_dcg_pretrain.py 로 DCG 를 CE+lambda*BPR 학습 (val AUC 로 best 선택) -> dcg_bpr.pth
#     2) eval_dcg.py 로 그 DCG 를 test 에 평가 -> predictions_test_diffmicv2.npz (표준 포맷)
#   이후 viz.sh b <tag> 로 집계 -> baseline_32(0.864) 와 비교.
#
# Usage:  bash tools/run_resnet18_bpr_clf_5fold.sh [tag]
# env: K=5  STAGE1_EPOCHS=100  PATIENCE=40
#      BPR_LAMBDA=1.0  BPR_PROTO=geomedian  BPR_PROTO_SCOPE=global  BPR_WARMUP_EPOCHS=5
set -e
TAG="${1:-resnet18_bpr_clf}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
K="${K:-5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-${BPR_STAGE1_EPOCHS:-100}}"
PATIENCE="${PATIENCE:-40}"
export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-global}"
export BPR_WARMUP_EPOCHS="${BPR_WARMUP_EPOCHS:-5}"
export BPR_ADV="${BPR_ADV:-0}"
export BPR_NUM_CLASSES=2
# sampler — DiffMICv2(baseline_32)와 동일하게. 기본 weighted ON. BALANCED=1 이면 balanced.
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMICV2_BALANCED_SAMPLER=1; export DIFFMICV2_WEIGHTED_SAMPLER=0
else
    export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"
fi

CFG_BASE="$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml"
RES="$ROOT/experiments/option_b_5fold/results"
DM="$ROOT/models/DiffMICv2-main"

echo "[resnet18-bpr-clf] tag=$TAG  STAGE1_EPOCHS=$STAGE1_EPOCHS  lambda=$BPR_LAMBDA  proto=$BPR_PROTO/$BPR_PROTO_SCOPE  warmup=$BPR_WARMUP_EPOCHS"

for ((i=0; i<K; i++)); do
    echo ""; echo "########## ResNet18+BPR clf  fold $i ##########"
    D="$ROOT/data/5fold/fold_$i/pkl"
    TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"; TE="$D/lesion_test.pkl"
    for f in "$TR" "$VA" "$TE"; do [ -f "$f" ] || { echo "[ERR] pkl 없음: $f"; exit 1; }; done

    OUTDIR="$RES/$TAG/diffmicv2/fold_$i"; mkdir -p "$OUTDIR"
    CFG="$OUTDIR/dcg_config.yml"
    python3 "$ROOT/tools/diffmic_config_swap.py" --in "$CFG_BASE" --out "$CFG" --traindata "$TR" --testdata "$VA"

    DCG_CKPT="$OUTDIR/dcg_bpr.pth"
    if [ "${TEST_PREVIEW:-0}" = "1" ]; then
        export TEST_PREVIEW_PKL="$TE"; export TEST_PREVIEW_EVERY="${TEST_PREVIEW_EVERY:-1}"
        echo "[resnet18-bpr-clf] TEST_PREVIEW on -> test_AUC 도 매 epoch 출력 (모니터링 전용, 선택 미반영)"
    fi
    ( cd "$DM" && PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
        python3 "$ROOT/tools/bpr/run_diffmicv2_dcg_pretrain.py" \
            --config "$CFG" --out "$DCG_CKPT" \
            --epochs "$STAGE1_EPOCHS" --early-stop-patience "$PATIENCE" )
    [ -f "$DCG_CKPT" ] || { echo "[ERR] DCG ckpt 미생성: $DCG_CKPT"; exit 1; }

    ( cd "$DM" && PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
        python3 "$DM/eval_dcg.py" \
            --config "$CFG" --ckpt "$DCG_CKPT" --test-pkl "$TE" \
            --out "$OUTDIR/predictions_test_diffmicv2.npz" )
done

echo ""
echo "[resnet18-bpr-clf] DONE. 집계:"
bash "$ROOT/viz.sh" b "$TAG"
echo "  → $TAG (ResNet-18 + BPR, diffusion 없음) 를 diffmicv2_baseline_32(AUC 0.864) 와 비교하세요."
