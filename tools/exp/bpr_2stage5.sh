#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

TAG="${TAG:-bpr_2stage}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
EPOCHS="${EPOCHS:-100}"
K="${K:-5}"
#MODELS="${MODELS:-medvit diffmic diffmicv2}"
MODELS="${MODELS:-medvit}"

echo "============================================================"
echo " [recipe] BPR 2-stage 5-fold CV  (Stage1=$STAGE1_EPOCHS / total=$EPOCHS, K=$K)"
echo " TAG = $TAG  (BALANCED)  MODELS = $MODELS"
echo "============================================================"

export BPR_TWO_STAGE=1
export BPR_STAGE1_EPOCHS="$STAGE1_EPOCHS"

#export BPR_ADV=1
export BPR_LAMBDA=1 # 0.1
#export BPR_HOOK=dual_gl
#export BPR_PROTO_SCOPE=global
export BPR_PROTO=geomedian
export DCG_UNFREEZE=attn
export DCG_LR_SCALE=0.1
export TECH=bpr

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BALANCED=1
export EPOCHS

for m in $MODELS; do
    K="$K" bash train.sh b "$m" "$TAG"
done

for m in $MODELS; do
    K="$K" bash test.sh b "$m" "${TAG}_balanced"
done

python3 experiments/option_b_5fold/aggregate.py --tag "${TAG}_balanced"

echo ""
echo "[done] results: experiments/option_b_5fold/results/${TAG}_balanced/"
echo "       summary.csv : fold mean±std + pooled metrics"
echo "       compare with joint 5-fold:"
echo "         bash tools/bpr/compare_variants.sh b outputs/twostage5_vs_joint5.png \\"
echo "           joint:bpr_dual_adv_balanced \\"
echo "           two_stage:${TAG}_balanced"
