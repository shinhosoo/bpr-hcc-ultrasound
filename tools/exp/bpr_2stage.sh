#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
if [ ! -f "$ROOT/train.sh" ]; then
    echo "[error] train.sh not found. Run from the project root or set PROJECT_ROOT=/path"
    exit 1
fi
cd "$ROOT"

TAG="${TAG:-bpr_2stage}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
EPOCHS="${EPOCHS:-100}"
MODELS="${MODELS:-medvit diffmic diffmicv2}"

echo "============================================================"
echo " [recipe] BPR 2-stage  (Stage1=$STAGE1_EPOCHS / total=$EPOCHS)"
echo " TAG = $TAG  (BALANCED)  MODELS = $MODELS"
echo "============================================================"

export BPR_TWO_STAGE=1
export BPR_STAGE1_EPOCHS="$STAGE1_EPOCHS"

export BPR_ADV=1
export BPR_LAMBDA=0.1
export BPR_HOOK=dual_gl
export BPR_PROTO_SCOPE=global
export BPR_PROTO=geomedian
export DCG_UNFREEZE=attn
export DCG_LR_SCALE=0.1
export TECH=bpr
export BALANCED=1

export EPOCHS

for m in $MODELS; do
    bash train.sh a "$m" "$TAG"
done

for m in $MODELS; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/${TAG}_balanced/"
echo "       compare with joint (bpr_dual_adv):"
echo "         bash tools/bpr/compare_variants.sh a outputs/twostage_vs_joint.png \\"
echo "           joint:bpr_dual_adv_balanced \\"
echo "           two_stage:${TAG}_balanced"
