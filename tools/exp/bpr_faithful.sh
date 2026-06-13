#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

TAG="${TAG:-bpr_faithful}"
EPOCHS="${EPOCHS:-100}"
MODELS="${MODELS:-medvit}"

echo "============================================================"
echo " [recipe] BPR HSQ-faithful  (joint, λ=1, no normalize, 2-step PGD)"
echo " TAG = $TAG  EPOCHS=$EPOCHS  MODELS=$MODELS"
echo "============================================================"

export BPR_FAITHFUL=1
export BPR_LAMBDA=1.0
export BPR_WARMUP_EPOCHS=0
export BPR_USE_PROJ=0
export BPR_ADV=1
export BPR_ADV_STEPS=2
export BPR_MODE=joint
export BPR_PROTO=mean
export BPR_PROTO_SCOPE=batch
export BPR_NUM_CLASSES=2
export TECH=bpr
export BALANCED=1

unset BPR_TWO_STAGE BPR_STAGE1_EPOCHS

export EPOCHS

for m in $MODELS; do
    bash train.sh a "$m" "$TAG"
done

for m in $MODELS; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/${TAG}_balanced/"
echo "       compare with current variant (joint, λ=0.1, normalize, 1-step):"
echo "         bash tools/bpr/compare_variants.sh a outputs/faithful_vs_current.png \\"
echo "           current:bpr_dual_adv_balanced \\"
echo "           faithful:${TAG}_balanced"
