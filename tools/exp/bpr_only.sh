#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

MODE="${MODE:-ssl}"  # ssl | joint
EPOCHS="${EPOCHS:-100}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-$((EPOCHS / 2))}"
TAG="${TAG:-bpr_only_${MODE}}"
MODELS="${MODELS:-medvit}"

echo "============================================================"
echo " [recipe] BPR-only learning  (MODE=$MODE)"
if [ "$MODE" = "ssl" ]; then
    echo "          Stage1: BPR only (representation)"
    echo "          Stage2: CE only  (linear probe, backbone frozen)"
    echo "          STAGE1_EPOCHS=$STAGE1_EPOCHS / total $EPOCHS"
else
    echo "          BPR-only through all epochs — classifier not trained (val acc meaningless)"
fi
echo " TAG = $TAG  MODELS=$MODELS"
echo "============================================================"

export BPR_ONLY=1
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
export EPOCHS

if [ "$MODE" = "ssl" ]; then
    export BPR_TWO_STAGE=1
    export BPR_STAGE1_EPOCHS="$STAGE1_EPOCHS"
else
    unset BPR_TWO_STAGE BPR_STAGE1_EPOCHS
fi

for m in $MODELS; do
    bash train.sh a "$m" "$TAG"
done

for m in $MODELS; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/${TAG}_balanced/"
if [ "$MODE" = "joint" ]; then
    echo "  Note: joint BPR-only does not train the classifier; val acc may be random-level."
fi
