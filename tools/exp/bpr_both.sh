#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

TAG="${TAG:-bpr_both}"
EPOCHS="${EPOCHS:-100}"
MODELS="${MODELS:-medvit}"
GW="${BPR_GLOBAL_W:-1.0}"
DW="${BPR_DENSE_W:-0.5}"

echo "============================================================"
echo " [recipe] BPR Global + Dense  (both)"
echo "    bpr = ${GW}·global  +  ${DW}·dense"
echo " TAG = $TAG  EPOCHS=$EPOCHS  MODELS=$MODELS"
echo "============================================================"

export BPR_DENSE=both
export BPR_GLOBAL_W="$GW"
export BPR_DENSE_W="$DW"

export BPR_LAMBDA="${BPR_LAMBDA:-0.1}"
export BPR_WARMUP_EPOCHS="${BPR_WARMUP_EPOCHS:-5}"
export BPR_USE_PROJ=0
export BPR_ADV="${BPR_ADV:-1}"
export BPR_ADV_STEPS="${BPR_ADV_STEPS:-1}"
export BPR_MODE=joint
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_PROTO_SCOPE=batch
export BPR_NUM_CLASSES=2
export TECH=bpr
export BALANCED=1
export EPOCHS

: "${BPR_TWO_STAGE:=0}"
: "${BPR_STAGE1_EPOCHS:=-1}"
export BPR_TWO_STAGE BPR_STAGE1_EPOCHS

for m in $MODELS; do
    bash train.sh a "$m" "$TAG"
done

for m in $MODELS; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/${TAG}_balanced/"
echo "       compare global-only / dense-only:"
echo "         bash tools/bpr/compare_variants.sh a outputs/both_vs_each.png \\"
echo "           global:bpr_dual_adv_balanced \\"
echo "           dense:bpr_dense_balanced \\"
echo "           both:${TAG}_balanced"
