#!/usr/bin/env bash
set -e
TAG="${1:-bpr_diffmicv2}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

export TECH=bpr
export BPR_HOOK=xweight
export BPR_VICREG=0
export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-global}"
export BPR_TWO_PHASE="${BPR_TWO_PHASE:-1}"
export BPR_PHASE1_EPOCHS="${BPR_PHASE1_EPOCHS:-50}"
export PATIENCE="${PATIENCE:-40}"

echo "============================================================"
echo " [diffmicv2 bpr]  tag=$TAG"
echo "   BPR_LAMBDA=$BPR_LAMBDA  proto=$BPR_PROTO/$BPR_PROTO_SCOPE  two_phase=$BPR_TWO_PHASE"
echo "============================================================"

ONLY="${STAGE_ONLY:-all}"
[ "$ONLY" = "all" ] || [ "$ONLY" = "train" ] && { echo "[bpr] TRAIN"; bash "$ROOT/train.sh" b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "test" ]  && { echo "[bpr] TEST";  bash "$ROOT/test.sh" b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "viz" ]   && { echo "[bpr] VIZ";   bash "$ROOT/viz.sh"  b "$TAG"; }
echo "[bpr] DONE tag=$TAG"
