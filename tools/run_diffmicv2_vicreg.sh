#!/usr/bin/env bash
set -e
TAG="${1:-bpr_diffmicv2_vicreg}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

export TECH=bpr
export BPR_HOOK=xweight
export BPR_VICREG=1
export VICREG_VAR_W="${VICREG_VAR_W:-1.0}"
export VICREG_COV_W="${VICREG_COV_W:-0.0}"
export VICREG_GAMMA="${VICREG_GAMMA:-1.0}"
export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-global}"
export BPR_TWO_PHASE="${BPR_TWO_PHASE:-1}"
export BPR_PHASE1_EPOCHS="${BPR_PHASE1_EPOCHS:-50}"
export PATIENCE="${PATIENCE:-40}"

echo "============================================================"
echo " [diffmicv2 vicreg]  tag=$TAG"
echo "   var_w=$VICREG_VAR_W  cov_w=$VICREG_COV_W  gamma=$VICREG_GAMMA"
echo "   BPR_LAMBDA=$BPR_LAMBDA  proto=$BPR_PROTO/$BPR_PROTO_SCOPE  two_phase=$BPR_TWO_PHASE"
echo "============================================================"

ONLY="${STAGE_ONLY:-all}"
[ "$ONLY" = "all" ] || [ "$ONLY" = "train" ] && { echo "[vicreg] TRAIN"; bash "$ROOT/train.sh" b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "test" ]  && { echo "[vicreg] TEST"; bash "$ROOT/test.sh" b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "viz" ]   && { echo "[vicreg] VIZ";  bash "$ROOT/viz.sh"  b "$TAG"; }
echo "[vicreg] DONE tag=$TAG"
