#!/usr/bin/env bash
set -e
TAG="${1:-bpr_diffmicv2_dual}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

export TECH=bpr
export BPR_HOOK=dual2ch
export BPR_DCLS="${BPR_DCLS:-128}"
export BPR_DUAL_DETACH="${BPR_DUAL_DETACH:-1}"
export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-mean}"
export BPR_PROTO_SCOPE="${BPR_PROTO_SCOPE:-global}"
export BPR_TWO_PHASE="${BPR_TWO_PHASE:-1}"
export BPR_PHASE1_EPOCHS="${BPR_PHASE1_EPOCHS:-50}"
export BPR_PHASE2_LR_SCALE="${BPR_PHASE2_LR_SCALE:-1}"
export PATIENCE="${PATIENCE:-40}"

echo "============================================================"
echo " [diffmicv2 dual2ch]  tag=$TAG"
echo "   BPR_DCLS=$BPR_DCLS  DUAL_DETACH=$BPR_DUAL_DETACH  LAMBDA=$BPR_LAMBDA"
echo "   PROTO=$BPR_PROTO/$BPR_PROTO_SCOPE  TWO_PHASE=$BPR_TWO_PHASE  PATIENCE=$PATIENCE"
echo "============================================================"

ONLY="${STAGE_ONLY:-all}"

if [ "$ONLY" = "all" ] || [ "$ONLY" = "train" ]; then
    echo "[dual] === TRAIN ==="
    bash "$ROOT/train.sh" b diffmicv2 "$TAG"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "test" ]; then
    echo "[dual] === TEST ==="
    bash "$ROOT/test.sh" b diffmicv2 "$TAG"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "viz" ]; then
    echo "[dual] === VIZ ==="
    bash "$ROOT/viz.sh" b "$TAG"
fi

echo ""
echo "[dual] DONE  tag=$TAG"
