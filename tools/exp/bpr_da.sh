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

TAG="bpr_dual_adv"

echo "============================================================"
echo " [recipe] dual_gl BPR + adversarial + warmup 10 (joint, λ=0.3)"
echo " TAG = $TAG  (BALANCED)"
echo "============================================================"

export BPR_ADV=1
export BPR_LAMBDA=0.1
export BPR_HOOK=dual_gl
export BALANCED=1
export BPR_PROTO_SCOPE=global
export BPR_PROTO=geomedian
export DCG_UNFREEZE=attn
export DCG_LR_SCALE=0.1
export TECH=bpr

TAG="bpr_dual_adv"
for m in medvit diffmic diffmicv2; do
    bash train.sh a "$m" "$TAG"
done

for m in medvit diffmic diffmicv2; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/$TAG/"
echo "       compare adv on/off:"
echo "         bash tools/bpr/compare_variants.sh a outputs/adv_ablation.png \\"
echo "           baseline:baseline \\"
echo "           no_adv:bpr_dual_warmup10 \\"
echo "           with_adv:$TAG"
