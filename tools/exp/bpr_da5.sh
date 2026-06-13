#!/usr/bin/env bash
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

TAG="bpr_dual_adv"

echo "============================================================"
echo " [recipe] dual_gl BPR + adversarial 5-fold CV"
echo " TAG = $TAG"
echo "============================================================"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BPR_ADV=1
export BPR_LAMBDA=0.1 # 
export BPR_HOOK=dual_gl
export BALANCED=1
export BPR_PROTO_SCOPE=global
export BPR_PROTO=geomedian
export DCG_UNFREEZE=attn
export DCG_LR_SCALE=0.1
export TECH=bpr

for m in diffmicv2; do
    bash train.sh b "$m" "$TAG"
done

for m in diffmicv2; do
    bash test.sh b "$m" "${TAG}_balanced"
done

python3 experiments/option_b_5fold/aggregate.py --tag "${TAG}_balanced"
python3 experiments/option_b_5fold/plot_5fold_bar.py --tag "${TAG}_balanced"

echo ""
echo "[done] results: experiments/option_b_5fold/results/${TAG}_balanced/"
echo "       compare: python3 experiments/option_b_5fold/compare.py \\"
echo "                --tag-a baseline_balanced --tag-b ${TAG}_balanced"