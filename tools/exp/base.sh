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

TAG="baseline"

echo "============================================================"
echo " [recipe] baseline — no BPR, no adversarial, no DCG hook"
echo " TAG = $TAG  (BALANCED sampler)"
echo "============================================================"

unset TECH
unset BPR_ADV BPR_LAMBDA BPR_HOOK BPR_MODE
unset BPR_PROTO BPR_PROTO_SCOPE BPR_PROTO_REFRESH BPR_PROTO_BS BPR_PROTO_EMA
unset BPR_BUFFER_SIZE BPR_PROJ_DIM BPR_PROJ_HIDDEN
unset BPR_WARMUP_EPOCHS BPR_T_MAX BPR_MIN_ACTIVE
unset BPR_HOOK BPR_BN_DIM BPR_BN_SKIP BPR_LOCAL_POOL
unset BPR_STAGE BPR_STAGE2_CKPT BPR_STAGE2_DIFF_W BPR_STAGE2_LR_SCALE BPR_STAGE2_FROM
unset DCG_UNFREEZE DCG_LR_SCALE DCG_WARMUP

export BALANCED=1

for m in medvit diffmic diffmicv2; do
    bash train.sh a "$m" "$TAG"
done

for m in medvit diffmic diffmicv2; do
    bash test.sh a "$m" "${TAG}_balanced"
done

echo ""
echo "[done] results: experiments/option_a_3way/results/${TAG}_balanced/"
echo "       compare: bash tools/bpr/compare_variants.sh a outputs/baseline_vs_bpr.png \\"
echo "                baseline:${TAG}_balanced \\"
echo "                with_bpr:bpr_dual_adv_balanced"