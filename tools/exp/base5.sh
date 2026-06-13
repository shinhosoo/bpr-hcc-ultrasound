#!/usr/bin/env bash
# Baseline 5-fold CV — no BPR, no adversarial, no DCG hook.
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${PROJECT_ROOT:-$HERE}"
while [ ! -f "$ROOT/train.sh" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/train.sh" ] || { echo "[error] train.sh not found"; exit 1; }
cd "$ROOT"

TAG="baseline"

echo "============================================================"
echo " [recipe] baseline 5-fold CV (BALANCED, no BPR)"
echo " TAG = $TAG"
echo "============================================================"

unset TECH
unset BPR_ADV BPR_LAMBDA BPR_HOOK BPR_MODE
unset BPR_PROTO BPR_PROTO_SCOPE BPR_PROTO_REFRESH BPR_PROTO_BS BPR_PROTO_EMA
unset BPR_BUFFER_SIZE BPR_PROJ_DIM BPR_PROJ_HIDDEN
unset BPR_WARMUP_EPOCHS BPR_T_MAX BPR_MIN_ACTIVE
unset BPR_BN_DIM BPR_BN_SKIP BPR_LOCAL_POOL
unset BPR_STAGE BPR_STAGE2_CKPT BPR_STAGE2_DIFF_W BPR_STAGE2_LR_SCALE BPR_STAGE2_FROM
unset DCG_UNFREEZE DCG_LR_SCALE DCG_WARMUP

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BALANCED=1

for m in medvit diffmic diffmicv2; do
    bash train.sh b "$m" "$TAG"
done

for m in medvit diffmic diffmicv2; do
    bash test.sh b "$m" "${TAG}_balanced"
done

python3 experiments/option_b_5fold/aggregate.py --tag "${TAG}_balanced"

echo ""
echo "[done] results: experiments/option_b_5fold/results/${TAG}_balanced/"
echo "       summary.csv : fold mean±std + pooled metrics"