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
K="${K:-5}"
MODELS="${MODELS:-medvit}"
GW="${BPR_GLOBAL_W:-1.0}"
DW="${BPR_DENSE_W:-0.5}"

echo "============================================================"
echo " [recipe] BPR Global + Dense 5-fold CV"
echo "    bpr = ${GW}·global  +  ${DW}·dense   (K=$K)"
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

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for m in $MODELS; do
    echo ""
    echo "############################################################"
    echo "###  MODEL: $m  —  train 5-fold → test 5-fold → aggregate"
    echo "############################################################"

    K="$K" bash train.sh b "$m" "$TAG"
    K="$K" bash test.sh  b "$m" "${TAG}_balanced"

    echo ""
    echo "===== [$m]  5-fold AGGREGATE on test set ====="
    _MODEL_CSV="experiments/option_b_5fold/results/${TAG}_balanced/summary_${m}.csv"
    python3 experiments/option_b_5fold/aggregate.py \
        --tag "${TAG}_balanced" --models "$m" \
        --out-csv "$_MODEL_CSV"
    echo "  → per-model CSV: $_MODEL_CSV"
done

if [ "$(echo $MODELS | wc -w)" -gt 1 ]; then
    echo ""
    echo "===== ALL MODELS  combined summary ====="
    python3 experiments/option_b_5fold/aggregate.py \
        --tag "${TAG}_balanced" --models $MODELS \
        --out-csv "experiments/option_b_5fold/results/${TAG}_balanced/summary.csv"
fi

python3 experiments/option_b_5fold/plot_aggregate.py \
    --tag "${TAG}_balanced" --models $MODELS \
    --out "experiments/option_b_5fold/results/${TAG}_balanced/fold_meanstd.png" \
    || echo "[plot] plot_aggregate.py failed — check matplotlib"

echo ""
echo "[done] results: experiments/option_b_5fold/results/${TAG}_balanced/"
echo "       summary_<m>.csv : fold mean±std + pooled per model"
echo "       fold_meanstd.png: 5-fold mean±std bar chart"
