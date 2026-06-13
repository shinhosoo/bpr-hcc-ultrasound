#!/usr/bin/env bash
# MedViT 5-fold: train one fold, immediately test that fold, save metrics CSV.
# Usage:
#   bash tools/train_test_medvit_5fold.sh [tag]
set -e

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
cd "$ROOT"

TAG="${1:-baseline}"
K="${K:-5}"

# Match train.sh/test.sh tag behavior for BALANCED=1.
if [ "${BALANCED:-0}" = "1" ]; then
    _tag_lc="$(echo "$TAG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_tag_lc" != *bal* ]]; then
        TAG="${TAG}_balanced"
        echo "[train_test_medvit_5fold] BALANCED=1 — TAG -> $TAG"
    else
        echo "[train_test_medvit_5fold] BALANCED=1 — TAG='$TAG' already includes bal"
    fi
fi

bash "$ROOT/experiments/option_b_5fold/prepare.sh"

for ((i=0; i<K; i++)); do
    DATA="$ROOT/data/5fold/fold_$i"
    RUN="$ROOT/experiments/option_b_5fold/results/$TAG/fold_$i"
    mkdir -p "$RUN"

    echo ""
    echo "########## FOLD $i / TRAIN MedViT ##########"
    if [ "${TECH:-}" = "bpr" ] || echo "$TAG" | grep -qi "bpr"; then
        SEED="${SEED:-$((42 + i))}" bash "$ROOT/tools/train_medvit_bpr.sh" \
            "$DATA/imagefolder/train" "$DATA/imagefolder/val" "$RUN/medvit"
    else
        SEED="${SEED:-$((42 + i))}" bash "$ROOT/tools/train_medvit.sh" \
            "$DATA/imagefolder/train" "$DATA/imagefolder/val" "$RUN/medvit"
    fi

    echo ""
    echo "########## FOLD $i / TEST MedViT ##########"
    bash "$ROOT/tools/test_medvit.sh" \
        "$RUN/medvit/checkpoint_best.pth" \
        "$DATA/imagefolder/test" \
        "$RUN/predictions_test_medvit.npz"

    echo ""
    echo "########## FOLD $i / METRICS ##########"
    python3 "$ROOT/tools/unified_eval.py" \
        --pred "$RUN/predictions_test_medvit.npz" --name MedViT \
        --out "$RUN/metrics_test.csv"

    mkdir -p "$RUN/viz"
    python3 "$ROOT/tools/visualize_results.py" \
        --pred "$RUN/predictions_test_medvit.npz" --name MedViT \
        --out "$RUN/viz" || echo "[viz] skipped or failed"

    echo "[fold $i done] CSV: $RUN/metrics_test.csv"
done

python3 "$ROOT/experiments/option_b_5fold/aggregate.py" \
    --tag "$TAG" --models medvit \
    --out-csv "$ROOT/experiments/option_b_5fold/results/$TAG/summary_medvit.csv"

echo ""
echo "[done] fold CSVs: experiments/option_b_5fold/results/$TAG/fold_*/metrics_test.csv"
echo "[done] summary : experiments/option_b_5fold/results/$TAG/summary_medvit.csv"
