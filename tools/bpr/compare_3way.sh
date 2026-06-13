#!/usr/bin/env bash
set -e
OPT="${1:?usage: bash tools/bpr/compare_3way.sh <a|b|c> [baseline_tag] [1stage_tag]}"
BASE_TAG="${2:-baseline}"
ONE_TAG="${3:-bpr1stage}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"
case "$OPT" in
    a) BASE="$ROOT/experiments/option_a_3way/results/$BASE_TAG"
       ONE="$ROOT/experiments/option_a_3way/results/$ONE_TAG";;
    b) BASE="$ROOT/experiments/option_b_5fold/results/$BASE_TAG/fold_0"
       ONE="$ROOT/experiments/option_b_5fold/results/$ONE_TAG/fold_0";;
    c) BASE="$ROOT/experiments/option_c_3way_multiseed/results/$BASE_TAG/seed_42"
       ONE="$ROOT/experiments/option_c_3way_multiseed/results/$ONE_TAG/seed_42";;
    *) echo "unknown opt: $OPT"; exit 1;;
esac

OUT="$ROOT/outputs/bpr_3way_${BASE_TAG}_vs_${ONE_TAG}.png"
python3 "$ROOT/tools/bpr/compare_3way.py" \
    --baseline-root  "$BASE" \
    --bpr1stage-root "$ONE" \
    --out "$OUT"
echo ""
echo "→ $OUT"
echo "→ ${OUT%.*}.csv"
