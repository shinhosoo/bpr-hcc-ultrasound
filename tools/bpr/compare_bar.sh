#!/usr/bin/env bash
# baseline vs +BPR 모델별 메트릭 바 그래프 한 줄.
# Usage: bash tools/bpr/compare_bar.sh <option> [tag]
set -e
OPT="${1:?usage: bash tools/bpr/compare_bar.sh <a|b|c> [tag]}"
TAG="${2:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"
case "$OPT" in
    a) R="$ROOT/experiments/option_a_3way/results/$TAG";;
    b) R="$ROOT/experiments/option_b_5fold/results/$TAG/fold_0";;
    c) R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG/seed_42";;
    *) echo "unknown opt: $OPT"; exit 1;;
esac
OUT="$R/bpr_compare_bar.png"
python3 "$ROOT/tools/bpr/compare_bar.py" \
    --results-root "$R" \
    --out "$OUT"
echo ""
echo "→ $OUT  +  $(basename "${OUT%.*}".csv)"
