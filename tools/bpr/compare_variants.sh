#!/usr/bin/env bash
set -e
OPT="${1:?usage: bash tools/bpr/compare_variants.sh <a|b|c> <out.png> <tag...>}"
OUT="${2:?out.png path required}"
shift 2
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"
case "$OPT" in
    a) BASE_ROOT="$ROOT/experiments/option_a_3way/results";;
    b) BASE_ROOT="$ROOT/experiments/option_b_5fold/results";;
    c) BASE_ROOT="$ROOT/experiments/option_c_3way_multiseed/results";;
    *) echo "unknown opt: $OPT"; exit 1;;
esac

TAGS=()
for t in "$@"; do TAGS+=( --tag "$t" ); done

python3 "$ROOT/tools/bpr/compare_variants.py" \
    --root "$BASE_ROOT" \
    "${TAGS[@]}" \
    --out "$OUT"
echo ""
echo "→ $OUT"
echo "→ ${OUT%.*}.csv"
