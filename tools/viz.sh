#!/usr/bin/env bash
set -e
OUT="${1:?usage: bash viz.sh <out_dir> <npz>:<name> [<npz>:<name> ...]}"; shift
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
ARGS=()
for arg in "$@"; do
    NPZ="${arg%%:*}"; NAME="${arg##*:}"
    ARGS+=(--pred "$NPZ" --name "$NAME")
done
mkdir -p "$OUT"
python3 "$ROOT/tools/visualize_results.py" "${ARGS[@]}" --out "$OUT"
python3 "$ROOT/tools/unified_eval.py" "${ARGS[@]}" --out "$OUT/metrics.csv"
echo "[viz] $OUT/{roc,pr,confusion,metrics_bar}.png + metrics.csv"
