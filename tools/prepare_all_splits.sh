#!/usr/bin/env bash
# Master preparation: creates both shared 3-way and 5-fold dataset folders.
# Idempotent — safe to re-run.
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

mkdir -p "$ROOT/data"

# 3-way split (shared by Option A and Option C)
if [ -f "$ROOT/data/3way/pkl/lesion_train.pkl" ] && \
   [ -f "$ROOT/data/3way/pkl/lesion_val.pkl" ] && \
   [ -f "$ROOT/data/3way/pkl/lesion_test.pkl" ]; then
    echo "[prepare_all] 3-way data already exists at $ROOT/data/3way — skipping."
else
    python3 "$ROOT/tools/prepare_binary_3way.py" \
        --source "$ROOT/image" --out "$ROOT/data/3way" \
        --val-ratio 0.15 --test-ratio 0.15 --seed 42
fi

# 5-fold split (Option B)
if [ -f "$ROOT/data/5fold/fold_0/pkl/lesion_train.pkl" ]; then
    echo "[prepare_all] 5-fold data already exists at $ROOT/data/5fold — skipping."
else
    python3 "$ROOT/tools/prepare_5fold.py" \
        --source "$ROOT/image" --out "$ROOT/data/5fold" \
        --k 5 --inner-val-ratio 0.15 --seed 42
fi

echo ""
echo "[prepare_all] done. Shared data under: $ROOT/data/"
echo "  - $ROOT/data/3way    (Option A, C)"
echo "  - $ROOT/data/5fold   (Option B)"
