#!/usr/bin/env bash
# Option B — prepare shared 5-fold data at test 2/data/5fold.
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"
SHARED="$ROOT/data/5fold"

if [ -f "$SHARED/fold_0/pkl/lesion_train.pkl" ]; then
    echo "[option_b/prepare] shared 5-fold data already exists at $SHARED — skipping."
    exit 0
fi

python3 "$ROOT/tools/prepare_5fold.py" \
    --source "$ROOT/image" --out "$SHARED" \
    --k 5 --inner-val-ratio 0.15 --seed 42
echo "[option_b/prepare] shared 5-fold data written to $SHARED"
