#!/usr/bin/env bash
die() { echo "[train_hsq] ERROR: $*" >&2; exit 1; }

FOLD="${1:-}"
OUT="${2:-}"
[ -n "$FOLD" ] || die "fold index required. Usage: bash tools/train_hsq.sh <fold_index> <out_dir>"
[ -n "$OUT"  ] || die "out_dir required."

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" || die "failed to resolve script path"
ROOT="$( cd "$HERE/.." && pwd )"                          || die "failed to resolve ROOT"
HSQ_DIR="$ROOT/models/HSQ"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-32}"
PATIENCE="${PATIENCE:-40}"
BASE="${BASE:-0}"
export BPR_AUX="${BPR_AUX:-0}"
export BPR_BN_DIM="${BPR_BN_DIM:-128}"
export BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
export BPR_WARMUP="${BPR_WARMUP:-5}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_NUM_CLASSES="${BPR_NUM_CLASSES:-2}"
HSQ_BPR="${HSQ_BPR:-0}"
export HSQ_PRETRAIN="${HSQ_PRETRAIN:-0}"

echo "[train_hsq] ============================================"
echo "[train_hsq] fold=$FOLD  seed=$SEED  epochs=$EPOCHS  bs=$BS"
echo "[train_hsq] ROOT=$ROOT"
echo "[train_hsq] HSQ_DIR=$HSQ_DIR"
echo "[train_hsq] OUT=$OUT"

[ -d "$HSQ_DIR" ] || die "HSQ directory not found: $HSQ_DIR"

mkdir -p "$OUT" || die "failed to create output directory: $OUT"
OUT_ABS="$(cd "$OUT" && pwd)" || die "failed to resolve absolute path for OUT: $OUT"
echo "[train_hsq] output → $OUT_ABS"

if [ "$HSQ_BPR" = "1" ]; then
    RUNNER="$HERE/run_hsq_bpr.py"
else
    RUNNER="$HERE/run_hsq.py"
fi
[ -f "$RUNNER" ] || die "runner not found: $RUNNER"

FOLD_DATA="$ROOT/${DATA_ROOT:-data}/5fold/fold_${FOLD}/imagefolder"
[ -d "$FOLD_DATA/train" ] || die "train folder not found: $FOLD_DATA/train"
[ -d "$FOLD_DATA/val"   ] || die "val folder not found: $FOLD_DATA/val"

TRAIN_ABS="$(cd "$FOLD_DATA/train" && pwd)" || die "failed to resolve train path"
VAL_ABS="$(  cd "$FOLD_DATA/val"   && pwd)" || die "failed to resolve val path"
if [ -d "$FOLD_DATA/test" ]; then
    TEST_OPT="--test-path $(cd "$FOLD_DATA/test" && pwd)"
else
    TEST_OPT=""
    echo "[train_hsq] test folder not found — falling back to val"
fi

echo "[train_hsq] standard: train=$TRAIN_ABS"
python3 "$RUNNER" \
    --train-path "$TRAIN_ABS" \
    --val-path   "$VAL_ABS" \
    $TEST_OPT \
    --output-dir "$OUT_ABS" \
    --seed       "$SEED" \
    --epochs     "$EPOCHS" \
    --patience   "$PATIENCE" \
    ${BASE:+--base} \
    --batch-size "$BS" \
    --fold-index "$FOLD" \
    || die "run_hsq.py failed (fold=$FOLD)"

echo "[train_hsq] fold $FOLD done — ckpt: $OUT_ABS/checkpoints/"
