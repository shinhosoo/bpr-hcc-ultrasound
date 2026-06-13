#!/usr/bin/env bash
set -e
SRC_TAG="${1:?usage: bash tools/refine_5fold.sh <src_tag> <out_tag>}"
OUT_TAG="${2:?usage: bash tools/refine_5fold.sh <src_tag> <out_tag>}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

K="${K:-5}"
LAM="${LAM:-0.3}"
PROTO="${PROTO:-mean}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
BS_REFINE="${BS_REFINE:-64}"
EXTRACT_BS="${EXTRACT_BS:-32}"
ADV_FLAG=""; [ "${USE_ADV:-0}" = "1" ] && ADV_FLAG="--use-adv"

CFG_BASE="$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml"
RESULTS="$ROOT/experiments/option_b_5fold/results"
FEAT_ROOT="$RESULTS/${OUT_TAG}_feats"

find_lightning_ckpt () {  # <run-dir-for-one-model-fold>
    local DIR="$1/lightning_logs"
    local CANDS=()
    CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null))
    CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null))
    local f SIZE
    for f in "${CANDS[@]}"; do
        [ -f "$f" ] || continue
        SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        if [ "$SIZE" -ge 1048576 ]; then echo "$f"; return; fi
    done
}

echo "[refine_5fold] src=$SRC_TAG  out=${OUT_TAG}_{bpr,nobpr}  K=$K  LAM=$LAM  ADV=${USE_ADV:-0}  PROTO=$PROTO"

for ((i=0; i<K; i++)); do
    echo ""; echo "########## REFINE  fold $i ##########"
    SRC_DIR="$RESULTS/$SRC_TAG/diffmicv2/fold_$i"
    CKPT="$(find_lightning_ckpt "$SRC_DIR")"
    if [ -z "$CKPT" ]; then
        echo "[refine_5fold] WARN: ckpt not found ($SRC_DIR/lightning_logs) — fold $i skip"; continue
    fi
    echo "[refine_5fold] ckpt: $CKPT"

    D="$ROOT/data/5fold/fold_$i/pkl"
    TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"; TE="$D/lesion_test.pkl"
    for f in "$TR" "$VA" "$TE"; do
        [ -f "$f" ] || { echo "[refine_5fold] ERROR: pkl 없음: $f"; exit 1; }
    done

    FDIR="$FEAT_ROOT/fold_$i"; mkdir -p "$FDIR"
    CFG="$FDIR/extract_config.yml"
    python3 "$ROOT/tools/diffmic_config_swap.py" --in "$CFG_BASE" --out "$CFG" --traindata "$TR" --testdata "$TR"

    python3 "$ROOT/tools/extract_features.py" --model diffmicv2 --ckpt "$CKPT" --config "$CFG" --data "$TR" --out "$FDIR/train.npz" --batch-size "$EXTRACT_BS"
    python3 "$ROOT/tools/extract_features.py" --model diffmicv2 --ckpt "$CKPT" --config "$CFG" --data "$VA" --out "$FDIR/val.npz"   --batch-size "$EXTRACT_BS"
    python3 "$ROOT/tools/extract_features.py" --model diffmicv2 --ckpt "$CKPT" --config "$CFG" --data "$TE" --out "$FDIR/test.npz"  --batch-size "$EXTRACT_BS"

    OB="$RESULTS/${OUT_TAG}_bpr/diffmicv2/fold_$i"; mkdir -p "$OB"
    python3 "$ROOT/tools/bpr/refine_head.py" \
        --features-train "$FDIR/train.npz" --features-val "$FDIR/val.npz" --features-test "$FDIR/test.npz" \
        --out "$OB/predictions_test_diffmicv2.npz" \
        --lam "$LAM" $ADV_FLAG --prototype "$PROTO" --num-classes 2 \
        --epochs "$EPOCHS" --seed "$SEED" --batch-size "$BS_REFINE"

    ON="$RESULTS/${OUT_TAG}_nobpr/diffmicv2/fold_$i"; mkdir -p "$ON"
    python3 "$ROOT/tools/bpr/refine_head.py" \
        --features-train "$FDIR/train.npz" --features-val "$FDIR/val.npz" --features-test "$FDIR/test.npz" \
        --out "$ON/predictions_test_diffmicv2.npz" \
        --lam "$LAM" --no-bpr --prototype "$PROTO" --num-classes 2 \
        --epochs "$EPOCHS" --seed "$SEED" --batch-size "$BS_REFINE"
done

echo ""
echo "[refine_5fold] DONE."
echo "  aggregate:"
echo "    bash viz.sh b ${OUT_TAG}_bpr"
echo "    bash viz.sh b ${OUT_TAG}_nobpr"
