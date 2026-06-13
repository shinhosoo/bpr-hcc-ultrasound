#!/usr/bin/env bash
set -e
OPT="${1:?usage: bash tools/bpr/run_bpr.sh <a|b|c> [model] [tag]}"
MODEL="${2:-all}"
TAG="${3:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"

BPR_LAMBDA="${BPR_LAMBDA:-0.3}"
BPR_EPOCHS="${BPR_EPOCHS:-100}"
BPR_LR="${BPR_LR:-0.001}"
BPR_USE_ADV="${BPR_USE_ADV:-0}"
NUM_CLASSES="${NUM_CLASSES:-2}"

case "$OPT" in
    a) R="$ROOT/experiments/option_a_3way/results/$TAG"; DATA="$ROOT/data/3way";;
    b) R="$ROOT/experiments/option_b_5fold/results/$TAG/fold_0"; DATA="$ROOT/data/5fold/fold_0";;
    c) R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG/seed_42"; DATA="$ROOT/data/3way";;
    *) echo "unknown opt: $OPT"; exit 1;;
esac
FEAT_DIR="$R/bpr_features"
mkdir -p "$FEAT_DIR"

want() { [ "$MODEL" = all ] || [ "$MODEL" = "$1" ]; }
ADV_FLAG=""; [ "$BPR_USE_ADV" = "1" ] && ADV_FLAG="--use-adv"
PROTO_FLAG="--prototype ${BPR_PROTO:-mean}"

do_refine_imagefolder () {  # <model_name> <ckpt_path> <train_dir> <val_dir> <test_dir>
    local NAME="$1" CKPT="$2" TR_DIR="$3" VA_DIR="$4" TE_DIR="$5"
    [ -f "$CKPT" ] || { echo "[bpr] WARN: $NAME ckpt not found — skip"; return; }
    echo ""; echo "===== $NAME BPR refine ====="
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$TR_DIR" --out "$FEAT_DIR/feats_${NAME}_train.npz"
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$VA_DIR" --out "$FEAT_DIR/feats_${NAME}_val.npz"
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$TE_DIR" --out "$FEAT_DIR/feats_${NAME}_test.npz"
    python3 "$ROOT/tools/bpr/refine_head.py" \
        --features-train "$FEAT_DIR/feats_${NAME}_train.npz" \
        --features-val   "$FEAT_DIR/feats_${NAME}_val.npz" \
        --features-test  "$FEAT_DIR/feats_${NAME}_test.npz" \
        --out "$R/predictions_test_${NAME}_bpr.npz" \
        --lam "$BPR_LAMBDA" --epochs "$BPR_EPOCHS" --lr "$BPR_LR" \
        --num-classes "$NUM_CLASSES" $ADV_FLAG $PROTO_FLAG
}

do_refine_pkl () {  # <model_name> <ckpt_arg> <train_pkl> <val_pkl> <test_pkl> [config]
    local NAME="$1" CKPT="$2" TR_PKL="$3" VA_PKL="$4" TE_PKL="$5" CFG="${6:-}"
    echo ""; echo "===== $NAME BPR refine ====="
    local EXTRA=""
    [ -n "$CFG" ] && EXTRA="--config $CFG"
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$TR_PKL" --out "$FEAT_DIR/feats_${NAME}_train.npz" $EXTRA
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$VA_PKL" --out "$FEAT_DIR/feats_${NAME}_val.npz" $EXTRA
    python3 "$ROOT/tools/extract_features.py" --model "$NAME" \
        --ckpt "$CKPT" --data "$TE_PKL" --out "$FEAT_DIR/feats_${NAME}_test.npz" $EXTRA
    python3 "$ROOT/tools/bpr/refine_head.py" \
        --features-train "$FEAT_DIR/feats_${NAME}_train.npz" \
        --features-val   "$FEAT_DIR/feats_${NAME}_val.npz" \
        --features-test  "$FEAT_DIR/feats_${NAME}_test.npz" \
        --out "$R/predictions_test_${NAME}_bpr.npz" \
        --lam "$BPR_LAMBDA" --epochs "$BPR_EPOCHS" --lr "$BPR_LR" \
        --num-classes "$NUM_CLASSES" $ADV_FLAG $PROTO_FLAG
}

if want medvit; then
    do_refine_imagefolder medvit "$R/medvit/checkpoint_best.pth" \
        "$DATA/imagefolder/train" "$DATA/imagefolder/val" "$DATA/imagefolder/test"
fi

if want medvitv2 && [ -d "$R/medvitv2" ]; then
    echo "[bpr] medvitv2 not yet supported in extract_features.py — skip"
fi

if want diffmic; then
    do_refine_pkl diffmic "$R/diffmic" \
        "$DATA/pkl/lesion_train.pkl" "$DATA/pkl/lesion_val.pkl" "$DATA/pkl/lesion_test.pkl"
fi

if want diffmicv2; then
    DMV2_CKPT=$(ls -t "$R"/diffmicv2/lightning_logs/*/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -n "$DMV2_CKPT" ]; then
        do_refine_pkl diffmicv2 "$DMV2_CKPT" \
            "$DATA/pkl/lesion_train.pkl" "$DATA/pkl/lesion_val.pkl" "$DATA/pkl/lesion_test.pkl" \
            "$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml"
    fi
fi

echo ""
echo "[bpr] DONE  → $R/predictions_test_<model>_bpr.npz"
echo "compare: bash tools/bpr/compare_bar.sh $OPT $TAG"
