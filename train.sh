#!/usr/bin/env bash
set -e

# === Determinism: cuBLAS workspace config (required when torch.use_deterministic_algorithms(True) is enabled) ===
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OPT="${1:?usage: bash train.sh <a|b|c> [all|medvit|medvitv2|diffmic|diffmicv2|diffmicv2_sam] [tag]}"
MODEL="${2:-all}"
TAG="${3:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$HERE"

if [ "${BALANCED:-0}" = "1" ]; then
    _tag_lc="$(echo "$TAG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_tag_lc" != *bal* ]]; then
        TAG="${TAG}_balanced"
        echo "[train.sh] BALANCED=1 — TAG → $TAG"
    else
        echo "[train.sh] BALANCED=1 — TAG='$TAG' already includes bal"
    fi
fi

want() { [ "$MODEL" = all ] || [ "$MODEL" = "$1" ]; }

train_one_3way () {  # <run-dir>
    local R="$1"
    local D="$ROOT/${DATA_ROOT:-data}/3way"
    mkdir -p "$R"
    local _bpr="0"
    if [ "${TECH:-}" = "bpr" ] || echo "${TAG}" | grep -qi "bpr"; then _bpr="1"; fi
    if [ "$_bpr" = "1" ]; then
        if want medvit;    then bash "$ROOT/tools/train_medvit_bpr.sh"    "$D/imagefolder/train" "$D/imagefolder/val" "$R/medvit"; fi
        if want diffmic;   then bash "$ROOT/tools/train_diffmic_bpr.sh"   "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmic"; fi
        if want diffmicv2; then CLEAN=1 bash "$ROOT/tools/train_diffmicv2_bpr.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmicv2"; fi
        if want medvitv2;  then bash "$ROOT/tools/train_medvitv2_bpr.sh" "$D/imagefolder/train" "$D/imagefolder/val" "$R/medvitv2"; fi
        if want diffmicv2_sam; then CLEAN=1 bash "$ROOT/tools/train_diffmicv2_sam.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmicv2_sam"; fi
    else
        if want medvit;    then bash "$ROOT/tools/train_medvit.sh"    "$D/imagefolder/train" "$D/imagefolder/val" "$R/medvit"; fi
        if want medvitv2;  then bash "$ROOT/tools/train_medvitv2.sh"  "$D/imagefolder/train" "$D/imagefolder/val" "$R/medvitv2"; fi
        if want diffmic;   then bash "$ROOT/tools/train_diffmic.sh"   "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmic"; fi
        if want diffmicv2; then CLEAN=1 bash "$ROOT/tools/train_diffmicv2.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmicv2"; fi
        if want diffmicv2_sam; then CLEAN=1 bash "$ROOT/tools/train_diffmicv2_sam.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$R/diffmicv2_sam"; fi
    fi
}

train_one_fold () {  # <fold_i> <base-tag-dir>
    local i="$1" BASE="$2"
    local D="$ROOT/${DATA_ROOT:-data}/5fold/fold_$i"
    local _seed=${SEED:-$((42 + i))}
    local _bpr="0"
    if [ "${TECH:-}" = "bpr" ] || echo "${TAG}" | grep -qi "bpr"; then _bpr="1"; fi
    if [ "$_bpr" = "1" ]; then
        if want medvit;    then mkdir -p "$BASE/medvit/fold_$i";    SEED=$_seed bash "$ROOT/tools/train_medvit_bpr.sh"    "$D/imagefolder/train" "$D/imagefolder/val" "$BASE/medvit/fold_$i"; fi
        if want diffmic;   then mkdir -p "$BASE/diffmic/fold_$i";   SEED=$_seed bash "$ROOT/tools/train_diffmic_bpr.sh"   "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmic/fold_$i"; fi
        if want diffmicv2; then mkdir -p "$BASE/diffmicv2/fold_$i"; SEED=$_seed CLEAN=1 bash "$ROOT/tools/train_diffmicv2_bpr.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmicv2/fold_$i"; fi
        if want medvitv2;  then mkdir -p "$BASE/medvitv2/fold_$i";  SEED=$_seed bash "$ROOT/tools/train_medvitv2_bpr.sh" "$D/imagefolder/train" "$D/imagefolder/val" "$BASE/medvitv2/fold_$i"; fi
        if want diffmicv2_sam; then mkdir -p "$BASE/diffmicv2_sam/fold_$i"; SEED=$_seed CLEAN=1 bash "$ROOT/tools/train_diffmicv2_sam.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmicv2_sam/fold_$i"; fi
        if want hsq; then
            mkdir -p "$BASE/hsq/fold_$i"
            SEED=$_seed bash "$ROOT/tools/train_hsq.sh" "$i" "$BASE/hsq/fold_$i" || { echo "[train.sh] train_hsq.sh failed (fold=$i)" >&2; exit 1; }
        fi
    else
        if want medvit;    then mkdir -p "$BASE/medvit/fold_$i";    SEED=$_seed bash "$ROOT/tools/train_medvit.sh"    "$D/imagefolder/train" "$D/imagefolder/val" "$BASE/medvit/fold_$i"; fi
        if want medvitv2;  then mkdir -p "$BASE/medvitv2/fold_$i";  SEED=$_seed bash "$ROOT/tools/train_medvitv2.sh"  "$D/imagefolder/train" "$D/imagefolder/val" "$BASE/medvitv2/fold_$i"; fi
        if want diffmic;   then mkdir -p "$BASE/diffmic/fold_$i";   SEED=$_seed bash "$ROOT/tools/train_diffmic.sh"   "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmic/fold_$i"; fi
        if want diffmicv2; then mkdir -p "$BASE/diffmicv2/fold_$i"; SEED=$_seed CLEAN=1 bash "$ROOT/tools/train_diffmicv2.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmicv2/fold_$i"; fi
        if want diffmicv2_sam; then mkdir -p "$BASE/diffmicv2_sam/fold_$i"; SEED=$_seed CLEAN=1 bash "$ROOT/tools/train_diffmicv2_sam.sh" "$D/pkl/lesion_train.pkl" "$D/pkl/lesion_val.pkl" "$BASE/diffmicv2_sam/fold_$i"; fi
        if want hsq; then
            mkdir -p "$BASE/hsq/fold_$i"
            SEED=$_seed bash "$ROOT/tools/train_hsq.sh" "$i" "$BASE/hsq/fold_$i" || { echo "[train.sh] train_hsq.sh failed (fold=$i)" >&2; exit 1; }
        fi
    fi
}

case "$OPT" in
    a)
        bash "$ROOT/experiments/option_a_3way/prepare.sh"
        R="$ROOT/experiments/option_a_3way/results/$TAG"
        train_one_3way "$R"
        ;;
    b)
        bash "$ROOT/experiments/option_b_5fold/prepare.sh"
        K=${K:-5}
        FOLD_START=${FOLD_START:-0}
        FOLD_END=${FOLD_END:-$((K-1))}
        BASE="$ROOT/experiments/option_b_5fold/results/$TAG"
        mkdir -p "$BASE"
        for ((i=FOLD_START; i<=FOLD_END; i++)); do
            echo ""; echo "########## TRAIN  option B  fold $i  (range $FOLD_START..$FOLD_END / K=$K) ##########"
            train_one_fold "$i" "$BASE"
        done
        ;;
    c)
        bash "$ROOT/experiments/option_c_3way_multiseed/prepare.sh"
        SEEDS=${SEEDS:-"42 43 44 45 46"}
        for s in $SEEDS; do
            R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG/seed_$s"
            echo ""; echo "########## TRAIN  option C  seed $s  ##########"
            SEED=$s train_one_3way "$R"
        done
        ;;
    *)
        echo "unknown option: $OPT (use a, b, or c)"; exit 1;;
esac
echo ""; echo "[train.sh] DONE  option=$OPT  model=$MODEL  tag=$TAG"
