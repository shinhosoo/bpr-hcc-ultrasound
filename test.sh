#!/usr/bin/env bash
set -e

# === Determinism: cuBLAS workspace config (required when torch.use_deterministic_algorithms(True) is enabled) ===
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OPT="${1:?usage: bash test.sh <a|b|c> [all|medvit|medvitv2|diffmic|diffmicv2|diffmicv2_sam] [tag]}"
MODEL="${2:-all}"
TAG="${3:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$HERE"

if [ "${BALANCED:-0}" = "1" ]; then
    _tag_lc="$(echo "$TAG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_tag_lc" != *bal* ]]; then
        TAG="${TAG}_balanced"
        echo "[test.sh] BALANCED=1 — TAG → $TAG"
    else
        echo "[test.sh] BALANCED=1 — TAG='$TAG' already includes bal"
    fi
fi

want() { [ "$MODEL" = all ] || [ "$MODEL" = "$1" ]; }

find_diffmicv2_ckpt_3way () {  # <run-dir>
    ls -t "$1"/diffmicv2/lightning_logs/*/version_*/checkpoints/*.ckpt 2>/dev/null | head -1
}

find_diffmicv2_sam_ckpt_3way () {
    ls -t "$1"/diffmicv2_sam/lightning_logs/*/version_*/checkpoints/*.ckpt 2>/dev/null | head -1
}

find_lightning_ckpt () {  # <run-dir-for-one-model-fold>
    local DIR="$1/lightning_logs"
    local CANDS=()
    if [ "${CKPT_PREFER:-best}" = "last" ]; then
        CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null))
        CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null))
    else
        CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null))
        CANDS+=($(ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null))
    fi
    for f in "${CANDS[@]}"; do
        [ -f "$f" ] || continue
        local SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        if [ "$SIZE" -ge 1048576 ]; then echo "$f"; return; fi
        echo "[test.sh] WARN: $f size $SIZE bytes — possibly corrupt, skip" >&2
    done
}

test_one_3way () {  # <run-dir> <test-imagefolder> <test-pkl>
    local R="$1" T_IMG="$2" T_PKL="$3"
    if want medvit;    then bash "$ROOT/tools/test_medvit.sh"    "$R/medvit/checkpoint_best.pth" "$T_IMG" "$R/predictions_test_medvit.npz"; fi
    if want medvitv2; then
        local BEST_V2=$(ls -t "$R"/medvitv2/MedViT_*_lesion_binary_best.pth 2>/dev/null | head -1)
        if [ -n "$BEST_V2" ]; then
            bash "$ROOT/tools/test_medvitv2.sh" "$BEST_V2" "$T_IMG" "$R/predictions_test_medvitv2.npz"
        else
            echo "[test.sh] WARN: MedViTV2 ckpt not found"
        fi
    fi
    if want diffmic;   then bash "$ROOT/tools/test_diffmic.sh"   "$R/diffmic" "$T_PKL" "$R/predictions_test_diffmic.npz"; fi
    if want diffmicv2; then
        local BEST=$(find_diffmicv2_ckpt_3way "$R")
        [ -n "$BEST" ] || { echo "[test.sh] WARN: DiffMICv2 ckpt not found in $R/diffmicv2/lightning_logs/"; return; }
        bash "$ROOT/tools/test_diffmicv2.sh" "$BEST" "$T_PKL" "$R/predictions_test_diffmicv2.npz"
    fi
    if want diffmicv2_sam; then
        local BEST_S=$(find_diffmicv2_sam_ckpt_3way "$R")
        if [ -z "$BEST_S" ]; then
            echo "[test.sh] WARN: DiffMICv2-SAM ckpt not found in $R/diffmicv2_sam/lightning_logs/"
        else
            bash "$ROOT/tools/test_diffmicv2_sam.sh" "$BEST_S" "$T_PKL" "$R/predictions_test_diffmicv2_sam.npz"
        fi
    fi
}

test_one_fold () {  # <base-tag-dir> <fold_i> <test-imagefolder> <test-pkl>
    local BASE="$1" i="$2" T_IMG="$3" T_PKL="$4"
    if want medvit; then
        local RM="$BASE/medvit/fold_$i"
        if [ -f "$RM/checkpoint_best.pth" ]; then
            bash "$ROOT/tools/test_medvit.sh" "$RM/checkpoint_best.pth" "$T_IMG" "$RM/predictions_test_medvit.npz"
        else
            echo "[test.sh] WARN: MedViT ckpt not found in $RM"
        fi
    fi
    if want medvitv2; then
        local RM="$BASE/medvitv2/fold_$i"
        local BEST_V2=$(ls -t "$RM"/MedViT_*_lesion_binary_best.pth 2>/dev/null | head -1)
        if [ -n "$BEST_V2" ]; then
            bash "$ROOT/tools/test_medvitv2.sh" "$BEST_V2" "$T_IMG" "$RM/predictions_test_medvitv2.npz"
        else
            echo "[test.sh] WARN: MedViTV2 ckpt not found in $RM"
        fi
    fi
    if want diffmic; then
        local RM="$BASE/diffmic/fold_$i"
        if [ -d "$RM" ]; then
            bash "$ROOT/tools/test_diffmic.sh" "$RM" "$T_PKL" "$RM/predictions_test_diffmic.npz"
        else
            echo "[test.sh] WARN: DiffMIC run dir not found: $RM"
        fi
    fi
    if want diffmicv2; then
        local RM="$BASE/diffmicv2/fold_$i"
        local BEST=$(find_lightning_ckpt "$RM")
        if [ -z "$BEST" ]; then
            echo "[test.sh] WARN: DiffMICv2 ckpt not found in $RM/lightning_logs/"
        else
            bash "$ROOT/tools/test_diffmicv2.sh" "$BEST" "$T_PKL" "$RM/predictions_test_diffmicv2.npz"
        fi
    fi
    if want diffmicv2_sam; then
        local RM="$BASE/diffmicv2_sam/fold_$i"
        local BEST_S=$(find_lightning_ckpt "$RM")
        if [ -z "$BEST_S" ]; then
            echo "[test.sh] WARN: DiffMICv2-SAM ckpt not found in $RM/lightning_logs/"
        else
            bash "$ROOT/tools/test_diffmicv2_sam.sh" "$BEST_S" "$T_PKL" "$RM/predictions_test_diffmicv2_sam.npz"
        fi
    fi
}

print_metrics () {  # <run-dir>
    local R="$1"
    local ARGS=()
    [ -f "$R/predictions_test_medvit.npz"     ] && ARGS+=(--pred "$R/predictions_test_medvit.npz"     --name MedViT)
    [ -f "$R/predictions_test_medvitv2.npz"   ] && ARGS+=(--pred "$R/predictions_test_medvitv2.npz"   --name MedViTV2)
    [ -f "$R/predictions_test_diffmic.npz"   ] && ARGS+=(--pred "$R/predictions_test_diffmic.npz"   --name DiffMIC)
    [ -f "$R/predictions_test_diffmicv2.npz"     ] && ARGS+=(--pred "$R/predictions_test_diffmicv2.npz"     --name DiffMICv2)
    [ -f "$R/predictions_test_diffmicv2_sam.npz" ] && ARGS+=(--pred "$R/predictions_test_diffmicv2_sam.npz" --name DiffMICv2-SAM)
    if [ ${#ARGS[@]} -gt 0 ]; then
        echo ""; echo "----- METRICS  ($R) -----"
        python3 "$ROOT/tools/unified_eval.py" "${ARGS[@]}" --out "$R/metrics_test.csv"
        mkdir -p "$R/viz"
        python3 "$ROOT/tools/visualize_results.py" "${ARGS[@]}" --out "$R/viz" ||             echo "[viz] visualize_results.py failed — check matplotlib/sklearn"
        echo "  → CSV : $R/metrics_test.csv"
        echo "  → PNGs: $R/viz/{roc,pr,confusion,metrics_bar}.png"
    fi
}

print_metrics_fold () {  # <base-tag-dir> <fold_i>
    local BASE="$1" i="$2"
    for m in medvit medvitv2 diffmic diffmicv2 diffmicv2_sam; do
        local RM="$BASE/$m/fold_$i"
        local NPZ="$RM/predictions_test_${m}.npz"
        [ -f "$NPZ" ] || continue
        local NAME
        case "$m" in
            medvit) NAME=MedViT;;
            medvitv2) NAME=MedViTV2;;
            diffmic) NAME=DiffMIC;;
            diffmicv2) NAME=DiffMICv2;;
            diffmicv2_sam) NAME=DiffMICv2-SAM;;
        esac
        echo ""; echo "----- METRICS  ($RM) -----"
        python3 "$ROOT/tools/unified_eval.py" --pred "$NPZ" --name "$NAME" --out "$RM/metrics_test.csv"
        mkdir -p "$RM/viz"
        python3 "$ROOT/tools/visualize_results.py" --pred "$NPZ" --name "$NAME" --out "$RM/viz" ||             echo "[viz] visualize_results.py failed — check matplotlib/sklearn"
        echo "  → CSV : $RM/metrics_test.csv"
        echo "  → PNGs: $RM/viz/{roc,pr,confusion,metrics_bar}.png"
    done
}

case "$OPT" in
    a)
        R="$ROOT/experiments/option_a_3way/results/$TAG"
        D="$ROOT/data/3way"
        test_one_3way "$R" "$D/imagefolder/test" "$D/pkl/lesion_test.pkl"
        print_metrics "$R"
        ;;
    b)
        K=${K:-5}
        BASE="$ROOT/experiments/option_b_5fold/results/$TAG"
        for ((i=0; i<K; i++)); do
            D="$ROOT/data/5fold/fold_$i"
            echo ""; echo "########## TEST  option B  fold $i  ##########"
            test_one_fold "$BASE" "$i" "$D/imagefolder/test" "$D/pkl/lesion_test.pkl"
            print_metrics_fold "$BASE" "$i"
        done
        ;;
    c)
        SEEDS=${SEEDS:-"42 43 44 45 46"}
        D="$ROOT/data/3way"
        for s in $SEEDS; do
            R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG/seed_$s"
            echo ""; echo "########## TEST  option C  seed $s  ##########"
            test_one_3way "$R" "$D/imagefolder/test" "$D/pkl/lesion_test.pkl"
            print_metrics "$R"
        done
        ;;
    *)
        echo "unknown option: $OPT"; exit 1;;
esac
echo ""; echo "[test.sh] DONE  option=$OPT  model=$MODEL  tag=$TAG"
