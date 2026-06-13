#!/usr/bin/env bash
set -e
OPT="${1:?usage: bash viz.sh <a|b|c> [tag]}"
TAG="${2:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$HERE"

if [ "${BALANCED:-0}" = "1" ]; then
    _tag_lc="$(echo "$TAG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$_tag_lc" != *bal* ]]; then
        TAG="${TAG}_balanced"
        echo "[viz.sh] BALANCED=1 — TAG → $TAG"
    else
        echo "[viz.sh] BALANCED=1 — TAG='$TAG' already includes bal"
    fi
fi

viz_compare () {  # <out-dir> <pred-dir> [label-prefix]
    local OUT="$1" PRED="$2" PREFIX="${3:-}"
    local ARGS=()
    [ -f "$PRED/predictions_test_medvit.npz"     ] && ARGS+=(--pred "$PRED/predictions_test_medvit.npz"     --name "${PREFIX}MedViT")
    [ -f "$PRED/predictions_test_medvitv2.npz"   ] && ARGS+=(--pred "$PRED/predictions_test_medvitv2.npz"   --name "${PREFIX}MedViTV2")
    [ -f "$PRED/predictions_test_diffmic.npz"   ] && ARGS+=(--pred "$PRED/predictions_test_diffmic.npz"   --name "${PREFIX}DiffMIC")
    [ -f "$PRED/predictions_test_diffmicv2.npz"     ] && ARGS+=(--pred "$PRED/predictions_test_diffmicv2.npz"     --name "${PREFIX}DiffMICv2")
    [ -f "$PRED/predictions_test_diffmicv2_sam.npz" ] && ARGS+=(--pred "$PRED/predictions_test_diffmicv2_sam.npz" --name "${PREFIX}DiffMICv2-SAM")
    if [ ${#ARGS[@]} -eq 0 ]; then echo "[viz] no npz in $PRED — skip"; return; fi
    mkdir -p "$OUT"
    python3 "$ROOT/tools/visualize_results.py" "${ARGS[@]}" --out "$OUT"
    python3 "$ROOT/tools/unified_eval.py"      "${ARGS[@]}" --out "$OUT/metrics.csv"
}

viz_pooled_per_model () {  # <results-root> <pattern_dir> <out-root>
    local R="$1" PAT="$2" OUT="$3"
    mkdir -p "$OUT"
    python3 - "$R" "$PAT" "$OUT" << 'PY'
import sys, glob, os
import numpy as np
root, pat, out = sys.argv[1], sys.argv[2], sys.argv[3]
models = ["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"]
for m in models:
    fs = sorted(set(
        glob.glob(os.path.join(root, m, pat, f"predictions_test_{m}.npz"))   # model-first
        + glob.glob(os.path.join(root, pat, f"predictions_test_{m}.npz"))    # legacy nested
    ))
    if not fs: continue
    y = np.concatenate([np.load(f, allow_pickle=True)["y_true"] for f in fs])
    s = np.concatenate([np.load(f, allow_pickle=True)["y_score"] for f in fs])
    op = os.path.join(out, f"predictions_test_{m}_pooled.npz")
    np.savez(op, y_true=y, y_score=s)
    print(f"pooled {len(y)} samples for {m} ({len(fs)} files) -> {op}")
PY
    ARGS=()
    [ -f "$OUT/predictions_test_medvit_pooled.npz"    ] && ARGS+=(--pred "$OUT/predictions_test_medvit_pooled.npz"    --name MedViT)
    [ -f "$OUT/predictions_test_medvitv2_pooled.npz"  ] && ARGS+=(--pred "$OUT/predictions_test_medvitv2_pooled.npz"  --name MedViTV2)
    [ -f "$OUT/predictions_test_diffmic_pooled.npz"   ] && ARGS+=(--pred "$OUT/predictions_test_diffmic_pooled.npz"   --name DiffMIC)
    [ -f "$OUT/predictions_test_diffmicv2_pooled.npz"     ] && ARGS+=(--pred "$OUT/predictions_test_diffmicv2_pooled.npz"     --name DiffMICv2)
    [ -f "$OUT/predictions_test_diffmicv2_sam_pooled.npz" ] && ARGS+=(--pred "$OUT/predictions_test_diffmicv2_sam_pooled.npz" --name DiffMICv2-SAM)
    [ ${#ARGS[@]} -gt 0 ] && python3 "$ROOT/tools/visualize_results.py" "${ARGS[@]}" --out "$OUT/comparison_pooled"
    [ ${#ARGS[@]} -gt 0 ] && python3 "$ROOT/tools/unified_eval.py"      "${ARGS[@]}" --out "$OUT/comparison_pooled/metrics.csv"
}

viz_compare_model_first () {  # <out-dir> <root> <index_dir>   (e.g. R, fold_0)
    local OUT="$1" R="$2" IDX="$3"
    mkdir -p "$OUT"
    local ARGS=()
    for m in medvit medvitv2 diffmic diffmicv2 diffmicv2_sam; do
        local NPZ="$R/$m/$IDX/predictions_test_${m}.npz"
        [ -f "$NPZ" ] || continue
        local NAME
        case "$m" in
            medvit) NAME=MedViT;;
            medvitv2) NAME=MedViTV2;;
            diffmic) NAME=DiffMIC;;
            diffmicv2) NAME=DiffMICv2;;
            diffmicv2_sam) NAME=DiffMICv2-SAM;;
        esac
        ARGS+=(--pred "$NPZ" --name "${IDX} ${NAME}")
    done
    if [ ${#ARGS[@]} -eq 0 ]; then echo "[viz] no npz for $IDX — skip"; return; fi
    python3 "$ROOT/tools/visualize_results.py" "${ARGS[@]}" --out "$OUT"
    python3 "$ROOT/tools/unified_eval.py"      "${ARGS[@]}" --out "$OUT/metrics.csv"
}

case "$OPT" in
    a)
        R="$ROOT/experiments/option_a_3way/results/$TAG"
        viz_compare "$R/comparison" "$R"
        ;;
    b)
        R="$ROOT/experiments/option_b_5fold/results/$TAG"
        K=${K:-5}
        for ((i=0; i<K; i++)); do
            viz_compare_model_first "$R/comparison_fold_$i" "$R" "fold_$i"
        done
        viz_pooled_per_model "$R" "fold_*" "$R/pooled"
        python3 "$ROOT/experiments/option_b_5fold/aggregate.py" --tag "$TAG" --results-root "$R"
        ;;
    c)
        R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG"
        SEEDS=${SEEDS:-"42 43 44 45 46"}
        for s in $SEEDS; do
            viz_compare "$R/seed_$s/comparison" "$R/seed_$s" "seed${s} "
        done
        viz_pooled_per_model "$R" "seed_*" "$R/pooled"
        python3 "$ROOT/experiments/option_c_3way_multiseed/aggregate.py" --tag "$TAG" --results-root "$R"
        ;;
    *)
        echo "unknown option: $OPT"; exit 1;;
esac
echo ""; echo "[viz.sh] DONE  option=$OPT  tag=$TAG"
