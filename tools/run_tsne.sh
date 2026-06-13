#!/usr/bin/env bash
# t-SNE 시각화 한 줄 실행.
# Usage:
#   bash tools/run_tsne.sh <option> [model] [tag]
#     option : a | b | c
#     model  : all | medvit | diffmic | diffmicv2  (default: all)
#     tag    : 기본 baseline
#
# 학습된 ckpt 가 있는 모델 만 latent 추출 → t-SNE PNG 생성.
set -e
OPT="${1:?usage: bash tools/run_tsne.sh <a|b|c> [all|medvit|diffmic|diffmicv2] [tag]}"
MODEL="${2:-all}"
TAG="${3:-baseline}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

want() { [ "$MODEL" = all ] || [ "$MODEL" = "$1" ]; }

# fold/seed 마다 추출하면 양이 많아 option B/C 는 fold_0 / seed_42 만 처리
case "$OPT" in
    a) R="$ROOT/experiments/option_a_3way/results/$TAG"; DATA="$ROOT/data/3way";;
    b) R="$ROOT/experiments/option_b_5fold/results/$TAG/fold_0"; DATA="$ROOT/data/5fold/fold_0";;
    c) R="$ROOT/experiments/option_c_3way_multiseed/results/$TAG/seed_42"; DATA="$ROOT/data/3way";;
    *) echo "unknown opt: $OPT"; exit 1;;
esac

OUT_DIR="$R/tsne"
mkdir -p "$OUT_DIR"
ARGS=()

# MedViT
if want medvit && [ -f "$R/medvit/checkpoint_best.pth" ]; then
    python3 "$ROOT/tools/extract_features.py" --model medvit \
        --ckpt "$R/medvit/checkpoint_best.pth" \
        --data "$DATA/imagefolder/test" \
        --out "$OUT_DIR/feats_medvit.npz"
    ARGS+=(--features "$OUT_DIR/feats_medvit.npz:MedViT")
elif want medvit; then
    echo "[run_tsne] WARN: MedViT ckpt 없음 (skip)"
fi

# DiffMIC
if want diffmic && [ -f "$R/diffmic/logs/lesion_binary/split_0/aux_ckpt_best.pth" ]; then
    python3 "$ROOT/tools/extract_features.py" --model diffmic \
        --ckpt "$R/diffmic" \
        --data "$DATA/pkl/lesion_test.pkl" \
        --out "$OUT_DIR/feats_diffmic.npz"
    ARGS+=(--features "$OUT_DIR/feats_diffmic.npz:DiffMIC")
elif want diffmic; then
    echo "[run_tsne] WARN: DiffMIC aux_ckpt_best.pth 없음 (skip)"
fi

# DiffMICv2
if want diffmicv2; then
    DMV2_CKPT=$(ls -t "$R"/diffmicv2/lightning_logs/*/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
    DMV2_CFG="$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml"
    if [ -n "$DMV2_CKPT" ]; then
        python3 "$ROOT/tools/extract_features.py" --model diffmicv2 \
            --ckpt "$DMV2_CKPT" --config "$DMV2_CFG" \
            --data "$DATA/pkl/lesion_test.pkl" \
            --out "$OUT_DIR/feats_diffmicv2.npz"
        ARGS+=(--features "$OUT_DIR/feats_diffmicv2.npz:DiffMICv2")
    else
        echo "[run_tsne] WARN: DiffMICv2 ckpt 없음 (skip)"
    fi
fi

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "[run_tsne] 처리한 모델 없음. 먼저 학습 또는 model 인자 확인 필요."
    exit 1
fi

python3 "$ROOT/tools/tsne_plot.py" "${ARGS[@]}" --out "$OUT_DIR/tsne.png"
echo ""
echo "[run_tsne] → $OUT_DIR/tsne.png"
