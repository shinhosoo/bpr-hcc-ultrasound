#!/usr/bin/env bash
# MedViTV2 테스트 (저장된 ckpt 로 test set 평가만).
# Usage: bash tools/test_medvitv2.sh <ckpt> <test_dir> <out_npz>
set -e
CKPT="${1:?}"; TEST_DIR="${2:?}"; OUT="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
MODEL="${MODEL:-MedViT_small}"
BS="${BS:-32}"; SEED="${SEED:-42}"

TEST_ABS="$( cd "$TEST_DIR" && pwd )"
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$( cd "$(dirname "$OUT")" && pwd )/$(basename "$OUT")"

cd "$ROOT/models/MedViTV2-main"
python3 main.py \
    --eval \
    --dataset lesion_binary \
    --train-path "$TEST_ABS" \
    --val-path   "$TEST_ABS" \
    --model_name "$MODEL" \
    --batch_size "$BS" --seed "$SEED" \
    --checkpoint_path "$CKPT" \
    --output-dir "$(dirname "$OUT_ABS")" \
    --save-predictions "$OUT_ABS"
echo "[test_medvitv2] wrote: $OUT_ABS"
