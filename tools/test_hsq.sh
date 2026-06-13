#!/usr/bin/env bash
# HSQ best ckpt 로 test set 평가 → predictions_test_hsq.npz 저장.
# 다른 test_*.sh 와 동일한 인터페이스.
#
# Usage:
#   bash tools/test_hsq.sh <ckpt_dir> <test_path> <out_npz> [fold_index]
#
# Example (단일 fold):
#   bash tools/test_hsq.sh \
#       experiments/option_b_5fold/results/<tag>/hsq/fold_0 \
#       data/5fold/fold_0/imagefolder/test \
#       experiments/option_b_5fold/results/<tag>/hsq/fold_0/predictions_test_hsq.npz \
#       0
#
# Example (5-fold 전체 루프):
#   TAG=my_tag
#   for i in 0 1 2 3 4; do
#     bash tools/test_hsq.sh \
#       experiments/option_b_5fold/results/$TAG/hsq/fold_$i \
#       data/5fold/fold_$i/imagefolder/test \
#       experiments/option_b_5fold/results/$TAG/hsq/fold_$i/predictions_test_hsq.npz \
#       $i
#   done
#
# env:
#   HSQ_BASE=1   → LENet_base 사용 (default: LENet proposed)
#   BS=32         → batch size

set -e
CKPT_DIR="${1:?usage: bash test_hsq.sh <ckpt_dir> <test_path> <out_npz> [fold_index]}"
TEST_PATH="${2:?usage: bash test_hsq.sh <ckpt_dir> <test_path> <out_npz> [fold_index]}"
OUT="${3:?usage: bash test_hsq.sh <ckpt_dir> <test_path> <out_npz> [fold_index]}"
FOLD="${4:-0}"

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

CKPT_ABS="$( cd "$CKPT_DIR" && pwd )"
TEST_ABS="$( cd "$TEST_PATH" && pwd )"
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$( cd "$(dirname "$OUT")" && pwd )/$(basename "$OUT")"

echo "[test_hsq] fold=$FOLD  ckpt_dir=$CKPT_ABS"
echo "[test_hsq] test=$TEST_ABS"
echo "[test_hsq] out=$OUT_ABS"

python3 "$ROOT/tools/test_hsq.py" \
    --ckpt-dir   "$CKPT_ABS" \
    --test-path  "$TEST_ABS" \
    --out        "$OUT_ABS" \
    --fold-index "$FOLD"

echo "[test_hsq] DONE → $OUT_ABS"
