#!/usr/bin/env bash
# 저장된 DiffMICv2 Lightning ckpt 로 test set 추론.
# Usage:
#   bash tools/test_diffmicv2.sh <ckpt> <test_pkl> <out_npz> [config_yml]
# Example:
#   bash tools/test_diffmicv2.sh \
#       results/diffmicv2_baseline/logs/lesion_binary/version_0/checkpoints/last.ckpt \
#       data/3way/pkl/lesion_test.pkl \
#       results/diffmicv2_baseline/predictions_test.npz
set -e
CKPT="${1:?usage: bash test_diffmicv2.sh <ckpt> <test_pkl> <out_npz> [config_yml]}"
TEST_PKL="${2:?usage: bash test_diffmicv2.sh <ckpt> <test_pkl> <out_npz> [config_yml]}"
OUT="${3:?usage: bash test_diffmicv2.sh <ckpt> <test_pkl> <out_npz> [config_yml]}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
CFG="${4:-$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml}"

mkdir -p "$(dirname "$OUT")"
OUT_ABS="$( cd "$(dirname "$OUT")" && pwd )/$(basename "$OUT")"
TEST_ABS="$( cd "$(dirname "$TEST_PKL")" && pwd )/$(basename "$TEST_PKL")"

cd "$ROOT/models/DiffMICv2-main"
python3 eval_only.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --test-pkl "$TEST_ABS" \
    --out "$OUT_ABS"
echo "[test_diffmicv2] wrote: $OUT_ABS"
