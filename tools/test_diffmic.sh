#!/usr/bin/env bash
# 저장된 DiffMIC 학습 결과로 test set 추론만 수행.
# Usage:
#   bash tools/test_diffmic.sh <exp_dir> <test_pkl> <out_npz>
# exp_dir: 학습 시 --exp 로 지정한 폴더. logs/<doc>/split_0/ 안에 ckpt_best.pth, aux_ckpt_best.pth, config.yml 이 있어야 함.
# Example:
#   bash tools/test_diffmic.sh \
#       results/diffmic_baseline \
#       data/3way/pkl/lesion_test.pkl \
#       results/diffmic_baseline/predictions_test.npz
set -e
EXP="${1:?usage: bash test_diffmic.sh <exp_dir> <test_pkl> <out_npz>}"
TEST_PKL="${2:?usage: bash test_diffmic.sh <exp_dir> <test_pkl> <out_npz>}"
OUT="${3:?usage: bash test_diffmic.sh <exp_dir> <test_pkl> <out_npz>}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DOC="${DOC:-lesion_binary}"
DEVICE="${DEVICE:-0}"
THREAD="${THREAD:-4}"

EXP_ABS="$( cd "$EXP" && pwd )"
TEST_ABS="$( cd "$(dirname "$TEST_PKL")" && pwd )/$(basename "$TEST_PKL")"
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$( cd "$(dirname "$OUT")" && pwd )/$(basename "$OUT")"

# 1) 학습 시 저장된 config.yml 찾기 — 여러 가능 위치 시도
DM_CFG=""
for _cand in \
    "$EXP_ABS/logs/$DOC/split_0/config.yml" \
    "$EXP_ABS/logs/$DOC/config.yml" \
    "$EXP_ABS/train_config.yml" \
    "$EXP_ABS/logs/config.yml"; do
    if [ -f "$_cand" ]; then DM_CFG="$_cand"; break; fi
done
if [ -z "$DM_CFG" ]; then
    echo "[test_diffmic] config not found — 다음 위치를 모두 시도했습니다:"
    echo "  - $EXP_ABS/logs/$DOC/split_0/config.yml"
    echo "  - $EXP_ABS/logs/$DOC/config.yml"
    echo "  - $EXP_ABS/train_config.yml"
    echo "  - $EXP_ABS/logs/config.yml"
    exit 1
fi
echo "[test_diffmic] using config: $DM_CFG"

# testdata 스왑 — 원본을 건드리지 않도록 split_0/ 에 작업본 생성
WORK_CFG="$EXP_ABS/logs/$DOC/split_0/config.yml"
mkdir -p "$(dirname "$WORK_CFG")"
python3 "$ROOT/tools/diffmic_config_swap.py" --in "$DM_CFG" --out "$WORK_CFG" --testdata "$TEST_ABS" --format namespace
DM_CFG="$WORK_CFG"

# 2) --test --eval_best 로 추론
cd "$ROOT/models/DiffMIC-main"
DIFFMIC_PRED_PATH="$OUT_ABS" \
python3 main.py --device "$DEVICE" --thread "$THREAD" --loss diffmic_conditional \
    --config "$EXP_ABS/logs/" --exp "$EXP_ABS" --doc "$DOC" \
    --n_splits 1 --test --eval_best
echo "[test_diffmic] wrote: $OUT_ABS"
