#!/usr/bin/env bash
# DiffMIC 학습만.
# Usage: bash tools/train_diffmic.sh <train_pkl> <val_pkl> <exp_dir>
set -e
TR="${1:?}"; VA="${2:?}"; EXP="${3:?}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DOC="${DOC:-lesion_binary}"
EPOCHS_PATIENCE_SEED_ENV="${SEED:-42}"
PATIENCE="${PATIENCE:-20}"; DEVICE="${DEVICE:-0}"; THREAD="${THREAD:-4}"
export DIFFMIC_PRETRAINED="${DIFFMIC_PRETRAINED:-1}"
# BALANCED=1 이면 balanced sampler, 아니면 weighted sampler
if [ "${BALANCED:-0}" = "1" ]; then
    export DIFFMIC_BALANCED_SAMPLER=1
    export DIFFMIC_WEIGHTED_SAMPLER=0
else
    export DIFFMIC_WEIGHTED_SAMPLER="${DIFFMIC_WEIGHTED_SAMPLER:-1}"
fi

mkdir -p "$EXP"
EXP_ABS="$( cd "$EXP" && pwd )"
TR_ABS="$( cd "$(dirname "$TR")" && pwd )/$(basename "$TR")"
VA_ABS="$( cd "$(dirname "$VA")" && pwd )/$(basename "$VA")"

CFG="$EXP_ABS/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" \
    --in "$ROOT/models/DiffMIC-main/configs/lesion_binary.yml" --out "$CFG" \
    --traindata "$TR_ABS" --testdata "$VA_ABS"

cd "$ROOT/models/DiffMIC-main"
export DIFFMIC_PRED_PATH="$EXP_ABS/predictions_val.npz"
python3 main.py --device "$DEVICE" --thread "$THREAD" --loss diffmic_conditional \
    --config "$CFG" --exp "$EXP_ABS" --doc "$DOC" \
    --n_splits 1 --ni --seed "$EPOCHS_PATIENCE_SEED_ENV" --early_stop_patience "$PATIENCE"
echo "[train_diffmic] ckpts in: $EXP_ABS/logs/$DOC/split_0/"
