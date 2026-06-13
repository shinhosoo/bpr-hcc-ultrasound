#!/usr/bin/env bash
# End-to-end driver for fair binary-lesion comparison of MedViT / DiffMIC / DiffMICv2.
# Same data split (seed=42, val 0.2). Same training budget (epochs=100, early-stop patience=20).
# Same evaluation/visualization pipeline.

set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}
PATIENCE=${PATIENCE:-20}
DEVICE=${DEVICE:-0}

cd "$ROOT"

echo "==[0/4] Prepare dataset (idempotent) ===================================="
python3 tools/prepare_binary_image_dataset.py \
    --source image --out prepared_binary_lesion \
    --val-ratio 0.2 --seed "$SEED"

# Common pretrained/sampler flags (toggle via env if you want to disable)
export DIFFMIC_PRETRAINED=${DIFFMIC_PRETRAINED:-1}
export DIFFMIC_WEIGHTED_SAMPLER=${DIFFMIC_WEIGHTED_SAMPLER:-1}
export DIFFMICV2_PRETRAINED=${DIFFMICV2_PRETRAINED:-1}
export DIFFMICV2_WEIGHTED_SAMPLER=${DIFFMICV2_WEIGHTED_SAMPLER:-1}

echo "==[1/4] Train MedViT ===================================================="
(
    cd models/MedViT-main/CustomDataset
    python3 main.py \
        --data-set image_folder \
        --data-path ../../prepared_binary_lesion/imagefolder/train \
        --eval-data-path ../../prepared_binary_lesion/imagefolder/val \
        --nb-classes 2 --model MedViT_small \
        --batch-size 32 --epochs "$EPOCHS" --seed "$SEED" \
        --pretrained --weighted-sampler \
        --early-stop-patience "$PATIENCE" \
        --output-dir ../../outputs/medvit_binary \
        --save-predictions ../../outputs/medvit_binary/predictions_val.npz
)

echo "==[2/4] Train DiffMIC ==================================================="
(
    cd models/DiffMIC-main
    rm -rf ./results_lesion_binary
    python3 main.py --device "$DEVICE" --thread 4 \
        --loss diffmic_conditional \
        --config configs/lesion_binary.yml \
        --exp ./results_lesion_binary --doc lesion_binary \
        --n_splits 1 --ni \
        --early_stop_patience "$PATIENCE" --seed "$SEED"
    # Run test() to write predictions_val.npz
    python3 main.py --device "$DEVICE" --thread 4 \
        --loss diffmic_conditional \
        --config ./results_lesion_binary/logs/ \
        --exp ./results_lesion_binary --doc lesion_binary \
        --n_splits 1 --test --eval_best
)

echo "==[3/4] Train DiffMICv2 ================================================="
(
    cd models/DiffMICv2-main
    rm -rf ./logs/lesion_binary
    python3 diffuser_trainer.py \
        --config configs/lesion_binary.yml \
        --early-stop-patience "$PATIENCE"
)

echo "==[4/4] Unified evaluation + visualization =============================="
MEDVIT_PRED="$ROOT/outputs/medvit_binary/predictions_val.npz"
DIFFMIC_PRED="$ROOT/models/DiffMIC-main/results_lesion_binary/logs/lesion_binary/split_0/predictions_val.npz"
DIFFMICV2_PRED="$ROOT/models/DiffMICv2-main/logs/predictions_val.npz"
OUT_DIR="$ROOT/outputs/comparison"
mkdir -p "$OUT_DIR"

python3 tools/unified_eval.py \
    --pred "$MEDVIT_PRED"  --name MedViT \
    --pred "$DIFFMIC_PRED" --name DiffMIC \
    --pred "$DIFFMICV2_PRED" --name DiffMICv2 \
    --out "$OUT_DIR/metrics.csv"

python3 tools/visualize_results.py \
    --pred "$MEDVIT_PRED"  --name MedViT \
    --pred "$DIFFMIC_PRED" --name DiffMIC \
    --pred "$DIFFMICV2_PRED" --name DiffMICv2 \
    --out "$OUT_DIR"

echo "Done. See $OUT_DIR/{roc,pr,confusion,metrics_bar}.png and metrics.csv"
