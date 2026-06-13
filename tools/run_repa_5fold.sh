#!/usr/bin/env bash
# DiffMICv2 + REPA — 5-fold 학습 + test 평가. conditioning encoder 를 강한 frozen 인코더에 정렬.
#   eval 은 표준(test_diffmicv2.sh) — repa_proj 는 학습 전용, diffusion forward 미사용.
#
# Usage:  bash tools/run_repa_5fold.sh [tag]
# env: K=5  PATIENCE=40  REPA_LAMBDA=0.5  REPA_ENCODER=resnet50  REPA_STRONG_DIM=2048
set -e
TAG="${1:-repa_diffmicv2}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DM="$ROOT/models/DiffMICv2-main"
RES="$ROOT/experiments/option_b_5fold/results"
K="${K:-5}"; PATIENCE="${PATIENCE:-40}"

export REPA_LAMBDA="${REPA_LAMBDA:-0.5}"
export REPA_ENCODER="${REPA_ENCODER:-resnet50}"
MEDVIT_TAG="${MEDVIT_TAG:-bpr_medvit_2stage_aux}"   # REPA_ENCODER=medvit 일 때 fold별 타깃 ckpt 출처
export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"

find_lightning_ckpt () {
    local DIR="$1/lightning_logs"; local f SIZE
    for f in $(ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null; ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null); do
        [ -f "$f" ] || continue
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo 0)
        [ "$SIZE" -ge 1048576 ] && { echo "$f"; return; }
    done
}

echo "[repa-5fold] tag=$TAG  lambda=$REPA_LAMBDA  encoder=$REPA_ENCODER"
for ((i=0; i<K; i++)); do
    echo ""; echo "########## REPA  fold $i ##########"
    D="$ROOT/data/5fold/fold_$i/pkl"
    TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"; TE="$D/lesion_test.pkl"
    for f in "$TR" "$VA" "$TE"; do [ -f "$f" ] || { echo "[ERR] pkl 없음: $f"; exit 1; }; done

    OUT="$RES/$TAG/diffmicv2/fold_$i"; mkdir -p "$OUT"
    if [ "$REPA_ENCODER" = "medvit" ]; then
        export MEDVIT_CKPT="$RES/$MEDVIT_TAG/medvit/fold_$i/checkpoint_best.pth"
        [ -f "$MEDVIT_CKPT" ] || { echo "[ERR] MedViT ckpt 없음: $MEDVIT_CKPT"; exit 1; }
        echo "[repa] fold $i target MedViT: $MEDVIT_CKPT"
    fi
    CFG="$OUT/train_config.yml"
    python3 "$ROOT/tools/diffmic_config_swap.py" --in "$DM/configs/lesion_binary.yml" --out "$CFG" --traindata "$TR" --testdata "$VA"

    ( cd "$DM" && [ -d ./logs ] && rm -rf ./logs; \
      DIFFMICV2_PRED_PATH="$OUT/predictions_val.npz" \
      PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
      python3 "$DM/run_diffmicv2_repa.py" --config "$CFG" --early-stop-patience "$PATIENCE"; \
      [ -d ./logs ] && cp -r ./logs "$OUT/lightning_logs" 2>/dev/null || true )

    CKPT="$(find_lightning_ckpt "$OUT")"
    [ -n "$CKPT" ] || { echo "[ERR] ckpt 없음: $OUT/lightning_logs"; exit 1; }
    # 표준 eval (repa_proj 는 strict=False 로 무시됨)
    bash "$ROOT/tools/test_diffmicv2.sh" "$CKPT" "$TE" "$OUT/predictions_test_diffmicv2.npz"
done

echo ""; echo "########## 집계 ##########"
bash "$ROOT/viz.sh" b "$TAG"
echo "[repa-5fold] DONE. $TAG AUC 를 baseline_32(0.864)와 비교하세요."
