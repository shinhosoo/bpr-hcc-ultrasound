#!/usr/bin/env bash
# DiffMICv2 + MedViT 백본 (arch=medvit) — 5-fold 학습+평가. 백본만 교체(BPR/REPA 없이 백본 효과 격리).
#   encoder_x = MedViT (강한 백본), local 은 resnet18 유지, diffusion/DCG 그대로.
#
# Usage:  bash tools/run_diffmic_medvitbb_5fold.sh [tag]
# env: K=5  PATIENCE=40  MEDVIT_MODEL=MedViT_small  MEDVIT_TAG=(옵션, fold별 warm-start)
set -e
TAG="${1:-diffmic_medvitbb}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DM="$ROOT/models/DiffMICv2-main"
RES="$ROOT/experiments/option_b_5fold/results"
K="${K:-5}"; PATIENCE="${PATIENCE:-40}"
export MEDVIT_MODEL="${MEDVIT_MODEL:-MedViT_small}"
MEDVIT_TAG="${MEDVIT_TAG:-}"
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

echo "[medvitbb] tag=$TAG  MEDVIT_MODEL=$MEDVIT_MODEL  warm-start=${MEDVIT_TAG:-none}"
for ((i=0; i<K; i++)); do
    echo ""; echo "########## DiffMIC+MedViTbb  fold $i ##########"
    D="$ROOT/data/5fold/fold_$i/pkl"
    TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"; TE="$D/lesion_test.pkl"
    for f in "$TR" "$VA" "$TE"; do [ -f "$f" ] || { echo "[ERR] pkl 없음: $f"; exit 1; }; done

    OUT="$RES/$TAG/diffmicv2/fold_$i"; mkdir -p "$OUT"
    if [ -n "$MEDVIT_TAG" ]; then
        export MEDVIT_CKPT="$RES/$MEDVIT_TAG/medvit/fold_$i/checkpoint_best.pth"
        [ -f "$MEDVIT_CKPT" ] || { echo "[ERR] warm-start MedViT 없음: $MEDVIT_CKPT"; exit 1; }
    fi
    CFG="$OUT/train_config.yml"
    python3 "$ROOT/tools/diffmic_config_swap.py" --in "$DM/configs/lesion_binary.yml" --out "$CFG" \
        --traindata "$TR" --testdata "$VA" --arch medvit

    ( cd "$DM" && [ -d ./logs ] && rm -rf ./logs; \
      DIFFMICV2_PRED_PATH="$OUT/predictions_val.npz" PYTHONPATH="$PWD:$PYTHONPATH" \
      python3 diffuser_trainer.py --config "$CFG" --early-stop-patience "$PATIENCE"; \
      [ -d ./logs ] && cp -r ./logs "$OUT/lightning_logs" 2>/dev/null || true )

    CKPT="$(find_lightning_ckpt "$OUT")"
    [ -n "$CKPT" ] || { echo "[ERR] ckpt 없음: $OUT/lightning_logs"; exit 1; }
    # eval — medvit config 사용(arch=medvit 로 인코더 재구성). pretrained 다운로드 불필요(ckpt 가 덮어씀)
    DIFFMICV2_PRETRAINED=0 bash "$ROOT/tools/test_diffmicv2.sh" "$CKPT" "$TE" "$OUT/predictions_test_diffmicv2.npz" "$CFG"
done

echo ""; echo "########## 집계 ##########"
bash "$ROOT/viz.sh" b "$TAG"
echo "[medvitbb] DONE. $TAG AUC 를 baseline_32(0.864)와 비교하세요 (MedViT 백본 효과)."
