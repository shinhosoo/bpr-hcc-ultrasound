#!/usr/bin/env bash
# Phase 1 스모크 — two-tower 학습이 끝까지 돌고 p_disc/head 가 저장되는지 1 fold 짧게 확인.
#   성공 기준: 크래시 없이 끝나고 "[twotower] p_disc saved: ..." 가 찍히면 OK.
# Usage:  bash tools/run_twotower_smoke.sh [tag]
# env: FOLD=0  EPOCHS=5  PATIENCE=999  BPR_LAMBDA=1.0  BPR_PROTO=geomedian
set -e
TAG="${1:-twotower_smoke}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-5}"
PATIENCE="${PATIENCE:-999}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DM="$ROOT/models/DiffMICv2-main"

export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_NUM_CLASSES=2
export DIFFMICV2_PRETRAINED="${DIFFMICV2_PRETRAINED:-1}"
export DIFFMICV2_WEIGHTED_SAMPLER="${DIFFMICV2_WEIGHTED_SAMPLER:-1}"

D="$ROOT/data/5fold/fold_$FOLD/pkl"
TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"
for f in "$TR" "$VA"; do [ -f "$f" ] || { echo "[ERR] pkl 없음: $f"; exit 1; }; done

OUT="$ROOT/experiments/option_b_5fold/results/$TAG/diffmicv2/fold_$FOLD"; mkdir -p "$OUT"
CFG="$OUT/train_config.yml"
python3 "$ROOT/tools/diffmic_config_swap.py" --in "$DM/configs/lesion_binary.yml" --out "$CFG" --traindata "$TR" --testdata "$VA" $( [ -n "${ARCH:-}" ] && echo "--arch $ARCH" )
# 스모크용으로 epoch 단축
sed -i "s/^\([[:space:]]*n_epochs:[[:space:]]*\).*/\1$EPOCHS/" "$CFG" || true
echo "[twotower-smoke] tag=$TAG fold=$FOLD epochs=$EPOCHS (n_epochs in cfg)"
grep -n "n_epochs" "$CFG" | head -1

cd "$DM"
[ -d ./logs ] && rm -rf ./logs
DIFFMICV2_PRED_PATH="$OUT/predictions_val.npz" \
PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
python3 "$DM/run_diffmicv2_twotower.py" --config "$CFG" --early-stop-patience "$PATIENCE"

echo ""
echo "=== 스모크 결과 확인 ==="
ls -la "$OUT"/predictions_val_disc.npz "$OUT"/predictions_val_disc_head.pth 2>/dev/null \
  && echo "[OK] p_disc + head 저장됨 — Phase 1 동작 확인" \
  || echo "[WARN] p_disc 파일 없음 — 로그에서 [twotower] 메시지 확인 필요"
