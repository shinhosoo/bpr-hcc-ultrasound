#!/usr/bin/env bash
# Two-Tower Phase 1.5 — 5-fold 정식 학습 + test p_gen/p_disc 평가 + 융합.
#   fold 마다: (1) two-tower 학습(diffusion + co-train 판별 head)
#             (2) eval_twotower 로 test 에서 p_gen(diffusion) + p_disc(판별) 평가
#   끝나면: gated_fusion 으로 p_gen ⊕ p_disc 융합 → baseline_32(0.864)와 비교.
#
# 결과 tag:
#   <tag>        : diffusion(p_gen)
#   <tag>_disc   : 판별 head(p_disc)
#   <tag>_fused  : 게이트 융합
#
# Usage:  bash tools/run_twotower_5fold.sh [tag]
# env: K=5  PATIENCE=40  BPR_LAMBDA=1.0  BPR_PROTO=geomedian  TT_PROJ_DIM=256  TT_LR=1e-3  GMAX=0.5
set -e
TAG="${1:-twotower}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
DM="$ROOT/models/DiffMICv2-main"
RES="$ROOT/experiments/option_b_5fold/results"
K="${K:-5}"; PATIENCE="${PATIENCE:-40}"; GMAX="${GMAX:-0.5}"
ARCH="${ARCH:-}"   # 비우면 config 기본(resnet18). resnet50|vit|densenet121 가능

export BPR_LAMBDA="${BPR_LAMBDA:-1.0}"
export BPR_PROTO="${BPR_PROTO:-geomedian}"
export BPR_NUM_CLASSES=2
export TT_PROJ_DIM="${TT_PROJ_DIM:-256}"
export TT_LR="${TT_LR:-1e-3}"
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

for ((i=0; i<K; i++)); do
    echo ""; echo "########## TWO-TOWER  fold $i ##########"
    D="$ROOT/data/5fold/fold_$i/pkl"
    TR="$D/lesion_train.pkl"; VA="$D/lesion_val.pkl"; TE="$D/lesion_test.pkl"
    for f in "$TR" "$VA" "$TE"; do [ -f "$f" ] || { echo "[ERR] pkl 없음: $f"; exit 1; }; done

    OUT="$RES/$TAG/diffmicv2/fold_$i"; mkdir -p "$OUT"
    DOUT="$RES/${TAG}_disc/diffmicv2/fold_$i"; mkdir -p "$DOUT"
    CFG="$OUT/train_config.yml"
    python3 "$ROOT/tools/diffmic_config_swap.py" --in "$DM/configs/lesion_binary.yml" --out "$CFG" --traindata "$TR" --testdata "$VA" $( [ -n "$ARCH" ] && echo "--arch $ARCH" )

    # (1) 학습
    ( cd "$DM" && [ -d ./logs ] && rm -rf ./logs; \
      DIFFMICV2_PRED_PATH="$OUT/predictions_val.npz" \
      PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
      python3 "$DM/run_diffmicv2_twotower.py" --config "$CFG" --early-stop-patience "$PATIENCE"; \
      [ -d ./logs ] && cp -r ./logs "$OUT/lightning_logs" 2>/dev/null || true )

    CKPT="$(find_lightning_ckpt "$OUT")"
    HEAD="$OUT/predictions_val_disc_head.pth"
    [ -n "$CKPT" ] || { echo "[ERR] diffusion ckpt 없음: $OUT/lightning_logs"; exit 1; }
    [ -f "$HEAD" ] || { echo "[ERR] disc head 없음: $HEAD"; exit 1; }

    # (2) test 평가 — p_gen + p_disc
    ( cd "$DM" && PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
      python3 "$DM/eval_twotower.py" --config "$CFG" --ckpt "$CKPT" --disc-head "$HEAD" \
        --test-pkl "$TE" \
        --out "$OUT/predictions_test_diffmicv2.npz" \
        --out-disc "$DOUT/predictions_test_diffmicv2.npz" )
done

echo ""; echo "########## 융합 + 집계 ##########"
python3 "$ROOT/tools/gated_fusion.py" --root "$RES" \
    --gen "$TAG" --disc "${TAG}_disc" --out "${TAG}_fused" --gmax "$GMAX" --k "$K"

bash "$ROOT/viz.sh" b "$TAG"          # diffusion 단독
bash "$ROOT/viz.sh" b "${TAG}_disc"   # 판별 단독
bash "$ROOT/viz.sh" b "${TAG}_fused"  # 융합

echo ""
echo "[two-tower] DONE. 비교:"
echo "  $TAG (diffusion) / ${TAG}_disc (판별) / ${TAG}_fused (융합)  vs  diffmicv2_baseline_32 (AUC 0.864)"
echo "  ${TAG}_fused 가 baseline 과 $TAG 둘 다 넘으면 → co-train 두-타워가 사후 앙상블(+0.0047)보다 효과적."
