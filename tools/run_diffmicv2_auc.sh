#!/usr/bin/env bash
# AUC-surrogate — DiffMICv2 독립 변형: train + test + viz 한 번에.
#   diffusion 예측 ŷ0 의 양성 점수가 음성보다 높게 랭킹되도록 pairwise surrogate 를
#   diffusion loss 에 가산(timestep-gated). 평가 AUC가 나오는 출력을 직접 최적화.
#   배관: run_diffmicv2_bpr.py (AUC_SURR_LAMBDA>0 이면 BPR injection 이전에 적용).
#
# Usage: bash tools/run_diffmicv2_auc.sh [tag]
# env: AUC_SURR_LAMBDA=1.0  AUC_SURR_MODE=logistic|hinge  AUC_SURR_TMAX=0.5
#      BPR_LAMBDA=0(기본=AUC만) / 1(BPR 병용)  VICREG_GAMMA(병용 시)  K=5  STAGE_ONLY=train|test|viz
set -e
TAG="${1:-auc_diffmicv2}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"

export TECH=bpr
export BPR_HOOK="${BPR_HOOK:-xweight}"     # 무해한 forward 캡쳐(파라미터 추가 없음)
export BPR_LAMBDA="${BPR_LAMBDA:-0}"       # 기본 0 = 순수 AUC-surrogate 효과만
export AUC_SURR_LAMBDA="${AUC_SURR_LAMBDA:-1.0}"
export AUC_SURR_MODE="${AUC_SURR_MODE:-logistic}"
export AUC_SURR_TMAX="${AUC_SURR_TMAX:-0.5}"
export BPR_TWO_PHASE="${BPR_TWO_PHASE:-0}"
export PATIENCE="${PATIENCE:-40}"

echo "============================================================"
echo " [diffmicv2 auc-surrogate]  tag=$TAG"
echo "   AUC_SURR_LAMBDA=$AUC_SURR_LAMBDA  mode=$AUC_SURR_MODE  tmax=$AUC_SURR_TMAX"
echo "   BPR_LAMBDA=$BPR_LAMBDA (0=AUC만)  hook=$BPR_HOOK"
echo "============================================================"

ONLY="${STAGE_ONLY:-all}"
[ "$ONLY" = "all" ] || [ "$ONLY" = "train" ] && { echo "[auc] TRAIN"; bash "$ROOT/train.sh" b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "test" ]  && { echo "[auc] TEST";  bash "$ROOT/test.sh"  b diffmicv2 "$TAG"; }
[ "$ONLY" = "all" ] || [ "$ONLY" = "viz" ]   && { echo "[auc] VIZ";   bash "$ROOT/viz.sh"   b "$TAG"; }
echo "[auc] DONE tag=$TAG  → baseline(0.8644)·vic_g3(0.8719)와 비교"
