#!/usr/bin/env bash
# 출력-융합(late fusion) probe — 학습 없음.
#   확산 예측(baseline) + BPR-판별 예측(refine)을 가중 평균해 baseline 을 넘는지 검증.
#   넘으면 two-tower 재설계 투자 가치 있음 / 못 넘으면 천장 → bs=64 로 마무리.
#
# Usage:
#   bash tools/run_ensemble_probe.sh [base_tag] [bpr_tag] [nobpr_tag]
#     base_tag  : 확산 예측 출처 (기본 diffmicv2_baseline_32)
#     bpr_tag   : refine BPR 예측 (기본 refine_base32_bpr)
#     nobpr_tag : refine no-BPR 예측 (기본 refine_base32_nobpr, 대조군)
# env: W=0.5 (base 가중치)   K=5
set -e
BASE_TAG="${1:-diffmicv2_baseline_32}"
BPR_TAG="${2:-refine_base32_bpr}"
NOBPR_TAG="${3:-refine_base32_nobpr}"
W="${W:-0.5}"
K="${K:-5}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
RES="$ROOT/experiments/option_b_5fold/results"

echo "[probe] base=$BASE_TAG  +  BPR=$BPR_TAG / noBPR=$NOBPR_TAG   (w=$W on base)"

python3 "$ROOT/tools/ensemble_preds.py" --root "$RES" \
    --src-a "$BASE_TAG" --src-b "$BPR_TAG"   --out "ens_${W}_base_x_bpr"   --weight "$W" --k "$K"
python3 "$ROOT/tools/ensemble_preds.py" --root "$RES" \
    --src-a "$BASE_TAG" --src-b "$NOBPR_TAG" --out "ens_${W}_base_x_nobpr" --weight "$W" --k "$K"

echo "[probe] 집계:"
bash "$ROOT/viz.sh" b "ens_${W}_base_x_bpr"
bash "$ROOT/viz.sh" b "ens_${W}_base_x_nobpr"

echo ""
echo "[probe] 해석:"
echo "  - ens_..._base_x_bpr 가 baseline_32 를 넘으면 → 출력 융합에 신호 有 → two-tower 가치 有."
echo "  - bpr 앙상블 > nobpr 앙상블 이면 → 그 이득이 BPR 덕분."
echo "  - 둘 다 baseline 못 넘으면 → 천장. bs=64 메인으로 마무리 권장."
