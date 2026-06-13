#!/usr/bin/env bash
set -e
GEN="${1:-diffmicv2_baseline_32}"
DBPR="${2:-refine_base32_bpr}"
DNOBPR="${3:-refine_base32_nobpr}"
GMAX_LIST="${GMAX:-0.5}"
K="${K:-5}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
RES="$ROOT/experiments/option_b_5fold/results"

echo "[dualpath] gen=$GEN  disc(bpr)=$DBPR  disc(nobpr)=$DNOBPR  GMAX=[$GMAX_LIST]"

for G in $GMAX_LIST; do
    TB="dualpath_g${G}_bpr"
    TN="dualpath_g${G}_nobpr"
    python3 "$ROOT/tools/gated_fusion.py" --root "$RES" --gen "$GEN" --disc "$DBPR"   --out "$TB" --gmax "$G" --k "$K"
    python3 "$ROOT/tools/gated_fusion.py" --root "$RES" --gen "$GEN" --disc "$DNOBPR" --out "$TN" --gmax "$G" --k "$K"
    bash "$ROOT/viz.sh" b "$TB"
    bash "$ROOT/viz.sh" b "$TN"
done

echo ""
echo "[dualpath] results:"
echo "  - dualpath_g*_bpr vs '$GEN': gate fusion contribution"
echo "  - dualpath_*_bpr vs dualpath_*_nobpr: BPR discriminative path contribution"
echo "  Note: do not tune GMAX on the test set; use 0.5 as default and treat sweep as reference."
