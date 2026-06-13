#!/usr/bin/env bash
set -e
TAG="${1:?usage: bash tools/bpr/compare_5fold_bar.sh <tag> [models...]}"
shift || true
MODELS_RAW="${*:-medvit diffmic diffmicv2}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/../.." && pwd )"

declare -A LBL=(
    [medvit]=MedViT
    [medvitv2]=MedViTv2
    [diffmic]=DiffMIC
    [diffmicv2]=DiffMICv2
    [diffmicv2_sam]="DiffMICv2-SAM"
    [translive]=TransLiver
)
MODELS=()
LABELS=()
for m in $MODELS_RAW; do
    MODELS+=("$m")
    LABELS+=("${LBL[$m]:-$m}")
done

R="$ROOT/experiments/option_b_5fold/results/$TAG"
if [ ! -d "$R" ]; then
    echo "[ERR] results dir not found: $R" >&2; exit 2
fi

OUT="$R/compare_5fold_bar.png"
K="${K:-5}"

python3 "$ROOT/tools/bpr/compare_5fold_bar.py" \
    --results-root "$R" \
    --models "${MODELS[@]}" \
    --labels "${LABELS[@]}" \
    --k "$K" \
    --out "$OUT" \
    --title "${K}-fold CV — $TAG"

echo ""
echo "→ $OUT"
echo "→ ${OUT%.*}.csv  (summary)"
echo "→ ${OUT%.*}_perfold.csv  (long format)"
