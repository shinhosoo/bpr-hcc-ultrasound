#!/usr/bin/env bash
# (B) ResNet-18 DCG+BPR 분류기 — 학습 중 test 미리보기. (탐색 전용; 최종 보고는 val-선택)
#   현재 저장된 dcg_bpr.pth(=val 기준 best)를 한 fold 의 test 에 평가 → 같은 fold baseline 과 비교.
#   기본은 baseline worst fold 자동 선택.
#
# Usage:  bash tools/preview_dcg.sh <tag> [fold]
# env:    WORST_FROM=diffmicv2_baseline_32   (worst fold + 비교 기준)
set -e
TAG="${1:?usage: bash tools/preview_dcg.sh <tag> [fold]}"
FOLD="${2:-}"
WORST_FROM="${WORST_FROM:-diffmicv2_baseline_32}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
RES="$ROOT/experiments/option_b_5fold/results"
DM="$ROOT/models/DiffMICv2-main"

if [ -z "$FOLD" ]; then
    FOLD=$(python3 - "$RES/$WORST_FROM/diffmicv2" <<'PY'
import sys, glob, os, numpy as np
from sklearn.metrics import roc_auc_score
base=sys.argv[1]; worst=None; wa=2.0
for d in sorted(glob.glob(os.path.join(base,"fold_*"))):
    f=os.path.join(d,"predictions_test_diffmicv2.npz")
    if not os.path.exists(f): continue
    z=np.load(f,allow_pickle=True); y=np.asarray(z["y_true"]).astype(int).ravel()
    s=np.asarray(z["y_score"],dtype=float); p=s[:,1] if s.ndim==2 and s.shape[1]==2 else s.ravel()
    try: a=roc_auc_score(y,p)
    except Exception: continue
    i=int(d.rsplit("_",1)[1])
    if a<wa: wa=a; worst=i
print(worst if worst is not None else 0)
PY
)
    echo "[preview-dcg] worst fold (by $WORST_FROM AUC) = $FOLD"
fi

OUTDIR="$RES/$TAG/diffmicv2/fold_$FOLD"
CKPT="$OUTDIR/dcg_bpr.pth"
CFG="$OUTDIR/dcg_config.yml"
[ -f "$CKPT" ] || { echo "[preview-dcg] dcg_bpr.pth 없음: $OUTDIR (아직 best 저장 전일 수 있음)"; exit 1; }
[ -f "$CFG" ]  || { echo "[preview-dcg] dcg_config.yml 없음: $OUTDIR"; exit 1; }
TE="$ROOT/data/5fold/fold_$FOLD/pkl/lesion_test.pkl"
OUT="$OUTDIR/predictions_preview_test.npz"
echo "[preview-dcg] tag=$TAG fold=$FOLD ckpt=dcg_bpr.pth (현재 best)"

( cd "$DM" && PYTHONPATH="$ROOT/tools/bpr:$PWD:$PYTHONPATH" \
    python3 "$DM/eval_dcg.py" --config "$CFG" --ckpt "$CKPT" --test-pkl "$TE" --out "$OUT" ) | tail -1

python3 - "$OUT" "$RES/$WORST_FROM/diffmicv2/fold_$FOLD/predictions_test_diffmicv2.npz" "$FOLD" <<'PY'
import sys, os, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
def load(f):
    z=np.load(f,allow_pickle=True); y=np.asarray(z["y_true"]).astype(int).ravel()
    s=np.asarray(z["y_score"],dtype=float); p=s[:,1] if s.ndim==2 and s.shape[1]==2 else s.ravel()
    return y,p
def met(y,p):
    yp=(p>=0.5).astype(int)
    try: auc=roc_auc_score(y,p)
    except Exception: auc=float("nan")
    return dict(ACC=accuracy_score(y,yp),BACC=balanced_accuracy_score(y,yp),AUC=auc,F1=f1_score(y,yp,average="macro"))
prev=met(*load(sys.argv[1])); fold=sys.argv[3]
line=lambda t,m: f"  {t:9s} AUC={m['AUC']:.4f}  ACC={m['ACC']:.4f}  BACC={m['BACC']:.4f}  F1={m['F1']:.4f}"
print(f"\n\033[93m[preview-dcg][fold {fold}] (test, 현재 best DCG - 탐색 전용)\033[0m")
print(line("THIS",prev))
bf=sys.argv[2]
if os.path.exists(bf):
    base=met(*load(bf)); print(line("baseline",base))
    d=prev['AUC']-base['AUC']
    print(f"  dAUC vs baseline(same fold) = {d:+.4f}  -> " + ("ok/대등" if d>-0.03 else "낮음 - 중단 고려"))
PY
