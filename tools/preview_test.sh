#!/usr/bin/env bash
# 학습 중 test 미리보기 — "가망 없으면 빨리 끊기"용. (탐색 전용; 최종 보고는 val-선택으로)
#
#   한 fold 만 현재 ckpt 로 test 평가 → ACC/AUC/BACC/F1 출력 + 같은 fold baseline 과 비교.
#   기본은 baseline 에서 '가장 안 나오는 fold'(worst by AUC)를 자동 선택 = 보수적 바닥 확인.
#
# Usage:
#   bash tools/preview_test.sh <tag> [fold]
#     fold 미지정 시 WORST_FROM(기본 diffmicv2_baseline_32)에서 worst fold 자동 선택.
#
# env:
#   WORST_FROM=diffmicv2_baseline_32   worst fold 산출 + 비교 기준 baseline tag
#   CKPT_PREFER=last                   학습 중이면 last.ckpt (기본), 끝났으면 best 도 가능
#   BPR_HOOK=...                       dual2ch/xweight_aux 등 파라미터 추가 모델이면 필수(eval 재구성)
#   BPR_DCLS / BPR_BN_DIM ...          해당 모델의 학습 때와 동일 값
set -e
TAG="${1:?usage: bash tools/preview_test.sh <tag> [fold]}"
FOLD="${2:-}"
WORST_FROM="${WORST_FROM:-diffmicv2_baseline_32}"
export CKPT_PREFER="${CKPT_PREFER:-last}"
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "$HERE/.." && pwd )"
RES="$ROOT/experiments/option_b_5fold/results"
CFG="$ROOT/models/DiffMICv2-main/configs/lesion_binary.yml"

# --- worst fold 자동 선택 (baseline per-fold AUC 최소) ---
if [ -z "$FOLD" ]; then
    FOLD=$(python3 - "$RES/$WORST_FROM/diffmicv2" <<'PY'
import sys, glob, os, numpy as np
from sklearn.metrics import roc_auc_score
base=sys.argv[1]; worst=None; worst_auc=2.0
for d in sorted(glob.glob(os.path.join(base,"fold_*"))):
    f=os.path.join(d,"predictions_test_diffmicv2.npz")
    if not os.path.exists(f): continue
    z=np.load(f,allow_pickle=True)
    y=np.asarray(z["y_true"]).astype(int).ravel()
    s=np.asarray(z["y_score"],dtype=float); p=s[:,1] if s.ndim==2 and s.shape[1]==2 else s.ravel()
    try: auc=roc_auc_score(y,p)
    except Exception: continue
    i=int(d.rsplit("_",1)[1])
    if auc<worst_auc: worst_auc=auc; worst=i
print(worst if worst is not None else 0)
PY
)
    echo "[preview] worst fold (by $WORST_FROM AUC) = $FOLD"
fi

# --- 현재 ckpt 찾기 (test.sh 와 동일 로직) ---
RM="$RES/$TAG/diffmicv2/fold_$FOLD"
# run 의 train_config.yml 이 있으면 그걸 사용 (백본/arch 자동 일치). 없으면 기본 config.
if [ -f "$RM/train_config.yml" ]; then
    CFG="$RM/train_config.yml"
    echo "[preview] run config 사용(arch 자동 일치): $CFG"
fi
DIR="$RM/lightning_logs"
CKPT=""
if [ "$CKPT_PREFER" = "last" ]; then
    CANDS=$(ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null; ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null)
else
    CANDS=$(ls -t "$DIR"/*/version_*/checkpoints/placental-*.ckpt 2>/dev/null; ls -t "$DIR"/*/version_*/checkpoints/last.ckpt 2>/dev/null)
fi
for f in $CANDS; do
    SZ=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    [ "$SZ" -ge 1048576 ] && { CKPT="$f"; break; }
done
[ -n "$CKPT" ] || { echo "[preview] ckpt 없음: $DIR (아직 1 epoch 저장 전일 수 있음)"; exit 1; }
echo "[preview] tag=$TAG  fold=$FOLD  ckpt=$(basename "$CKPT")  (prefer=$CKPT_PREFER)"

# --- test 평가 → preview 예측 npz ---
TPKL="$ROOT/data/5fold/fold_$FOLD/pkl/lesion_test.pkl"
OUT="$RM/predictions_preview_test.npz"
( cd "$ROOT/models/DiffMICv2-main" && python3 eval_only.py --config "$CFG" --ckpt "$CKPT" --test-pkl "$TPKL" --out "$OUT" ) >/dev/null 2>&1 || \
( cd "$ROOT/models/DiffMICv2-main" && python3 eval_only.py --config "$CFG" --ckpt "$CKPT" --test-pkl "$TPKL" --out "$OUT" )

# --- 지표 계산 + 같은 fold baseline 비교 ---
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
print(f"\n[preview][fold {fold}]  (test, 현재 ckpt — 탐색 전용)")
print(line("THIS",prev))
bf=sys.argv[2]
if os.path.exists(bf):
    base=met(*load(bf)); print(line("baseline",base))
    d=prev['AUC']-base['AUC']
    print(f"  ΔAUC vs baseline(same fold) = {d:+.4f}  → " + ("가망 있음/대등" if d> -0.03 else "확연히 낮음 — 중단 고려"))
else:
    print("  (baseline same-fold 예측 없음 — 비교 생략)")
PY
