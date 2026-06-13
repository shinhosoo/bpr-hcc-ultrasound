#!/usr/bin/env python3
"""fold_*/predictions_*.npz 를 합쳐 pooled npz 생성 (ROC/PR 겹쳐그리기용).

각 --fold-dir 아래 fold_*/<pred-name> 들을 모아 <fold-dir>/<out-name> 에 저장.
y_true, y_score 를 concat. (HSQ 는 predictions_val.npz, 타 모델은 predictions_test_<m>.npz)

Usage:
  python3 tools/pool_preds.py \\
    --fold-dir experiments/option_b_5fold/results/hsq_with_pretrained/hsq \\
    --fold-dir experiments/option_b_5fold/results/hsq_base_bpr_with_pretrained/hsq
"""
import argparse, glob, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fold-dir", action="append", required=True,
                help="fold_* 폴더를 포함한 디렉터리 (반복 가능)")
ap.add_argument("--pred-name", default="predictions_val.npz",
                help="fold 안의 npz 파일명 (기본 predictions_val.npz)")
ap.add_argument("--out-name", default="predictions_pooled.npz",
                help="저장할 pooled npz 파일명")
a = ap.parse_args()

for root in a.fold_dir:
    fs = sorted(glob.glob(os.path.join(root, "fold_*", a.pred_name)))
    if not fs:
        print(f"[skip] no {a.pred_name} under {root}")
        continue
    ys, ss = [], []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        ys.append(np.asarray(d["y_true"]))
        ss.append(np.asarray(d["y_score"]))
    out = os.path.join(root, a.out_name)
    np.savez(out, y_true=np.concatenate(ys), y_score=np.concatenate(ss))
    print(f"[ok] {out}  folds={len(fs)}  N={len(np.concatenate(ys))}")
