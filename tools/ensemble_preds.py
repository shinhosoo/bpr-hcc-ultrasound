#!/usr/bin/env python3
"""두 소스의 per-fold 예측 npz 를 가중 평균으로 앙상블 → 새 tag 로 저장.

설계 검증용 probe: 'BPR-강화 판별 예측을 conditioning 밖에서 확산 예측과
출력 단계에서 융합하면 baseline 을 넘는가?' 를 학습 없이 테스트한다.

각 fold:
  pA = pos-prob( <root>/<src_a>/diffmicv2/fold_i/predictions_test_diffmicv2.npz )
  pB = pos-prob( <root>/<src_b>/diffmicv2/fold_i/... )
  p_ens = w*pA + (1-w)*pB
  → <root>/<out>/diffmicv2/fold_i/predictions_test_diffmicv2.npz  (y_true, y_score=[1-p,p])
이후 `bash viz.sh b <out>` 로 집계.
"""
import argparse, os
import numpy as np


def _pos(s):
    s = np.asarray(s, dtype=float)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results root (experiments/option_b_5fold/results)")
    ap.add_argument("--src-a", required=True, help="tag A (예: diffmicv2_baseline_32)")
    ap.add_argument("--src-b", required=True, help="tag B (예: refine_base32_bpr)")
    ap.add_argument("--out",   required=True, help="출력 tag (예: ens_base_x_bpr)")
    ap.add_argument("--model", default="diffmicv2")
    ap.add_argument("--weight", type=float, default=0.5, help="w (A 가중치). p=w*A+(1-w)*B")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    fn = f"predictions_test_{a.model}.npz"
    done = 0
    for i in range(a.k):
        pa = os.path.join(a.root, a.src_a, a.model, f"fold_{i}", fn)
        pb = os.path.join(a.root, a.src_b, a.model, f"fold_{i}", fn)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"[ensemble] fold {i} skip — 누락: "
                  f"{'A' if not os.path.exists(pa) else ''}{'B' if not os.path.exists(pb) else ''}")
            continue
        da = np.load(pa, allow_pickle=True); db = np.load(pb, allow_pickle=True)
        ya = np.asarray(da["y_true"]).astype(int).ravel()
        yb = np.asarray(db["y_true"]).astype(int).ravel()
        if ya.shape != yb.shape:
            raise SystemExit(f"[ensemble] fold {i}: N 불일치 A={ya.shape} B={yb.shape} — 같은 test set 인지 확인")
        if not np.array_equal(ya, yb):
            raise SystemExit(f"[ensemble] fold {i}: y_true 순서/내용 불일치 — 두 소스의 샘플 정렬이 다름. "
                             f"(같은 lesion_test.pkl, shuffle=False 인지 확인)")
        pA = _pos(da["y_score"]); pB = _pos(db["y_score"])
        p = a.weight * pA + (1.0 - a.weight) * pB
        p = np.clip(p, 0.0, 1.0)
        y_score = np.stack([1.0 - p, p], axis=1)

        out_dir = os.path.join(a.root, a.out, a.model, f"fold_{i}")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, fn), y_true=ya, y_score=y_score)
        done += 1
        print(f"[ensemble] fold {i}: N={len(ya)}  w={a.weight}  → {out_dir}/{fn}")

    if done == 0:
        raise SystemExit("[ensemble] 앙상블된 fold 0개 — --root/--src-a/--src-b 경로 확인")
    print(f"[ensemble] DONE  {done}/{a.k} folds  out tag='{a.out}'  →  bash viz.sh b {a.out}")


if __name__ == "__main__":
    main()
