#!/usr/bin/env python3
"""Dual-Path 융합 — 불확실성 게이트 late fusion (학습 없음).

설계 4블록 중 '출력 융합' 부품. 고정 가중 앙상블(실패)과 달리, 생성 예측 p_gen 이
**불확실한 샘플에서만** 판별 예측 p_disc 를 신뢰한다.

  c    = |2*p_gen - 1|
  g    = gmax * (1 - c)
  p    = (1 - g) * p_gen + g * p_disc

per-fold 로 <root>/<gen>/.../predictions_test_diffmicv2.npz (p_gen) 와
<root>/<disc>/... (p_disc) 를 읽어 융합 → <root>/<out>/... 저장. 이후 viz.sh 로 집계.

p_gen=DiffMICv2 baseline, p_disc=refine BPR head 가 표준 사용.
"""
import argparse, os
import numpy as np


def _pos(s):
    s = np.asarray(s, dtype=float)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gen",  required=True, help="p_gen tag (예: diffmicv2_baseline_32)")
    ap.add_argument("--disc", required=True, help="p_disc tag (예: refine_base32_bpr)")
    ap.add_argument("--out",  required=True, help="출력 tag")
    ap.add_argument("--model", default="diffmicv2")
    ap.add_argument("--gmax", type=float, default=0.5, help="disc 최대 가중치 (불확실 샘플에서)")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    fn = f"predictions_test_{a.model}.npz"
    done = 0
    for i in range(a.k):
        pg = os.path.join(a.root, a.gen,  a.model, f"fold_{i}", fn)
        pd = os.path.join(a.root, a.disc, a.model, f"fold_{i}", fn)
        if not (os.path.exists(pg) and os.path.exists(pd)):
            print(f"[gated] fold {i} skip — 누락 "
                  f"{'gen ' if not os.path.exists(pg) else ''}{'disc' if not os.path.exists(pd) else ''}")
            continue
        dg = np.load(pg, allow_pickle=True); dd = np.load(pd, allow_pickle=True)
        yg = np.asarray(dg["y_true"]).astype(int).ravel()
        yd = np.asarray(dd["y_true"]).astype(int).ravel()
        if yg.shape != yd.shape or not np.array_equal(yg, yd):
            raise SystemExit(f"[gated] fold {i}: y_true 불일치 — 두 소스의 test 정렬이 다름 "
                             f"(같은 lesion_test.pkl, shuffle=False 인지 확인)")
        p_gen = _pos(dg["y_score"]); p_disc = _pos(dd["y_score"])
        c = np.abs(2.0 * p_gen - 1.0)            # p_gen confidence
        g = a.gmax * (1.0 - c)
        p = (1.0 - g) * p_gen + g * p_disc
        p = np.clip(p, 0.0, 1.0)
        y_score = np.stack([1.0 - p, p], axis=1)

        out_dir = os.path.join(a.root, a.out, a.model, f"fold_{i}")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, fn), y_true=yg, y_score=y_score)
        done += 1
        print(f"[gated] fold {i}: N={len(yg)}  gmax={a.gmax}  mean_g={g.mean():.3f}  → {out_dir}/{fn}")

    if done == 0:
        raise SystemExit("[gated] 융합된 fold 0개 — --root/--gen/--disc 경로 확인")
    print(f"[gated] DONE  {done}/{a.k}  out='{a.out}'  →  bash viz.sh b {a.out}")


if __name__ == "__main__":
    main()
