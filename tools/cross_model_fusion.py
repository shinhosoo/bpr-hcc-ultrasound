#!/usr/bin/env python3
"""Cross-model 융합 — MedViT(imagefolder 순서) + DiffMICv2(pkl 순서) test 예측을 파일명으로 정렬 후 융합.

두 모델은 같은 5-fold test 이미지를 평가하지만 순서가 다르다(MedViT=ImageFolder sorted,
DiffMICv2=pkl list). pkl 의 img_root 가 imagefolder 경로를 가리키므로 **basename 으로 정렬**한다.
정렬 안전장치: 정렬된 쌍의 라벨이 일치하지 않으면 즉시 에러(잘못된 정렬 방지).

융합:
  gated : g = gmax*(1-|2*p_diff-1|), p = (1-g)*p_diff + g*p_med   (diffusion 불확실할 때만 MedViT)
  mean  : p = w*p_diff + (1-w)*p_med
저장: <root>/<out>/diffmicv2/fold_i/predictions_test_diffmicv2.npz  → viz.sh b <out> 로 집계.
"""
import argparse, os, pickle
import numpy as np


def _pos(s):
    s = np.asarray(s, dtype=float)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def _imagefolder_order(test_dir):
    """torchvision ImageFolder 와 동일 순서: class 정렬 → class 내 파일 정렬."""
    order = []
    if not os.path.isdir(test_dir):
        return order
    for cls in sorted(os.listdir(test_dir)):
        cdir = os.path.join(test_dir, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            order.append((os.path.basename(f), cls))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results root")
    ap.add_argument("--data-root", required=True, help="data/5fold 경로")
    ap.add_argument("--diff-tag", required=True, help="DiffMICv2 tag (예: diffmicv2_baseline_32)")
    ap.add_argument("--med-tag", required=True, help="MedViT tag (예: bpr_medvit_2stage_aux)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gated", "mean"], default="gated")
    ap.add_argument("--gmax", type=float, default=0.5)
    ap.add_argument("--weight", type=float, default=0.6, help="mean 모드 diffusion 가중")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    done = 0
    for i in range(a.k):
        pkl_path = os.path.join(a.data_root, f"fold_{i}", "pkl", "lesion_test.pkl")
        d_npz = os.path.join(a.root, a.diff_tag, "diffmicv2", f"fold_{i}", "predictions_test_diffmicv2.npz")
        m_npz = os.path.join(a.root, a.med_tag, "medvit", f"fold_{i}", "predictions_test_medvit.npz")
        img_dir = os.path.join(a.data_root, f"fold_{i}", "imagefolder", "test")
        for f in (pkl_path, d_npz, m_npz):
            if not os.path.exists(f):
                print(f"[xfuse] fold {i} skip — 누락: {f}"); break
        else:
            pkl = pickle.load(open(pkl_path, "rb"))
            dd = np.load(d_npz, allow_pickle=True); mm = np.load(m_npz, allow_pickle=True)
            p_d = _pos(dd["y_score"]); y_d = np.asarray(dd["y_true"]).astype(int).ravel()
            p_m = _pos(mm["y_score"]); y_m = np.asarray(mm["y_true"]).astype(int).ravel()
            if len(pkl) != len(p_d):
                raise SystemExit(f"[xfuse] fold {i}: pkl({len(pkl)}) != diff preds({len(p_d)})")
            # diff: basename -> (p_diff, label)
            dmap = {}
            for k, e in enumerate(pkl):
                bn = os.path.basename(e["img_root"])
                dmap[bn] = (p_d[k], int(e["label"]))
            order = _imagefolder_order(img_dir)
            if len(order) != len(p_m):
                raise SystemExit(f"[xfuse] fold {i}: imagefolder({len(order)}) != med preds({len(p_m)}) "
                                 f"— ImageFolder 순서 재구성 불일치")
            mmap = {bn: (p_m[j], y_m[j]) for j, (bn, cls) in enumerate(order)}
            bns = list(dmap.keys())
            miss = [b for b in bns if b not in mmap]
            if miss:
                raise SystemExit(f"[xfuse] fold {i}: med 에 없는 파일 {len(miss)}개 (예 {miss[:3]})")
            pd = np.array([dmap[b][0] for b in bns])
            pm = np.array([mmap[b][0] for b in bns])
            yd = np.array([dmap[b][1] for b in bns])
            ym = np.array([mmap[b][1] for b in bns])
            if not np.array_equal(yd, ym):
                nbad = int((yd != ym).sum())
                raise SystemExit(f"[xfuse] fold {i}: 정렬 후 라벨 불일치 {nbad}개 — 정렬 오류! "
                                 f"(pkl basename ↔ imagefolder 매핑 확인)")
            if a.mode == "gated":
                g = a.gmax * (1.0 - np.abs(2.0 * pd - 1.0))
                p = (1.0 - g) * pd + g * pm
            else:
                p = a.weight * pd + (1.0 - a.weight) * pm
            p = np.clip(p, 0.0, 1.0)
            y_score = np.stack([1.0 - p, p], axis=1)
            outdir = os.path.join(a.root, a.out, "diffmicv2", f"fold_{i}"); os.makedirs(outdir, exist_ok=True)
            np.savez(os.path.join(outdir, "predictions_test_diffmicv2.npz"),
                     y_true=yd.astype(int), y_score=y_score)
            done += 1
            print(f"[xfuse] fold {i}: N={len(bns)}  mode={a.mode}  라벨정렬 OK  → {outdir}")
    if done == 0:
        raise SystemExit("[xfuse] 융합된 fold 0개 — tag/경로 확인")
    print(f"[xfuse] DONE {done}/{a.k}  out='{a.out}'  →  bash viz.sh b {a.out}")


if __name__ == "__main__":
    main()
