"""baseline / +BPR (2-stage) / +BPR (1-stage) — 세 변형을 모델별로 한 바 그래프에 비교.

입력:
  --baseline-root <path>    : baseline 결과 폴더 (예: results/baseline)
                              predictions_test_<model>.npz + predictions_test_<model>_bpr.npz (2-stage) 를 가짐
  --bpr1stage-root <path>   : 1-stage BPR 결과 폴더 (예: results/bpr1stage)
                              predictions_test_<model>.npz 를 가짐

출력:
  --out <path.png>           : 바 그래프 PNG
  자동: 같은 경로에 .csv 도 저장
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def _pos(s):
    s = np.asarray(s)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return dict(
        ACC=accuracy_score(y, yp),
        BACC=balanced_accuracy_score(y, yp),
        AUC=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        AP=average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        F1=f1_score(y, yp, average='macro'),
        Precision=prec, Recall=sens, Specificity=spec,
    )


def load(path):
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return np.asarray(d['y_true']).astype(int).ravel(), _pos(d['y_score']).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--bpr1stage-root", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["ACC", "BACC", "AUC", "F1", "Precision", "Recall", "Specificity"])
    args = ap.parse_args()

    rows = []
    for m in args.models:
        p_base = os.path.join(args.baseline_root,  f"predictions_test_{m}.npz")
        p_2st  = os.path.join(args.baseline_root,  f"predictions_test_{m}_bpr.npz")
        p_1st  = os.path.join(args.bpr1stage_root, f"predictions_test_{m}.npz")
        y_b, s_b = load(p_base)
        y_2, s_2 = load(p_2st)
        y_1, s_1 = load(p_1st)
        if y_b is None and y_2 is None and y_1 is None:
            print(f"[skip {m}] 어떤 결과도 없음")
            continue
        rec = {"model": m}
        if y_b is not None: rec["baseline"] = metrics(y_b, s_b)
        if y_2 is not None: rec["bpr2stage"] = metrics(y_2, s_2)
        if y_1 is not None: rec["bpr1stage"] = metrics(y_1, s_1)
        rows.append(rec)
        print(f"\n[{m}]")
        for k in args.metrics:
            vals = []
            for tag in ("baseline", "bpr2stage", "bpr1stage"):
                if tag in rec: vals.append(f"{tag}={rec[tag][k]:.4f}")
            print(f"  {k:12s}: " + "  ".join(vals))

    if not rows:
        raise SystemExit("[compare_3way] no data")

    variants = [("baseline", "baseline", "#88AACC"),
                ("bpr2stage", "+BPR (2-stage)", "#CC8855"),
                ("bpr1stage", "+BPR (1-stage)", "#55AA77")]
    n_models = len(rows); n_metrics = len(args.metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 5.5), squeeze=False)
    axes = axes[0]
    x = np.arange(n_metrics)
    n_var = len(variants); width = 0.85 / n_var

    for i, r in enumerate(rows):
        ax = axes[i]
        for vi, (key, label, color) in enumerate(variants):
            if key not in r:
                continue
            vals = [r[key][k] for k in args.metrics]
            offset = (vi - (n_var - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=label, color=color)
            for j, v in enumerate(vals):
                if v == v:
                    ax.text(x[j] + offset, v + 0.012, f"{v:.3f}",
                            ha='center', va='bottom', fontsize=7,
                            rotation=45, rotation_mode='anchor')
        ax.set_xticks(x); ax.set_xticklabels(args.metrics, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.18); ax.set_title(r["model"])
        ax.set_ylabel("Score"); ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\n[compare_3way] saved PNG: {args.out}")

    # CSV
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["model", "variant"] + list(args.metrics)
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            for key, label, _c in variants:
                if key in r:
                    w.writerow([r["model"], label] + [f"{r[key][k]:.4f}" for k in args.metrics])
    print(f"[compare_3way] saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
