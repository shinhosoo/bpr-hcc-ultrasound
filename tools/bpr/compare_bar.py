"""각 모델별로 baseline (predictions_test_<m>.npz) 과 +BPR (predictions_test_<m>_bpr.npz)
   메트릭을 비교하는 바 그래프 + CSV 생성."""
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
        Precision=prec,
        Recall=sens,
        Specificity=spec,
    )


def load(path):
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return np.asarray(d['y_true']).astype(int).ravel(), _pos(d['y_score']).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="experiments/option_a_3way/results/<tag>/ 같은 폴더")
    ap.add_argument("--models", nargs="+",
                    default=["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"],
                    help="비교할 모델 목록")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--metrics", nargs="+",
                    default=["ACC", "BACC", "AUC", "F1", "Precision", "Recall", "Specificity"])
    args = ap.parse_args()

    rows = []
    for m in args.models:
        base_path = os.path.join(args.results_root, f"predictions_test_{m}.npz")
        bpr_path  = os.path.join(args.results_root, f"predictions_test_{m}_bpr.npz")
        y_b, p_b = load(base_path)
        y_r, p_r = load(bpr_path)
        if y_b is None or y_r is None:
            print(f"[compare] skip {m}: baseline={y_b is not None}  bpr={y_r is not None}")
            continue
        mb = metrics(y_b, p_b)
        mr = metrics(y_r, p_r)
        print(f"\n[{m}]")
        for k in args.metrics:
            print(f"  {k:12s}: baseline={mb[k]:.4f}  +BPR={mr[k]:.4f}  Δ={mr[k]-mb[k]:+.4f}")
        rows.append({"model": m, "baseline": mb, "bpr": mr})

    if not rows:
        raise SystemExit("[compare] 비교할 데이터 없음 — baseline/bpr npz 확인")

    n_models = len(rows); n_metrics = len(args.metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 5), squeeze=False)
    axes = axes[0]
    x = np.arange(n_metrics)
    width = 0.36
    for i, r in enumerate(rows):
        ax = axes[i]
        vals_b = [r["baseline"][k] for k in args.metrics]
        vals_r = [r["bpr"][k]      for k in args.metrics]
        b1 = ax.bar(x - width/2, vals_b, width, label="baseline", color="#88AACC")
        b2 = ax.bar(x + width/2, vals_r, width, label="+BPR",     color="#CC8855")
        for j, (v1, v2) in enumerate(zip(vals_b, vals_r)):
            ax.text(x[j] - width/2, v1 + 0.012, f"{v1:.3f}",
                    ha='center', va='bottom', fontsize=7, rotation=45, rotation_mode='anchor')
            ax.text(x[j] + width/2, v2 + 0.012, f"{v2:.3f}",
                    ha='center', va='bottom', fontsize=7, rotation=45, rotation_mode='anchor')
        ax.set_xticks(x); ax.set_xticklabels(args.metrics, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_title(r["model"])
        ax.set_ylabel("Score"); ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\n[compare] saved bar PNG: {args.out}")

    # CSV — long format
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["model", "variant"] + list(args.metrics) + ["delta_summary"]
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r["model"], "baseline"] + [f"{r['baseline'][k]:.4f}" for k in args.metrics] + [""])
            deltas = {k: r["bpr"][k] - r["baseline"][k] for k in args.metrics}
            delta_str = "; ".join(f"{k}{v:+.3f}" for k, v in deltas.items())
            w.writerow([r["model"], "+BPR"] + [f"{r['bpr'][k]:.4f}" for k in args.metrics] + [delta_str])
    print(f"[compare] saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
