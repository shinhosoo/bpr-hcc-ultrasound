"""임의 개수의 BPR 변형을 모델별로 한 바 그래프에 비교.

예시 입력:
  --tag baseline:baseline
  --tag bpr_2stage:baseline:_bpr
  --tag bpr_mean:bpr1stage                             # 1-stage mean
  --tag bpr_geomedian:bpr1stage_gm                     # 1-stage geomedian
  --tag bpr_sinkhorn:bpr1stage_sk                      # 1-stage sinkhorn
  --tag bpr_pcgrad_gm:bpr1stage_pcgrad_gm              # 1-stage pcgrad + geomedian

각 --tag 의 포맷:
  <표시이름>:<폴더태그>[:<파일접미사>]
    - 결과 파일 = <root>/<폴더태그>/predictions_test_<model>[<접미사>].npz
    - 접미사 미지정 = 그냥 .npz

--root <experiments/option_a_3way/results> 처럼 공통 루트를 지정.
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


PALETTE = ["#88AACC", "#CC8855", "#55AA77", "#9966CC",
           "#CCAA44", "#44AACC", "#CC5577"]


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


def load_npz(path):
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    return np.asarray(d['y_true']).astype(int).ravel(), _pos(d['y_score']).astype(float)


def parse_tag(spec):
    """'label:dir[:suffix]' → (label, dir, suffix)"""
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise SystemExit(f"--tag 잘못된 포맷: {spec}  (label:dir[:suffix])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="실험 결과 공통 루트 (예: experiments/option_a_3way/results)")
    ap.add_argument("--tag", action="append", required=True,
                    help="label:dir[:suffix] 형식. 여러 번 지정 가능.")
    ap.add_argument("--models", nargs="+",
                    default=["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["ACC", "BACC", "AUC", "F1", "Precision", "Recall", "Specificity"])
    args = ap.parse_args()

    variants = [parse_tag(t) for t in args.tag]
    n_var = len(variants)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_var)]

    rows = []
    for m in args.models:
        rec = {"model": m}
        for (label, d, sfx) in variants:
            path = os.path.join(args.root, d, f"predictions_test_{m}{sfx}.npz")
            y, s = load_npz(path)
            if y is not None:
                rec[label] = metrics(y, s)
        if len(rec) > 1:
            rows.append(rec)
            print(f"\n[{m}]")
            for k in args.metrics:
                vals = []
                for (label, _d, _sfx) in variants:
                    if label in rec:
                        vals.append(f"{label}={rec[label][k]:.4f}")
                print(f"  {k:12s}: " + "  ".join(vals))

    if not rows:
        raise SystemExit("[compare_variants] no data — --root / --tag 확인하세요.")

    n_models = len(rows); n_metrics = len(args.metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(5.2 * n_models, 5.8), squeeze=False)
    axes = axes[0]
    x = np.arange(n_metrics)
    width = 0.85 / n_var

    for i, r in enumerate(rows):
        ax = axes[i]
        for vi, (label, _d, _sfx) in enumerate(variants):
            if label not in r:
                continue
            vals = [r[label][k] for k in args.metrics]
            offset = (vi - (n_var - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=label, color=colors[vi])
            for j, v in enumerate(vals):
                if v == v:
                    ax.text(x[j] + offset, v + 0.012, f"{v:.3f}",
                            ha='center', va='bottom', fontsize=6,
                            rotation=45, rotation_mode='anchor')
        ax.set_xticks(x); ax.set_xticklabels(args.metrics, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.20); ax.set_title(r["model"])
        ax.set_ylabel("Score"); ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\n[compare_variants] saved PNG: {args.out}")

    # CSV
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["model", "variant"] + list(args.metrics)
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            for (label, _d, _sfx) in variants:
                if label in r:
                    w.writerow([r["model"], label] + [f"{r[label][k]:.4f}" for k in args.metrics])
    print(f"[compare_variants] saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
