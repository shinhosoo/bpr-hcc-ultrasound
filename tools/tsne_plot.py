#!/usr/bin/env python3
"""features.npz 의 latent 을 t-SNE/UMAP 으로 2D 시각화 + 정량 클러스터 분리도 지표.

각 모델 panel 제목에 다음을 같이 표시:
  - silhouette score (high-D 원본 feature 기준)
  - k-means clustering accuracy (2-cluster k-means → label 매칭)
  - linear probing accuracy (logistic regression on features)

Usage:
    python3 tools/tsne_plot.py \
        --features outputs/feats_medvit.npz:MedViT \
        --features outputs/feats_diffmic.npz:DiffMIC \
        --out outputs/tsne.png

각 npz: 'features' (N, D), 'labels' (N,).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


CLASS_NAME = {0: "Benign", 1: "Malignant"}
CLASS_COLOR = {0: "#1f77b4", 1: "#d62728"}


def cluster_metrics(X, y, seed=42):
    """X: (N, D) features, y: (N,) labels. 세 가지 지표 계산."""
    metrics = {}
    # 1) Silhouette score on high-D features
    try:
        if len(np.unique(y)) > 1 and len(y) >= 4:
            metrics["sil"] = float(silhouette_score(X, y, metric="euclidean"))
        else:
            metrics["sil"] = float("nan")
    except Exception:
        metrics["sil"] = float("nan")

    try:
        km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(X)
        pred = km.labels_
        acc_a = float((pred == y).mean())
        acc_b = float((1 - pred == y).mean())
        metrics["km_acc"] = max(acc_a, acc_b)
    except Exception:
        metrics["km_acc"] = float("nan")

    try:
        if len(y) > 10 and len(np.unique(y)) > 1:
            n_folds = min(5, int(np.bincount(y).min()))
            if n_folds >= 2:
                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, random_state=seed),
                )
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
                metrics["lp_acc"] = float(scores.mean())
                metrics["lp_std"] = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
            else:
                metrics["lp_acc"] = float("nan"); metrics["lp_std"] = 0.0
        else:
            metrics["lp_acc"] = float("nan"); metrics["lp_std"] = 0.0
    except Exception:
        metrics["lp_acc"] = float("nan"); metrics["lp_std"] = 0.0
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="append", required=True,
                    help="features.npz:DisplayName (반복 가능)")
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True,
                    help="output PNG 경로. --separate-files 면 디렉토리로 해석")
    ap.add_argument("--separate-files", action="store_true",
                    help="모델별로 독립된 PNG 파일을 만듦")
    args = ap.parse_args()

    items = []
    for f in args.features:
        path, name = (f.split(":", 1) + [None])[:2]
        if not name:
            name = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        X = d["features"]
        y = d["labels"].astype(int).ravel()
        m = cluster_metrics(X, y, seed=args.seed)
        items.append((name, X, y, m))
        print(f"[tsne] {name}: feats={X.shape}  N={len(y)}  "
              f"silhouette={m['sil']:.3f}  kmeans-acc={m['km_acc']:.3f}  "
              f"linear-probe={m['lp_acc']:.3f}±{m.get('lp_std',0):.3f}")

    if args.method == "tsne":
        from sklearn.manifold import TSNE
        def reduce(X):
            n = X.shape[0]
            perp = min(args.perplexity, max(5, (n - 1) // 3))
            return TSNE(n_components=2, perplexity=perp, random_state=args.seed,
                        init="pca", learning_rate="auto").fit_transform(X)
    else:
        import umap
        def reduce(X):
            return umap.UMAP(n_components=2, random_state=args.seed).fit_transform(X)

    def title_for(name, y, m):
        return (f"{name}  ({args.method.upper()}, N={len(y)})\n"
                f"silhouette={m['sil']:.3f}  k-means acc={m['km_acc']:.3f}  "
                f"linear probe={m['lp_acc']:.3f}±{m.get('lp_std',0):.3f}")

    if args.separate_files:
        out_dir = os.path.abspath(args.out)
        os.makedirs(out_dir, exist_ok=True)
        for name, X, y, m in items:
            Z = reduce(X)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            for c in sorted(set(y.tolist())):
                mask = (y == c)
                ax.scatter(Z[mask, 0], Z[mask, 1],
                           c=CLASS_COLOR.get(c, None), label=CLASS_NAME.get(c, str(c)),
                           s=18, alpha=0.75, edgecolors='none')
            ax.set_title(title_for(name, y, m), fontsize=11)
            ax.set_xlabel(f"{args.method}-1"); ax.set_ylabel(f"{args.method}-2")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            safe_name = name.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(out_dir, f"tsne_{safe_name}.png")
            plt.savefig(out_path, dpi=150); plt.close(fig)
            print(f"[tsne] saved: {out_path}")
        import csv
        csv_path = os.path.join(out_dir, "cluster_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "N", "silhouette", "kmeans_acc", "linear_probe_acc", "linear_probe_std"])
            for name, X, y, m in items:
                w.writerow([name, len(y), f"{m['sil']:.4f}",
                            f"{m['km_acc']:.4f}", f"{m['lp_acc']:.4f}"])
        print(f"[tsne] metrics CSV: {csv_path}")
        return

    n_plots = len(items)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.0 * rows), squeeze=False)

    for i, (name, X, y, m) in enumerate(items):
        Z = reduce(X)
        ax = axes[i // cols][i % cols]
        for c in sorted(set(y.tolist())):
            mask = (y == c)
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=CLASS_COLOR.get(c, None), label=CLASS_NAME.get(c, str(c)),
                       s=18, alpha=0.75, edgecolors='none')
        ax.set_title(title_for(name, y, m), fontsize=11)
        ax.set_xlabel(f"{args.method}-1"); ax.set_ylabel(f"{args.method}-2")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.25)

    for j in range(n_plots, rows * cols):
        axes[j // cols][j % cols].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[tsne] saved: {args.out}")

    import csv
    csv_path = os.path.splitext(args.out)[0] + "_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "silhouette", "kmeans_acc", "linear_probe_acc", "linear_probe_std"])
        for name, X, y, m in items:
            w.writerow([name, len(y), f"{m['sil']:.4f}",
                        f"{m['km_acc']:.4f}", f"{m['lp_acc']:.4f}",
                        f"{m.get('lp_std',0):.4f}"])
    print(f"[tsne] metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
