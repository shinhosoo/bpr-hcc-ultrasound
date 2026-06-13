#!/usr/bin/env python3
"""
5-fold CV 결과를 모델별 mean ± std bar graph로 시각화.

Usage:
    python3 experiments/option_b_5fold/plot_5fold_bar.py --tag baseline_balanced

    python3 experiments/option_b_5fold/plot_5fold_bar.py \
        --tag baseline_balanced --tag2 bpr_dual_adv_balanced

옵션:
    --metrics ACC AUC AP F1_macro Sens Spec   (default: ACC AUC AP F1_macro)
    --out <path.png>     (default: <results>/<tag>/metrics_bar_5fold.png)
    --error std|sem|ci95 (default: std)
"""
import argparse, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, confusion_matrix)

ALL_MODELS = ["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"]
MODEL_DISPLAY = {"medvit": "MedViT", "medvitv2": "MedViTV2",
                 "diffmic": "DiffMIC", "diffmicv2": "DiffMICv2",
                 "diffmicv2_sam": "DiffMICv2-SAM"}
DEFAULT_METRICS = ["ACC", "Precision", "Recall", "F1", "AUC"]
ALL_METRICS = ["ACC", "BACC", "AUC", "AP",
               "Precision", "Recall", "F1",
               "Precision_macro", "Recall_macro", "F1_macro",
               "Sens", "Spec", "NPV"]

MODELS = ALL_MODELS


def _pos(s):
    s = np.asarray(s)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def metrics_of(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    multi_class = len(np.unique(y)) > 1
    return dict(
        ACC=accuracy_score(y, yp),
        BACC=balanced_accuracy_score(y, yp),
        AUC=(roc_auc_score(y, p) if multi_class else float("nan")),
        AP=(average_precision_score(y, p) if multi_class else float("nan")),
        Precision=precision_score(y, yp, pos_label=1, zero_division=0),
        Recall=recall_score(y, yp, pos_label=1, zero_division=0),
        F1=f1_score(y, yp, pos_label=1, zero_division=0),
        Precision_macro=precision_score(y, yp, average="macro", zero_division=0),
        Recall_macro=recall_score(y, yp, average="macro", zero_division=0),
        F1_macro=f1_score(y, yp, average="macro", zero_division=0),
        Sens=tp / (tp + fn) if tp + fn else float("nan"),
        Spec=tn / (tn + fp) if tn + fp else float("nan"),
        NPV=tn / (tn + fn) if tn + fn else float("nan"),
    )


def per_fold_metrics(root, model):
    """Return list of metric dicts, one per fold that has predictions.
    두 레이아웃 모두 지원:
      model-first: <root>/<model>/fold_*/predictions_test_<model>.npz   (현재 표준)
      legacy:      <root>/fold_*/predictions_test_<model>.npz
    """
    rows = []
    candidates = sorted(set(
        glob.glob(os.path.join(root, model, "fold_*", f"predictions_test_{model}.npz"))
        + glob.glob(os.path.join(root, "fold_*", f"predictions_test_{model}.npz"))
    ))
    for f in candidates:
        d = np.load(f, allow_pickle=True)
        y = np.asarray(d["y_true"]).astype(int).ravel()
        p = _pos(d["y_score"]).astype(float)
        rows.append(metrics_of(y, p))
    return rows


def aggregate(rows, metric, error="std"):
    vals = np.array([r[metric] for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    if error == "std":
        err = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    elif error == "sem":
        err = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    elif error == "ci95":
        # 95% CI half-width using t-distribution (small n)
        try:
            from scipy import stats
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            err = float(stats.t.ppf(0.975, len(vals) - 1) * sem) if len(vals) > 1 else 0.0
        except ImportError:
            err = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    else:
        err = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, err, len(vals)


def make_single_tag_plot(rows_per_model, metrics, error_label, tag, out_path, models):
    """Bar chart: x=metric, group=model. Mean bar + error bar + value label."""
    n_metrics = len(metrics)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * n_metrics), 5.5))
    colors = plt.get_cmap("tab10").colors
    ax.set_ylim(0, 1.18)

    for i, model in enumerate(models):
        rows = rows_per_model.get(model, [])
        means, errs = [], []
        for m in metrics:
            mean, err, _ = aggregate(rows, m, error=error_label)
            means.append(mean); errs.append(err)
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2, means,
                      width=width * 0.95, yerr=errs, capsize=4,
                      label=MODEL_DISPLAY[model], color=colors[i],
                      edgecolor="black", linewidth=0.5)
        for b, mv, er in zip(bars, means, errs):
            if np.isnan(mv): continue
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + er + 0.012,
                    f"{mv:.3f}\n±{er:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, linespacing=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=(0 if n_metrics <= 6 else 30), ha="right" if n_metrics > 6 else "center")
    ax.set_ylabel("Score")
    ax.set_title(f"5-fold CV — {tag}  (mean ± {error_label}, value labels on bars)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_dual_tag_plot(rows_per_model_a, rows_per_model_b, metrics,
                        error_label, tag_a, tag_b, out_path, models):
    """Two TAG comparison: x=metric, group=(model x tag) interleaved.
    각 막대 위에 mean 만 표시 (양/음 짝이라 std 까지 쓰면 겹침)."""
    n_metrics = len(metrics)
    n_models = len(models)
    bar_w = 0.8 / (n_models * 2)
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(11, 2.2 * n_metrics), 6))
    cmap = plt.get_cmap("tab10").colors
    ax.set_ylim(0, 1.15)

    def _label(bars, means, errs):
        for b, mv, er in zip(bars, means, errs):
            if np.isnan(mv): continue
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + er + 0.008,
                    f"{mv:.3f}",
                    ha="center", va="bottom", fontsize=7,
                    rotation=90 if n_models * 2 > 4 else 0)

    for i, model in enumerate(models):
        # tag A (solid)
        rows_a = rows_per_model_a.get(model, [])
        means_a, errs_a = [], []
        for m in metrics:
            mean, err, _ = aggregate(rows_a, m, error=error_label)
            means_a.append(mean); errs_a.append(err)
        off_a = (2 * i) * bar_w - (n_models * 2 - 1) * bar_w / 2
        bars_a = ax.bar(x + off_a, means_a, width=bar_w * 0.95, yerr=errs_a, capsize=3,
                        label=f"{MODEL_DISPLAY[model]} ({tag_a})",
                        color=cmap[i], edgecolor="black", linewidth=0.5)
        _label(bars_a, means_a, errs_a)

        # tag B (hatched)
        rows_b = rows_per_model_b.get(model, [])
        means_b, errs_b = [], []
        for m in metrics:
            mean, err, _ = aggregate(rows_b, m, error=error_label)
            means_b.append(mean); errs_b.append(err)
        off_b = (2 * i + 1) * bar_w - (n_models * 2 - 1) * bar_w / 2
        bars_b = ax.bar(x + off_b, means_b, width=bar_w * 0.95, yerr=errs_b, capsize=3,
                        label=f"{MODEL_DISPLAY[model]} ({tag_b})",
                        color=cmap[i], edgecolor="black", linewidth=0.5, hatch="//")
        _label(bars_b, means_b, errs_b)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=(0 if n_metrics <= 6 else 30), ha="right" if n_metrics > 6 else "center")
    ax.set_ylabel("Score")
    ax.set_title(f"5-fold CV comparison  (mean ± {error_label})\n"
                 f"solid = {tag_a}   hatched = {tag_b}")
    ax.legend(loc="lower right", ncol=max(1, n_models), fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(rows_per_model, metrics, error_label, tag, models):
    print(f"\n[{tag}]")
    for model in models:
        rows = rows_per_model.get(model, [])
        if not rows:
            print(f"  {MODEL_DISPLAY[model]}: no fold data found")
            continue
        line = f"  {MODEL_DISPLAY[model]:12s}  (n={len(rows)} folds): "
        parts = []
        for m in metrics:
            mean, err, _ = aggregate(rows, m, error=error_label)
            parts.append(f"{m}={mean:.3f}±{err:.3f}")
        print(line + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="result tag (folder name under results/)")
    ap.add_argument("--tag2", default=None, help="optional second TAG for comparison")
    ap.add_argument("--results-root", default=None,
                    help="defaults to <this dir>/results/")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help=f"metrics to plot (subset of: {ALL_METRICS}). default: {DEFAULT_METRICS}")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <results>/<tag>/metrics_bar_5fold.png)")
    ap.add_argument("--error", choices=["std", "sem", "ci95"], default="std",
                    help="error bar type (default: std)")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"which models to plot (subset of {ALL_MODELS}). "
                         f"기본: 결과 폴더에 npz 가 실제 있는 모델만 자동 선택")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results_root = a.results_root or os.path.join(here, "results")

    # Collect per-fold metrics for tag A
    root_a = os.path.join(results_root, a.tag)
    if not os.path.isdir(root_a):
        print(f"[error] not a directory: {root_a}", file=sys.stderr); sys.exit(1)

    if a.models:
        models = [m for m in a.models if m in ALL_MODELS]
        bad = [m for m in a.models if m not in ALL_MODELS]
        if bad:
            print(f"[warn] unknown model(s) ignored: {bad}", file=sys.stderr)
    else:
        models = [m for m in ALL_MODELS if per_fold_metrics(root_a, m)]
        if not models:
            print(f"[error] no predictions_test_*.npz found under {root_a}", file=sys.stderr)
            print("       먼저 'bash test.sh b <model> <tag>' 로 npz 생성 필요.", file=sys.stderr)
            sys.exit(1)
        print(f"[info] auto-detected models: {models}")

    rows_a = {m: per_fold_metrics(root_a, m) for m in models}

    if a.tag2:
        root_b = os.path.join(results_root, a.tag2)
        if not os.path.isdir(root_b):
            print(f"[error] not a directory: {root_b}", file=sys.stderr); sys.exit(1)
        rows_b = {m: per_fold_metrics(root_b, m) for m in models}
        print_summary(rows_a, a.metrics, a.error, a.tag, models)
        print_summary(rows_b, a.metrics, a.error, a.tag2, models)
        out = a.out or os.path.join(results_root,
                                     f"metrics_bar_5fold__{a.tag}__vs__{a.tag2}.png")
        make_dual_tag_plot(rows_a, rows_b, a.metrics, a.error, a.tag, a.tag2, out, models)
    else:
        print_summary(rows_a, a.metrics, a.error, a.tag, models)
        out = a.out or os.path.join(root_a, "metrics_bar_5fold.png")
        make_single_tag_plot(rows_a, a.metrics, a.error, a.tag, out, models)


if __name__ == "__main__":
    main()
