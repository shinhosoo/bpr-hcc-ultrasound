#!/usr/bin/env python3
"""
Visualize prediction results saved as .npz (keys: y_true, y_score) from
MedViT / DiffMIC / DiffMICv2 binary lesion classifiers.

Produces:
- ROC curves overlay         -> <out>/roc.png
- PR curves overlay          -> <out>/pr.png
- Confusion matrices grid    -> <out>/confusion.png
- Key metrics bar chart      -> <out>/metrics_bar.png
- Per-model summary CSV      -> <out>/metrics.csv

Usage:
    python3 tools/visualize_results.py \
        --pred outputs/medvit_binary/predictions_val.npz \
        --pred models/DiffMIC-main/results_lesion_binary/logs/lesion_binary/split_0/predictions_val.npz \
        --pred models/DiffMICv2-main/logs/predictions_val.npz \
        --name MedViT --name DiffMIC --name DiffMICv2 \
        --out outputs/comparison
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix,
)


def _to_pos(y_score):
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score[:, 0]
    raise ValueError(f"Unsupported y_score shape: {y_score.shape}")


def compute_metrics(y_true, pos_score, threshold=0.5):
    y_pred = (pos_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(y_true, pos_score) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, pos_score) if len(np.unique(y_true)) > 1 else float("nan")
    return dict(
        N=int(len(y_true)),
        ACC=accuracy_score(y_true, y_pred),
        BACC=balanced_accuracy_score(y_true, y_pred),
        AUC=auc, AP=ap,
        Precision=tp / (tp + fp) if (tp + fp) > 0 else float('nan'),
        Recall=sens,
        NPV=tn / (tn + fn) if (tn + fn) > 0 else float('nan'),
        F1_macro=f1_score(y_true, y_pred, average='macro'),
        F1_pos=f1_score(y_true, y_pred, pos_label=1),
        Sensitivity=sens, Specificity=spec,
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
    )


def load(p):
    d = np.load(p, allow_pickle=True)
    y_true = np.asarray(d["y_true"]).astype(int).ravel()
    pos = _to_pos(d["y_score"]).astype(float)
    return y_true, pos


def plot_roc(records, out_path):
    plt.figure(figsize=(6, 5.5))
    for r in records:
        fpr, tpr, _ = roc_curve(r['y'], r['p'])
        plt.plot(fpr, tpr, lw=2, label=f"{r['name']}  (AUC={r['m']['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(records, out_path):
    plt.figure(figsize=(6, 5.5))
    for r in records:
        pr, rc, _ = precision_recall_curve(r['y'], r['p'])
        plt.plot(rc, pr, lw=2, label=f"{r['name']}  (AP={r['m']['AP']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(records, out_path):
    n = len(records)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for i, r in enumerate(records):
        ax = axes[i // cols][i % cols]
        m = r['m']
        cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
        im = ax.imshow(cm, cmap='Blues')
        for (yy, xx), v in np.ndenumerate(cm):
            ax.text(xx, yy, str(int(v)), ha='center', va='center',
                    color='white' if v > cm.max() / 2 else 'black', fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign(0)', 'Malignant(1)'])
        ax.set_yticklabels(['Benign(0)', 'Malignant(1)'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{r['name']}  (Acc {m['ACC']:.3f})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # hide unused
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_bar(records, out_path):
    metric_keys = ['ACC', 'BACC', 'AUC', 'AP', 'Precision', 'Recall', 'Specificity', 'NPV', 'F1_macro', 'F1_pos']
    names = [r['name'] for r in records]
    vals = np.array([[r['m'][k] for k in metric_keys] for r in records])
    x = np.arange(len(metric_keys))
    width = 0.8 / max(1, len(records))
    plt.figure(figsize=(max(8, 1.4 * len(metric_keys)), 5))
    for i, (n, row) in enumerate(zip(names, vals)):
        plt.bar(x + i * width, row, width=width, label=n)
        for j, v in enumerate(row):
            if not np.isnan(v):
                plt.text(x[j] + i * width, v + 0.005, f"{v:.3f}",
                         ha='center', va='bottom', fontsize=8, rotation=0)
    plt.xticks(x + width * (len(records) - 1) / 2, metric_keys)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Model comparison")
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_csv(records, out_path):
    cols = ['Model', 'N', 'ACC', 'BACC', 'AUC', 'AP',
            'Precision', 'Sensitivity', 'Specificity', 'NPV', 'F1_macro', 'F1_pos',
            'TP', 'FP', 'FN', 'TN', 'File']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {'Model': r['name'], 'File': r['path']}
            row.update(r['m'])
            w.writerow({k: row.get(k, '') for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', action='append', required=True,
                    help='predictions .npz file (repeat for each model)')
    ap.add_argument('--name', action='append', default=None,
                    help='display name per --pred (same order)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--separate-files', action='store_true',
                    help='모델별로 roc/pr/confusion PNG 를 따로 저장')
    args = ap.parse_args()

    names = args.name or []
    if names and len(names) != len(args.pred):
        raise SystemExit("--name count must match --pred count")

    records = []
    for i, p in enumerate(args.pred):
        name = names[i] if i < len(names) else os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        y, pos = load(p)
        m = compute_metrics(y, pos, threshold=args.threshold)
        records.append({'name': name, 'path': p, 'y': y, 'p': pos, 'm': m})
        print(f"[{name}]  N={m['N']}  ACC={m['ACC']:.3f}  AUC={m['AUC']:.3f}  F1_macro={m['F1_macro']:.3f}  "
              f"Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    if args.separate_files:
        from sklearn.metrics import roc_curve, precision_recall_curve
        for r in records:
            safe = r['name'].replace(' ', '_').replace('/', '_')
            # ROC
            fpr, tpr, _ = roc_curve(r['y'], r['p'])
            plt.figure(figsize=(5.5, 5))
            plt.plot(fpr, tpr, lw=2, label=f"AUC={r['m']['AUC']:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f"ROC — {r['name']}"); plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out, f'roc_{safe}.png'), dpi=150); plt.close()
            # PR
            pr, rc, _ = precision_recall_curve(r['y'], r['p'])
            plt.figure(figsize=(5.5, 5))
            plt.plot(rc, pr, lw=2, label=f"AP={r['m']['AP']:.3f}")
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title(f"PR — {r['name']}"); plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out, f'pr_{safe}.png'), dpi=150); plt.close()
            # Confusion
            m_ = r['m']
            cm = np.array([[m_['TN'], m_['FP']], [m_['FN'], m_['TP']]])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.imshow(cm, cmap='Blues')
            for (yy, xx), v in np.ndenumerate(cm):
                ax.text(xx, yy, str(int(v)), ha='center', va='center',
                        color='white' if v > cm.max() / 2 else 'black', fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Benign(0)', 'Malignant(1)'])
            ax.set_yticklabels(['Benign(0)', 'Malignant(1)'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f"{r['name']}  (Acc {m_['ACC']:.3f})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f'confusion_{safe}.png'), dpi=150); plt.close()
            print(f"saved per model: roc_{safe}.png  pr_{safe}.png  confusion_{safe}.png")
        plot_metrics_bar(records, os.path.join(args.out, 'metrics_bar.png'))
        write_csv(records, os.path.join(args.out, 'metrics.csv'))
        print(f"\nAlso saved: metrics_bar.png, metrics.csv (통합)")
    else:
        plot_roc(records, os.path.join(args.out, 'roc.png'))
        plot_pr(records, os.path.join(args.out, 'pr.png'))
        plot_confusion(records, os.path.join(args.out, 'confusion.png'))
        plot_metrics_bar(records, os.path.join(args.out, 'metrics_bar.png'))
        write_csv(records, os.path.join(args.out, 'metrics.csv'))
        print(f"\nSaved: {args.out}/roc.png, pr.png, confusion.png, metrics_bar.png, metrics.csv")


if __name__ == '__main__':
    main()
