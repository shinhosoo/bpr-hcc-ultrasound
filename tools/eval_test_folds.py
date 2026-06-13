#!/usr/bin/env python3
"""
5-fold test set 성능 평가 (mean ± std)

학습 중 val set 으로 체크포인트가 선택되므로, 최종 보고에는
별도 test set 예측 파일(predictions_test_*.npz)을 사용해야 합니다.
이 스크립트는 test 예측 파일만 읽어 fold 별 성능을 집계합니다.

사용법
------
python3 tools/eval_test_folds.py \
    --fold-dir experiments/option_b_5fold/results/<tag>/medvitv2 \
    --pred-file predictions_test_medvitv2.npz \
    --name MyModel

python3 tools/eval_test_folds.py \
    --fold-dir .../medvitv2    --pred-file predictions_test_medvitv2.npz    --name MedViTV2 \
    --fold-dir .../diffmicv2   --pred-file predictions_test_diffmicv2.npz   --name DiffMICv2 \
    --fold-dir .../medvit      --pred-file predictions_test_medvit.npz      --name MedViT \
    --fold-dir .../diffmic     --pred-file predictions_test_diffmic.npz     --name DiffMIC \
    --out results/test_comparison.csv

옵션
----
--k           폴드 수 (default: 5)
--threshold   양성 클래스 임계값 (default: 0.5)
--out         결과 CSV 저장 경로 (optional)
--ci          95% 신뢰구간 추가 출력 (optional)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, f1_score, confusion_matrix,
)

METRICS = ["ACC", "BACC", "AUC", "AP", "Precision", "Recall",
           "Specificity", "NPV", "F1_macro", "F1_pos"]


def _to_pos_score(y_score: np.ndarray) -> np.ndarray:
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score[:, 0]
    raise ValueError(f"지원하지 않는 y_score shape: {y_score.shape}")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    pos    = _to_pos_score(y_score)
    y_pred = (pos >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    auc  = roc_auc_score(y_true, pos) if len(np.unique(y_true)) > 1 else float("nan")
    ap   = average_precision_score(y_true, pos) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "N":           int(len(y_true)),
        "ACC":         accuracy_score(y_true, y_pred),
        "BACC":        balanced_accuracy_score(y_true, y_pred),
        "AUC":         auc,
        "AP":          ap,
        "Precision":   prec,
        "Recall":      sens,
        "Specificity": spec,
        "NPV":         npv,
        "F1_macro":    f1_score(y_true, y_pred, average="macro"),
        "F1_pos":      f1_score(y_true, y_pred, pos_label=1),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    if "y_true" not in d or "y_score" not in d:
        raise KeyError(f"{path}: 'y_true' 와 'y_score' 키가 필요합니다")
    return d["y_true"], d["y_score"]


def eval_test_folds(
    fold_dir: str,
    pred_file: str,
    k: int = 5,
    threshold: float = 0.5,
) -> list[dict]:
    """fold_0 ~ fold_{k-1} 의 pred_file 을 읽어 per-fold 메트릭 반환."""
    results = []
    missing = []
    for i in range(k):
        p = os.path.join(fold_dir, f"fold_{i}", pred_file)
        if not os.path.isfile(p):
            missing.append(p)
            continue
        y_true, y_score = load_npz(p)
        m = compute_metrics(y_true, y_score, threshold)
        m["fold"] = i
        m["path"] = p
        results.append(m)

    if missing:
        print(f"  [경고] 누락된 파일 {len(missing)}개:", file=sys.stderr)
        for mp in missing:
            print(f"    {mp}", file=sys.stderr)

    if not results:
        raise FileNotFoundError(
            f"'{pred_file}' 파일을 {fold_dir} 아래 어디서도 찾지 못했습니다."
        )
    return results


def summarize(results: list[dict], ci: bool = False) -> dict[str, tuple]:
    """mean, std, (선택) 95% CI 반환."""
    summary = {}
    for m in METRICS:
        vals = [r[m] for r in results if not math.isnan(r[m])]
        if not vals:
            summary[m] = (float("nan"), float("nan"))
            continue
        arr = np.array(vals)
        mn, sd = arr.mean(), arr.std(ddof=0)
        if ci and len(arr) > 1:
            from scipy import stats as _stats
            se = arr.std(ddof=1) / math.sqrt(len(arr))
            t  = _stats.t.ppf(0.975, df=len(arr) - 1)
            summary[m] = (mn, sd, mn - t * se, mn + t * se)
        else:
            summary[m] = (mn, sd)
    return summary


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def print_summary(name: str, results: list[dict], summary: dict, ci: bool = False) -> None:
    n_folds = len(results)
    print(f"\n{'='*70}")
    print(f"  {name}  ({n_folds} folds, TEST SET)")
    print(f"{'='*70}")
    if ci:
        print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>8}  {'95% CI':^17}  per-fold values")
        print(f"  {'-'*66}")
    else:
        print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>8}  per-fold values")
        print(f"  {'-'*62}")

    for m in METRICS:
        vals = [r[m] for r in results]
        fold_str = "  ".join(_fmt(v) for v in vals)
        s = summary[m]
        mn_s = _fmt(s[0])
        sd_s = _fmt(s[1])
        if ci and len(s) == 4:
            ci_s = f"[{_fmt(s[2])}, {_fmt(s[3])}]"
            print(f"  {m:<14} {mn_s:>8}  {sd_s:>8}  {ci_s:^17}  [{fold_str}]")
        else:
            print(f"  {m:<14} {mn_s:>8}  {sd_s:>8}  [{fold_str}]")

    tp = sum(r["TP"] for r in results)
    fp = sum(r["FP"] for r in results)
    fn = sum(r["FN"] for r in results)
    tn = sum(r["TN"] for r in results)
    total_n = sum(r["N"] for r in results)
    print(f"\n  {'합계 (all folds)':<14}  N={total_n}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def write_csv(path: str, all_data: list[tuple]) -> None:
    """(name, results, summary) 리스트를 CSV로 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    max_k = max(len(r) for _, r, _ in all_data)
    mean_cols = [f"{m}_mean" for m in METRICS]
    std_cols  = [f"{m}_std"  for m in METRICS]
    fold_cols = [f"fold_{i}_{m}" for i in range(max_k) for m in METRICS]

    fieldnames = ["model"] + mean_cols + std_cols + fold_cols
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for name, results, summary in all_data:
            row: dict = {"model": name}
            for m in METRICS:
                s = summary[m]
                row[f"{m}_mean"] = _fmt(s[0])
                row[f"{m}_std"]  = _fmt(s[1])
            for r in results:
                i = r["fold"]
                for m in METRICS:
                    row[f"fold_{i}_{m}"] = _fmt(r[m])
            w.writerow(row)
    print(f"\nCSV 저장: {path}")


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="5-fold test set 평가 (mean ± std)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--fold-dir",   action="append", required=True,
                   help="모델 결과 루트 (fold_0~N 포함). 반복 가능.")
    p.add_argument("--pred-file",  action="append", required=True,
                   help="각 fold 디렉터리 안의 test npz 파일명 (fold-dir 와 같은 순서).")
    p.add_argument("--name",       action="append", default=None,
                   help="표시 이름 (fold-dir 와 같은 순서).")
    p.add_argument("--k",          type=int, default=5,   help="폴드 수 (default: 5)")
    p.add_argument("--threshold",  type=float, default=0.5, help="양성 임계값 (default: 0.5)")
    p.add_argument("--out",        type=str, default=None, help="CSV 저장 경로")
    p.add_argument("--ci",         action="store_true",   help="95% 신뢰구간 출력")
    return p.parse_args()


def main():
    args = parse_args()

    n = len(args.fold_dir)
    if len(args.pred_file) != n:
        sys.exit(f"오류: --fold-dir ({n}개) 와 --pred-file ({len(args.pred_file)}개) 수가 다릅니다.")

    names = args.name or []
    if names and len(names) != n:
        sys.exit(f"오류: --name ({len(names)}개) 와 --fold-dir ({n}개) 수가 다릅니다.")

    all_data = []
    for i, (fd, pf) in enumerate(zip(args.fold_dir, args.pred_file)):
        name = names[i] if i < len(names) else os.path.basename(fd.rstrip("/")) or fd
        try:
            results = eval_test_folds(fd, pf, k=args.k, threshold=args.threshold)
        except FileNotFoundError as e:
            print(f"[오류] {e}", file=sys.stderr)
            continue
        summary = summarize(results, ci=args.ci)
        print_summary(name, results, summary, ci=args.ci)
        all_data.append((name, results, summary))

    if args.out and all_data:
        write_csv(args.out, all_data)


if __name__ == "__main__":
    main()
