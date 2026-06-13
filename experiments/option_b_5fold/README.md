# Option B — 5-fold Stratified Cross-Validation

전체 635장을 stratified 5-fold로 쪼개기. 각 fold의 test에서 평가, 5번 학습 결과를 mean ± std로 보고.

## 데이터셋 위치 (공유)
`test 2/data/5fold/fold_0..4/` 에서 불러옴.

## 모델 단독 한 줄 실행 (권장)
```bash
# MedViT 만
bash experiments/option_b_5fold/run_medvit.sh            # tag=baseline
bash experiments/option_b_5fold/run_medvit.sh supcon     # 기법 적용 후

# DiffMIC 만
bash experiments/option_b_5fold/run_diffmic.sh
bash experiments/option_b_5fold/run_diffmic.sh supcon

# DiffMICv2 만
bash experiments/option_b_5fold/run_diffmicv2.sh
bash experiments/option_b_5fold/run_diffmicv2.sh supcon
```
각 스크립트는 데이터 준비 → 5-fold 학습 → 5-fold 테스트 평가 → 집계 (mean ± std) → per-fold ROC 오버레이 시각화 → pooled 시각화까지 모두 수행.

환경변수로 옵션 조절 가능:
```bash
EPOCHS=50 PATIENCE=10 K=3 bash experiments/option_b_5fold/run_medvit.sh baseline
SEED_BASE=100 bash experiments/option_b_5fold/run_medvit.sh baseline
```

## 세 모델 동시 5-fold
```bash
bash experiments/option_b_5fold/run_5fold.sh baseline
bash experiments/option_b_5fold/run_5fold.sh supcon
```

## 기법 적용 전후 비교
```bash
python3 experiments/option_b_5fold/compare.py --tag-a baseline --tag-b supcon
```
→ 각 모델별 메트릭의 Δ (mean ± std) + paired t-test p-value.

## 결과 위치 (model-first 레이아웃)
```
experiments/option_b_5fold/results/<tag>/
├── medvit/
│   ├── fold_0/
│   │   ├── checkpoint_best.pth
│   │   ├── predictions_test_medvit.npz
│   │   ├── metrics_test.csv
│   │   └── viz/{roc,pr,confusion,metrics_bar}.png
│   ├── fold_1/ … fold_4/
├── diffmic/      (fold_0/ … fold_4/ — 동일 구조)
├── diffmicv2/    (fold_0/ … fold_4/ — lightning_logs 포함)
├── comparison_fold_0/      ← viz.sh가 만듦 (세 모델 같은 fold ROC)
│   ├── roc.png  pr.png  confusion.png  metrics_bar.png  metrics.csv
├── comparison_fold_1/ … comparison_fold_4/
├── pooled/                  ← 5 folds 합친 pooled
│   ├── predictions_test_<m>_pooled.npz   (모델별)
│   └── comparison_pooled/{roc,pr,…}.png
└── summary.csv              ← aggregate.py (fold mean ± std + pooled)
```

## Migration 참고
이전엔 `fold_i/<model>/...` 였지만 model-first로 영구 전환됨.
기존 fold-first 결과가 있으면 `aggregate.py` / `viz.sh` 는 두 레이아웃 모두 인식합니다 (legacy fallback).
