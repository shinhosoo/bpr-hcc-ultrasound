# Data-Efficient Hepatocellular Carcinoma Detection via Prototype-Anchored Ranking-Based Representation Learning

**Paper:** Data-Efficient Hepatocellular Carcinoma Detection in Ultrasound Images via Prototype-Anchored Ranking-Based Representation Learning  
**Authors:** Hosoo Shin, Seongyeon Son, Minhee Park, Lin Xia, Eunchan Kim  
**Affiliation:** Hanyang University, Seoul, Republic of Korea  
**Corresponding:** eckim@hanyang.ac.kr

---

## Abstract

We propose a prototype-anchored pairwise ranking representation learning framework for HCC detection from ultrasound images. The method adopts Bayesian Personalized Ranking (BPR) loss to structure the latent space such that same-class prototype relevance consistently exceeds opposite-class prototype relevance, promoting discriminative intra-class compactness under data-scarce conditions. We investigate two class prototype estimation strategies—arithmetic mean and geometric median—and validate the framework on three architecturally distinct medical imaging backbones (DiffMIC v2, MedViT v2, HSQformer) under standard 5-fold cross-validation.

---

## Method Overview

The proposed framework is applied as a training-time regularizer on top of pretrained backbones, without modifying their architecture:

1. **Feature extraction**: A backbone-specific hook extracts a latent representation vector **z** per sample.
   - *Feature map-based* (DiffMIC v2, MedViT v2): a two-layer projection module maps backbone features to a 512-dim latent vector.
   - *Token-based* (HSQformer): a `forward_pre_hook` on the classification head captures the pre-head token sequence and applies token-mean aggregation.

2. **BPR loss**: For each sample, the relevance score to the same-class prototype must exceed the relevance to the opposite-class prototype:

   $$\mathcal{L}_\text{bpr} = -\frac{1}{N_S}\sum_{i=1}^{N_S}\log\sigma(r_i - r_{i^c}),\quad r_i = \mathbf{z}^{(i)\top}\,\mu(C_i)$$

3. **Prototype estimation**: Two strategies for class centroid **μ**:
   - *Arithmetic mean* — efficient; sensitive to outlier embeddings.
   - *Geometric median* — robust to outliers; computed via Weiszfeld's algorithm.

4. **Training schedule**:
   - **DiffMIC v2**: single-phase (CE + BPR + VICReg variance penalty); no backbone freezing.
   - **MedViT v2**: two-phase — Phase 1 trains with CE + BPR (5-epoch CE-only warm-up); Phase 2 freezes backbone and trains classification head with CE only.
   - **HSQformer**: single-phase (CE + BPR, 20-epoch warm-up).

---

## Repository Structure

```
.
├── train.sh                      # Top-level training entry point
├── test.sh                       # Top-level evaluation entry point
├── data/                         # Dataset directory (not included; see below)
│   ├── 5fold/fold_{0..4}/
│   │   ├── imagefolder/{train,val,test}/   # ImageFolder format
│   │   └── pkl/{lesion_train,lesion_val,lesion_test}.pkl
│   └── 3way/
│       ├── imagefolder/{train,val,test}/
│       └── pkl/{lesion_train,lesion_val,lesion_test}.pkl
├── models/
│   ├── DiffMICv2-main/           # DiffMIC v2 backbone (IEEE TMI'25)
│   │   ├── bpr_arch_hook.py      # BPR projection module for DiffMIC v2
│   │   ├── diffuser_trainer.py   # Training loop
│   │   └── ...
│   ├── MedViTV2-main/            # MedViT v2 backbone (Appl. Soft Comput.'25)
│   │   ├── main.py               # Training + evaluation loop
│   │   └── ...
│   └── HSQ/                      # HSQformer backbone (arXiv'25)
│       ├── train.py              # Training entry point
│       └── ...
├── tools/
│   ├── bpr/
│   │   ├── bpr_loss.py           # BPR loss with geometric median / mean prototype
│   │   ├── run_diffmicv2_bpr.py  # Runner: DiffMIC v2 + BPR
│   │   └── run_bpr.sh            # Shell launcher
│   ├── train_hsq.sh             # HSQformer training launcher
│   ├── train_medvitv2_bpr.sh     # MedViT v2 + BPR launcher
│   ├── train_diffmicv2_bpr.sh    # DiffMIC v2 + BPR launcher
│   ├── balanced_sampler.py       # Class-balanced sampling
│   └── exp/                      # Experiment sweep scripts
└── experiments/
    └── option_b_5fold/           # 5-fold cross-validation setup
        ├── prepare.sh
        ├── aggregate.py          # Aggregate fold results
        └── results/              # Training outputs
```

---

## Dataset

All experiments use the publicly available liver ultrasound dataset from the **Third Affiliated Hospital of Sun Yat-sen University** (benign/malignant binary classification), released with HSQformer [[She et al., 2025]](https://arxiv.org/abs/...).

Organize the data as follows before training:

```
data/
└── 5fold/
    ├── fold_0/
    │   ├── imagefolder/
    │   │   ├── train/{benign,malignant}/
    │   │   ├── val/{benign,malignant}/
    │   │   └── test/{benign,malignant}/
    │   └── pkl/
    │       ├── lesion_train.pkl
    │       ├── lesion_val.pkl
    │       └── lesion_test.pkl
    ├── fold_1/ ...
    ...
```

- `imagefolder/` format is used by MedViT v2 and HSQformer.
- `.pkl` format is used by DiffMIC v2.

---

## Installation

### DiffMIC v2

```bash
conda create -n diffmicv2 python=3.8
conda activate diffmicv2
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==2.0.8 diffusers==0.20.2 timm==0.9.12
pip install scikit-learn scikit-image einops tqdm pyyaml
```

> Full dependency list: `models/DiffMICv2-main/requirements.txt`

### MedViT v2

```bash
conda create -n medvitv2 python=3.9
conda activate medvitv2
pip install torch torchvision
pip install einops timm medmnist numpy pandas scikit-learn scikit-image tqdm Pillow
pip install natten  # required for deterministic GPU execution
```

> Full dependency list: `models/MedViTV2-main/requirements.txt`

### HSQformer

```bash
conda create -n hsq python=3.9
conda activate hsq
pip install torch torchvision
pip install einops timm scikit-learn tqdm
```

---

## Training

### 5-fold Cross-Validation (primary experiments)

```bash
# Baseline — DiffMIC v2
bash train.sh b diffmicv2 baseline

# Baseline — MedViT v2
bash train.sh b medvitv2 baseline

# Baseline — HSQformer
HSQ_PRETRAIN=1 BASE=1 bash train.sh b hsq hsq_baseline

# Proposed (BPR) — DiffMIC v2
BPR_TWO_PHASE=0 VICREG_GAMMA=3 VICREG_VAR_W=1 \
  bash tools/run_diffmicv2_vicreg.sh vic_g3

# Proposed (BPR) — MedViT v2
BPR_STAGE1_EPOCHS=50 BPR_HOOK=aux BPR_LAMBDA=1.0 BPR_WARMUP_EPOCHS=5 \
  BPR_PROTO_SCOPE=global BPR_PROTO=geomedian \
  TECH=bpr bash train.sh b medvitv2 bpr_medvitv2_2stage_aux

# Proposed (BPR) — HSQformer
HSQ_BPR=1 HSQ_PRETRAIN=1 BASE=1 \
  BPR_LAMBDA=0.05 BPR_WARMUP=20 BPR_PROTO=geomedian \
  EPOCHS=100 PATIENCE=40 \
  bash train.sh b hsq hsq_base_bpr
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `K` | `5` | Number of folds |
| `SEED` | `42+i` | Random seed (per fold) |
| `FOLD_START` / `FOLD_END` | `0` / `K-1` | Run a subset of folds |
| `BALANCED` | `0` | Set to `1` to enable class-balanced sampling |
| `HSQ_PRETRAIN` | `0` | Set to `1` to load ImageNet pretrained backbone weights (HSQformer only) |
| `BASE` | `0` | Set to `1` to train HSQformer baseline — `LENet_base`, CE only |
| `HSQ_BPR` | `0` | Set to `1` to enable BPR training for HSQformer |
| `BPR_LAMBDA` | `1.0` | BPR loss weight |
| `BPR_PROTO` | `geomedian` | Prototype estimation strategy (`geomedian` or `mean`) |
| `BPR_WARMUP` / `BPR_WARMUP_EPOCHS` | `5` | Warm-up epochs before BPR loss is applied |
| `VICREG_GAMMA` | `1` | VICReg variance penalty coefficient (DiffMIC v2 only) |

Results are saved under `experiments/option_b_5fold/results/<tag>/`.

### Single Train-Val-Test Split (option a)

```bash
bash train.sh a diffmicv2 baseline
```

---

## Evaluation

```bash
# Evaluate all folds for a given tag and model
bash test.sh b diffmicv2 baseline
bash test.sh b medvitv2 bpr_geo

# Aggregate 5-fold results
python experiments/option_b_5fold/aggregate.py \
    --results experiments/option_b_5fold/results/bpr_geo
```

Per-fold outputs: `predictions_test_<model>.npz`, `metrics_test.csv`, `viz/{roc,pr,confusion,metrics_bar}.png`.

---

## Results

Performance on the HCC liver ultrasound dataset (mean ± std, 5-fold cross-validation, batch size 32).  
**Bold** = improvement over baseline. Shading = best variant within each model group.

| Metric | DiffMIC v2 Base | DiffMIC v2 Prop. (Geo) | MedViT v2 Base | MedViT v2 Prop. (Geo) | HSQformer Base | HSQformer Prop. (Geo) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| ACC    | 0.7984 ± 0.0406 | **0.8142 ± 0.0388** | 0.7260 ± 0.0486 | **0.7480 ± 0.0376** | 0.7748 ± 0.0226 | **0.7811 ± 0.0116** |
| BACC   | 0.7813 ± 0.0447 | **0.8063 ± 0.0255** | 0.6839 ± 0.0829 | **0.7148 ± 0.0437** | 0.7343 ± 0.0278 | **0.7430 ± 0.0236** |
| AUC    | 0.8644 ± 0.0355 | **0.8719 ± 0.0302** | 0.7720 ± 0.0570 | **0.7915 ± 0.0476** | 0.8349 ± 0.0297 | 0.8307 ± 0.0212 |
| AP     | 0.9262 ± 0.0211 | **0.9303 ± 0.0268** | 0.8684 ± 0.0346 | 0.8670 ± 0.0365 | 0.9146 ± 0.0280 | 0.9077 ± 0.0220 |
| Precision | 0.8724 ± 0.0332 | **0.8965 ± 0.0294** | 0.8080 ± 0.0579 | **0.8326 ± 0.0572** | 0.8319 ± 0.0257 | **0.8384 ± 0.0257** |
| Recall | 0.8276 ± 0.0482 | 0.8276 ± 0.0822 | 0.7977 ± 0.0527 | **0.8046 ± 0.1083** | 0.8437 ± 0.0474 | **0.8460 ± 0.0452** |
| F1 (macro) | 0.7726 ± 0.0431 | **0.7927 ± 0.0351** | 0.6750 ± 0.0843 | **0.7072 ± 0.0319** | 0.7356 ± 0.0243 | **0.7432 ± 0.0158** |

DiffMIC v2 Prop. additionally applies VICReg variance penalty (γ=3). MedViT v2 uses two-phase training with NATTEN determinism. HSQformer is evaluated supplementarily (single training run per fold).

---

## Citation

Citation will be added upon publication.

---

## Acknowledgements

This work builds upon [DiffMIC v2](https://github.com/...) (IEEE TMI'25), [MedViT v2](https://github.com/...) (Appl. Soft Comput.'25), and [HSQformer](https://github.com/...) (arXiv'25). We thank the Third Affiliated Hospital of Sun Yat-sen University for making the HCC ultrasound dataset publicly available.
