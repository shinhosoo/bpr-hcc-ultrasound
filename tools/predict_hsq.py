#!/usr/bin/env python3
"""
HSQ best checkpoint → predictions_val.npz 재생성 스크립트.
학습 없이 저장된 checkpoint만으로 예측을 다시 뽑습니다.

Usage
-----
python3 tools/predict_hsq.py \
    --fold-dir experiments/option_b_5fold/results/hsq/hsq \
    --val-dir  data/5fold \
    --base

python3 tools/predict_hsq.py \
    --fold-dir experiments/option_b_5fold/results/hsq_bpr_only/hsq \
    --val-dir  data/5fold

"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HSQ_DIR = os.path.join(ROOT, "models", "HSQ")

if not os.path.isdir(HSQ_DIR):
    sys.exit(f"[predict_hsq] ERROR: HSQ directory not found: {HSQ_DIR}")

_ORIG_CWD = os.getcwd()

sys.path.insert(0, HSQ_DIR)
os.chdir(HSQ_DIR)

import numpy as np
import torch
from torchvision import datasets as _tds, transforms as _T
import argparse as _ap

def make_hsq_args(proposed: bool):
    return _ap.Namespace(
        model_name         = "hsq",
        model              = "LENet",
        proposed           = proposed,
        adv_bpr            = False,
        patience           = 40,
        serial_parallel    = "serial",
        sparse_dense       = "sparse_token",
        num_experts        = 4,
        top_k              = 1,
        head_type          = "linear",
        cat_moe_head       = False,
        q_former_depths    = [1, 1, 1, 1],
        stage_dims         = [96, 192, 384, 768],
        q_former_head_num  = [3, 6, 12, 24],
        num_query_tokens   = 200,
        query_dim          = 384,
        model_ema          = True,
        model_ema_decay    = 0.9998,
        model_ema_warmup   = False,
        model_ema_force_cpu= False,
        visual_feature     = False,
        num_classes        = 2,
        categories         = "binary",
    )


def run_one_fold(fold_idx: int, fold_dir: str, val_dir: str,
                 proposed: bool, batch_size: int, device):
    ckpt_dir  = os.path.join(fold_dir, f"fold_{fold_idx}", "checkpoints")
    pred_path = os.path.join(fold_dir, f"fold_{fold_idx}", "predictions_val.npz")
    ckpt_path = os.path.join(ckpt_dir, f"hsq_val{fold_idx}.pth")
    val_path  = os.path.join(val_dir, f"fold_{fold_idx}", "imagefolder", "val")

    if not os.path.isfile(ckpt_path):
        print(f"  [fold {fold_idx}] WARNING: checkpoint not found → {ckpt_path}  (skipped)")
        return
    if not os.path.isdir(val_path):
        print(f"  [fold {fold_idx}] WARNING: val dir not found → {val_path}  (skipped)")
        return

    print(f"  [fold {fold_idx}] ckpt  : {ckpt_path}")
    print(f"  [fold {fold_idx}] val   : {val_path}")
    print(f"  [fold {fold_idx}] out   : {pred_path}")

    hsq_args = make_hsq_args(proposed)
    from models.LivNet import LENet, LENet_base

    if proposed:
        model = LENet(args=hsq_args)
    else:
        model = LENet_base(args=hsq_args)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    _MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
    _tf = _T.Compose([
        _T.Resize((224, 224)),
        _T.ToTensor(),
        _T.Normalize(_MEAN, _STD),
    ])

    class _IFP(_tds.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            return sample, target, path

    val_ds     = _IFP(val_path, transform=_tf)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ytrue, yscore, paths = [], [], []
    with torch.no_grad():
        for img, lbl, p in val_loader:
            img = img.to(device)
            if proposed:
                out = model(img, "val")
            else:
                out = model(img)
            prob = torch.softmax(out, 1).cpu().numpy()
            ytrue.extend(lbl.numpy())
            yscore.extend(prob[:, 1])
            paths.extend(p)

    np.savez(pred_path,
             y_true  = np.array(ytrue,  dtype=np.int64),
             y_score = np.array(yscore, dtype=np.float32),
             paths   = np.array(paths))
    print(f"  [fold {fold_idx}] saved  n={len(ytrue)}  pos={sum(ytrue)}")


def main():
    parser = argparse.ArgumentParser(description="HSQ predictions_val.npz 재생성")
    parser.add_argument("--fold-dir",   required=True,
                        help="모델 결과 루트 (fold_0~N 포함), e.g. experiments/.../results/hsq/hsq")
    parser.add_argument("--val-dir",    required=True,
                        help="5-fold 데이터 루트, e.g. data/5fold  (fold_0/imagefolder/val 구조)")
    parser.add_argument("--base",       action="store_true", default=False,
                        help="LENet_base 체크포인트 사용 (BASE=1 로 학습된 경우)")
    parser.add_argument("--k",          type=int, default=5,
                        help="전체 fold 수 (default 5)")
    parser.add_argument("--fold",       type=int, default=None,
                        help="특정 fold 만 처리 (미지정 시 0..k-1 전체)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    proposed = not args.base

    fold_dir = os.path.normpath(os.path.join(_ORIG_CWD, args.fold_dir))
    val_dir  = os.path.normpath(os.path.join(_ORIG_CWD, args.val_dir))

    print(f"[predict_hsq] fold_dir : {fold_dir}")
    print(f"[predict_hsq] val_dir  : {val_dir}")
    print(f"[predict_hsq] proposed : {proposed}  device: {device}")

    folds = [args.fold] if args.fold is not None else list(range(args.k))
    for i in folds:
        run_one_fold(i, fold_dir, val_dir, proposed, args.batch_size, device)

    print("[predict_hsq] DONE")


if __name__ == "__main__":
    main()
