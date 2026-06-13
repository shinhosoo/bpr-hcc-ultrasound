#!/usr/bin/env python3
"""DCG(ResNet-18 분류기) 단독 평가 — diffusion 없이 DCG 의 분류 예측을 test 에 평가.

(B) 실험: 같은 ResNet-18 백본에서 'BPR 분류기 vs DiffMICv2(diffusion)' 공정 비교.
예측 = softmax(0.5*(y_global + y_local))  ← run_diffmicv2_dcg_pretrain.py 의 _val_auc 와 동일.
출력 npz: y_true(int), y_score(N,2)  ← aggregate.py/viz.sh 표준 포맷.
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from utils import get_dataset
from pretraining.dcg import DCG as AuxCls


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="dcg_bpr.pth (torch.save([state_dict]))")
    ap.add_argument("--test-pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu"
    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.data.testdata = a.test_pkl
    cfg.data.traindata = a.test_pkl
    cfg.device = device

    dcg = AuxCls(cfg).to(device)
    sd = _torch_load(a.ckpt, device)
    state = sd[0] if isinstance(sd, (list, tuple)) else sd
    missing, unexpected = dcg.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[eval_dcg] load_state_dict (strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    dcg.eval()

    _, _, test_ds = get_dataset(cfg)
    loader = DataLoader(test_ds, batch_size=cfg.testing.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_fusion, y_global, y_local, patches, attns, attn_map = dcg(x)
            prob = F.softmax(0.5 * (y_global + y_local), dim=1)
            y = y.detach().cpu().numpy()
            if y.ndim > 1 and y.shape[1] > 1:
                y = y.argmax(1)
            ys.append(y.astype(int).ravel())
            ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(int)
    y_score = np.concatenate(ps).astype(float)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, y_true=y_true, y_score=y_score)
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        _auc = roc_auc_score(y_true, y_score[:, 1]) if y_score.shape[1] == 2 else float("nan")
        _acc = accuracy_score(y_true, (y_score[:, 1] >= 0.5).astype(int)) if y_score.shape[1] == 2 else float("nan")
        print(f"[eval_dcg] wrote {a.out}  N={len(y_true)}  AUC={_auc:.4f}  ACC={_acc:.4f}")
    except Exception:
        print(f"[eval_dcg] wrote {a.out}  N={len(y_true)}")


if __name__ == "__main__":
    main()
