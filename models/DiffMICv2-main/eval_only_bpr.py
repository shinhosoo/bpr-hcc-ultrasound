#!/usr/bin/env python3
"""DiffMICv2 — evaluate-only mode.

Loads a Lightning ckpt and runs validation_step on a designated test pkl,
saving predictions to <out> as a standardized .npz (y_true, y_score).

Usage:
    DIFFMICV2_PRED_PATH=./logs/predictions_test.npz \
    python3 eval_only.py --config configs/lesion_binary.yml \
        --ckpt logs/<runname>/version_0/checkpoints/last.ckpt \
        --test-pkl /path/to/lesion_test.pkl \
        --out logs/predictions_test.npz
"""
import argparse, os, sys
import yaml
from easydict import EasyDict
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    import argparse as _ap
    torch.serialization.add_safe_globals([EasyDict, _ap.Namespace])
except Exception:
    pass

from diffuser_trainer import CoolSystem
from utils import get_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-pkl", required=True, help="path to lesion_test.pkl")
    ap.add_argument("--out", required=True, help="path to write predictions npz")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.data.testdata = a.test_pkl
    cfg.data.traindata = a.test_pkl
    os.environ["DIFFMICV2_PRED_PATH"] = a.out

    seed = int(os.environ.get("SEED", getattr(cfg.data, 'seed', 42)))
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[eval_only][seed] fixed at {seed}")

    try:
        import bpr_arch_hook
        bpr_arch_hook.apply_arch_hook()
    except Exception as _e:
        print(f"[eval_only][bpr-arch-hook] skipped: {_e}")

    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig(*args, **kwargs)
    torch.load = _patched
    try:
        model = CoolSystem.load_from_checkpoint(a.ckpt, hparams=cfg, strict=False)
    finally:
        torch.load = _orig
    model.eval()

    _, _, test_dataset = get_dataset(cfg)
    loader = DataLoader(test_dataset, batch_size=cfg.testing.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    trainer = pl.Trainer(
        accelerator='cpu' if a.cpu else 'gpu',
        devices=1, logger=False, enable_progress_bar=True,
        deterministic=True,
    )
    trainer.validate(model, dataloaders=loader)
    print(f"[eval_only] predictions saved to {a.out}")


if __name__ == "__main__":
    main()
