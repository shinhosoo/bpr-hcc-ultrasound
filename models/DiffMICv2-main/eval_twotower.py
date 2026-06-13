"""Two-tower test 평가 — diffusion(p_gen) + 판별 head(p_disc) 둘 다 test 에 평가.

학습된 diffusion CoolSystem ckpt + 저장된 disc head(.pth)를 로드해서 test set 에 대해:
  - p_gen  : 원본 validation 경로(diffusion sampling) → DIFFMICV2_PRED_PATH(=--out)
  - p_disc : ConditionalModel.x_weight 캡처 → disc head → TT_DISC_PRED_PATH(=--out-disc)
둘 다 표준 npz(y_true,y_score). 이후 gated_fusion.py 로 융합.
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    import argparse as _ap
    torch.serialization.add_safe_globals([EasyDict, _ap.Namespace])
except Exception:
    pass

import diffuser_trainer as DT
import model as MDL
from diffuser_trainer import CoolSystem
from utils import get_dataset

_cap = {"x_weight": None}
_tt  = {"proj": None, "clf": None}


def _cm_fwd(self, x, y, t, x_l, attn):
    bz, np_, I, J = x_l.shape
    x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
    x_l_feat = self.encoder_x_l(x_l_in); x_l_feat = self.norm_l(x_l_feat)
    x_g = self.encoder_x(x); x_g = self.norm(x_g)
    x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
    x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)
    w = torch.softmax(self.cond_weight, dim=2)
    x_weight = torch.sum(x_cat * w, dim=-1)
    _cap["x_weight"] = x_weight
    y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
    y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
    y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
    y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
    return self.lin4(y2)
MDL.ConditionalModel.forward = _cm_fwd


_orig_vs = DT.CoolSystem.validation_step
def _vs(self, batch, batch_idx):
    _orig_vs(self, batch, batch_idx)
    xw = _cap.get("x_weight")
    if xw is not None and _tt["clf"] is not None:
        with torch.no_grad():
            p = F.softmax(_tt["clf"](_tt["proj"](xw.float())), dim=-1)
        if not hasattr(self, "_tt_disc"):
            self._tt_disc = []
        self._tt_disc.append(p.detach().cpu())
DT.CoolSystem.validation_step = _vs


_orig_ovee = DT.CoolSystem.on_validation_epoch_end
def _ovee(self):
    disc = None
    if getattr(self, "_tt_disc", None):
        disc = torch.cat(self._tt_disc, dim=0).numpy()
    _orig_ovee(self)
    if disc is not None:
        dp = os.environ.get("TT_DISC_PRED_PATH", "predictions_test_disc.npz")
        os.makedirs(os.path.dirname(os.path.abspath(dp)) or ".", exist_ok=True)
        yt = None
        gen_path = os.environ.get("DIFFMICV2_PRED_PATH", "")
        if gen_path and os.path.exists(gen_path):
            try: yt = np.load(gen_path, allow_pickle=True)["y_true"]
            except Exception: yt = None
        if yt is not None and len(yt) == len(disc):
            np.savez(dp, y_true=np.asarray(yt).astype(int), y_score=disc.astype(float))
        else:
            np.savez(dp, y_score=disc.astype(float))
        print(f"[eval-twotower] p_disc saved: {dp}  shape={disc.shape}")
    self._tt_disc = []
DT.CoolSystem.on_validation_epoch_end = _ovee


def _load(path, device):
    try: return torch.load(path, map_location=device, weights_only=False)
    except TypeError: return torch.load(path, map_location=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="diffusion CoolSystem ckpt")
    ap.add_argument("--disc-head", required=True, help="predictions_*_disc_head.pth")
    ap.add_argument("--test-pkl", required=True)
    ap.add_argument("--out", required=True, help="p_gen npz")
    ap.add_argument("--out-disc", required=True, help="p_disc npz")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu"
    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.data.testdata = a.test_pkl
    cfg.data.traindata = a.test_pkl
    os.environ["DIFFMICV2_PRED_PATH"] = a.out
    os.environ["TT_DISC_PRED_PATH"] = a.out_disc
    pl.seed_everything(int(os.environ.get("SEED", 42)), workers=True)

    hd = _load(a.disc_head, device)
    fdim = hd["proj"]["0.weight"].shape[1]
    pdim = hd["proj"]["0.weight"].shape[0]
    proj = nn.Sequential(nn.Linear(fdim, pdim), nn.ReLU(inplace=False))
    proj.load_state_dict(hd["proj"])
    nc = hd["clf"]["weight"].shape[0]
    clf = nn.Linear(pdim, nc); clf.load_state_dict(hd["clf"])
    _tt["proj"] = proj.to(device).eval()
    _tt["clf"]  = clf.to(device).eval()
    print(f"[eval-twotower] disc head loaded: {fdim}->{pdim}->{nc}")

    _orig = torch.load
    def _patched(*ar, **kw):
        kw["weights_only"] = False
        return _orig(*ar, **kw)
    torch.load = _patched
    try:
        model = CoolSystem.load_from_checkpoint(a.ckpt, hparams=cfg, strict=False)
    finally:
        torch.load = _orig
    model.eval()

    _, _, test_ds = get_dataset(cfg)
    loader = DataLoader(test_ds, batch_size=cfg.testing.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)
    trainer = pl.Trainer(accelerator='cpu' if a.cpu else 'gpu', devices=1,
                         logger=False, enable_progress_bar=True, deterministic=True)
    trainer.validate(model, dataloaders=loader)
    print(f"[eval-twotower] done. p_gen={a.out}  p_disc={a.out_disc}")


if __name__ == "__main__":
    main()
