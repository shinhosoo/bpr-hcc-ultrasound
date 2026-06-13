"""DiffMICv2 Two-Tower (Phase 1) — 공유 백본 위 판별 head 공동학습 + p_disc 출력.

생성 타워(diffusion)는 그대로. ConditionalModel 의 라이브 conditioning feature(x_weight)를
**stop-grad** 로 읽는 판별 head(proj+clf)를 CE + lambda*BPR 로 **자체 옵티마이저** 학습한다.
stop-grad 라 판별/BPR gradient 가 encoder/conditioning 에 닿지 않음 → 생성 안 해침.

Phase 1 (스모크) 목표: 학습이 끝까지 돌고, validation 마다 p_disc 를 별도 npz 로 저장 +
판별 head state 저장. 융합은 사후(gated_fusion / val-stacking)로. Phase 2 에서 in-graph
학습형 게이트 추가 예정.

모든 판별 연산은 try/except 로 감싸 diffusion 학습은 절대 안 멈춘다.

env: BPR_LAMBDA(1.0) BPR_PROTO(geomedian) BPR_NUM_CLASSES(2)
     TT_PROJ_DIM(256) TT_LR(1e-3)
     DIFFMICV2_PRED_PATH (p_gen val npz; 여기서 _disc 파생)
실행: train_diffmicv2_bpr.sh 와 동일하게 --config --early-stop-patience 받음 (DT.main()).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffuser_trainer as DT
import model as MDL
from bpr_loss import total_bpr_loss
from utils import cast_label_to_one_hot_and_prototype as _cast_oh

BPR_LAMBDA  = float(os.environ.get("BPR_LAMBDA", "1.0"))
BPR_PROTO   = os.environ.get("BPR_PROTO", "geomedian")
NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
TT_PROJ_DIM = int(os.environ.get("TT_PROJ_DIM", "256"))
TT_LR       = float(os.environ.get("TT_LR", "1e-3"))

_cap = {"x_weight": None}
_tt  = {"proj": None, "clf": None, "opt": None}

print(f"[twotower] Phase1  lambda={BPR_LAMBDA} proto={BPR_PROTO} proj_dim={TT_PROJ_DIM} lr={TT_LR}")


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
print("[twotower] ConditionalModel.forward patched (x_weight capture)")


def _ensure_tt(feat_dim, device):
    if _tt["proj"] is None:
        _tt["proj"] = nn.Sequential(nn.Linear(feat_dim, TT_PROJ_DIM), nn.ReLU(inplace=False)).to(device)
        _tt["clf"]  = nn.Linear(TT_PROJ_DIM, NUM_CLASSES).to(device)
        _tt["opt"]  = torch.optim.Adam(
            list(_tt["proj"].parameters()) + list(_tt["clf"].parameters()),
            lr=TT_LR, weight_decay=1e-4)
        print(f"[twotower] disc head init: {feat_dim} -> {TT_PROJ_DIM} -> {NUM_CLASSES}")
    return _tt["proj"], _tt["clf"]


_orig_ts = DT.CoolSystem.training_step
def _ts(self, batch, batch_idx):
    out = _orig_ts(self, batch, batch_idx)
    try:
        x_batch, y_raw = batch
        y_oh, _ = _cast_oh(y_raw, self.params)
        labels = y_oh.argmax(-1).to(self.device).long()
        xw = _cap.get("x_weight")
        if xw is not None and xw.size(0) == labels.size(0):
            proj, clf = _ensure_tt(xw.size(-1), xw.device)
            proj.train(); clf.train()
            feat = proj(xw.detach().float())
            logits = clf(feat)
            ce = F.cross_entropy(logits, labels)
            zp = F.normalize(feat, dim=-1)
            try:
                bpr = total_bpr_loss(zp, labels, num_classes=NUM_CLASSES, prototype=BPR_PROTO)
            except Exception:
                bpr = torch.zeros((), device=xw.device)
            disc_loss = ce + BPR_LAMBDA * bpr
            _tt["opt"].zero_grad(); disc_loss.backward(); _tt["opt"].step()
            try: self.log("tt_ce", float(ce.item()), prog_bar=True)
            except Exception: pass
    except Exception as _e:
        print(f"[twotower] disc train step skipped: {_e}")
    return out
DT.CoolSystem.training_step = _ts


_orig_vs = DT.CoolSystem.validation_step
def _vs(self, batch, batch_idx):
    _orig_vs(self, batch, batch_idx)
    try:
        xw = _cap.get("x_weight")
        if xw is not None and _tt["clf"] is not None:
            _tt["proj"].eval(); _tt["clf"].eval()
            with torch.no_grad():
                p = F.softmax(_tt["clf"](_tt["proj"](xw.float())), dim=-1)
            if not hasattr(self, "_tt_disc"):
                self._tt_disc = []
            self._tt_disc.append(p.detach().cpu())
    except Exception as _e:
        print(f"[twotower] val disc skipped: {_e}")
DT.CoolSystem.validation_step = _vs


_orig_ovee = DT.CoolSystem.on_validation_epoch_end
def _ovee(self):
    disc = None
    try:
        if getattr(self, "_tt_disc", None):
            disc = torch.cat(self._tt_disc, dim=0).numpy()
    except Exception:
        disc = None
    _orig_ovee(self)
    try:
        if disc is not None:
            base = os.environ.get("DIFFMICV2_PRED_PATH", "predictions_val.npz")
            disc_path = os.environ.get("TT_DISC_PRED_PATH", base.replace(".npz", "_disc.npz"))
            os.makedirs(os.path.dirname(os.path.abspath(disc_path)) or ".", exist_ok=True)
            np.savez(disc_path, y_score=disc.astype(float))
            if _tt["clf"] is not None:
                torch.save({"proj": _tt["proj"].state_dict(), "clf": _tt["clf"].state_dict(),
                            "proj_dim": TT_PROJ_DIM},
                           disc_path.replace(".npz", "_head.pth"))
            print(f"[twotower] p_disc saved: {disc_path}  shape={disc.shape}")
        self._tt_disc = []
    except Exception as _e:
        print(f"[twotower] p_disc save skipped: {_e}")
DT.CoolSystem.on_validation_epoch_end = _ovee


print("[twotower] patches applied — launching DT.main()")
DT.main()
