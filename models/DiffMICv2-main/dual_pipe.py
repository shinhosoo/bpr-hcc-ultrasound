"""Dual-channel conditioning — narrow class pipe + wide instance pipe.

The instance channel (x_weight, 6144-d) follows the multiplication path unchanged
and is kept BPR-free. The class channel e_cls = cls_head(x_weight.detach()) is a
narrow d_cls-dimensional projection; detach prevents BPR gradients from flowing back
into the encoder. The run script reads _state['z']=e_cls for BPR. e_to_y(e_cls) is
added to y2 (separate from the multiplicative x_weight path), with e_to_y zero-init
so training starts identical to baseline.

BPR gradients reach only cls_head. encoder_x(_l), norm, cond_weight, and lin1-4 are
never touched by BPR.

This module defines the architecture (__init__ + forward) only and is shared by
train and eval. BPR loss/buffer/optimizer wiring lives in run_diffmicv2_bpr.py.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_fallback_state = {"z": None}


def apply(state=None, d_cls=None, detach_instance=None, verbose=True):
    """Monkey-patch dual-channel conditioning onto ConditionalModel.

    Must be called before model instantiation. e_cls is written to state['z'].
    Env vars: BPR_DCLS (default 128), BPR_DUAL_DETACH (default 1).
    """
    if state is None:
        state = _fallback_state
    if d_cls is None:
        d_cls = int(os.environ.get("BPR_DCLS", "128"))
    if detach_instance is None:
        detach_instance = os.environ.get("BPR_DUAL_DETACH", "1") == "1"

    import model as MDL

    _orig_init = MDL.ConditionalModel.__init__

    def _cm_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        feat_dim = self.encoder_x.g.out_features  # 6144
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, d_cls), nn.ReLU(inplace=False)
        )
        self.e_to_y = nn.Linear(d_cls, feat_dim)
        nn.init.zeros_(self.e_to_y.weight)
        nn.init.zeros_(self.e_to_y.bias)
    MDL.ConditionalModel.__init__ = _cm_init

    def _cm_fwd(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape
        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_feat = self.encoder_x_l(x_l_in)
        x_l_feat = self.norm_l(x_l_feat)
        x_g = self.encoder_x(x)
        x_g = self.norm(x_g)
        x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(x_cat * w, dim=-1)

        src = x_weight.detach() if detach_instance else x_weight
        e_cls = self.cls_head(src)                            # (B, d_cls)
        state["z"] = e_cls

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = y2 + self.e_to_y(e_cls).unsqueeze(-1).unsqueeze(-1)
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd

    if verbose:
        print(f"[dual_pipe] dual-channel conditioning applied "
              f"(d_cls={d_cls}, detach_instance={detach_instance}, e_to_y zero-init)")
    return True
