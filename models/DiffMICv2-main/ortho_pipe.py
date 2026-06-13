"""Orthogonal subspace conditioning — decoupled objective design.

x_weight (6144-d) is complementarily decomposed into:
  z_cls  = cls_proj(x_weight.detach())
  x_cls  = cls_back(z_cls)
  z_inst = x_weight - x_cls

Denoiser uses z_inst in the multiplicative path (pure instance conditioning,
free from BPR collapse) and re-injects the class axis as e_to_y(z_cls)
additively (zero-init), so training starts identical to baseline.

Orthogonality penalty (aux_loss): per-sample cos(z_inst, x_cls)^2, optimizing
cls_proj/cls_back only while protecting the encoder.

Invariant: BPR/ortho gradients reach only cls_proj/cls_back/e_to_y.
encoder_x(_l), norm, cond_weight, and lin1-4 are never touched.

train: run_diffmicv2_bpr.py reads _state["z"]=z_cls (BPR) and _state["aux_loss"].
eval : bpr_arch_hook.py reconstructs the same modules from the checkpoint.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_fallback_state = {"z": None, "aux_loss": None}


def apply(state=None, d_cls=None, detach_instance=None, verbose=True):
    if state is None:
        state = _fallback_state
    if d_cls is None:
        d_cls = int(os.environ.get("BPR_DCLS", "128"))
    if detach_instance is None:
        detach_instance = os.environ.get("BPR_ORTHO_DETACH", "1") == "1"
    mult_mode = os.environ.get("BPR_ORTHO_MULT", "zinst")

    import model as MDL
    _orig_init = MDL.ConditionalModel.__init__

    def _cm_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        feat_dim = self.encoder_x.g.out_features  # 6144
        self.cls_proj = nn.Linear(feat_dim, d_cls, bias=False)
        self.cls_back = nn.Linear(d_cls, feat_dim, bias=False)
        self.e_to_y = nn.Linear(d_cls, feat_dim)
        nn.init.zeros_(self.cls_back.weight)
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
        z_cls = self.cls_proj(src)                               # (B, d_cls)  ← BPR collapse
        x_cls = self.cls_back(z_cls)
        z_inst = x_weight - x_cls

        state["z"] = z_cls
        resid_d = src - x_cls
        cos = F.cosine_similarity(resid_d, x_cls, dim=-1)        # (B,)
        state["aux_loss"] = (cos ** 2).mean()

        x_mult = x_weight if mult_mode == "full" else z_inst
        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_mult.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = y2 + self.e_to_y(z_cls).unsqueeze(-1).unsqueeze(-1)
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd

    if verbose:
        print(f"[ortho_pipe] orthogonal-subspace conditioning applied "
              f"(d_cls={d_cls}, mult={mult_mode}, detach_instance={detach_instance}, zero-init)")
    return True
