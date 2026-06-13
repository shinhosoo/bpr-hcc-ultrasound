"""Architecture-only BPR hooks (parameter-adding) — shared by EVAL.

xweight_aux / xweight_bn add submodules (aux_down/aux_up or bn_down/bn_up) to
ConditionalModel.__init__ and accumulate their outputs into x_weight. Because
these modules are saved in the checkpoint, eval must reconstruct the same
architecture and patched forward to avoid train/test mismatch.

This module reproduces the architecture (__init__ + forward) only.
BPR loss / prototype buffer / training_step patches are not included here.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_state = {"z": None}

PARAM_ADDING_HOOKS = ("xweight_aux", "xweight_bn")


def needs_arch_rebuild(hook: str) -> bool:
    return hook in PARAM_ADDING_HOOKS


def apply_arch_hook(hook=None, bn_dim=None, detach_main=None, bn_skip=None, verbose=True):
    """Apply xweight_aux / xweight_bn architecture patch to ConditionalModel.

    Must be called before model instantiation so that __init__ creates the
    submodules and the checkpoint can populate them via strict=False.
    All values are read from env vars and must match training:
      BPR_HOOK, BPR_BN_DIM (default 512), BPR_BN_SKIP (xweight_bn only), BPR_AUX_DETACH.
    """
    if hook is None:
        hook = os.environ.get("BPR_HOOK", "attn")
    if not needs_arch_rebuild(hook):
        if verbose:
            print(f"[bpr-arch-hook] hook={hook} — no parameters added, eval rebuild not required")
        return False

    if bn_dim is None:
        bn_dim = int(os.environ.get("BPR_BN_DIM", "512"))
    if detach_main is None:
        detach_main = os.environ.get("BPR_AUX_DETACH", "0") == "1"
    if bn_skip is None:
        bn_skip = os.environ.get("BPR_BN_SKIP", "0") == "1"

    import model as MDL

    if hook == "xweight_aux":
        _orig_init = MDL.ConditionalModel.__init__

        def _cm_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            feat_dim = self.encoder_x.g.out_features  # 6144
            self.aux_down = nn.Linear(feat_dim, bn_dim)
            self.aux_relu = nn.ReLU(inplace=False)
            self.aux_up = nn.Linear(bn_dim, feat_dim)
            nn.init.zeros_(self.aux_up.weight)
            nn.init.zeros_(self.aux_up.bias)
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
            x_weight = torch.sum(x_cat * w, dim=-1)                 # (B, 6144)
            src = x_weight.detach() if detach_main else x_weight
            x_aux_mid = self.aux_down(src)                          # (B, bn_dim)
            _state["z"] = x_aux_mid
            x_aux_out = self.aux_up(self.aux_relu(x_aux_mid))
            x_weight = x_weight + x_aux_out
            y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
            y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
            y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
            y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
            return self.lin4(y2)
        MDL.ConditionalModel.forward = _cm_fwd

        if verbose:
            print(f"[bpr-arch-hook] xweight_aux REBUILT for eval "
                  f"(bn_dim={bn_dim}, detach_main={detach_main}, zero-init residual)")
        return True

    if hook == "xweight_bn":
        _orig_init = MDL.ConditionalModel.__init__

        def _cm_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            feat_dim = self.encoder_x.g.out_features  # 6144
            self.bn_down = nn.Linear(feat_dim, bn_dim)
            self.bn_relu = nn.ReLU(inplace=False)
            self.bn_up = nn.Linear(bn_dim, feat_dim)
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
            x_weight = torch.sum(x_cat * w, dim=-1)                 # (B, 6144)
            x_bn = self.bn_down(x_weight)
            _state["z"] = x_bn
            x_act = self.bn_relu(x_bn)
            x_up = self.bn_up(x_act)
            x_weight = (x_weight + x_up) if bn_skip else x_up
            y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
            y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
            y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
            y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
            return self.lin4(y2)
        MDL.ConditionalModel.forward = _cm_fwd

        if verbose:
            print(f"[bpr-arch-hook] xweight_bn REBUILT for eval "
                  f"(bn_dim={bn_dim}, skip={bn_skip})")
        return True

    return False
