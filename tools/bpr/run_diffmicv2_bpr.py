"""DiffMICv2 BPR integration — monkey-patches CoolSystem.training_step and
CoolSystem.configure_optimizers without modifying the original code.

Environment variables:
  BPR_HOOK     = attn | prelin4 | enc512 | enc512_local | xweight
      - attn        : capture DCG.AttentionModule.z (B, 512)
      - prelin4     : capture ConditionalModel pre-lin4 feature, spatial mean -> (B, 6144)
      - enc512      : capture ConditionalModel.encoder_x ResNet18 output (B, 512)
      - enc512_local: capture ConditionalModel.encoder_x_l ResNet18 output, K-crop mean -> (B, 512)
      - xweight     : capture fused conditioning vector (B, 6144)
      - xweight_bn  : insert in-path bottleneck Linear(6144->BN_DIM)->ReLU->Linear(BN_DIM->6144),
                      capture mid (B, BN_DIM); BN_SKIP=1 adds residual
      - xweight_aux : parallel auxiliary bottleneck (additive), capture mid (B, BN_DIM);
                      aux_up zero-initialized so auxiliary output starts at zero
      - dual_gl     : independent BPR on global encoder_x (B, 512) and local encoder_x_l mean (B, 512)
  BPR_BN_DIM   = int (default 512)
  BPR_BN_SKIP  = 0 | 1 (default 0) — xweight_bn only: 0=pure, 1=residual
  BPR_T_MAX    = float (0, 1] (default 1.0) — only include samples with mean timestep fraction < BPR_T_MAX
  BPR_WARMUP_EPOCHS = int (default 0) — epochs before BPR loss is activated
  BPR_MIN_ACTIVE    = int (default 2) — skip BPR if fewer than this many samples pass the gate
  BPR_STAGE    = 1 | 2 (default 1)
      1 = joint training (diffusion + BPR)
      2 = post-hoc refinement: load stage-1 checkpoint, reduce diffusion weight, train BPR only
  BPR_STAGE2_CKPT     = str (required when BPR_STAGE=2)
  BPR_STAGE2_DIFF_W   = float (default 0.0)
  BPR_STAGE2_LR_SCALE = float (default 0.1)
  DCG_UNFREEZE = 0 | attn | local | all  # joint | mgda | pcgrad
  DCG_LR_SCALE = float (default 0.1)
  DCG_WARMUP   = int   (default 0)
"""
import os, sys, argparse
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models/DiffMICv2-main"))

import torch
import torch.nn.functional as F
from bpr_loss import bpr_prototype_loss, total_bpr_loss

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.1"))
BPR_ORTHO_LAMBDA = float(os.environ.get("BPR_ORTHO_LAMBDA", "1.0"))
BPR_VICREG = os.environ.get("BPR_VICREG", "0") == "1"
VICREG_VAR_W = float(os.environ.get("VICREG_VAR_W", "1.0"))
VICREG_COV_W = float(os.environ.get("VICREG_COV_W", "0.0"))
VICREG_CLASSWISE = os.environ.get("VICREG_CLASSWISE", "0") == "1"
VICREG_GAMMA = float(os.environ.get("VICREG_GAMMA", "1.0"))
AUC_SURR_LAMBDA = float(os.environ.get("AUC_SURR_LAMBDA", "0"))       # 0=off
AUC_SURR_MODE   = os.environ.get("AUC_SURR_MODE", "logistic")         # logistic | hinge
AUC_SURR_MARGIN = float(os.environ.get("AUC_SURR_MARGIN", "0.1"))
AUC_SURR_TMAX   = float(os.environ.get("AUC_SURR_TMAX", "0.5"))
BPR_ADV = os.environ.get("BPR_ADV", "0") == "1"
NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
BPR_PROTO = os.environ.get("BPR_PROTO", "mean")   # mean | geomedian | sinkhorn
BPR_MODE = os.environ.get("BPR_MODE", "joint")   # joint | mgda | pcgrad
BPR_PROTO_SCOPE = os.environ.get("BPR_PROTO_SCOPE", "batch")   # batch | global
BPR_BUFFER_SIZE = int(os.environ.get("BPR_BUFFER_SIZE", "512"))
from mgda import mgda_step
from pcgrad import pcgrad_step
from bpr_loss import geometric_median, sinkhorn_centroid
BPR_PROJ_DIM = int(os.environ.get("BPR_PROJ_DIM", "128"))
BPR_PROJ_HIDDEN = int(os.environ.get("BPR_PROJ_HIDDEN", "512"))

DCG_UNFREEZE = os.environ.get("DCG_UNFREEZE", "0")          # 0 | attn | local | all
DCG_LR_SCALE = float(os.environ.get("DCG_LR_SCALE", "0.1"))
DCG_WARMUP   = int(os.environ.get("DCG_WARMUP", "0"))

BPR_HOOK = os.environ.get("BPR_HOOK", "attn")               # attn | prelin4 | enc512 | enc512_local | xweight | xweight_bn | xweight_aux | dual_gl

BPR_BN_DIM  = int(os.environ.get("BPR_BN_DIM", "512"))
BPR_BN_SKIP = os.environ.get("BPR_BN_SKIP", "0") == "1"
BPR_AUX_DETACH = os.environ.get("BPR_AUX_DETACH", "0") == "1"

BPR_LOCAL_POOL = os.environ.get("BPR_LOCAL_POOL", "mean")

BPR_T_MAX           = float(os.environ.get("BPR_T_MAX", "1.0"))
BPR_WARMUP_EPOCHS   = int(os.environ.get("BPR_WARMUP_EPOCHS", "0"))
BPR_MIN_ACTIVE      = int(os.environ.get("BPR_MIN_ACTIVE", "2"))

BPR_STAGE             = int(os.environ.get("BPR_STAGE", "1"))
BPR_STAGE2_CKPT       = os.environ.get("BPR_STAGE2_CKPT", "")
BPR_STAGE2_DIFF_W     = float(os.environ.get("BPR_STAGE2_DIFF_W", "0.0"))
BPR_STAGE2_LR_SCALE   = float(os.environ.get("BPR_STAGE2_LR_SCALE", "0.1"))

BPR_TWO_PHASE         = os.environ.get("BPR_TWO_PHASE", "0") == "1"
BPR_PHASE1_EPOCHS     = int(os.environ.get("BPR_PHASE1_EPOCHS", "-1"))
BPR_PHASE2_LR_SCALE   = float(os.environ.get("BPR_PHASE2_LR_SCALE", "0.1"))
_phase_state = {"frozen": False, "phase1_done_at": -1}

# _buffer[branch] = {"cls": {c: [tensors]}, "cap": N}
_buffer = {}
def _ensure_buffer(branch):
    if branch not in _buffer:
        _buffer[branch] = {"cls": {c: [] for c in range(NUM_CLASSES)}, "cap": BPR_BUFFER_SIZE}
    return _buffer[branch]

def _update_buffer(branch, zp, labels):
    if BPR_PROTO_SCOPE != "global":
        return
    buf = _ensure_buffer(branch)
    for c in range(NUM_CLASSES):
        m = (labels == c)
        if m.any():
            buf["cls"][c].append(zp[m].detach().cpu())
            n = sum(t.size(0) for t in buf["cls"][c])
            while n > buf["cap"] and len(buf["cls"][c]) > 1:
                drop = buf["cls"][c].pop(0)
                n -= drop.size(0)

def _buffer_prototypes(branch, device):
    if BPR_PROTO_SCOPE != "global" or branch not in _buffer:
        return None
    buf = _buffer[branch]
    protos = []
    for c in range(NUM_CLASSES):
        if not buf["cls"][c]:
            return None
        X = torch.cat(buf["cls"][c], dim=0).to(device)
        if X.size(0) < 8:
            return None
        if BPR_PROTO == "geomedian": p = geometric_median(X)
        elif BPR_PROTO == "sinkhorn": p = sinkhorn_centroid(X)
        else: p = X.mean(0)
        p = torch.nn.functional.normalize(p, dim=-1)
        protos.append(p)
    return torch.stack(protos, dim=0)

import torch.nn as nn
_proj = {}
def _ensure_projection(branch, feat_dim, device):
    if branch not in _proj:
        head = nn.Sequential(
            nn.Linear(feat_dim, BPR_PROJ_HIDDEN), nn.ReLU(inplace=False),
            nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
        ).to(device)
        _proj[branch] = {
            "head": head,
            "opt":  torch.optim.Adam(head.parameters(), lr=1e-3),
        }
    return _proj[branch]["head"], _proj[branch]["opt"]

_state = {"z": None, "zg": None, "zl": None}

if BPR_HOOK == "attn":
    import pretraining.modules as M
    _orig_am = M.AttentionModule.forward
    def _am_patched(self, h_crops):
        z, attn, y = _orig_am(self, h_crops)
        _state["z"] = z
        return z, attn, y
    M.AttentionModule.forward = _am_patched
    print(f"[bpr-hook] attn  — DCG.AttentionModule.z capture")

elif BPR_HOOK == "prelin4":
    import model as MDL
    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape

        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_in = self.encoder_x_l(x_l_in)
        x_l_in = self.norm_l(x_l_in)

        x_g = self.encoder_x(x)
        x_g = self.norm(x_g)

        y2 = self.lin1(y, t)
        y2 = self.unetnorm1(y2)
        y2 = F.softplus(y2)

        x_l_in = x_l_in.reshape(bz, np_, x_l_in.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g.unsqueeze(-1), x_l_in], dim=-1)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(x_cat * w, dim=-1)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2

        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)

        _state["z"] = y2.mean(dim=(-2, -1))

        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] prelin4 — ConditionalModel pre-lin4 feature (spatial-mean) capture")

elif BPR_HOOK == "dual_gl":
    import model as MDL
    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape

        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_512_raw = self.encoder_x_l.f(x_l_in)
        x_l_512 = torch.flatten(x_l_512_raw, start_dim=1)         # (B*K, 512)
        x_l_feat = self.encoder_x_l.g(x_l_512)                    # (B*K, 6144)
        x_l_feat = self.norm_l(x_l_feat)

        x_512_raw = self.encoder_x.f(x)
        x_512 = torch.flatten(x_512_raw, start_dim=1)             # (B, 512)
        x_g_feat = self.encoder_x.g(x_512)
        x_g_feat = self.norm(x_g_feat)

        # ★ DUAL capture
        _state["zg"] = x_512                                       # global (B, 512)
        if BPR_LOCAL_POOL == "parallel":
            _state["zl"] = x_l_512
        else:
            _state["zl"] = x_l_512.view(bz, np_, -1).mean(dim=1)  # local pooled (B, 512)

        x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g_feat.unsqueeze(-1), x_l_feat], dim=-1)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(x_cat * w, dim=-1)

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] dual_gl — global encoder_x + local encoder_x_l (K=6 mean) dual capture, each (B, 512)")

elif BPR_HOOK == "xweight_aux":
    import model as MDL
    _orig_cm_init = MDL.ConditionalModel.__init__
    def _cm_init_with_aux(self, *args, **kwargs):
        _orig_cm_init(self, *args, **kwargs)
        feat_dim = self.encoder_x.g.out_features
        self.aux_down = nn.Linear(feat_dim, BPR_BN_DIM)
        self.aux_relu = nn.ReLU(inplace=False)
        self.aux_up   = nn.Linear(BPR_BN_DIM, feat_dim)
        nn.init.zeros_(self.aux_up.weight)
        nn.init.zeros_(self.aux_up.bias)
        print(f"[bpr-hook] xweight_aux — added parallel bottleneck {feat_dim} → {BPR_BN_DIM} → {feat_dim} (aux_up zero-init)")
    MDL.ConditionalModel.__init__ = _cm_init_with_aux

    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
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

        _aux_src = x_weight.detach() if BPR_AUX_DETACH else x_weight
        x_aux_mid = self.aux_down(_aux_src)                                # (B, BN_DIM=512)
        _state["z"] = x_aux_mid
        x_aux_out = self.aux_up(self.aux_relu(x_aux_mid))
        x_weight = x_weight + x_aux_out

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] xweight_aux — parallel auxiliary bottleneck (B, {BPR_BN_DIM}) capture, zero-init residual")

elif BPR_HOOK == "xweight_bn":
    import model as MDL
    _orig_cm_init = MDL.ConditionalModel.__init__
    def _cm_init_with_bn(self, *args, **kwargs):
        _orig_cm_init(self, *args, **kwargs)
        feat_dim = self.encoder_x.g.out_features  # 6144
        self.bn_down = nn.Linear(feat_dim, BPR_BN_DIM)
        self.bn_relu = nn.ReLU(inplace=False)
        self.bn_up   = nn.Linear(BPR_BN_DIM, feat_dim)
        print(f"[bpr-hook] xweight_bn — added bottleneck {feat_dim} → {BPR_BN_DIM} → {feat_dim}  (skip={BPR_BN_SKIP})")
    MDL.ConditionalModel.__init__ = _cm_init_with_bn

    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape

        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_feat = self.encoder_x_l(x_l_in)
        x_l_feat = self.norm_l(x_l_feat)

        x_g = self.encoder_x(x)
        x_g = self.norm(x_g)

        x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)         # (B, 6144, 7)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(x_cat * w, dim=-1)                           # (B, 6144)

        # ★ In-path bottleneck: 6144 → BN_DIM → 6144
        x_bn  = self.bn_down(x_weight)                                    # (B, BN_DIM)
        _state["z"] = x_bn
        x_act = self.bn_relu(x_bn)
        x_up  = self.bn_up(x_act)                                         # (B, 6144)
        x_weight = (x_weight + x_up) if BPR_BN_SKIP else x_up              # pure / residual

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] xweight_bn — in-path bottleneck capture (B, {BPR_BN_DIM}), skip={BPR_BN_SKIP}")

elif BPR_HOOK == "xweight":
    import model as MDL
    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape

        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_feat = self.encoder_x_l(x_l_in)
        x_l_feat = self.norm_l(x_l_feat)

        x_g = self.encoder_x(x)
        x_g = self.norm(x_g)

        x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)         # (B, 6144, 7)
        w = torch.softmax(self.cond_weight, dim=2)                        # (1, 6144, 7)
        x_weight = torch.sum(x_cat * w, dim=-1)                           # (B, 6144)

        _state["z"] = x_weight

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] xweight — ConditionalModel fused conditioning vector (B, 6144) capture")

elif BPR_HOOK in ("enc512", "enc512_local"):
    import model as MDL
    _orig_cm_fwd = MDL.ConditionalModel.forward
    def _cm_fwd_patched(self, x, y, t, x_l, attn):
        bz, np_, I, J = x_l.shape

        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_512_raw = self.encoder_x_l.f(x_l_in)
        x_l_512 = torch.flatten(x_l_512_raw, start_dim=1)         # (B*K, 512)
        x_l_feat = self.encoder_x_l.g(x_l_512)                    # (B*K, 6144)
        x_l_feat = self.norm_l(x_l_feat)

        x_512_raw = self.encoder_x.f(x)
        x_512 = torch.flatten(x_512_raw, start_dim=1)             # (B, 512)
        x_g = self.encoder_x.g(x_512)                             # (B, 6144)
        x_g = self.norm(x_g)

        # ★ 512-d hook capture
        if BPR_HOOK == "enc512":
            _state["z"] = x_512                                   # (B, 512)
        else:
            if BPR_LOCAL_POOL == "parallel":
                _state["z"] = x_l_512                             # (B*K, 512)
            else:
                _state["z"] = x_l_512.view(bz, np_, -1).mean(dim=1)

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
        x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(x_cat * w, dim=-1)
        y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd_patched
    print(f"[bpr-hook] {BPR_HOOK} — ConditionalModel encoder pre-projection 512-d capture")

elif BPR_HOOK == "dual2ch":
    import dual_pipe
    dual_pipe.apply(_state)
    print(f"[bpr-hook] dual2ch — dual-channel conditioning (instance x_weight BPR-free + narrow class pipe BPR-shaped)")

elif BPR_HOOK == "ortho":
    import ortho_pipe
    ortho_pipe.apply(_state)
    print(f"[bpr-hook] ortho")

else:
    raise ValueError(f"BPR_HOOK={BPR_HOOK} not supported (attn | prelin4 | enc512 | enc512_local | xweight | xweight_bn | xweight_aux | dual_gl | dual2ch | ortho)")

import diffuser_trainer as DT
from utils import cast_label_to_one_hot_and_prototype as _cast_oh

if BPR_STAGE == 2:
    _orig_init = DT.CoolSystem.__init__
    def _init_stage2(self, hparams):
        _orig_init(self, hparams)
        if BPR_STAGE2_LR_SCALE != 1.0:
            try:
                self.params.optim.lr = float(self.params.optim.lr) * BPR_STAGE2_LR_SCALE
            except Exception:
                pass
            self.initlr = float(self.initlr) * BPR_STAGE2_LR_SCALE
            print(f"[stage2] LR scaled by {BPR_STAGE2_LR_SCALE} → initlr={self.initlr:.2e}")
        if not BPR_STAGE2_CKPT:
            raise RuntimeError("[stage2] BPR_STAGE2_CKPT not set.")
        if not os.path.exists(BPR_STAGE2_CKPT):
            raise FileNotFoundError(f"[stage2] BPR_STAGE2_CKPT not found: {BPR_STAGE2_CKPT}")
        try:
            sd = torch.load(BPR_STAGE2_CKPT, map_location="cpu", weights_only=False)
        except TypeError:
            sd = torch.load(BPR_STAGE2_CKPT, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[stage2] loaded {BPR_STAGE2_CKPT}")
        print(f"[stage2] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    DT.CoolSystem.__init__ = _init_stage2

def _enter_phase2_if_due(self):
    if not BPR_TWO_PHASE or _phase_state["frozen"]:
        return
    try:
        total = int(self.trainer.max_epochs) if self.trainer is not None else 100
    except Exception:
        total = 100
    s1 = BPR_PHASE1_EPOCHS if BPR_PHASE1_EPOCHS > 0 else max(1, total // 2)
    if self.current_epoch < s1:
        return

    n_frozen = n_trainable = 0
    head_params = []
    HEAD_PREFIXES = {"lin1", "lin2", "lin3", "lin4",
                     "unetnorm1", "unetnorm2", "unetnorm3"}
    for name, p in self.model.named_parameters():
        top = name.split(".", 1)[0]
        if top in HEAD_PREFIXES:
            p.requires_grad = True
            head_params.append(p)
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    for _, p in self.aux_model.named_parameters():
        p.requires_grad = False
        n_frozen += p.numel()

    added = False
    try:
        opts = self.trainer.optimizers
        lr_new = float(self.initlr) * BPR_PHASE2_LR_SCALE
        for pg in opts[0].param_groups:
            pg["lr"] = 0.0
        if head_params:
            opts[0].add_param_group({"params": head_params, "lr": lr_new})
            added = True
        print(f"[bpr-2phase] LR for lin1-4 head = {lr_new:.2e}")
    except Exception as _e:
        print(f"[bpr-2phase] optimizer update failed ({_e})")

    print(f"[bpr-2phase] === Phase 2 begin at epoch {self.current_epoch} ===")
    print(f"[bpr-2phase] frozen={n_frozen/1e6:.2f}M | trainable={n_trainable/1e6:.4f}M (lin1-4 + unetnorm)")
    print(f"[bpr-2phase] head -> optimizer: {'OK' if added else 'FAILED'}")
    print(f"[bpr-2phase] BPR DISABLED in phase 2")
    _phase_state["frozen"] = True
    _phase_state["phase1_done_at"] = s1

if BPR_TWO_PHASE:
    print(f"[bpr-2phase] ENABLED — phase1_epochs={BPR_PHASE1_EPOCHS if BPR_PHASE1_EPOCHS > 0 else 'auto (max_epochs/2)'}, "
          f"phase2_lr_scale={BPR_PHASE2_LR_SCALE}")

_orig_ts = DT.CoolSystem.training_step
def _ts_patched(self, batch, batch_idx):
    _enter_phase2_if_due(self)
    in_phase2 = BPR_TWO_PHASE and _phase_state["frozen"]

    self.model.train()
    do_unfreeze = (not in_phase2) and ((DCG_UNFREEZE != "0") and (self.current_epoch >= DCG_WARMUP))
    self.aux_model.train() if do_unfreeze else self.aux_model.eval()

    x_batch, y_batch_raw = batch
    y_batch_oh, _ = _cast_oh(y_batch_raw, self.params)
    y_batch_oh = y_batch_oh.cuda(); x_batch = x_batch.cuda()

    if do_unfreeze:
        y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
    else:
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

    bz, nc, H, W = attn_map.size()
    bz, npp = attns.size()
    y_map = y_batch_oh.unsqueeze(1).expand(-1, npp*npp, -1).reshape(bz*npp*npp, nc)
    noise = torch.randn_like(y_map).to(self.device)
    timesteps = torch.randint(
        0, self.DiffSampler.scheduler.config.num_train_timesteps,
        (bz*npp*npp,), device=self.device,
    ).long()
    noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
    noisy_y = noisy_y.view(bz, npp*npp, -1).permute(0, 2, 1).reshape(bz, nc, npp, npp)
    y0_cond = self.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, npp)
    y_fusion = torch.cat([y0_cond, noisy_y], dim=1)
    attns2 = attns.unsqueeze(-1)
    attns2 = (attns2 * attns2.transpose(1, 2)).unsqueeze(1)
    noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns2)
    noise = noise.view(bz, npp*npp, -1).permute(0, 2, 1).reshape(bz, nc, npp, npp)
    loss = self.diffusion_focal_loss(y0_aux, y_batch_oh, noise_pred, noise)
    if BPR_STAGE == 2 and BPR_STAGE2_DIFF_W != 1.0:
        loss = BPR_STAGE2_DIFF_W * loss
    self.log("train_loss", loss, prog_bar=True)
    if in_phase2:
        self.log("train_phase", 2.0, prog_bar=True)
    out = {"loss": loss}

    if AUC_SURR_LAMBDA > 0:
        try:
            import auc_loss as _al
            _acp = self.DiffSampler.scheduler.alphas_cumprod.to(noisy_y.device, noisy_y.dtype)
            _a = _acp[timesteps].view(bz, npp, npp).unsqueeze(1).clamp_min(1e-8)   # (bz,1,np,np)
            _x0 = (noisy_y - (1.0 - _a).sqrt() * noise_pred) / _a.sqrt()
            _score = torch.softmax(_x0.float(), dim=1).mean(dim=[2, 3])[:, 1]
            _labs = y_batch_raw.view(-1).to(_score.device).float()
            if AUC_SURR_TMAX < 1.0:
                _T = float(self.DiffSampler.scheduler.config.num_train_timesteps)
                _tf = timesteps.view(bz, npp * npp).float().mean(dim=1) / _T
                _keep = _tf < AUC_SURR_TMAX
                _score, _labs = _score[_keep], _labs[_keep]
            _aucl = _al.auc_surrogate(_score, _labs, mode=AUC_SURR_MODE, margin=AUC_SURR_MARGIN)
            if torch.is_tensor(_aucl) and torch.isfinite(_aucl) and _aucl.item() != 0.0:
                out["loss"] = out["loss"] + AUC_SURR_LAMBDA * _aucl
                self.log("train_auc_surr", float(_aucl.item()), prog_bar=True)
        except Exception as _e:
            if self.current_epoch == 0 and batch_idx == 0:
                print(f"[auc-surrogate] skipped: {_e}")

    if in_phase2:
        return out
    if self.current_epoch < BPR_WARMUP_EPOCHS:
        return out

    if BPR_HOOK == "dual_gl":
        _branch_keys = [("zg", "g"), ("zl", "l")]
    else:
        _branch_keys = [("z",  "main")]

    try:
        labels_full = y_batch_raw.view(-1).to(self.device)
    except Exception:
        return out

    _gate_mask = None
    if BPR_T_MAX < 1.0:
        try:
            T_total = float(self.DiffSampler.scheduler.config.num_train_timesteps)
            t_per_img = timesteps.view(bz, npp * npp).float().mean(dim=1)
            t_frac = t_per_img / T_total
            _gate_mask = (t_frac < BPR_T_MAX)
            n_active = int(_gate_mask.sum().item())
            self.log("bpr_n_active", float(n_active), prog_bar=False)
            if n_active < BPR_MIN_ACTIVE:
                return out
        except Exception:
            _gate_mask = None

    _bpr_per_branch = []
    for state_key, branch_name in _branch_keys:
        z = _state.get(state_key)
        if z is None:
            continue
        labels_b = labels_full
        if z.size(0) == bz:
            if _gate_mask is not None:
                z = z[_gate_mask]
                labels_b = labels_full[_gate_mask]
        elif z.size(0) % bz == 0:
            K = z.size(0) // bz
            labels_b = labels_full.repeat_interleave(K)
            if _gate_mask is not None:
                mask_exp = _gate_mask.repeat_interleave(K)
                z = z[mask_exp]
                labels_b = labels_b[mask_exp]
        else:
            continue
        if z.size(0) < 2:
            continue
        try:
            head, _ = _ensure_projection(branch_name, z.size(-1), z.device)
            head.train()
            zp = head(z.float())
            zp = torch.nn.functional.normalize(zp, dim=-1)
            _update_buffer(branch_name, zp, labels_b)
            ext_proto = _buffer_prototypes(branch_name, zp.device) if BPR_PROTO_SCOPE == "global" else None
            bpr_b = total_bpr_loss(zp, labels_b, num_classes=NUM_CLASSES,
                                    use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                    external_prototypes=ext_proto)
            _bpr_per_branch.append((branch_name, bpr_b))
        except Exception:
            continue

    if not _bpr_per_branch:
        return out

    bpr = sum(b for _, b in _bpr_per_branch) / len(_bpr_per_branch)
    if len(_bpr_per_branch) > 1:
        for n, b in _bpr_per_branch:
            self.log(f"train_bpr_{n}", float(b.item()), prog_bar=False)
    labels = labels_full
    if BPR_HOOK in ("prelin4", "enc512", "enc512_local", "xweight", "xweight_bn", "xweight_aux", "dual_gl", "dual2ch", "ortho"):
        _grad_target = [p for p in self.model.parameters() if p.requires_grad]
    else:
        _grad_target = [p for p in self.aux_model.parameters() if p.requires_grad]
        if not _grad_target:
            _grad_target = [p for p in self.model.parameters() if p.requires_grad]

    if BPR_MODE == "pcgrad":
        try:
            params = _grad_target
            g_ce  = torch.autograd.grad(out["loss"], params, retain_graph=True, allow_unused=True)
            g_bpr = torch.autograd.grad(BPR_LAMBDA * bpr, params, retain_graph=True, allow_unused=True)
            g1 = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                             for p, g in zip(params, g_ce)])
            g2 = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                             for p, g in zip(params, g_bpr)])
            dot = (g1 * g2).sum()
            cos = (dot / (g1.norm() * g2.norm() + 1e-12)).item()
        except Exception:
            cos = 1.0
        if cos < 0:
            w = max(0.0, 1.0 + cos)
            out["loss"] = out["loss"] + w * BPR_LAMBDA * bpr
        else:
            out["loss"] = out["loss"] + BPR_LAMBDA * bpr
        self.log("pcgrad_cos", float(cos), prog_bar=True)
    elif BPR_MODE == "mgda":
        try:
            params = _grad_target
            g_ce  = torch.autograd.grad(out["loss"], params, retain_graph=True,
                                          create_graph=False, allow_unused=True)
            g_bpr = torch.autograd.grad(BPR_LAMBDA * bpr, params, retain_graph=True,
                                          create_graph=False, allow_unused=True)
            g1 = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                             for p, g in zip(params, g_ce)])
            g2 = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                             for p, g in zip(params, g_bpr)])
            from mgda import mgda_two_task_alpha
            alpha = mgda_two_task_alpha(g1, g2)
        except Exception:
            alpha = 1.0 - BPR_LAMBDA / (1.0 + BPR_LAMBDA)
        out["loss"] = alpha * out["loss"] + (1.0 - alpha) * BPR_LAMBDA * bpr
        self.log("alpha_mgda", float(alpha), prog_bar=True)
    else:
        out["loss"] = out["loss"] + BPR_LAMBDA * bpr

    if _state.get("aux_loss") is not None:
        out["loss"] = out["loss"] + BPR_ORTHO_LAMBDA * _state["aux_loss"]
        self.log("train_ortho", float(_state["aux_loss"].item()), prog_bar=False)
        _state["aux_loss"] = None
    if BPR_VICREG and _state.get("z") is not None:
        import vicreg_loss as _vc
        _zc = _state["z"].float()
        if VICREG_CLASSWISE:
            _vv = _vc.variance_loss_classwise(_zc, labels_full, num_classes=NUM_CLASSES, gamma=VICREG_GAMMA)
        else:
            _vv = _vc.variance_loss(_zc, gamma=VICREG_GAMMA)
        out["loss"] = out["loss"] + VICREG_VAR_W * _vv
        self.log("train_vic_var", float(_vv.item()), prog_bar=False)
        if VICREG_COV_W > 0:
            _cc = _vc.covariance_loss(_zc)
            out["loss"] = out["loss"] + VICREG_COV_W * _cc
            self.log("train_vic_cov", float(_cc.item()), prog_bar=False)
    self.log("train_bpr", float(bpr.item()), prog_bar=True)
    return out
DT.CoolSystem.training_step = _ts_patched

_orig_cfg = DT.CoolSystem.configure_optimizers
def _cfg_with_aux(self):
    ret = _orig_cfg(self)
    if DCG_UNFREEZE == "0":
        return ret

    for p in self.aux_model.parameters():
        p.requires_grad = False
    if DCG_UNFREEZE == "all":
        for p in self.aux_model.parameters():
            p.requires_grad = True
    elif DCG_UNFREEZE == "attn":
        for n, p in self.aux_model.named_parameters():
            if any(k in n for k in ("attention_module", "mil_attn", "classifier_linear")):
                p.requires_grad = True
    elif DCG_UNFREEZE == "local":
        for n, p in self.aux_model.named_parameters():
            if any(k in n for k in ("local_network", "attention_module", "mil_attn", "classifier_linear")):
                p.requires_grad = True
    else:
        print(f"[dcg-unfreeze] WARN: unknown DCG_UNFREEZE={DCG_UNFREEZE} — keeping frozen")
        return ret

    aux_params = [p for p in self.aux_model.parameters() if p.requires_grad]
    if not aux_params:
        print(f"[dcg-unfreeze] mode={DCG_UNFREEZE} — no matching params, keeping frozen")
        return ret

    if isinstance(ret, tuple) and len(ret) >= 1:
        opts = ret[0] if isinstance(ret[0], list) else [ret[0]]
    elif isinstance(ret, list):
        opts = ret
    else:
        opts = [ret]

    aux_lr = DCG_LR_SCALE * self.initlr
    opts[0].add_param_group({"params": aux_params, "lr": aux_lr})
    print(f"[dcg-unfreeze] mode={DCG_UNFREEZE}  n_params={sum(p.numel() for p in aux_params)}  "
          f"lr={aux_lr:.2e}  warmup_epochs={DCG_WARMUP}")
    return ret
DT.CoolSystem.configure_optimizers = _cfg_with_aux

def _on_after_backward_patched(self):
    for branch, info in _proj.items():
        if info.get("opt") is not None:
            info["opt"].step()
            info["opt"].zero_grad(set_to_none=True)
DT.CoolSystem.on_after_backward = _on_after_backward_patched
print(f"[bpr] DiffMICv2 training_step + configure_optimizers monkey-patched "
      f"(stage={BPR_STAGE}, hook={BPR_HOOK}, lambda={BPR_LAMBDA}, mode={BPR_MODE}, proto={BPR_PROTO}, scope={BPR_PROTO_SCOPE}, "
      f"t_max={BPR_T_MAX}, bpr_warmup={BPR_WARMUP_EPOCHS}, "
      f"dcg_unfreeze={DCG_UNFREEZE}, dcg_lr_scale={DCG_LR_SCALE}, dcg_warmup={DCG_WARMUP})")
if BPR_STAGE == 2:
    print(f"[bpr] stage2 — diff_w={BPR_STAGE2_DIFF_W}  lr_scale={BPR_STAGE2_LR_SCALE}  ckpt={BPR_STAGE2_CKPT}")

DT.main()
