"""MedViT BPR training — joint (1-stage) or 2-stage.

joint (default): optimizes CE + lambda*BPR together.
2-stage: Stage1 = CE+BPR, Stage2 = backbone frozen, classifier (proj_head) CE only.

Activation:
  env: BPR_TWO_STAGE=1 BPR_STAGE1_EPOCHS=30
  CLI: --two-stage --stage1-epochs 30
"""
import os, sys, argparse, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models/MedViT-main", "CustomDataset"))

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)
import engine as _eng
import utils as _u
from timm.utils import accuracy, ModelEma
from timm.data import Mixup
from bpr_loss import total_bpr_loss, geometric_median, sinkhorn_centroid

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.1"))
BPR_ADV = os.environ.get("BPR_ADV", "0") == "1"
NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
BPR_PROTO = os.environ.get("BPR_PROTO", "mean")  # mean | geomedian | sinkhorn
BPR_MODE = os.environ.get("BPR_MODE", "joint")  # joint | mgda | pcgrad
BPR_PROTO_SCOPE = os.environ.get("BPR_PROTO_SCOPE", "batch")  # batch | global
BPR_PROTO_REFRESH = int(os.environ.get("BPR_PROTO_REFRESH", "1"))
BPR_PROTO_BS = int(os.environ.get("BPR_PROTO_BS", "64"))
BPR_PROTO_EMA = float(os.environ.get("BPR_PROTO_EMA", "0.0"))
from mgda import mgda_step
from pcgrad import pcgrad_step
BPR_PROJ_DIM = int(os.environ.get("BPR_PROJ_DIM", "128"))
BPR_PROJ_HIDDEN = int(os.environ.get("BPR_PROJ_HIDDEN", "512"))
BPR_WARMUP_EPOCHS = int(os.environ.get("BPR_WARMUP_EPOCHS", "5"))
BPR_USE_PROJ = os.environ.get("BPR_USE_PROJ", "0") == "1"

_dense_raw = os.environ.get("BPR_DENSE", "0").lower()
BPR_DENSE = _dense_raw in ("1", "true", "yes", "dense", "both", "2", "all")
BPR_DENSE_MIX = _dense_raw in ("both", "2", "all")
BPR_DENSE_W = float(os.environ.get("BPR_DENSE_W", "1.0"))
BPR_GLOBAL_W = float(os.environ.get("BPR_GLOBAL_W", "1.0"))

BPR_FAITHFUL = os.environ.get("BPR_FAITHFUL", "0") == "1"
BPR_ADV_STEPS = int(os.environ.get("BPR_ADV_STEPS", "2" if BPR_FAITHFUL else "1"))

TWO_STAGE = os.environ.get("BPR_TWO_STAGE", "0") == "1"
STAGE1_EPOCHS = int(os.environ.get("BPR_STAGE1_EPOCHS", "-1"))

BPR_ONLY = os.environ.get("BPR_ONLY", "0") == "1"

BPR_HOOK = os.environ.get("BPR_HOOK", "global")

BPR_BN_DIM = int(os.environ.get("BPR_BN_DIM", "512"))
BPR_LOCAL_STAGE = int(os.environ.get("BPR_LOCAL_STAGE", "2"))
BPR_LOCAL_POOL = os.environ.get("BPR_LOCAL_POOL", "mean")  # mean (B,C) | parallel (B*H*W, C)
BPR_DUAL_GLOBAL_W = float(os.environ.get("BPR_DUAL_GLOBAL_W", "1.0"))
BPR_DUAL_LOCAL_W  = float(os.environ.get("BPR_DUAL_LOCAL_W", "1.0"))

if BPR_HOOK not in ("global", "aux", "dual_gl", "proto"):
    raise ValueError(f"BPR_HOOK={BPR_HOOK} not supported (global | aux | dual_gl | proto)")

import torch.nn as nn
_proj = {"head": None, "opt": None}
_global_proto = {"tensor": None, "last_epoch": -1}
_stage_state = {"frozen": False, "announced_stage2": False}

_proj_local = {"head": None, "opt": None}
_global_proto_local = {"tensor": None, "last_epoch": -1}

_aux_state = {"down": None, "relu": None, "up": None, "opt": None, "attached": False}

def _attach_aux(model, in_dim, device):
    if _aux_state["attached"]:
        return
    _m = model.module if hasattr(model, 'module') else model
    down = nn.Linear(in_dim, BPR_BN_DIM).to(device)
    relu = nn.ReLU(inplace=False)
    up   = nn.Linear(BPR_BN_DIM, in_dim).to(device)
    nn.init.zeros_(up.weight); nn.init.zeros_(up.bias)
    _m.add_module('bpr_aux_down', down)
    _m.add_module('bpr_aux_up',   up)
    _aux_state["down"] = down
    _aux_state["relu"] = relu
    _aux_state["up"]   = up
    _aux_state["opt"]  = torch.optim.Adam(
        list(down.parameters()) + list(up.parameters()), lr=1e-3)
    _aux_state["attached"] = True
    print(f"[bpr-hook] aux — attached bottleneck {in_dim} → {BPR_BN_DIM} → {in_dim} (up zero-init)")

def _ensure_projection(feat_dim, device):
    if _proj["head"] is None:
        _proj["head"] = nn.Sequential(
            nn.Linear(feat_dim, BPR_PROJ_HIDDEN), nn.ReLU(inplace=True),
            nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
        ).to(device)
        _proj["opt"] = torch.optim.Adam(_proj["head"].parameters(), lr=1e-3)
    return _proj["head"], _proj["opt"]


def _ensure_projection_local(feat_dim, device):
    if _proj_local["head"] is None:
        _proj_local["head"] = nn.Sequential(
            nn.Linear(feat_dim, BPR_PROJ_HIDDEN), nn.ReLU(inplace=True),
            nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
        ).to(device)
        _proj_local["opt"] = torch.optim.Adam(_proj_local["head"].parameters(), lr=1e-3)
    return _proj_local["head"], _proj_local["opt"]


def _project_for_bpr_branch(feat, branch="global"):
    """branch: 'global' | 'local'"""
    if BPR_USE_PROJ:
        if branch == "local":
            head, _ = _ensure_projection_local(feat.size(1), feat.device)
        else:
            head, _ = _ensure_projection(feat.size(1), feat.device)
        head.train()
        z = head(feat.float())
    else:
        z = feat.float()
    if BPR_FAITHFUL:
        return z
    return torch.nn.functional.normalize(z, dim=-1)


def _project_for_bpr(feat):
    if BPR_USE_PROJ:
        head, _ = _ensure_projection(feat.size(1), feat.device)
        head.train()
        z = head(feat.float())
    else:
        z = feat.float()
    if BPR_FAITHFUL:
        return z
    return torch.nn.functional.normalize(z, dim=-1)


def _classifier_prototypes(model):
    m = model.module if hasattr(model, "module") else model
    head = getattr(m, "proj_head", None)
    if isinstance(head, nn.Linear):
        return head.weight
    if isinstance(head, nn.Sequential):
        for layer in reversed(head):
            if isinstance(layer, nn.Linear):
                return layer.weight
    return None


def _aggregate_proto(X, kind):
    if X.size(0) == 0:
        return None
    if X.size(0) == 1:
        return X[0]
    if kind == "geomedian":
        return geometric_median(X)
    if kind == "sinkhorn":
        return sinkhorn_centroid(X)
    return X.mean(0)


@torch.no_grad()
def _refresh_global_prototypes(model, dataset, head, device, num_classes, kind,
                               feat_transform=None, target_module_attr="proj_head",
                               ema_cache=None, branch_tag=""):
    """Compute per-class prototypes from full training set.

    feat_transform: callable applied to captured features (e.g., aux_down).
    target_module_attr: module attr to hook ('proj_head' or 'features_stage').
    ema_cache: dict with 'tensor' key for EMA blending.
    branch_tag: logging tag.
    """
    if BPR_USE_PROJ and head is None:
        return None
    was_train_model = model.training
    was_train_head = head.training if head is not None else False
    model.eval()
    if head is not None:
        head.eval()

    seq_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BPR_PROTO_BS, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False,
    )

    _m = model.module if hasattr(model, 'module') else model
    feat_holder = {}
    if target_module_attr == "features_stage":
        def _hook(_mod, _in, _out):
            feat_holder['feat'] = _out
        try:
            lidx = _m.stage_out_idx[BPR_LOCAL_STAGE]
            h = _m.features[lidx].register_forward_hook(_hook)
        except (AttributeError, IndexError) as _e:
            print(f"[bpr-global:{branch_tag}] features stage hook failed ({_e}) — skipping refresh")
            if was_train_model: model.train()
            if head is not None and was_train_head: head.train()
            return None
    else:
        def _hook(_mod, _in):
            feat_holder['feat'] = _in[0]
        h = _m.proj_head.register_forward_pre_hook(_hook)

    per_class = {c: [] for c in range(num_classes)}
    for batch in seq_loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        if torch.is_tensor(y):
            y = y.view(-1).cpu().long()
        else:
            y = torch.tensor(y, dtype=torch.long)
        with torch.cuda.amp.autocast():
            _ = model(x)
        feat = feat_holder.get('feat')
        if feat is None:
            continue
        feat = feat.float()
        if target_module_attr == "features_stage" and feat.dim() == 4:
            B, C, H, W = feat.shape
            if BPR_LOCAL_POOL == "parallel":
                feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                y = y.view(-1, 1).expand(B, H * W).reshape(-1)
            else:
                feat = feat.mean(dim=[2, 3])
        if feat_transform is not None:
            feat = feat_transform(feat)
        if BPR_WARMUP_EPOCHS < 5:
            valid = ~torch.isnan(feat).any(dim=-1)
            if not valid.all():
                print(f"[bpr-global:{branch_tag}] NaN features in {(~valid).sum().item()} samples — skipped")
            feat = feat[valid]
            y = y[valid]
            if feat.size(0) == 0:
                continue
        if BPR_USE_PROJ:
            z = head(feat)
            z = torch.nn.functional.normalize(z, dim=-1)
        else:
            z = torch.nn.functional.normalize(feat, dim=-1)
        for c in range(num_classes):
            m = (y == c)
            if m.any():
                per_class[c].append(z[m].detach())
    h.remove()
    if was_train_model: model.train()
    if head is not None and was_train_head: head.train()

    if any(len(v) == 0 for v in per_class.values()):
        print(f"[bpr-global:{branch_tag}] WARN: some classes have 0 samples, skipping refresh")
        return None

    protos = []
    for c in range(num_classes):
        X = torch.cat(per_class[c], dim=0)
        proto = _aggregate_proto(X, kind)
        proto = torch.nn.functional.normalize(proto, dim=-1)
        protos.append(proto)
    new_protos = torch.stack(protos, dim=0).to(device)
    if BPR_WARMUP_EPOCHS < 5 and torch.isnan(new_protos).any():
        print(f"[bpr-global:{branch_tag}] NaN in prototypes — refresh skipped")
        if was_train_model: model.train()
        if head is not None and was_train_head: head.train()
        return None

    cache = ema_cache if ema_cache is not None else _global_proto
    if BPR_PROTO_EMA > 0 and cache.get("tensor") is not None:
        old = cache["tensor"].to(device)
        new_protos = (1 - BPR_PROTO_EMA) * new_protos + BPR_PROTO_EMA * old
        new_protos = torch.nn.functional.normalize(new_protos, dim=-1)
    print(f"[bpr-global:{branch_tag or 'global'}] refreshed {num_classes} prototypes "
          f"(kind={kind}, n_per_class={[len(v) for v in per_class.values()]})")
    return new_protos


def _bpr_train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                          loss_scaler, max_norm=0, model_ema=None, mixup_fn=None,
                          set_training_mode=True):
    in_stage2 = TWO_STAGE and STAGE1_EPOCHS > 0 and epoch >= STAGE1_EPOCHS
    if in_stage2 and not _stage_state["frozen"]:
        _mm = model.module if hasattr(model, 'module') else model
        n_frozen = n_trainable = 0
        for name, p in _mm.named_parameters():
            if name.startswith('proj_head'):
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        _stage_state["frozen"] = True
        print(f"[bpr-2stage] === Stage 2 begin at epoch {epoch} ===")
        print(f"[bpr-2stage] frozen={n_frozen/1e6:.2f}M | trainable={n_trainable/1e6:.2f}M")

    model.train(set_training_mode)
    metric_logger = _u.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', _u.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('bpr', _u.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('alpha', _u.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.add_meter('alpha', _u.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    stage_tag = "S2" if in_stage2 else ("S1" if TWO_STAGE else "")
    header = f'BPR{stage_tag} Epoch: [{epoch}]' if stage_tag else f'BPR Epoch: [{epoch}]'

    latent_buf = {}
    _m = model.module if hasattr(model, 'module') else model
    hook_handles = []

    if BPR_HOOK == "aux":
        in_dim = None
        if isinstance(_m.proj_head, nn.Sequential) and len(_m.proj_head) > 0 \
                and isinstance(_m.proj_head[0], nn.Linear):
            in_dim = _m.proj_head[0].in_features
        elif isinstance(_m.proj_head, nn.Linear):
            in_dim = _m.proj_head.in_features
        if in_dim is None:
            raise RuntimeError("[bpr-hook:aux] failed to infer proj_head in_features")
        _attach_aux(model, in_dim, device)

        def _aux_pre_hook(_mod, _in):
            x = _in[0]
            mid = _aux_state["down"](x)
            latent_buf['feat_aux'] = mid
            out = x + _aux_state["up"](_aux_state["relu"](mid))
            return (out,)
        hook_handles.append(_m.proj_head.register_forward_pre_hook(_aux_pre_hook))

    elif BPR_HOOK == "dual_gl":
        def _g_pre_hook(_mod, _in):
            latent_buf['feat_g'] = _in[0]
        hook_handles.append(_m.proj_head.register_forward_pre_hook(_g_pre_hook))

        if hasattr(_m, 'features') and hasattr(_m, 'stage_out_idx'):
            try:
                lidx = _m.stage_out_idx[BPR_LOCAL_STAGE]
                def _l_fwd_hook(_mod, _in, _out):
                    latent_buf['feat_l'] = _out
                hook_handles.append(_m.features[lidx].register_forward_hook(_l_fwd_hook))
            except (IndexError, AttributeError) as _e:
                print(f"[bpr-hook:dual_gl] WARN: local stage hook failed ({_e}) — global only")
        else:
            print("[bpr-hook:dual_gl] WARN: model has no features/stage_out_idx — global only")

    else:
        def _pre_hook(_mod, _in):
            latent_buf['feat'] = _in[0]
        hook_handles.append(_m.proj_head.register_forward_pre_hook(_pre_hook))
        if BPR_DENSE:
            def _pre_hook_dense(_mod, _in):
                latent_buf['feat_dense'] = _in[0]
            if hasattr(_m, 'avgpool'):
                hook_handles.append(_m.avgpool.register_forward_pre_hook(_pre_hook_dense))
            else:
                print("[bpr-dense] WARN: model has no avgpool module — dense BPR disabled")

    _refresh_due = (
        not in_stage2
        and BPR_PROTO_SCOPE == "global"
        and epoch >= BPR_WARMUP_EPOCHS
        and (epoch - BPR_WARMUP_EPOCHS) % max(BPR_PROTO_REFRESH, 1) == 0
    )
    if _refresh_due and BPR_HOOK == "global" and _proj["head"] is not None \
            and _global_proto["last_epoch"] != epoch:
        try:
            ds = data_loader.dataset
            protos = _refresh_global_prototypes(
                model, ds, _proj["head"], device, NUM_CLASSES, BPR_PROTO,
                ema_cache=_global_proto, branch_tag="global")
            if protos is not None:
                _global_proto["tensor"] = protos
                _global_proto["last_epoch"] = epoch
        except Exception as _e:
            print(f"[bpr-global] refresh failed — falling back to batch prototype: {_e}")
    elif _refresh_due and BPR_HOOK == "aux" and _aux_state["attached"] \
            and _global_proto["last_epoch"] != epoch:
        try:
            ds = data_loader.dataset
            protos = _refresh_global_prototypes(
                model, ds, _proj.get("head"), device, NUM_CLASSES, BPR_PROTO,
                feat_transform=_aux_state["down"], ema_cache=_global_proto, branch_tag="aux")
            if protos is not None:
                _global_proto["tensor"] = protos
                _global_proto["last_epoch"] = epoch
        except Exception as _e:
            print(f"[bpr-global:aux] refresh failed — falling back to batch prototype: {_e}")
    elif _refresh_due and BPR_HOOK == "dual_gl":
        # Global branch
        if _proj["head"] is not None and _global_proto["last_epoch"] != epoch:
            try:
                ds = data_loader.dataset
                protos = _refresh_global_prototypes(
                    model, ds, _proj["head"], device, NUM_CLASSES, BPR_PROTO,
                    ema_cache=_global_proto, branch_tag="dual_gl/g")
                if protos is not None:
                    _global_proto["tensor"] = protos
                    _global_proto["last_epoch"] = epoch
            except Exception as _e:
                print(f"[bpr-global:dual_gl/g] refresh failed: {_e}")
        if _proj_local["head"] is not None and _global_proto_local["last_epoch"] != epoch:
            try:
                ds = data_loader.dataset
                protos_l = _refresh_global_prototypes(
                    model, ds, _proj_local["head"], device, NUM_CLASSES, BPR_PROTO,
                    target_module_attr="features_stage",
                    ema_cache=_global_proto_local, branch_tag="dual_gl/l")
                if protos_l is not None:
                    _global_proto_local["tensor"] = protos_l
                    _global_proto_local["last_epoch"] = epoch
            except Exception as _e:
                print(f"[bpr-global:dual_gl/l] refresh failed: {_e}")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        orig_targets = targets.clone()

        if mixup_fn is not None:
            samples_use, targets_use = mixup_fn(samples, targets)
        else:
            samples_use, targets_use = samples, targets

        with torch.cuda.amp.autocast():
            outputs = model(samples_use)
            ce_loss = criterion(samples_use, outputs, targets_use)

        bpr = torch.tensor(0.0, device=device)
        _bpr_active_epoch = epoch >= (0 if BPR_ONLY else BPR_WARMUP_EPOCHS)

        if BPR_HOOK == "aux":
            feat = latent_buf.get('feat_aux')
        elif BPR_HOOK == "dual_gl":
            feat = latent_buf.get('feat_g')
        else:
            feat = latent_buf.get('feat')
        feat_dense = latent_buf.get('feat_dense') if (BPR_HOOK == "global" and BPR_DENSE) else None

        if not in_stage2 and feat is not None and _bpr_active_epoch \
                and feat.size(0) == orig_targets.size(0):

            if BPR_HOOK == "aux":
                ext_proto = _global_proto["tensor"] if BPR_PROTO_SCOPE == "global" else None
                z = _project_for_bpr_branch(feat, branch="global")
                bpr = total_bpr_loss(z, orig_targets,
                                      num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                      prototype=BPR_PROTO, external_prototypes=ext_proto,
                                      num_adv_steps=BPR_ADV_STEPS)

            elif BPR_HOOK == "dual_gl":
                ext_g = _global_proto["tensor"] if BPR_PROTO_SCOPE == "global" else None
                z_g = _project_for_bpr_branch(feat, branch="global")
                bpr_g = total_bpr_loss(z_g, orig_targets,
                                        num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                        prototype=BPR_PROTO, external_prototypes=ext_g,
                                        num_adv_steps=BPR_ADV_STEPS)
                feat_l = latent_buf.get('feat_l')
                if feat_l is not None and feat_l.dim() == 4:
                    B, C, H, W = feat_l.shape
                    if BPR_LOCAL_POOL == "parallel":
                        z_l_raw = feat_l.permute(0, 2, 3, 1).reshape(B * H * W, C)
                        labels_l = orig_targets.view(-1, 1).expand(B, H * W).reshape(-1)
                    else:
                        z_l_raw = feat_l.mean(dim=[2, 3])
                        labels_l = orig_targets
                    ext_l = _global_proto_local["tensor"] if BPR_PROTO_SCOPE == "global" else None
                    z_l = _project_for_bpr_branch(z_l_raw, branch="local")
                    bpr_l = total_bpr_loss(z_l, labels_l,
                                            num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                            prototype=BPR_PROTO, external_prototypes=ext_l,
                                            num_adv_steps=BPR_ADV_STEPS)
                    bpr = BPR_DUAL_GLOBAL_W * bpr_g + BPR_DUAL_LOCAL_W * bpr_l
                else:
                    bpr = bpr_g

            elif BPR_HOOK == "proto":
                Wc = _classifier_prototypes(model)
                if Wc is not None and Wc.size(-1) == feat.size(-1):
                    z = torch.nn.functional.normalize(feat.float(), dim=-1)
                    _Wc = Wc.float().detach() if os.environ.get("BPR_PROTO_DETACH","0")=="1" else Wc.float()
                    P = torch.nn.functional.normalize(_Wc, dim=-1)
                    bpr = total_bpr_loss(z, orig_targets,
                                          num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                          prototype=BPR_PROTO, external_prototypes=P,
                                          num_adv_steps=BPR_ADV_STEPS)
                else:
                    print(f"[bpr-hook:proto] proj_head weight dim mismatch ({None if Wc is None else Wc.size(-1)} vs {feat.size(-1)}) — global fallback")
                    _ext = _global_proto["tensor"] if BPR_PROTO_SCOPE == "global" else None
                    z = _project_for_bpr(feat)
                    bpr = total_bpr_loss(z, orig_targets,
                                          num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                          prototype=BPR_PROTO, external_prototypes=_ext,
                                          num_adv_steps=BPR_ADV_STEPS)

            else:
                ext_proto = _global_proto["tensor"] if BPR_PROTO_SCOPE == "global" else None

                def _bpr_global():
                    z = _project_for_bpr(feat)
                    return total_bpr_loss(z, orig_targets,
                                           num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                           prototype=BPR_PROTO, external_prototypes=ext_proto,
                                           num_adv_steps=BPR_ADV_STEPS)

                def _bpr_dense():
                    B, C, H, W = feat_dense.shape
                    feat_tokens = feat_dense.permute(0, 2, 3, 1).reshape(B * H * W, C)
                    labels_tokens = orig_targets.view(-1, 1).expand(B, H * W).reshape(-1)
                    z = _project_for_bpr(feat_tokens)
                    return total_bpr_loss(z, labels_tokens,
                                           num_classes=NUM_CLASSES, use_adversarial=BPR_ADV,
                                           prototype=BPR_PROTO, external_prototypes=ext_proto,
                                           num_adv_steps=BPR_ADV_STEPS)

                _dense_ok = (BPR_DENSE and feat_dense is not None and feat_dense.dim() == 4)
                if BPR_DENSE_MIX and _dense_ok:
                    bpr = BPR_GLOBAL_W * _bpr_global() + BPR_DENSE_W * _bpr_dense()
                elif _dense_ok:
                    bpr = _bpr_dense()
                else:
                    bpr = _bpr_global()
        _ce_for_log = 0.0 if BPR_ONLY else float(ce_loss.item())
        loss_value = _ce_for_log + BPR_LAMBDA * float(bpr.item() if torch.is_tensor(bpr) else 0.0)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping"); sys.exit(1)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        alpha_log = -1.0
        if in_stage2:
            optimizer.zero_grad()
            trainable = [p for p in model.parameters() if p.requires_grad]
            loss_scaler(ce_loss, optimizer, clip_grad=max_norm,
                        parameters=trainable, create_graph=is_second_order)
        elif BPR_ONLY:
            if not torch.is_tensor(bpr) or bpr.item() == 0.0:
                optimizer.zero_grad()
                continue
            loss = BPR_LAMBDA * bpr
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        elif epoch < BPR_WARMUP_EPOCHS or feat is None or not torch.is_tensor(bpr) or bpr.item() == 0.0:
            optimizer.zero_grad()
            loss_scaler(ce_loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        elif BPR_MODE == "mgda":
            params = [p for p in model.parameters() if p.requires_grad]
            info = mgda_step(ce_loss, BPR_LAMBDA * bpr, params, optimizer)
            alpha_log = info["alpha"]
        elif BPR_MODE == "pcgrad":
            params = [p for p in model.parameters() if p.requires_grad]
            info = pcgrad_step(ce_loss, BPR_LAMBDA * bpr, params, optimizer)
            alpha_log = info["cos"]
        else:
            loss = ce_loss + BPR_LAMBDA * bpr
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(bpr=float(bpr.item() if torch.is_tensor(bpr) else bpr))
        metric_logger.update(alpha=alpha_log)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if not in_stage2 and feat is not None and epoch >= BPR_WARMUP_EPOCHS:
            if BPR_USE_PROJ and _proj.get("opt") is not None:
                _proj["opt"].step(); _proj["opt"].zero_grad()
            if BPR_USE_PROJ and BPR_HOOK == "dual_gl" and _proj_local.get("opt") is not None:
                _proj_local["opt"].step(); _proj_local["opt"].zero_grad()
            if BPR_HOOK == "aux" and _aux_state.get("opt") is not None:
                _aux_state["opt"].step(); _aux_state["opt"].zero_grad()

    for _h in hook_handles:
        try: _h.remove()
        except Exception: pass
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


_eng.train_one_epoch = _bpr_train_one_epoch
_hook_extra = ""
if BPR_HOOK == "aux":
    _hook_extra = f", bn_dim={BPR_BN_DIM}"
elif BPR_HOOK == "dual_gl":
    _hook_extra = (f", local_stage={BPR_LOCAL_STAGE}, local_pool={BPR_LOCAL_POOL}, "
                   f"w_g={BPR_DUAL_GLOBAL_W}, w_l={BPR_DUAL_LOCAL_W}")
print(f"[bpr] MedViT train_one_epoch monkey-patched "
      f"(hook={BPR_HOOK}{_hook_extra}, "
      f"lambda={BPR_LAMBDA}, adv={BPR_ADV}, adv_steps={BPR_ADV_STEPS}, mode={BPR_MODE}, "
      f"proto={BPR_PROTO}, scope={BPR_PROTO_SCOPE}, use_proj={BPR_USE_PROJ}, "
      f"dense={'both' if BPR_DENSE_MIX else BPR_DENSE}"
      + (f"(global_w={BPR_GLOBAL_W},dense_w={BPR_DENSE_W})" if BPR_DENSE_MIX else "")
      + f", faithful={BPR_FAITHFUL}"
      + (f", refresh={BPR_PROTO_REFRESH}ep, ema={BPR_PROTO_EMA}" if BPR_PROTO_SCOPE == "global" else "")
      + ")")
if BPR_FAITHFUL:
    print("[bpr-faithful] HSQ-faithful mode — F.normalize OFF, PGD steps=", BPR_ADV_STEPS,
          " | recommended: BPR_LAMBDA=1.0, BPR_WARMUP_EPOCHS=0, BPR_USE_PROJ=0, BALANCED=1, BPR_ADV=1")

import main as _m
if hasattr(_m, 'get_args_parser'):
    parser = argparse.ArgumentParser('MedViT BPR', parents=[_m.get_args_parser()])
    parser.add_argument('--two-stage', dest='two_stage', action='store_true',
                        default=TWO_STAGE,
                        help='2-stage: Stage1 CE+BPR, Stage2 classifier CE only (backbone frozen)')
    parser.add_argument('--stage1-epochs', dest='stage1_epochs', type=int,
                        default=STAGE1_EPOCHS,
                        help='Stage 1 epochs (default: half of total epochs)')
    parser.add_argument('--bpr-only', dest='bpr_only', action='store_true',
                        default=BPR_ONLY,
                        help='train backbone with BPR loss only (no CE). Combine with 2-stage.')
    args = parser.parse_args()

    if getattr(args, 'two_stage', False):
        TWO_STAGE = True
        s1 = int(getattr(args, 'stage1_epochs', -1) or -1)
        if s1 <= 0:
            s1 = max(1, int(getattr(args, 'epochs', 0)) // 2)
        STAGE1_EPOCHS = s1
        total = int(getattr(args, 'epochs', 0))
        print(f"[bpr-2stage] ENABLED — stage1_epochs={STAGE1_EPOCHS}, "
              f"stage2_epochs={max(0, total - STAGE1_EPOCHS)}, total={total}")
    else:
        print(f"[bpr-2stage] disabled — joint training (CE + lambda*BPR)")
    if getattr(args, 'bpr_only', False):
        BPR_ONLY = True
        print("[bpr-only] ENABLED — CE excluded, backbone updated with lambda*BPR only")
        if not TWO_STAGE:
            print("[bpr-only] WARN: 2-stage disabled — classifier (proj_head) not trained. "
                  "val accuracy may be meaningless (consider running a separate linear probe)")
    _m.main(args)
else:
    if hasattr(_m, 'args'):
        _m.main(_m.args)
    else:
        raise SystemExit("[bpr] could not find get_args_parser or args in MedViT main")
