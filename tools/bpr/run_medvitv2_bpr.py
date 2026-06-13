"""MedViTV2 + BPR — monkey-patches main.train_other_with_early_stop to inject BPR loss
on the feature immediately before proj_head.

BPR_HOOK = global | aux | dual_gl
BPR_TWO_STAGE=1: Stage 1 = backbone+head+BPR, Stage 2 = backbone frozen, proj_head CE only
"""
import os, sys, argparse, copy
import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models", "MedViTV2-main"))

from bpr_loss import total_bpr_loss
try:
    from bpr_loss import geometric_median, sinkhorn_centroid
except ImportError:
    geometric_median = sinkhorn_centroid = None
try:
    from supcon_loss import supcon_loss
except ImportError:
    supcon_loss = None

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.3"))
BPR_ADV = os.environ.get("BPR_ADV", "0") == "1"
NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
BPR_PROTO = os.environ.get("BPR_PROTO", "mean")
BPR_PROTO_SCOPE = os.environ.get("BPR_PROTO_SCOPE", "batch")
BPR_HOOK = os.environ.get("BPR_HOOK", "global")  # global | aux | dual_gl
BPR_BN_DIM = int(os.environ.get("BPR_BN_DIM", "512"))

BPR_LOCAL_STAGE = int(os.environ.get("BPR_LOCAL_STAGE", "2"))
BPR_LOCAL_POOL = os.environ.get("BPR_LOCAL_POOL", "mean")  # mean (B,C) | parallel (B*H*W, C)
BPR_DUAL_GLOBAL_W = float(os.environ.get("BPR_DUAL_GLOBAL_W", "1.0"))
BPR_DUAL_LOCAL_W  = float(os.environ.get("BPR_DUAL_LOCAL_W", "0.5"))
BPR_WARMUP_EPOCHS = int(os.environ.get("BPR_WARMUP_EPOCHS", "5"))

BPR_PROTO_REFRESH = int(os.environ.get("BPR_PROTO_REFRESH", "1"))
BPR_PROTO_BS = int(os.environ.get("BPR_PROTO_BS", "64"))
BPR_PROTO_EMA = float(os.environ.get("BPR_PROTO_EMA", "0.0"))

BPR_USE_PROJ = os.environ.get("BPR_USE_PROJ", "0") == "1"
BPR_PROJ_DIM = int(os.environ.get("BPR_PROJ_DIM", "128"))
BPR_PROJ_HIDDEN = int(os.environ.get("BPR_PROJ_HIDDEN", "512"))
BPR_FAITHFUL = os.environ.get("BPR_FAITHFUL", "0") == "1"

BPR_SUPCON_LAMBDA = float(os.environ.get("BPR_SUPCON_LAMBDA", "0.0"))
BPR_SUPCON_TEMP = float(os.environ.get("BPR_SUPCON_TEMP", "0.1"))

TWO_STAGE = os.environ.get("BPR_TWO_STAGE", "0") == "1"
STAGE1_EPOCHS = int(os.environ.get("BPR_STAGE1_EPOCHS", "-1"))

if BPR_HOOK not in ("global", "aux", "dual_gl"):
    raise ValueError(f"BPR_HOOK={BPR_HOOK} not supported (global | aux | dual_gl)")

_aux_state = {"attached": False, "down": None, "up": None, "relu": None}
_proj_state = {"attached": False, "head": None}
_proj_state_local = {"attached": False, "head": None}
_global_proto = {"tensor": None, "last_epoch": -1}
_global_proto_local = {"tensor": None, "last_epoch": -1}


def _project_for_bpr(feat, branch="global"):
    """branch: 'global' | 'local'"""
    state = _proj_state_local if branch == "local" else _proj_state
    if BPR_USE_PROJ and state["head"] is not None:
        state["head"].train()
        z = state["head"](feat.float())
    else:
        z = feat.float()
    if BPR_FAITHFUL:
        return z
    return F.normalize(z, dim=-1)


def _aggregate_proto(X, kind):
    if X.size(0) == 0:
        return None
    if X.size(0) == 1:
        return X[0]
    if kind == "geomedian" and geometric_median is not None:
        return geometric_median(X)
    if kind == "sinkhorn" and sinkhorn_centroid is not None:
        return sinkhorn_centroid(X)
    return X.mean(0)


@torch.no_grad()
def _refresh_global_prototypes(model, dataset, device, num_classes, kind,
                               branch="global", ema_cache=None):
    """Refresh global prototypes for the given branch ('global' or 'local')."""
    was_train_model = model.training
    model.eval()
    proj_state = _proj_state_local if branch == "local" else _proj_state
    was_train_proj = False
    if proj_state.get("head") is not None:
        was_train_proj = proj_state["head"].training
        proj_state["head"].eval()

    seq_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BPR_PROTO_BS, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False,
    )

    _mm = model.module if hasattr(model, 'module') else model
    feat_holder = {}

    if branch == "local":
        try:
            lidx = _mm.stage_out_idx[BPR_LOCAL_STAGE]
        except (AttributeError, IndexError) as _e:
            print(f"[bpr-global:local] stage_out_idx access failed ({_e}) — skipping refresh")
            if was_train_model: model.train()
            if was_train_proj and proj_state.get("head") is not None:
                proj_state["head"].train()
            return None
        def _hook(_mod, _in, _out):
            feat_holder['feat'] = _out
        h = _mm.features[lidx].register_forward_hook(_hook)
    else:
        def _hook(_mod, _in):
            feat_holder['feat'] = _in[0]
        h = _mm.proj_head.register_forward_pre_hook(_hook)

    per_class = {c: [] for c in range(num_classes)}
    try:
        for batch in seq_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]
            if torch.is_tensor(y):
                y = y.view(-1).cpu().long()
            else:
                y = torch.tensor(y, dtype=torch.long)
            _ = model(x)
            feat = feat_holder.get('feat')
            if feat is None:
                continue
            feat = feat.float()

            if branch == "local":
                if feat.dim() != 4:
                    print(f"[bpr-global:local] WARN: unexpected feat.dim()={feat.dim()}")
                    continue
                B, C, H, W = feat.shape
                if BPR_LOCAL_POOL == "parallel":
                    feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                    y_use = y.view(-1, 1).expand(B, H * W).reshape(-1)
                else:  # mean
                    feat = feat.mean(dim=[2, 3])
                    y_use = y
            else:
                if BPR_HOOK == "aux" and _aux_state["attached"]:
                    feat = _aux_state["down"](feat)
                y_use = y

            if BPR_USE_PROJ and proj_state["head"] is not None:
                z = proj_state["head"](feat)
            else:
                z = feat
            if not BPR_FAITHFUL:
                z = F.normalize(z, dim=-1)
            for c in range(num_classes):
                m = (y_use == c)
                if m.any():
                    per_class[c].append(z[m].detach())
    finally:
        h.remove()
        if was_train_model:
            model.train()
        if was_train_proj and proj_state.get("head") is not None:
            proj_state["head"].train()

    if any(len(v) == 0 for v in per_class.values()):
        print(f"[bpr-global:{branch}] WARN: some classes have 0 samples, skipping refresh")
        return None

    protos = []
    for c in range(num_classes):
        X = torch.cat(per_class[c], dim=0)
        proto = _aggregate_proto(X, kind)
        if not BPR_FAITHFUL:
            proto = F.normalize(proto, dim=-1)
        protos.append(proto)
    new_protos = torch.stack(protos, dim=0).to(device)

    cache = ema_cache if ema_cache is not None else (_global_proto_local if branch == "local" else _global_proto)
    if BPR_PROTO_EMA > 0 and cache.get("tensor") is not None:
        old = cache["tensor"].to(device)
        new_protos = (1 - BPR_PROTO_EMA) * new_protos + BPR_PROTO_EMA * old
        if not BPR_FAITHFUL:
            new_protos = F.normalize(new_protos, dim=-1)

    n_per = [sum(t.size(0) for t in per_class[c]) for c in range(num_classes)]
    print(f"[bpr-global:{branch}] refreshed {num_classes} prototypes "
          f"(kind={kind}, n_per_class={n_per}, ema={BPR_PROTO_EMA})")
    return new_protos


def _attach_aux(model, in_dim, device):
    if _aux_state["attached"]:
        return
    down = nn.Linear(in_dim, BPR_BN_DIM).to(device)
    up = nn.Linear(BPR_BN_DIM, in_dim).to(device)
    nn.init.zeros_(up.weight); nn.init.zeros_(up.bias)
    relu = nn.ReLU(inplace=False)
    _m = model.module if hasattr(model, 'module') else model
    _m.add_module('bpr_aux_down', down)
    _m.add_module('bpr_aux_up', up)
    _aux_state.update({"down": down, "up": up, "relu": relu, "attached": True})
    print(f"[bpr-medvitv2] aux bottleneck attached: {in_dim} → {BPR_BN_DIM} → {in_dim} (zero-init)")


import main as _m


def _bpr_train_loop(epochs, net, train_loader, test_loader, optimizer, scheduler,
                    loss_function, device, save_path, patience=20):
    BEST_BY = "auc"
    print(f"[bpr-medvitv2] best ckpt by {BEST_BY.upper()}")

    best_score = 0.0
    wait = 0
    _mm = net.module if hasattr(net, 'module') else net

    in_dim = None
    if isinstance(_mm.proj_head, nn.Sequential):
        for layer in _mm.proj_head:
            if isinstance(layer, nn.Linear):
                in_dim = layer.in_features; break
    elif isinstance(_mm.proj_head, nn.Linear):
        in_dim = _mm.proj_head.in_features
    if in_dim is None:
        raise RuntimeError("[bpr-medvitv2] failed to infer proj_head in_features")

    if BPR_HOOK == "aux":
        _attach_aux(_mm, in_dim, device)
        aux_params = list(_aux_state["down"].parameters()) + list(_aux_state["up"].parameters())
        optimizer.add_param_group({"params": aux_params, "lr": optimizer.param_groups[0]['lr']})
        print(f"[bpr-medvitv2] aux params {sum(p.numel() for p in aux_params)} added to optimizer")

    if BPR_USE_PROJ and _proj_state["head"] is None:
        proj_in_dim = BPR_BN_DIM if BPR_HOOK == "aux" else in_dim
        head = nn.Sequential(
            nn.Linear(proj_in_dim, BPR_PROJ_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
        ).to(device)
        _proj_state["head"] = head
        _proj_state["attached"] = True
        proj_params = list(head.parameters())
        optimizer.add_param_group({"params": proj_params, "lr": optimizer.param_groups[0]['lr']})
        print(f"[bpr-medvitv2] projection head (global) attached: "
              f"{proj_in_dim} → {BPR_PROJ_HIDDEN} → {BPR_PROJ_DIM} "
              f"({sum(p.numel() for p in proj_params)} params → optimizer)")

    local_in_dim = None
    if BPR_HOOK == "dual_gl":
        try:
            local_in_dim = _mm.stage_out_channels[BPR_LOCAL_STAGE][-1]
        except (AttributeError, IndexError) as _e:
            raise RuntimeError(
                f"[bpr-medvitv2:dual_gl] stage_out_channels[{BPR_LOCAL_STAGE}] access failed: {_e}"
            )
        print(f"[bpr-medvitv2:dual_gl] local stage={BPR_LOCAL_STAGE}, "
              f"features[{_mm.stage_out_idx[BPR_LOCAL_STAGE]}], in_dim={local_in_dim}")
        if BPR_USE_PROJ and _proj_state_local["head"] is None:
            head_l = nn.Sequential(
                nn.Linear(local_in_dim, BPR_PROJ_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
            ).to(device)
            _proj_state_local["head"] = head_l
            _proj_state_local["attached"] = True
            local_params = list(head_l.parameters())
            optimizer.add_param_group({"params": local_params, "lr": optimizer.param_groups[0]['lr']})
            print(f"[bpr-medvitv2] projection head (local) attached: "
                  f"{local_in_dim} → {BPR_PROJ_HIDDEN} → {BPR_PROJ_DIM} "
                  f"({sum(p.numel() for p in local_params)} params → optimizer)")

    latent_buf = {}
    hook_handles = []

    if BPR_HOOK == "dual_gl":
        def _g_pre_hook(_mod, _in):
            latent_buf['feat_g'] = _in[0]
        hook_handles.append(_mm.proj_head.register_forward_pre_hook(_g_pre_hook))
        try:
            lidx = _mm.stage_out_idx[BPR_LOCAL_STAGE]
            def _l_fwd_hook(_mod, _in, _out):
                latent_buf['feat_l'] = _out
            hook_handles.append(_mm.features[lidx].register_forward_hook(_l_fwd_hook))
        except (AttributeError, IndexError) as _e:
            print(f"[bpr-medvitv2:dual_gl] WARN: local stage hook failed ({_e}) — global only")
    else:
        def _pre_hook(_mod, _in):
            x = _in[0]
            if BPR_HOOK == "aux":
                mid = _aux_state["down"](x)
                latent_buf['feat_aux'] = mid
                return (x + _aux_state["up"](_aux_state["relu"](mid)),)
            else:
                latent_buf['feat'] = x
                return None
        hook_handles.append(_mm.proj_head.register_forward_pre_hook(_pre_hook))

    s1 = STAGE1_EPOCHS if (TWO_STAGE and STAGE1_EPOCHS > 0) else max(1, epochs // 2)
    stage2_entered = False

    _norm_tag = "raw(faithful)" if BPR_FAITHFUL else "L2"
    _proj_tag = (f"proj({BPR_PROJ_HIDDEN}->{BPR_PROJ_DIM})" if BPR_USE_PROJ else "no-proj")
    _refresh_tag = (f"every {BPR_PROTO_REFRESH}ep, bs={BPR_PROTO_BS}, ema={BPR_PROTO_EMA}"
                    if BPR_PROTO_SCOPE == "global" else "n/a")
    _dual_tag = (f"local_stage={BPR_LOCAL_STAGE}, pool={BPR_LOCAL_POOL}, "
                 f"w_g={BPR_DUAL_GLOBAL_W}, w_l={BPR_DUAL_LOCAL_W}"
                 if BPR_HOOK == "dual_gl" else "")
    _supcon_print = (f"SUPCON[λ={BPR_SUPCON_LAMBDA}, τ={BPR_SUPCON_TEMP}]"
                     if BPR_SUPCON_LAMBDA > 0 else "SUPCON=off")
    print(f"[bpr-medvitv2] HOOK={BPR_HOOK}  λ={BPR_LAMBDA}  ADV={BPR_ADV}  "
          f"PROTO={BPR_PROTO}/{BPR_PROTO_SCOPE}  WARMUP={BPR_WARMUP_EPOCHS}  "
          f"NORM={_norm_tag}  PROJ={_proj_tag}  REFRESH={_refresh_tag}  "
          f"{_supcon_print}"
          + (f"  DUAL[{_dual_tag}]" if _dual_tag else ""))
    if TWO_STAGE:
        print(f"[bpr-medvitv2] 2-stage: Stage1=0..{s1-1} (CE+BPR), Stage2={s1}..end (proj_head only, CE)")

    for epoch in range(epochs):
        in_stage2 = TWO_STAGE and epoch >= s1
        if in_stage2 and not stage2_entered:
            n_frozen = n_train = 0
            for name, p in _mm.named_parameters():
                if name.startswith('proj_head'):
                    p.requires_grad = True
                    n_train += p.numel()
                else:
                    p.requires_grad = False
                    n_frozen += p.numel()
            print(f"[bpr-medvitv2] === Stage 2 begin at epoch {epoch} ===")
            print(f"[bpr-medvitv2] frozen={n_frozen/1e6:.2f}M | trainable={n_train/1e6:.4f}M")
            stage2_entered = True

        net.train()
        bpr_active = (not in_stage2) and (epoch >= BPR_WARMUP_EPOCHS)

        _refresh_due = (
            bpr_active
            and BPR_PROTO_SCOPE == "global"
            and (epoch - BPR_WARMUP_EPOCHS) % max(BPR_PROTO_REFRESH, 1) == 0
        )
        if _refresh_due:
            if _global_proto["last_epoch"] != epoch:
                try:
                    ds = train_loader.dataset
                    protos = _refresh_global_prototypes(
                        net, ds, device, NUM_CLASSES, BPR_PROTO, branch="global")
                    if protos is not None:
                        _global_proto["tensor"] = protos
                        _global_proto["last_epoch"] = epoch
                except Exception as _e:
                    print(f"[bpr-global] refresh failed — falling back to batch prototype: {_e}")
            if BPR_HOOK == "dual_gl" and _global_proto_local["last_epoch"] != epoch:
                try:
                    ds = train_loader.dataset
                    protos_l = _refresh_global_prototypes(
                        net, ds, device, NUM_CLASSES, BPR_PROTO, branch="local")
                    if protos_l is not None:
                        _global_proto_local["tensor"] = protos_l
                        _global_proto_local["last_epoch"] = epoch
                except Exception as _e:
                    print(f"[bpr-global:local] refresh failed: {_e}")

        running_ce = running_bpr = running_supcon = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss_ce = loss_function(outputs, labels)
            loss_bpr = torch.zeros(1, device=device)

            if bpr_active:
                bt_supcon = torch.zeros(1, device=device)

                if BPR_HOOK == "dual_gl":
                    feat_g = latent_buf.get('feat_g')
                    bt_g = torch.zeros(1, device=device)
                    if feat_g is not None and feat_g.requires_grad:
                        ext_g = (_global_proto["tensor"]
                                 if BPR_PROTO_SCOPE == "global" else None)
                        z_g = _project_for_bpr(feat_g, branch="global")
                        bt_g = total_bpr_loss(z_g, labels, num_classes=NUM_CLASSES,
                                              use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                              external_prototypes=ext_g)
                        if BPR_SUPCON_LAMBDA > 0 and supcon_loss is not None:
                            bt_supcon = bt_supcon + supcon_loss(z_g, labels,
                                                                temperature=BPR_SUPCON_TEMP)

                    feat_l = latent_buf.get('feat_l')
                    bt_l = torch.zeros(1, device=device)
                    if feat_l is not None and feat_l.dim() == 4 and feat_l.requires_grad:
                        B, C, H, W = feat_l.shape
                        if BPR_LOCAL_POOL == "parallel":
                            z_l_raw = feat_l.permute(0, 2, 3, 1).reshape(B * H * W, C)
                            labels_l = labels.view(-1, 1).expand(B, H * W).reshape(-1)
                        else:
                            z_l_raw = feat_l.mean(dim=[2, 3])
                            labels_l = labels
                        ext_l = (_global_proto_local["tensor"]
                                 if BPR_PROTO_SCOPE == "global" else None)
                        z_l = _project_for_bpr(z_l_raw, branch="local")
                        bt_l = total_bpr_loss(z_l, labels_l, num_classes=NUM_CLASSES,
                                              use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                              external_prototypes=ext_l)
                        if BPR_SUPCON_LAMBDA > 0 and supcon_loss is not None:
                            bt_supcon = bt_supcon + supcon_loss(z_l, labels_l,
                                                                temperature=BPR_SUPCON_TEMP)

                    bt = BPR_DUAL_GLOBAL_W * bt_g + BPR_DUAL_LOCAL_W * bt_l
                    loss_bpr = bt + BPR_SUPCON_LAMBDA * bt_supcon
                    running_bpr += float(bt.item() if torch.is_tensor(bt) else 0.0)
                    running_supcon += float(bt_supcon.item() if torch.is_tensor(bt_supcon) else 0.0)
                else:
                    key = 'feat_aux' if BPR_HOOK == "aux" else 'feat'
                    feat = latent_buf.get(key)
                    if feat is not None and feat.requires_grad:
                        ext_proto = (_global_proto["tensor"]
                                     if BPR_PROTO_SCOPE == "global" else None)
                        z = _project_for_bpr(feat, branch="global")
                        bt = total_bpr_loss(z, labels, num_classes=NUM_CLASSES,
                                            use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                            external_prototypes=ext_proto)
                        if BPR_SUPCON_LAMBDA > 0 and supcon_loss is not None:
                            bt_supcon = supcon_loss(z, labels, temperature=BPR_SUPCON_TEMP)
                        loss_bpr = bt + BPR_SUPCON_LAMBDA * bt_supcon
                        running_bpr += float(bt.item())
                        running_supcon += float(bt_supcon.item() if torch.is_tensor(bt_supcon) else 0.0)

            loss = loss_ce + (BPR_LAMBDA * loss_bpr if bpr_active else torch.zeros(1, device=device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_ce += float(loss_ce.item())

        net.eval()
        ys_list, scores_list = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = net(x)
                prob = torch.softmax(out, dim=1)
                ys_list.append(y.cpu().numpy().astype(int).ravel())
                if prob.shape[1] == 2:
                    scores_list.append(prob[:, 1].cpu().numpy())
                else:
                    scores_list.append(prob.cpu().numpy())
        ys = np.concatenate(ys_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        if scores.ndim == 1:
            preds = (scores >= 0.5).astype(int)
        else:
            preds = scores.argmax(axis=1)
        val_acc = float((preds == ys).mean())
        try:
            if scores.ndim == 1:
                val_auc = float(roc_auc_score(ys, scores)) if len(set(ys.tolist())) > 1 else 0.0
            else:
                val_auc = float(roc_auc_score(ys, scores, multi_class='ovr'))
        except Exception:
            val_auc = 0.0
        val_f1 = float(f1_score(ys, preds, average='macro'))
        val_score = {"acc": val_acc, "auc": val_auc, "f1": val_f1}[BEST_BY]

        tag = "S2" if in_stage2 else ("S1" if TWO_STAGE else "joint")
        _supcon_str = (f"  supcon={running_supcon/max(1,len(train_loader)):.4f}"
                       if BPR_SUPCON_LAMBDA > 0 else "")
        print(f"[{tag}] epoch {epoch+1:3d}/{epochs}  "
              f"ce={running_ce/len(train_loader):.4f}  "
              f"bpr={running_bpr/max(1,len(train_loader)):.4f}{_supcon_str}  "
              f"acc={val_acc:.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}  "
              f"best[{BEST_BY}]={best_score:.4f}")

        if val_score > best_score + 1e-6:
            best_score = val_score; wait = 0
            torch.save(net.state_dict(), save_path)
            print(f"  → best updated [{BEST_BY}]={best_score:.4f}, saved. (epoch={epoch})")
        else:
            wait += 1
            if patience and wait >= patience:
                print(f"[early-stop] no improvement for {patience} epochs — stopping at epoch {epoch+1}")
                break

    for _h in hook_handles:
        try: _h.remove()
        except Exception: pass
    print(f"[bpr-medvitv2] training done. best_{BEST_BY}={best_score:.4f}")


_m.train_other_with_early_stop = _bpr_train_loop
print(f"[bpr-medvitv2] monkey-patched main.train_other_with_early_stop with BPR")

if hasattr(_m, 'get_args_parser') or True:
    parser = argparse.ArgumentParser(description='MedViTV2 + BPR')
    parser.add_argument('--model_name', type=str, default='MedViT_small')
    parser.add_argument('--dataset', type=str, default='lesion_binary')
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    from distutils.util import strtobool
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--save-predictions', type=str, default='')
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--weighted-sampler', action='store_true')
    parser.add_argument('--balanced-sampler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    _m.main(args)
