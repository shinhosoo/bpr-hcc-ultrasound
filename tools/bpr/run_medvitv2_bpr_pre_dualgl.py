"""MedViTV2 + BPR — v2 의 main.train_other_with_early_stop 을 monkey-patch 하여
proj_head 직전 feature 에 BPR loss 를 합산.

지원:
- BPR_HOOK=global (proj_head 입력 직접 BPR) | aux (zero-init residual bottleneck)
- BPR_TWO_STAGE=1  → Stage 1 = backbone+head+BPR, Stage 2 = backbone freeze + proj_head + CE only
- BPR_LAMBDA, BPR_ADV, BPR_PROTO, BPR_PROTO_SCOPE, BPR_WARMUP_EPOCHS 등
"""
import os, sys, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)                                      # bpr_loss
sys.path.insert(0, os.path.join(ROOT, "models", "MedViTV2-main"))

from bpr_loss import total_bpr_loss
try:
    from bpr_loss import geometric_median, sinkhorn_centroid
except ImportError:
    geometric_median = sinkhorn_centroid = None

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.3"))
BPR_ADV = os.environ.get("BPR_ADV", "0") == "1"
NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
BPR_PROTO = os.environ.get("BPR_PROTO", "mean")
BPR_PROTO_SCOPE = os.environ.get("BPR_PROTO_SCOPE", "batch")
BPR_HOOK = os.environ.get("BPR_HOOK", "global")              # global | aux
BPR_BN_DIM = int(os.environ.get("BPR_BN_DIM", "512"))
BPR_WARMUP_EPOCHS = int(os.environ.get("BPR_WARMUP_EPOCHS", "5"))

BPR_PROTO_REFRESH = int(os.environ.get("BPR_PROTO_REFRESH", "1"))
BPR_PROTO_BS = int(os.environ.get("BPR_PROTO_BS", "64"))
BPR_PROTO_EMA = float(os.environ.get("BPR_PROTO_EMA", "0.0"))

BPR_USE_PROJ = os.environ.get("BPR_USE_PROJ", "0") == "1"
BPR_PROJ_DIM = int(os.environ.get("BPR_PROJ_DIM", "128"))
BPR_PROJ_HIDDEN = int(os.environ.get("BPR_PROJ_HIDDEN", "512"))
BPR_FAITHFUL = os.environ.get("BPR_FAITHFUL", "0") == "1"

# 2-stage
TWO_STAGE = os.environ.get("BPR_TWO_STAGE", "0") == "1"
STAGE1_EPOCHS = int(os.environ.get("BPR_STAGE1_EPOCHS", "-1"))

if BPR_HOOK not in ("global", "aux"):
    raise ValueError(f"BPR_HOOK={BPR_HOOK} 미지원 (global | aux 중 택1)")

_aux_state = {"attached": False, "down": None, "up": None, "relu": None}
_proj_state = {"attached": False, "head": None}
_global_proto = {"tensor": None, "last_epoch": -1}


def _project_for_bpr(feat):
    """BPR loss 에 넘기기 전 feature 변환.

    - BPR_USE_PROJ=1 이면 lazy 생성된 _proj_state['head'] (MLP) 통과
    - BPR_FAITHFUL=1 이면 raw 그대로 (HSQ 원본 재현)
    - 그 외 기본: L2 정규화로 magnitude 통제 (v1 의 _project_for_bpr 동작 일치)
    """
    if BPR_USE_PROJ and _proj_state["head"] is not None:
        _proj_state["head"].train()
        z = _proj_state["head"](feat.float())
    else:
        z = feat.float()
    if BPR_FAITHFUL:
        return z
    return F.normalize(z, dim=-1)


def _aggregate_proto(X, kind):
    """v1 _aggregate_proto 와 동일."""
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
def _refresh_global_prototypes(model, dataset, device, num_classes, kind):
    """v1 의 _refresh_global_prototypes 를 v2 wrapper 용으로 포팅.

    전체 train_dataset 을 sequential loader 로 한 번 더 forward 해서
    클래스별 prototype 을 재계산. BPR_HOOK 에 따라:
      global: proj_head 입력 (B, in_dim) 캡쳐
      aux:    proj_head 입력 캡쳐 → _aux_state["down"] 적용 (bottleneck mid)
    BPR_USE_PROJ=1 이면 그 위에 projection head 통과.
    BPR_FAITHFUL=0 이면 L2 정규화 후 _aggregate_proto.
    BPR_PROTO_EMA>0 이면 이전 prototype 과 보간.
    """
    was_train_model = model.training
    model.eval()
    was_train_proj = False
    if _proj_state.get("head") is not None:
        was_train_proj = _proj_state["head"].training
        _proj_state["head"].eval()

    seq_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BPR_PROTO_BS, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False,
    )

    _mm = model.module if hasattr(model, 'module') else model
    feat_holder = {}
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
            if BPR_HOOK == "aux" and _aux_state["attached"]:
                feat = _aux_state["down"](feat)
            if BPR_USE_PROJ and _proj_state["head"] is not None:
                z = _proj_state["head"](feat)
            else:
                z = feat
            if not BPR_FAITHFUL:
                z = F.normalize(z, dim=-1)
            for c in range(num_classes):
                m = (y == c)
                if m.any():
                    per_class[c].append(z[m].detach())
    finally:
        h.remove()
        if was_train_model:
            model.train()
        if was_train_proj and _proj_state.get("head") is not None:
            _proj_state["head"].train()

    if any(len(v) == 0 for v in per_class.values()):
        print(f"[bpr-global] WARN: 일부 클래스 표본 0개, refresh 건너뜀")
        return None

    protos = []
    for c in range(num_classes):
        X = torch.cat(per_class[c], dim=0)
        proto = _aggregate_proto(X, kind)
        if not BPR_FAITHFUL:
            proto = F.normalize(proto, dim=-1)
        protos.append(proto)
    new_protos = torch.stack(protos, dim=0).to(device)

    if BPR_PROTO_EMA > 0 and _global_proto.get("tensor") is not None:
        old = _global_proto["tensor"].to(device)
        new_protos = (1 - BPR_PROTO_EMA) * new_protos + BPR_PROTO_EMA * old
        if not BPR_FAITHFUL:
            new_protos = F.normalize(new_protos, dim=-1)

    n_per = [sum(t.size(0) for t in per_class[c]) for c in range(num_classes)]
    print(f"[bpr-global] refreshed {num_classes} prototypes "
          f"(kind={kind}, n_per_class={n_per}, ema={BPR_PROTO_EMA})")
    return new_protos


def _attach_aux(model, in_dim, device):
    if _aux_state["attached"]:
        return
    down = nn.Linear(in_dim, BPR_BN_DIM).to(device)
    up = nn.Linear(BPR_BN_DIM, in_dim).to(device)
    nn.init.zeros_(up.weight); nn.init.zeros_(up.bias)   # zero-init residual path
    relu = nn.ReLU(inplace=False)
    _m = model.module if hasattr(model, 'module') else model
    _m.add_module('bpr_aux_down', down)
    _m.add_module('bpr_aux_up', up)
    _aux_state.update({"down": down, "up": up, "relu": relu, "attached": True})
    print(f"[bpr-medvitv2] aux bottleneck attached: {in_dim} → {BPR_BN_DIM} → {in_dim} (zero-init)")


import main as _m


def _bpr_train_loop(epochs, net, train_loader, test_loader, optimizer, scheduler,
                    loss_function, device, save_path, patience=20):
    """v2 의 train_other_with_early_stop 을 대체. BPR + (옵션) 2-stage 지원.

    BEST_BY 환경변수로 best ckpt 선택 기준 지정 (default: auc).
    main.py 의 train_other_with_early_stop 과 일관된 동작 — baseline / BPR 비교 fair.
    """
    BEST_BY = os.environ.get("BEST_BY", "auc").lower()
    if BEST_BY not in ("auc", "f1", "acc"):
        print(f"[bpr-medvitv2] warn: BEST_BY={BEST_BY} 미지원 → 'auc' fallback")
        BEST_BY = "auc"
    print(f"[bpr-medvitv2] best ckpt 기준 = {BEST_BY.upper()}  (override: env BEST_BY=acc|f1|auc)")

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
        raise RuntimeError("[bpr-medvitv2] proj_head 의 in_features 자동 감지 실패")

    if BPR_HOOK == "aux":
        _attach_aux(_mm, in_dim, device)
        aux_params = list(_aux_state["down"].parameters()) + list(_aux_state["up"].parameters())
        optimizer.add_param_group({"params": aux_params, "lr": optimizer.param_groups[0]['lr']})
        print(f"[bpr-medvitv2] aux params {sum(p.numel() for p in aux_params)} → optimizer 추가")

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
        print(f"[bpr-medvitv2] projection head attached: "
              f"{proj_in_dim} → {BPR_PROJ_HIDDEN} → {BPR_PROJ_DIM} "
              f"({sum(p.numel() for p in proj_params)} params → optimizer)")

    latent_buf = {}
    def _pre_hook(_mod, _in):
        x = _in[0]
        if BPR_HOOK == "aux":
            mid = _aux_state["down"](x)
            latent_buf['feat_aux'] = mid       # BPR target (B, BN_DIM)
            return (x + _aux_state["up"](_aux_state["relu"](mid)),)
        else:
            latent_buf['feat'] = x
            return None
    h = _mm.proj_head.register_forward_pre_hook(_pre_hook)

    s1 = STAGE1_EPOCHS if (TWO_STAGE and STAGE1_EPOCHS > 0) else max(1, epochs // 2)
    stage2_entered = False

    _norm_tag = "raw(faithful)" if BPR_FAITHFUL else "L2"
    _proj_tag = (f"proj({BPR_PROJ_HIDDEN}->{BPR_PROJ_DIM})" if BPR_USE_PROJ else "no-proj")
    _refresh_tag = (f"every {BPR_PROTO_REFRESH}ep, bs={BPR_PROTO_BS}, ema={BPR_PROTO_EMA}"
                    if BPR_PROTO_SCOPE == "global" else "n/a")
    print(f"[bpr-medvitv2] HOOK={BPR_HOOK}  λ={BPR_LAMBDA}  ADV={BPR_ADV}  "
          f"PROTO={BPR_PROTO}/{BPR_PROTO_SCOPE}  WARMUP={BPR_WARMUP_EPOCHS}  "
          f"NORM={_norm_tag}  PROJ={_proj_tag}  REFRESH={_refresh_tag}")
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
            print(f"[bpr-medvitv2] frozen={n_frozen/1e6:.2f}M | trainable={n_train/1e6:.4f}M (proj_head only)")
            stage2_entered = True

        net.train()
        bpr_active = (not in_stage2) and (epoch >= BPR_WARMUP_EPOCHS)

        _refresh_due = (
            bpr_active
            and BPR_PROTO_SCOPE == "global"
            and (epoch - BPR_WARMUP_EPOCHS) % max(BPR_PROTO_REFRESH, 1) == 0
            and _global_proto["last_epoch"] != epoch
        )
        if _refresh_due:
            try:
                ds = train_loader.dataset
                protos = _refresh_global_prototypes(net, ds, device, NUM_CLASSES, BPR_PROTO)
                if protos is not None:
                    _global_proto["tensor"] = protos
                    _global_proto["last_epoch"] = epoch
            except Exception as _e:
                print(f"[bpr-global] refresh 실패 — batch prototype 으로 fallback: {_e}")

        running_ce = running_bpr = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss_ce = loss_function(outputs, labels)
            loss_bpr = torch.zeros(1, device=device)

            if bpr_active:
                key = 'feat_aux' if BPR_HOOK == "aux" else 'feat'
                feat = latent_buf.get(key)
                if feat is not None and feat.requires_grad:
                    ext_proto = (_global_proto["tensor"]
                                 if BPR_PROTO_SCOPE == "global" else None)
                    z = _project_for_bpr(feat)
                    bt = total_bpr_loss(z, labels, num_classes=NUM_CLASSES,
                                        use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                        external_prototypes=ext_proto)
                    loss_bpr = bt
                    running_bpr += float(bt.item())

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
        print(f"[{tag}] epoch {epoch+1:3d}/{epochs}  "
              f"ce={running_ce/len(train_loader):.4f}  bpr={running_bpr/max(1,len(train_loader)):.4f}  "
              f"acc={val_acc:.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}  "
              f"best[{BEST_BY}]={best_score:.4f}")

        if val_score > best_score + 1e-6:
            best_score = val_score; wait = 0
            torch.save(net.state_dict(), save_path)
            print(f"  → best updated [{BEST_BY}]={best_score:.4f}, saved. (epoch={epoch})")
        else:
            wait += 1
            if patience and wait >= patience:
                print(f"[early-stop] no improvement for {patience} epochs — STOP at epoch {epoch+1}")
                break

    h.remove()
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
