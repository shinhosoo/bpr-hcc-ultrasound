"""HSQ BPR runner — monkey-patches utils.train_eval.train_one_epoch_base to inject
BPR loss on the latent immediately before model.head, then runs run_hsq.py unchanged.

BPR target: token-mean of head_norm output -> (B, query_dim).
env: BPR_LAMBDA(0.3)  BPR_WARMUP(5)  BPR_PROTO(geomedian)  BPR_NUM_CLASSES(2)
"""
import os, sys, math, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HSQ_DIR = os.path.join(ROOT, "models", "HSQ")
sys.path.insert(0, HSQ_DIR)
sys.path.insert(0, os.path.join(ROOT, "tools", "bpr"))
os.chdir(HSQ_DIR)

import torch
from bpr_loss import total_bpr_loss
import utils.train_eval as TE
from tqdm import tqdm

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.3"))
BPR_WARMUP = int(os.environ.get("BPR_WARMUP", "5"))
BPR_PROTO = os.environ.get("BPR_PROTO", "geomedian")
BPR_NUM_CLASSES = int(os.environ.get("BPR_NUM_CLASSES", "2"))
HSQ_PRETRAIN = os.environ.get("HSQ_PRETRAIN", "0") == "1"
BPR_TWO_STAGE      = os.environ.get("BPR_TWO_STAGE", "0") == "1"   # 2-phase training on/off
BPR_STAGE1_EPOCHS  = int(os.environ.get("BPR_STAGE1_EPOCHS", "50"))

_pre = {"done": False}

_YEL = "\033[93m"
_RED = "\033[91m"
_RST = "\033[0m"

def _maybe_load_pretrained(model):
    """Inject ImageNet Swin-S / ConvNeXt-S weights into backbone. HSQ_PRETRAIN=1 only."""
    if _pre["done"]:
        return
    if not HSQ_PRETRAIN:
        print(f"{_RED}[hsq-pretrain] HSQ_PRETRAIN=0 — no pretrained weights loaded (random init){_RST}")
        _pre["done"] = True
        return
    _pre["done"] = True
    try:
        import timm
    except Exception as e:
        print(f"{_RED}[hsq-pretrain] timm import failed — using random init: {e}{_RST}")
        return
    import torch
    m = model.module if hasattr(model, "module") else model
    _URLS = {
        "swin": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "convnext": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    }
    def _load(dst, url, tag):
        if dst is None:
            print(f"{_RED}[hsq-pretrain] backbone not found (skip {tag}){_RST}"); return
        try:
            ck = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
        except Exception as e:
            print(f"{_RED}[hsq-pretrain] {tag} download failed: {e}{_RST}"); return
        ref = ck.get("model", ck) if isinstance(ck, dict) else ck
        cur = dst.state_dict()
        match = {k: v for k, v in ref.items() if k in cur and v.shape == cur[k].shape}
        dst.load_state_dict(match, strict=False)
        print(f"{_YEL}[hsq-pretrain] {tag} matched {len(match)}/{len(cur)} keys "
              f"(ref={len(ref)}){_RST}")
    _load(getattr(m, "swintransformer", None), _URLS["swin"], "swin_small(MS)")
    _load(getattr(m, "convnext", None), _URLS["convnext"], "convnext_small(FB)")

_lat = {}
_hook = {"done": False}

_phase = {"current": 1}

def _set_backbone_frozen(model, frozen):
    """swintransformer / convnext backbone freeze/unfreeze."""
    m = model.module if hasattr(model, "module") else model
    n_frozen = 0
    for name, param in m.named_parameters():
        if "swintransformer" in name or "convnext" in name:
            param.requires_grad = not frozen
            n_frozen += 1
    state = "frozen" if frozen else "unfrozen"
    print(f"{_YEL}[hsq-2phase] backbone {state} ({n_frozen} params){_RST}")


def _register_latent_hook(model):
    if _hook["done"]:
        return
    m = model.module if hasattr(model, "module") else model
    head = getattr(m, "head", None)
    if head is None:
        print("[hsq-bpr] WARN: model.head not found — BPR disabled")
        _hook["done"] = True
        return
    def _pre(_mod, _inp):
        f = _inp[0]
        if f.dim() == 3:
            f = f.mean(dim=1)
        _lat["z"] = f
    head.register_forward_pre_hook(_pre)
    _hook["done"] = True
    print(f"[hsq-bpr] latent hook on model.head  (lambda={BPR_LAMBDA}, warmup={BPR_WARMUP}, proto={BPR_PROTO})")


def _bpr_train_one_epoch_base(model, optimizer, metric_collection=None, data_loader=None,
                              device=0, num_updates=0, epoch=0, criterion_focal=None,
                              scheduler=None, criterion=None, mixup_fn=None, scaler=None,
                              aux_loss=None, model_ema=None, ema_updata_epoch=None):
    _maybe_load_pretrained(model)
    _register_latent_hook(model)
    if BPR_TWO_STAGE:
        if epoch < BPR_STAGE1_EPOCHS and _phase['current'] != 1:
            _phase['current'] = 1
            _set_backbone_frozen(model, False)
            print(f'{_YEL}[hsq-2phase] → Phase 1 (CE+BPR, backbone unfrozen){_RST}')
        elif epoch >= BPR_STAGE1_EPOCHS and _phase['current'] != 2:
            _phase['current'] = 2
            _set_backbone_frozen(model, True)
            print(f'{_YEL}[hsq-2phase] → Phase 2 (CE only, backbone frozen){_RST}')
    metric_collection.reset()
    model.train()
    total_loss = 0.
    data_loader = tqdm(data_loader, desc=' train')
    for step, (img, labels, img_path) in enumerate(data_loader):
        target = labels
        img = img.to(device)
        labels = labels.to(device)
        if mixup_fn is not None:
            img, labels = mixup_fn(img, labels)
        _use_amp = (scaler is not None)
        with torch.cuda.amp.autocast(enabled=_use_amp):
            output = model(img)
            if aux_loss:
                loss = criterion(output, labels) + criterion_focal(output, labels, target)
            else:
                loss = criterion(output, labels)
            _z = _lat.get("z")
            _in_phase1 = (not BPR_TWO_STAGE) or (epoch < BPR_STAGE1_EPOCHS)
            if BPR_LAMBDA > 0 and epoch >= BPR_WARMUP and _z is not None and _z.size(0) == target.size(0) and _in_phase1:
                _bpr = total_bpr_loss(_z.float(), target.to(_z.device),
                                      num_classes=BPR_NUM_CLASSES, use_adversarial=False,
                                      prototype=BPR_PROTO)
                if torch.is_tensor(_bpr) and torch.isfinite(_bpr):
                    loss = loss + BPR_LAMBDA * _bpr
        if not math.isfinite(loss.item()):
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        if _use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        with torch.no_grad():
            total_loss = total_loss + loss.item()
            data_loader.set_description('loss:{:.6f}'.format(loss.item()))
            pred = torch.softmax(output, 1)
            metric_collection.update(pred[:, 1].detach().cpu(), target.detach().cpu())
        scheduler.step()
    return total_loss


TE.train_one_epoch_base = _bpr_train_one_epoch_base
print(f"[hsq-bpr] train_one_epoch_base monkey-patched (latent BPR, lambda={BPR_LAMBDA})")

runpy.run_path(os.path.join(HERE, "run_hsq.py"), run_name="__main__")
