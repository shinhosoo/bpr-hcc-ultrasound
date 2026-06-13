"""DiffMICv2 Stage-1: DCG(aux_model) standalone pretrain with CE + BPR.

DiffMIC-v1 식 2-stage 를 v2 구조 그대로 재현하는 Stage-1.
  Stage 1 (이 스크립트): DCG(보조 분류기)를 CE + lambda*BPR 로 단독 표현학습.
  Stage 2 (diffuser_trainer.py): 이 DCG 를 aux_ckpt_path 로 로드->freeze, diffusion 만 학습.

체크포인트 포맷은 CoolSystem.init_weight 가 읽는 형식과 일치:
  저장: torch.save([dcg.state_dict()], out)
  로드: torch.load(path)[0]  -> state_dict (init_weight 가 aux_model 키만 필터)

환경변수:
  BPR_LAMBDA(0.1) BPR_ADV(0) BPR_NUM_CLASSES(2)
  BPR_PROTO(mean|geomedian|sinkhorn) BPR_PROTO_SCOPE(batch|global)
  BPR_BUFFER_SIZE(512) BPR_PROJ_DIM(128) BPR_PROJ_HIDDEN(512)
  BPR_WARMUP_EPOCHS(0) SEED(42)
CLI: --config --out --epochs --early-stop-patience --cpu
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)                                   # bpr_loss
sys.path.insert(0, os.path.join(ROOT, "models/DiffMICv2-main"))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader

from pretraining.dcg import DCG as AuxCls
from utils import get_dataset
from bpr_loss import total_bpr_loss, geometric_median, sinkhorn_centroid

BPR_LAMBDA      = float(os.environ.get("BPR_LAMBDA",       "0.1"))
BPR_ADV         = os.environ.get("BPR_ADV",          "0") == "1"
NUM_CLASSES     = int(os.environ.get("BPR_NUM_CLASSES",   "2"))
BPR_PROTO       = os.environ.get("BPR_PROTO",        "mean")
BPR_PROTO_SCOPE = os.environ.get("BPR_PROTO_SCOPE",  "batch")
BPR_BUFFER_SIZE = int(os.environ.get("BPR_BUFFER_SIZE",  "512"))
BPR_PROJ_DIM    = int(os.environ.get("BPR_PROJ_DIM",     "128"))
BPR_PROJ_HIDDEN = int(os.environ.get("BPR_PROJ_HIDDEN",  "512"))
BPR_WARMUP      = int(os.environ.get("BPR_WARMUP_EPOCHS", "0"))

_buffer = {"cls": {c: [] for c in range(NUM_CLASSES)}, "cap": BPR_BUFFER_SIZE}

def _update_buffer(zp, labels):
    if BPR_PROTO_SCOPE != "global":
        return
    for c in range(NUM_CLASSES):
        m = (labels == c)
        if m.any():
            _buffer["cls"][c].append(zp[m].detach().cpu())
            n = sum(t.size(0) for t in _buffer["cls"][c])
            while n > _buffer["cap"] and len(_buffer["cls"][c]) > 1:
                n -= _buffer["cls"][c].pop(0).size(0)

def _buffer_prototypes(device):
    if BPR_PROTO_SCOPE != "global":
        return None
    protos = []
    for c in range(NUM_CLASSES):
        if not _buffer["cls"][c]:
            return None
        X = torch.cat(_buffer["cls"][c], dim=0).to(device)
        if X.size(0) < 8:
            return None
        if BPR_PROTO == "geomedian":
            p = geometric_median(X)
        elif BPR_PROTO == "sinkhorn":
            p = sinkhorn_centroid(X)
        else:
            p = X.mean(0)
        protos.append(F.normalize(p, dim=-1))
    return torch.stack(protos, dim=0)

_proj = {"head": None, "opt": None}

def _ensure_projection(feat_dim, device):
    if _proj["head"] is None:
        _proj["head"] = nn.Sequential(
            nn.Linear(feat_dim, BPR_PROJ_HIDDEN), nn.ReLU(inplace=False),
            nn.Linear(BPR_PROJ_HIDDEN, BPR_PROJ_DIM),
        ).to(device)
        _proj["opt"] = torch.optim.Adam(_proj["head"].parameters(), lr=1e-3)
    return _proj["head"], _proj["opt"]


_ATTN_CAP = {"z": None}

def _patch_attention_module():
    import pretraining.modules as _M
    if getattr(_M.AttentionModule.forward, "_bpr_patched", False):
        return
    _orig = _M.AttentionModule.forward
    def _am_patched(self, h_crops):
        z, attn, y = _orig(self, h_crops)
        _ATTN_CAP["z"] = z
        return z, attn, y
    _am_patched._bpr_patched = True
    _M.AttentionModule.forward = _am_patched


def _val_auc(dcg, loader, device):
    dcg.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_fusion, y_global, y_local, patches, attns, attn_map = dcg(x)
            prob = F.softmax(0.5 * (y_global + y_local), dim=1)
            ys.append(y.detach().cpu().numpy().astype(int))
            ps.append(prob.detach().cpu().numpy())
    yt = np.concatenate(ys); pp = np.concatenate(ps)
    try:
        from sklearn.metrics import roc_auc_score as _roc
        if pp.shape[1] == 2:
            return float(_roc(yt, pp[:, 1])), yt, pp
        return float(_roc(yt, pp, average='macro', multi_class='ovr')), yt, pp
    except Exception as e:
        print(f"[dcg-pretrain][auc] failed -> 0.0 ({e})")
        return 0.0, yt, pp


def main():
    ap = argparse.ArgumentParser(description="DiffMICv2 Stage-1 DCG BPR pretrain")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="저장 경로 (torch.save([state_dict]))")
    ap.add_argument("--epochs", type=int, default=-1, help="-1 이면 config.training.n_epochs")
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    seed = int(os.environ.get("SEED", getattr(config.data, "seed", 42)))
    torch.manual_seed(seed); np.random.seed(seed)
    import random as _r; _r.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    epochs = args.epochs if args.epochs > 0 else int(config.training.n_epochs)
    patience = int(args.early_stop_patience)

    _, train_ds, test_ds = get_dataset(config)
    _use_balanced = os.environ.get('DIFFMICV2_BALANCED_SAMPLER', '0') == '1'
    _use_weighted = os.environ.get('DIFFMICV2_WEIGHTED_SAMPLER', '0') == '1'
    if _use_balanced:
        import sys as _sys
        _tools = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root/tools
        if _tools not in _sys.path: _sys.path.insert(0, _tools)
        from balanced_sampler import BalancedBatchSampler
        _labels = [int(s['label']) for s in train_ds.data_list]
        _bsampler = BalancedBatchSampler(_labels, batch_size=config.training.batch_size,
                                         num_classes=config.data.num_classes,
                                         seed=int(os.environ.get("SEED", getattr(config.data, 'seed', 42))))
        print(f"[dcg-pretrain] balanced-sampler activated, per_class={_bsampler.per_class}")
        train_loader = DataLoader(train_ds, batch_sampler=_bsampler,
                                  num_workers=config.data.num_workers)
    elif _use_weighted:
        from torch.utils.data import WeightedRandomSampler as _WRS
        _labels = [int(s['label']) for s in train_ds.data_list]
        _cnt = np.bincount(_labels, minlength=config.data.num_classes)
        _w = 1.0 / np.maximum(_cnt, 1)
        _sw = _w[_labels]
        _sampler = _WRS(_sw.tolist(), num_samples=len(_labels), replacement=True)
        print(f"[dcg-pretrain] weighted-sampler class_count={_cnt.tolist()} class_weight={_w.tolist()}")
        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size,
                                  sampler=_sampler, num_workers=config.data.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size,
                                  shuffle=True, num_workers=config.data.num_workers)
    val_loader = DataLoader(test_ds, batch_size=config.testing.batch_size,
                            shuffle=False, num_workers=config.data.num_workers)

    _tp_loader = None
    _tp_pkl = os.environ.get("TEST_PREVIEW_PKL", "")
    _tp_every = max(1, int(os.environ.get("TEST_PREVIEW_EVERY", "1")))
    if _tp_pkl:
        try:
            import copy as _copy
            _cfg2 = _copy.deepcopy(config); _cfg2.data.testdata = _tp_pkl
            _, _, _tp_ds = get_dataset(_cfg2)
            _tp_loader = DataLoader(_tp_ds, batch_size=config.testing.batch_size,
                                    shuffle=False, num_workers=config.data.num_workers)
            print(f"[test-preview] loader from {_tp_pkl} (N={len(_tp_ds)})")
        except Exception as _e:
            print(f"[test-preview] loader skipped: {_e}"); _tp_loader = None

    dcg = AuxCls(config).to(device)

    _patch_attention_module()

    opt = torch.optim.Adam(dcg.parameters(), lr=float(config.aux_optim.lr))
    ce = nn.CrossEntropyLoss()

    print(f"[dcg-pretrain] device={device} epochs={epochs} patience={patience} "
          f"lambda={BPR_LAMBDA} adv={BPR_ADV} proto={BPR_PROTO}/{BPR_PROTO_SCOPE} warmup={BPR_WARMUP}")

    best_auc, wait = -1.0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    for epoch in range(epochs):
        dcg.train()
        run_ce = run_bpr = 0.0; nb = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).long()
            y_fusion, y_global, y_local, patches, attns, attn_map = dcg(x)
            loss_ce = ce(y_global, y) + ce(y_local, y)

            bpr_term = torch.zeros((), device=device)
            z = _ATTN_CAP.get("z")
            if epoch >= BPR_WARMUP and z is not None:
                head, hopt = _ensure_projection(z.size(-1), device)
                head.train()
                zp = F.normalize(head(z.float()), dim=-1)
                _update_buffer(zp, y)
                ext = _buffer_prototypes(zp.device) if BPR_PROTO_SCOPE == "global" else None
                try:
                    bpr_term = total_bpr_loss(zp, y, num_classes=NUM_CLASSES,
                                              use_adversarial=BPR_ADV, prototype=BPR_PROTO,
                                              external_prototypes=ext)
                except Exception as _e:
                    print(f"[dcg-pretrain][bpr] skipped: {_e}")
                    bpr_term = torch.zeros((), device=device)

            total = loss_ce + BPR_LAMBDA * bpr_term
            opt.zero_grad()
            if _proj["opt"] is not None: _proj["opt"].zero_grad()
            total.backward()
            opt.step()
            if _proj["opt"] is not None: _proj["opt"].step()

            run_ce += float(loss_ce.item())
            run_bpr += float(bpr_term.item()) if torch.is_tensor(bpr_term) else 0.0
            nb += 1

        auc, _, _ = _val_auc(dcg, val_loader, device)
        print(f"[dcg-pretrain] epoch {epoch+1}/{epochs}  ce={run_ce/max(1,nb):.4f}  "
              f"bpr={run_bpr/max(1,nb):.4f}  val_AUC={auc:.4f}  best={max(best_auc,auc):.4f}  wait={wait}/{patience}")

        if _tp_loader is not None and (epoch % _tp_every == 0):
            try:
                _tauc, _, _ = _val_auc(dcg, _tp_loader, device)
                print(f"\033[93m[test-preview] epoch {epoch+1}  test_AUC={_tauc:.4f}  [PREVIEW ONLY - not used for selection]\033[0m")
            except Exception as _e:
                print(f"[test-preview] skipped: {_e}")

        if auc > best_auc + 1e-6:
            best_auc = auc; wait = 0
            torch.save([dcg.state_dict()], args.out)
            print(f"  -> best DCG saved (AUC={best_auc:.4f}) : {args.out}")
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"[dcg-pretrain] early-stop at epoch {epoch+1} (best AUC={best_auc:.4f})")
                break

    if best_auc < 0:
        torch.save([dcg.state_dict()], args.out)
        print(f"[dcg-pretrain] no AUC improvement recorded; saved last weights to {args.out}")
    print(f"[dcg-pretrain] DONE  best_val_AUC={best_auc:.4f}  ckpt={args.out}")


if __name__ == "__main__":
    main()
