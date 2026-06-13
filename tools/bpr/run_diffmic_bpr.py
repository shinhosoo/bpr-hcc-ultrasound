"""DiffMIC BPR runner — joint 또는 2-stage 학습 지원.

원본 코드 무수정. monkey-patch 두 가지:
  1) AttentionModule.forward — z_weighted_avg 를 글로벌 buffer 에 보관
  2) nonlinear_guidance_model_train_step — CE + BPR loss 로 교체
     ★ 모듈 load 시점에 패치 → pre-train loop 에서도 BPR 적용됨 (구버전과 다른 점)

환경변수:
  BPR_LAMBDA      BPR 가중치 (default: 0.1)
  BPR_ADV         adversarial perturbation 사용 여부 (0/1, default: 0)
  BPR_PROTO       prototype 종류: mean | geomedian | sinkhorn (default: mean)
  BPR_PROTO_SCOPE feature buffer 범위: batch | global (default: batch)
  BPR_BUFFER_SIZE global proto 용 rolling buffer 크기 (default: 512)
  BPR_MODE        gradient 처리: joint | mgda | pcgrad (default: joint)
  BPR_PROJ_DIM    projection head 출력 차원 (default: 128)
  BPR_PROJ_HIDDEN projection head 중간 차원 (default: 512)

  BPR_TWO_STAGE        1이면 MedViT와 동일한 단일 run 2-stage:
                         Stage1 = DCG를 BPR+CE로 BPR_STAGE1_EPOCHS epoch 학습
                         Stage2 = DCG freeze, diffusion만 학습
                         → train.sh 명령 한 줄로 실행 가능
  BPR_STAGE1_EPOCHS    BPR_TWO_STAGE=1 시 Stage1 epoch 수 (default: 50)

  BPR_STAGE       0=joint(기본) | 1=DCG만 BPR+CE pre-train 후 종료 (2-run 방식)
                                | 2=pre-train된 DCG 로드 후 diffusion만 학습 (2-run 방식)
  BPR_PRETRAIN_EPOCHS  BPR_STAGE=1 에서 pre-train epoch 수 (0이면 config 기본값)
  BPR_STAGE1_LOG  BPR_STAGE=2 에서 불러올 Stage 1 log 디렉터리

단일 run 2-stage 사용 예 (MedViT 방식과 동일):
  PATIENCE=40 BPR_TWO_STAGE=1 BPR_STAGE1_EPOCHS=50 \\
  BPR_LAMBDA=1.0 BPR_ADV=1 \\
  BPR_PROTO_SCOPE=global BPR_PROTO=geomedian \\
  TECH=bpr bash train.sh b diffmic bpr_diffmic_2stage
"""
import os, sys
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
sys.path.insert(0, os.path.join(ROOT, "models/DiffMIC-main"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from bpr_loss import bpr_prototype_loss, total_bpr_loss, geometric_median, sinkhorn_centroid
from mgda import mgda_step
from pcgrad import pcgrad_step

BPR_LAMBDA       = float(os.environ.get("BPR_LAMBDA",       "0.1"))
BPR_ADV          = os.environ.get("BPR_ADV",          "0") == "1"
NUM_CLASSES      = int(os.environ.get("BPR_NUM_CLASSES", "2"))
BPR_PROTO        = os.environ.get("BPR_PROTO",        "mean")   # mean|geomedian|sinkhorn
BPR_MODE         = os.environ.get("BPR_MODE",         "joint")  # joint|mgda|pcgrad
BPR_PROTO_SCOPE  = os.environ.get("BPR_PROTO_SCOPE",  "batch")  # batch|global
BPR_BUFFER_SIZE  = int(os.environ.get("BPR_BUFFER_SIZE", "512"))
BPR_PROJ_DIM     = int(os.environ.get("BPR_PROJ_DIM",    "128"))
BPR_PROJ_HIDDEN  = int(os.environ.get("BPR_PROJ_HIDDEN", "512"))

BPR_TWO_STAGE       = os.environ.get("BPR_TWO_STAGE",    "0") == "1"
BPR_STAGE1_EPOCHS   = int(os.environ.get("BPR_STAGE1_EPOCHS",   "50"))

BPR_STAGE           = int(os.environ.get("BPR_STAGE",           "0"))
BPR_PRETRAIN_EPOCHS = int(os.environ.get("BPR_PRETRAIN_EPOCHS", "0"))
BPR_STAGE1_LOG      = os.environ.get("BPR_STAGE1_LOG", "").strip()

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

import pretraining.modules as M
_orig_am_fwd = M.AttentionModule.forward
_state = {"z": None}

def _am_patched(self, h_crops):
    z, attn, y = _orig_am_fwd(self, h_crops)
    _state["z"] = z
    return z, attn, y

M.AttentionModule.forward = _am_patched

import diffusion_trainer as DT

_orig_aux_step = DT.Diffusion.nonlinear_guidance_model_train_step

def _aux_step_with_bpr(self, x_batch, y_one_hot_batch, aux_optimizer):
    y_pred, y_global, y_local = self.compute_guiding_prediction(x_batch)
    ce = self.aux_cost_function(y_pred, y_one_hot_batch)

    z = _state.get("z")
    bpr_term = torch.zeros((), device=ce.device)
    if z is not None:
        labels = y_one_hot_batch.argmax(dim=-1)
        try:
            head, head_opt = _ensure_projection(z.size(-1), z.device)
            head.train()
            zp = F.normalize(head(z.float()), dim=-1)
            _update_buffer(zp, labels)
            ext_proto = _buffer_prototypes(zp.device) if BPR_PROTO_SCOPE == "global" else None
            bpr_term = total_bpr_loss(
                zp, labels,
                num_classes=NUM_CLASSES,
                use_adversarial=BPR_ADV,
                prototype=BPR_PROTO,
                external_prototypes=ext_proto,
            )
        except Exception as _e:
            import logging
            logging.warning(f"[bpr] BPR term skipped: {_e}")
            bpr_term = torch.zeros((), device=ce.device)

    bpr_val = float(bpr_term.item()) if torch.is_tensor(bpr_term) else 0.0

    if bpr_val == 0.0:
        total = ce
        aux_optimizer.zero_grad()
        if _proj["opt"] is not None: _proj["opt"].zero_grad()
        total.backward()
        aux_optimizer.step()
        if _proj["opt"] is not None: _proj["opt"].step()

    elif BPR_MODE == "mgda":
        params = [p for p in self.cond_pred_model.parameters() if p.requires_grad]
        if _proj["head"] is not None:
            params += [p for p in _proj["head"].parameters() if p.requires_grad]
        info = mgda_step(ce, BPR_LAMBDA * bpr_term, params, aux_optimizer)
        if _proj["opt"] is not None:
            _proj["opt"].step(); _proj["opt"].zero_grad()
        total = ce

    elif BPR_MODE == "pcgrad":
        params = [p for p in self.cond_pred_model.parameters() if p.requires_grad]
        if _proj["head"] is not None:
            params += [p for p in _proj["head"].parameters() if p.requires_grad]
        info = pcgrad_step(ce, BPR_LAMBDA * bpr_term, params, aux_optimizer)
        if _proj["opt"] is not None:
            _proj["opt"].step(); _proj["opt"].zero_grad()
        total = ce

    else:
        # joint (default)
        total = ce + BPR_LAMBDA * bpr_term
        aux_optimizer.zero_grad()
        if _proj["opt"] is not None: _proj["opt"].zero_grad()
        total.backward()
        aux_optimizer.step()
        if _proj["opt"] is not None: _proj["opt"].step()

    return float(total.item())

DT.Diffusion.nonlinear_guidance_model_train_step = _aux_step_with_bpr

_orig_train = DT.Diffusion.train

def bpr_train(self):
    import logging

    # Stage1 = DCG BPR+CE pre-train (n_pretrain_epochs = BPR_STAGE1_EPOCHS)
    if BPR_TWO_STAGE:
        self.config.diffusion.aux_cls.n_pretrain_epochs = BPR_STAGE1_EPOCHS
        self.config.diffusion.aux_cls.joint_train = False
        logging.info(
            f"[bpr] 2-stage (single run) — "
            f"Stage1: DCG BPR+CE {BPR_STAGE1_EPOCHS}ep → "
            f"Stage2: diffusion (DCG frozen)"
        )
        return _orig_train(self)

    if BPR_STAGE == 1:
        print(f"[bpr] Stage 1 — DCG BPR pre-train only "
              f"(lambda={BPR_LAMBDA}, proto={BPR_PROTO}, scope={BPR_PROTO_SCOPE})")
        if BPR_PRETRAIN_EPOCHS > 0:
            self.config.diffusion.aux_cls.n_pretrain_epochs = BPR_PRETRAIN_EPOCHS
            logging.info(f"[bpr] n_pretrain_epochs overridden → {BPR_PRETRAIN_EPOCHS}")
        self.args.train_guidance_only = True
        return _orig_train(self)

    if BPR_STAGE == 2:
        if not BPR_STAGE1_LOG:
            raise ValueError(
                "[bpr] Stage 2 requires BPR_STAGE1_LOG=<stage1 log dir> "
                "(contains aux_ckpt.pth saved by Stage 1)"
            )
        print(f"[bpr] Stage 2 — loading DCG from '{BPR_STAGE1_LOG}', "
              f"diffusion-only training")
        self.config.diffusion.trained_aux_cls_log_path = BPR_STAGE1_LOG
        self.config.diffusion.aux_cls.joint_train = False
        return _orig_train(self)

    print(f"[bpr] DiffMIC joint BPR train "
          f"(lambda={BPR_LAMBDA}, adv={BPR_ADV}, mode={BPR_MODE}, "
          f"proto={BPR_PROTO}, scope={BPR_PROTO_SCOPE})")
    return _orig_train(self)

DT.Diffusion.train = bpr_train

_mode_tag = (
    f"two_stage(stage1={BPR_STAGE1_EPOCHS}ep)" if BPR_TWO_STAGE else
    f"stage={BPR_STAGE}" + (f"(pretrain={BPR_PRETRAIN_EPOCHS}ep)" if BPR_STAGE == 1 and BPR_PRETRAIN_EPOCHS > 0 else "")
                         + (f"(log={BPR_STAGE1_LOG})" if BPR_STAGE == 2 else "")
)
print(
    f"[bpr] DiffMIC patches active — "
    f"{_mode_tag}, lambda={BPR_LAMBDA}, adv={BPR_ADV}, "
    f"mode={BPR_MODE}, proto={BPR_PROTO}, scope={BPR_PROTO_SCOPE}"
    + (f", buffer={BPR_BUFFER_SIZE}" if BPR_PROTO_SCOPE == "global" else "")
)

import main as _main
import sys as _sys

_split_sfx = "/split_" + str(_main.args.split)
if not _main.args.doc.endswith(_split_sfx):
    _main.args.doc = _main.args.doc + _split_sfx
    print(f"[bpr] applied split suffix: args.doc = {_main.args.doc}")

if hasattr(_main, 'main'):
    _sys.exit(_main.main())
