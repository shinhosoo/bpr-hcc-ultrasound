"""DiffMICv2 1-stage end-to-end BPR — Lightning training_step 에 BPR loss 합산.
원본 코드 무수정. AttentionModule.forward 또는 ConditionalModel.forward,
CoolSystem.training_step, CoolSystem.configure_optimizers 를 monkey-patch.

추가 옵션 (env var):
  BPR_HOOK     = attn | prelin4 | enc512 | enc512_local | xweight
      - attn   (기본) : DCG.AttentionModule.z (B, 512) 캡쳐
                        — DCG 가 frozen 이면 표현학습 신호가 DCG 까지 안 흐름
                        (DCG_UNFREEZE 와 함께 써야 backbone 까지 학습)
      - prelin4       : ConditionalModel 의 lin4 직전 feature 캡쳐
                        — (B, feature_dim, np, np) → spatial mean → (B, feature_dim=6144)
                        — diffusion 본체 (trainable) 가 BPR signal 로 학습됨
                        — timestep 의존적 (BPR_T_MAX 의 의미 있음)
      - enc512        : ConditionalModel.encoder_x (global) 의 ResNet18 backbone 직후 512-d
                        — (B, 512), Linear(512→6144) 직전. trainable backbone.
                        — image-only, **timestep 무관** (BPR_T_MAX 무의미)
      - enc512_local  : ConditionalModel.encoder_x_l (local crops) 의 ResNet18 직후 512-d,
                        K=6 crop 평균 → (B, 512). image-only, timestep 무관.
      - xweight       : ConditionalModel 의 fused conditioning vector 캡쳐
                        — (B, 6144), global + local + cond_weight fusion 결과
                        — encoder_x, encoder_x_l, cond_weight 셋 다 trainable
                        — image-only, timestep 무관, 라벨 stream 진입 전.
      - xweight_bn    : xweight 위치에 in-path bottleneck 삽입
                        — Linear(6144→BN_DIM) → ReLU → Linear(BN_DIM→6144)
                        — 중간 (B, BN_DIM) 캡쳐 → BPR 표현학습 대상
                        — 분류가 그 압축점에 의존하므로 표현학습 신호가 강력
                        — BN_SKIP=1 이면 residual (x_weight + bn(x_weight))
      - xweight_aux   : 병렬 보조 bottleneck branch (additive, by design)
                        — 원본 x_weight (B, 6144) 은 그대로 유지
                        — 보조 branch: aux_down(6144→BN_DIM) → ReLU → aux_up(BN_DIM→6144)
                        — 중간 (B, BN_DIM) 캡쳐 → BPR 표현학습 대상
                        — 보조 출력을 x_weight 에 ADD (병렬 residual)
                        — aux_up 의 weight/bias ZERO INIT → 시작 시점에 보조 출력 = 0
                        → 학습 초기엔 baseline 과 완전 동일, BPR 가 bottleneck 을 점진적 학습
                        — 가장 안전한 representation learning 설계
      - dual_gl       : 글로벌 (encoder_x) + 로컬 (encoder_x_l mean) 두 표현을 각각 독립 BPR
                        — 각 branch (B, 512), 별도 projection head + 별도 prototype buffer
                        — 두 BPR loss 평균이 최종 BPR loss
                        — DCG 의 dual-granularity 철학과 정렬됨
  BPR_BN_DIM   = int (기본 512) — xweight_bn / xweight_aux 의 bottleneck 중간 차원
  BPR_BN_SKIP  = 0 | 1 (기본 0) — xweight_bn 만 적용: 0: pure / 1: residual
  BPR_T_MAX           = float (0, 1] 기본 1.0
      이미지별 평균 diffusion timestep 이 T*BPR_T_MAX 미만일 때만 BPR loss 합산.
      prelin4 처럼 feature 가 timestep 에 의존할 때, 노이즈가 덜한 step 에서만 학습 가능.
      예) 0.3 → 평균 t 가 하위 30% 인 샘플만 BPR 에 참여.
  BPR_WARMUP_EPOCHS   = int   기본 0
      첫 N epoch 동안은 BPR loss 비활성 (diffusion 만 학습) → 본체 안정화 후 BPR 가세.
  BPR_MIN_ACTIVE      = int   기본 2
      게이트 통과 샘플이 이 값 미만이면 그 step 의 BPR skip.
  BPR_STAGE           = 1 | 2 (기본 1)
      1 = joint training (diffusion + BPR 동시 학습)
      2 = post-hoc refinement — stage1 ckpt 로드 후 diffusion 가중치 ↓ + BPR 만 학습
  BPR_STAGE2_CKPT     = str (BPR_STAGE=2 일 때 필수)
      stage1 baseline ckpt 경로. shell 스크립트가 BPR_STAGE2_FROM 으로 자동 탐색.
  BPR_STAGE2_DIFF_W   = float (기본 0.0)
      stage2 의 diffusion loss 가중치. 0 = 순수 BPR refinement, 0.1 = 가벼운 regularization.
  BPR_STAGE2_LR_SCALE = float (기본 0.1)
      stage2 의 main LR 배수 (작게 시작).
  DCG_UNFREEZE = 0 | attn | local | all
      - 0    : 기본. DCG (aux_model) frozen 유지 (원본 동작).
      - attn : attention_module + mil_attn + classifier_linear 만 unfreeze
      - local: local_network + 위 항목들까지 unfreeze
      - all  : DCG 전체 unfreeze
  DCG_LR_SCALE = float (기본 0.1) — main LR 대비 DCG 파라미터 LR 배수
  DCG_WARMUP   = int   (기본 0)   — 첫 N epoch 동안은 frozen, 그 후 unfreeze
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models/DiffMICv2-main"))

import torch
import torch.nn.functional as F
from bpr_loss import bpr_prototype_loss, total_bpr_loss

BPR_LAMBDA = float(os.environ.get("BPR_LAMBDA", "0.1"))
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

BPR_STAGE             = int(os.environ.get("BPR_STAGE", "1"))           # 1 (joint) | 2 (post-hoc)
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
_proj = {}  # branch -> {"head": nn.Sequential, "opt": torch.optim.Adam}
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

        # local: ResNet18 .f → flatten → (B*K, 512)
        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_512_raw = self.encoder_x_l.f(x_l_in)
        x_l_512 = torch.flatten(x_l_512_raw, start_dim=1)         # (B*K, 512)
        x_l_feat = self.encoder_x_l.g(x_l_512)                    # (B*K, 6144)
        x_l_feat = self.norm_l(x_l_feat)

        # global: ResNet18 .f → flatten → (B, 512)
        x_512_raw = self.encoder_x.f(x)
        x_512 = torch.flatten(x_512_raw, start_dim=1)             # (B, 512)
        x_g_feat = self.encoder_x.g(x_512)
        x_g_feat = self.norm(x_g_feat)

        # ★ DUAL capture
        _state["zg"] = x_512                                       # global (B, 512)
        if BPR_LOCAL_POOL == "parallel":
            _state["zl"] = x_l_512
        else:
            _state["zl"] = x_l_512.view(bz, np_, -1).mean(dim=1)   # local pooled (B, 512)

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
        feat_dim = self.encoder_x.g.out_features  # 6144
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

        # encoder_x_l: ResNet18 backbone .f → flatten → .g(512→6144)
        x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
        x_l_512_raw = self.encoder_x_l.f(x_l_in)
        x_l_512 = torch.flatten(x_l_512_raw, start_dim=1)         # (B*K, 512)
        x_l_feat = self.encoder_x_l.g(x_l_512)                    # (B*K, 6144)
        x_l_feat = self.norm_l(x_l_feat)

        # encoder_x: ResNet18 backbone .f → flatten → .g
        x_512_raw = self.encoder_x.f(x)
        x_512 = torch.flatten(x_512_raw, start_dim=1)             # (B, 512)
        x_g = self.encoder_x.g(x_512)                             # (B, 6144)
        x_g = self.norm(x_g)

        # ★ 512-d hook capture
        if BPR_HOOK == "enc512":
            _state["z"] = x_512                                   # (B, 512)
        else:  # enc512_local
            if BPR_LOCAL_POOL == "parallel":
                _state["z"] = x_l_512                             # parallel: (B*K, 512)
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

else:
    raise ValueError(f"BPR_HOOK={BPR_HOOK} 미지원 (attn | prelin4 | enc512 | enc512_local | xweight | xweight_bn | xweight_aux | dual_gl 중 택1)")

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
            raise RuntimeError("[stage2] BPR_STAGE2_CKPT not set. shell 스크립트가 자동탐색하거나 직접 지정하세요.")
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
    """Phase 2 진입 조건 검사 + 1회 freeze + fusion_dnn 을 optimizer 에 등록."""
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
        top = name.split(".", 1)[0]   # e.g. "lin1.lin.weight" → "lin1"
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
        print(f"[bpr-2phase] optimizer 갱신 실패 ({_e}) — head 업데이트 안 될 수 있음")

    print(f"[bpr-2phase] === Phase 2 begin at epoch {self.current_epoch} (Option C — head fine-tune) ===")
    print(f"[bpr-2phase] frozen={n_frozen/1e6:.2f}M (encoder_x/x_l/norm/cond_weight + aux_model) | "
          f"trainable={n_trainable/1e6:.4f}M (lin1-4 + unetnorm)")
    print(f"[bpr-2phase] head → optimizer: {'OK' if added else 'FAILED'}")
    print(f"[bpr-2phase] loss = diffusion_focal_loss (원래 그대로), BPR DISABLED")
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

    if in_phase2:
        return out
    # training-step warmup
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
    if BPR_HOOK in ("prelin4", "enc512", "enc512_local", "xweight", "xweight_bn", "xweight_aux", "dual_gl"):
        _grad_target = [p for p in self.model.parameters() if p.requires_grad]
    else:  # attn
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
        print(f"[dcg-unfreeze] WARN: unknown DCG_UNFREEZE={DCG_UNFREEZE} — frozen 유지")
        return ret

    aux_params = [p for p in self.aux_model.parameters() if p.requires_grad]
    if not aux_params:
        print(f"[dcg-unfreeze] mode={DCG_UNFREEZE}  → 매칭 파라미터 0개, frozen 유지")
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
    """Lightning 의 backward 가 끝난 뒤 모든 branch 의 projector optimizer step."""
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
