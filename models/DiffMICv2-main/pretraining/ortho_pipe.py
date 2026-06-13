"""직교 부분공간 conditioning — "목적 분리(decouple the objective)" 설계.

문제의식
--------
dual_pipe(detach 분리 채널)는 해는 없앴지만 클래스 정보가 DCG prior 와 중복돼 무익했다.
근본 원인: BPR(분산 축소)을 conditioning 에 직접 걸면 denoiser 가 쓰는 instance richness
가 파괴된다. 공간만 분리하면 BPR 이 diffusion 에 닿지 못한다.

설계 (dual_pipe 와 다른 점)
---------------------------
같은 conditioning 벡터 x_weight(6144) 를 **상보 분해**한다:
  z_cls  = cls_proj(x_weight.detach())
  x_cls  = cls_back(z_cls)
  z_inst = x_weight - x_cls

denoiser:
  - 곱셈 경로에 **z_inst** 사용 (x_weight 대신) → 엉켜있던 클래스 축을 *제거한* 순수 인스턴스
    conditioning. BPR collapse 영향 없음(encoder 는 diffusion loss 로만 학습).
  - 클래스 축은 **e_to_y(z_cls)** 로 *가산* 재주입 (zero-init) → BPR 로 또렷해진 깨끗한
    클래스 신호. dual_pipe 처럼 중복이 아니라, 곱셈경로에서 뺀 축을 정제해서 되돌림.

직교 패널티(aux_loss): per-sample cos(z_inst, x_cls)^2 → 두 부분공간이 실제로 직교(상보)
하도록 cls_proj/cls_back 만 학습 (encoder 보호 위해 detached 잔차로 계산).

불변식
  · BPR/ortho gradient 는 cls_proj/cls_back/e_to_y 에만. encoder_x(_l)·norm·cond_weight·
    lin1~4 에는 닿지 않는다 (z_cls 가 x_weight.detach() 에서 나오므로).
  · diffusion loss 는 z_inst 를 통해 encoder 를 정상 학습(분산 보존).

train: run_diffmicv2_bpr.py 가 _state["z"]=z_cls (BPR), _state["aux_loss"]=직교패널티 사용.
eval : bpr_arch_hook.py 가 동일 모듈 재구성(체크포인트의 cls_proj/cls_back/e_to_y 로드).
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

    import model as MDL
    _orig_init = MDL.ConditionalModel.__init__

    def _cm_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        feat_dim = self.encoder_x.g.out_features  # 6144
        self.cls_proj = nn.Linear(feat_dim, d_cls, bias=False)
        self.cls_back = nn.Linear(d_cls, feat_dim, bias=False)
        self.e_to_y = nn.Linear(d_cls, feat_dim)
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

        y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
        y2 = z_inst.unsqueeze(-1).unsqueeze(-1) * y2
        y2 = y2 + self.e_to_y(z_cls).unsqueeze(-1).unsqueeze(-1)
        y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
        y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
        return self.lin4(y2)
    MDL.ConditionalModel.forward = _cm_fwd

    if verbose:
        print(f"[ortho_pipe] orthogonal-subspace conditioning applied "
              f"(d_cls={d_cls}, detach_instance={detach_instance}, e_to_y zero-init)")
    return True
