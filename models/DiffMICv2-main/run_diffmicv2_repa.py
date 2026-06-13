"""DiffMICv2 + REPA — conditioning encoder 표현을 강한 frozen 인코더에 정렬(distill).

REPA(Yu et al., ICLR'25) 정신: 디노이저가 쓰는 표현을 강한 사전학습 인코더의
'깨끗한 이미지 표현'에 cosine 정렬시키면, 표현이 풍부해져 생성이 개선된다.
여기서는 ConditionalModel 의 conditioning feature(x_weight, 6144-d)를
강한 frozen 인코더(timm) feature 에 정렬한다 → conditioning 을 collapse(BPR)가 아니라
ENRICH → diffusion 이 환영. (BPR 충돌과 정반대 방향)

손실: L = diffusion_focal_loss + REPA_LAMBDA * (1 - cos(proj(x_weight), f_strong))
  - f_strong = StrongEncoder(image)  (frozen, detached)
  - proj = self.model.repa_proj (MLP) — self.model 에 등록 → 메인 옵티마이저가 학습
  - x_weight 는 grad 가 살아있어 → DiffMICv2 인코더가 강한 표현을 닮도록 학습됨

env:
  REPA_LAMBDA(0.5)  REPA_ENCODER(timm 모델명, 기본 resnet50)  REPA_STRONG_DIM(인코더 출력차원)
  REPA_PROJ_HID(2048)
모든 REPA 연산은 try/except — diffusion 학습은 안 멈춤.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffuser_trainer as DT
import model as MDL
from timm.models import create_model

REPA_LAMBDA  = float(os.environ.get("REPA_LAMBDA", "0.5"))
REPA_ENCODER = os.environ.get("REPA_ENCODER", "resnet50")
REPA_PROJ_HID = int(os.environ.get("REPA_PROJ_HID", "2048"))
_DIM_GUESS = {"resnet50": 2048, "resnet18": 512, "convnext_small": 768,
              "convnext_tiny": 768, "vit_small_patch14_dinov2.lvd142m": 384,
              "vit_base_patch14_dinov2.lvd142m": 768, "vit_small_patch16_224": 384}
REPA_STRONG_DIM = int(os.environ.get("REPA_STRONG_DIM", str(_DIM_GUESS.get(REPA_ENCODER, 2048))))

_cap = {"x_weight": None}
_strong = {"model": None}

print(f"[repa] lambda={REPA_LAMBDA} encoder={REPA_ENCODER} strong_dim={REPA_STRONG_DIM}")


def _cm_fwd(self, x, y, t, x_l, attn):
    bz, np_, I, J = x_l.shape
    x_l_in = x_l.view(bz * np_, I, J).unsqueeze(1).expand(-1, 3, -1, -1)
    x_l_feat = self.encoder_x_l(x_l_in); x_l_feat = self.norm_l(x_l_feat)
    x_g = self.encoder_x(x); x_g = self.norm(x_g)
    x_l_feat = x_l_feat.reshape(bz, np_, x_l_feat.shape[1]).permute(0, 2, 1)
    x_cat = torch.cat([x_g.unsqueeze(-1), x_l_feat], dim=-1)
    w = torch.softmax(self.cond_weight, dim=2)
    x_weight = torch.sum(x_cat * w, dim=-1)
    _cap["x_weight"] = x_weight
    y2 = self.lin1(y, t); y2 = self.unetnorm1(y2); y2 = F.softplus(y2)
    y2 = x_weight.unsqueeze(-1).unsqueeze(-1) * y2
    y2 = self.lin2(y2, t); y2 = self.unetnorm2(y2); y2 = F.softplus(y2)
    y2 = self.lin3(y2, t); y2 = self.unetnorm3(y2); y2 = F.softplus(y2)
    return self.lin4(y2)
MDL.ConditionalModel.forward = _cm_fwd


def _get_strong(device):
    if _strong["model"] is None:
        if REPA_ENCODER == "medvit":
            import sys as _sys
            _mv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "MedViT-main", "CustomDataset")
            if _mv not in _sys.path:
                _sys.path.insert(0, _mv)
            import importlib.util as _ilu
            _saved_utils = _sys.modules.get("utils")
            try:
                _uspec = _ilu.spec_from_file_location("utils", os.path.join(_mv, "utils.py"))
                _umod = _ilu.module_from_spec(_uspec)
                _sys.modules["utils"] = _umod
                _uspec.loader.exec_module(_umod)
                import MedViT
            finally:
                if _saved_utils is not None:
                    _sys.modules["utils"] = _saved_utils
                else:
                    _sys.modules.pop("utils", None)
            from timm.models import create_model as _cm
            mname = os.environ.get("MEDVIT_MODEL", "MedViT_small")
            ckpt = os.environ.get("MEDVIT_CKPT", "")
            m = _cm(mname, num_classes=int(os.environ.get("BPR_NUM_CLASSES", "2")))
            if ckpt and os.path.exists(ckpt):
                _ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                _state = _ck.get("model", _ck) if isinstance(_ck, dict) else _ck
                miss, unexp = m.load_state_dict(_state, strict=False)
                print(f"[repa] MedViT loaded from {ckpt} (missing={len(miss)} unexpected={len(unexp)})")
            else:
                print(f"[repa] WARN: MEDVIT_CKPT 없음/미존재 ({ckpt}) — 미학습 MedViT (타깃 무익할 수 있음)")
            m.eval()
            for _p in m.parameters():
                _p.requires_grad = False
            _feat = {}
            def _ph(_mod, _in):
                _feat["f"] = _in[0]
            m.proj_head.register_forward_pre_hook(_ph)
            _strong["model"] = m.to(device)
            _strong["feat"] = _feat
            _strong["is_medvit"] = True
        else:
            m = create_model(REPA_ENCODER, pretrained=True, num_classes=0)  # feature extractor
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            _strong["model"] = m.to(device)
            _strong["is_medvit"] = False
            print(f"[repa] strong encoder '{REPA_ENCODER}' loaded (frozen)")
    return _strong["model"]


_orig_init = DT.CoolSystem.__init__
def _init(self, hparams):
    _orig_init(self, hparams)
    feat_dim = self.model.encoder_x.g.out_features  # 6144
    self.model.repa_proj = nn.Sequential(
        nn.Linear(feat_dim, REPA_PROJ_HID), nn.GELU(),
        nn.Linear(REPA_PROJ_HID, REPA_STRONG_DIM))
    print(f"[repa] repa_proj added: {feat_dim} -> {REPA_PROJ_HID} -> {REPA_STRONG_DIM}")
DT.CoolSystem.__init__ = _init


_orig_ts = DT.CoolSystem.training_step
def _ts(self, batch, batch_idx):
    out = _orig_ts(self, batch, batch_idx)
    try:
        x_batch, _ = batch
        x_batch = x_batch.to(self.device)
        xw = _cap.get("x_weight")
        if xw is not None and hasattr(self.model, "repa_proj"):
            enc = _get_strong(xw.device)
            with torch.no_grad():
                out_enc = enc(x_batch)
                if _strong.get("is_medvit"):
                    f_strong = _strong["feat"]["f"]
                else:
                    f_strong = out_enc
                if f_strong.dim() > 2:
                    f_strong = f_strong.mean(dim=tuple(range(2, f_strong.dim())))
                f_strong = F.normalize(f_strong.float(), dim=-1)
            z = F.normalize(self.model.repa_proj(xw.float()), dim=-1)
            align = (1.0 - (z * f_strong).sum(-1)).mean()
            out["loss"] = out["loss"] + REPA_LAMBDA * align
            try: self.log("repa_align", float(align.item()), prog_bar=True)
            except Exception: pass
    except Exception as _e:
        print(f"[repa] align skipped: {_e}")
    return out
DT.CoolSystem.training_step = _ts


print("[repa] patches applied — launching DT.main()")
DT.main()
