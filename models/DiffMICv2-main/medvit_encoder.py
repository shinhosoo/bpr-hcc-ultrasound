"""MedViT 를 DiffMICv2 의 인코더로 사용 (arch=medvit).

ResNetEncoder 와 동일 인터페이스: forward(x) -> (B, feature_dim).
MedViT 의 proj_head 입력(pooled feature)을 hook 으로 받아 g(Linear)로 feature_dim 에 투영.
backbone 은 trainable (encoder_x 가 self.model 에 속해 메인 옵티마이저가 학습) → diffusion 과 공동학습.

env:
  MEDVIT_MODEL (MedViT_small)  MEDVIT_CKPT (옵션: lesion 학습 ckpt 로 warm-start, fold별)
  DIFFMICV2_PRETRAINED (1이면 ckpt 없을 때 ImageNet pretrained)
"""
import os
import torch
import torch.nn as nn


def _import_medvit():
    """utils 모듈명 충돌을 피해 MedViT 를 import (MedViT.py 의 'from utils import merge_pre_bn')."""
    import sys, importlib.util as _ilu
    mv = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "MedViT-main", "CustomDataset"))
    if mv not in sys.path:
        sys.path.insert(0, mv)
    _saved = sys.modules.get("utils")
    try:
        _spec = _ilu.spec_from_file_location("utils", os.path.join(mv, "utils.py"))
        _umod = _ilu.module_from_spec(_spec)
        sys.modules["utils"] = _umod
        _spec.loader.exec_module(_umod)
        import MedViT
        from timm.models import create_model
    finally:
        if _saved is not None:
            sys.modules["utils"] = _saved
        else:
            sys.modules.pop("utils", None)
    return create_model


class MedViTEncoder(nn.Module):
    def __init__(self, feature_dim=6144, config=None, local=False):
        super().__init__()
        model_name = os.environ.get("MEDVIT_MODEL", "MedViT_small")
        ckpt = os.environ.get("MEDVIT_CKPT", "")
        nbc = int(os.environ.get("BPR_NUM_CLASSES", "2"))
        create_model = _import_medvit()
        use_pre = (os.environ.get("DIFFMICV2_PRETRAINED", "0") == "1") and not (ckpt and os.path.exists(ckpt))
        m = create_model(model_name, pretrained=use_pre)
        if ckpt and os.path.exists(ckpt):
            _ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            _state = _ck.get("model", _ck) if isinstance(_ck, dict) else _ck
            _miss, _unexp = m.load_state_dict(_state, strict=False)
            print(f"[medvit-enc] warm-start from {ckpt} (missing={len(_miss)} unexpected={len(_unexp)})")
        ph = getattr(m, "proj_head", None)
        featdim = None
        if isinstance(ph, nn.Linear):
            featdim = ph.in_features
        elif isinstance(ph, nn.Sequential):
            for _l in ph:
                if isinstance(_l, nn.Linear):
                    featdim = _l.in_features; break
        if featdim is None:
            featdim = 1024
        self.featdim = featdim
        self.backbone = m
        self._feat = {}
        def _hook(_mod, _in):
            self._feat["f"] = _in[0]
        m.proj_head.register_forward_pre_hook(_hook)
        self.g = nn.Linear(self.featdim, feature_dim)
        print(f"[medvit-enc] {model_name} as encoder_x (featdim={self.featdim} -> {feature_dim}, local={local}, pretrained={use_pre})")

    def forward(self, x):
        self.backbone(x)
        return self.g(self._feat["f"].float())
