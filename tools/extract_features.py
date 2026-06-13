#!/usr/bin/env python3
"""각 모델의 학습된 ckpt 에서 latent embedding 을 추출해 features.npz 로 저장.
   tools/tsne_plot.py 가 이 npz 를 받아 2D t-SNE/UMAP 시각화."""
import argparse, os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def extract_medvit(ckpt, imagefolder_dir, model_name, nb_classes, batch_size, device):
    """MedViT — proj_head 직전 (B, 1024) 추출."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "MedViT-main", "CustomDataset"))
    sys.path.insert(0, root)
    from timm.models import create_model
    import MedViT  # noqa: registers MedViT_small/base/large

    model = create_model(model_name, num_classes=nb_classes)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    features = []
    def pre_hook(module, _input):
        features.append(_input[0].detach().cpu())
    model.proj_head.register_forward_pre_hook(pre_hook)

    # ImageFolder val/test loader
    from torchvision import datasets, transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    ds = datasets.ImageFolder(imagefolder_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    labels = []
    with torch.no_grad():
        for x, y in loader:
            _ = model(x.to(device))
            labels.append(y.numpy())
    feats = torch.cat(features, dim=0).numpy()
    labs = np.concatenate(labels, axis=0)
    return feats, labs


def _diffmic_load_and_hook(model_root, exp_dir, doc, test_pkl, device, is_v2=False):
    """공통 — DiffMIC / DiffMICv2 에서 DCG aux 의 AttentionModule.forward 를
    monkey-patch 해서 z_weighted_avg 를 캡쳐."""
    sys.path.insert(0, model_root)
    import yaml, argparse as ap
    from easydict import EasyDict

    cfg_path = os.path.join(exp_dir, "logs", doc, "split_0", "config.yml") if not is_v2 else exp_dir
    if is_v2:
        with open(cfg_path) as f:
            cfg = EasyDict(yaml.safe_load(f))
    else:
        with open(cfg_path) as f:
            try:
                cfg = yaml.unsafe_load(f)
            except Exception:
                f.seek(0); cfg = yaml.safe_load(f)
        def _to_ns(o):
            if isinstance(o, dict):
                n = ap.Namespace()
                for k, v in o.items(): setattr(n, k, _to_ns(v))
                return n
            return o
        if isinstance(cfg, dict): cfg = _to_ns(cfg)
        cfg.device = device

    cfg.data.testdata = test_pkl

    captured = []
    if is_v2:
        import pretraining.modules as M
        orig = M.AttentionModule.forward
        def patched(self, h_crops):
            z, attn, y = orig(self, h_crops)
            captured.append(z.detach().cpu())
            return z, attn, y
        M.AttentionModule.forward = patched
    else:
        import pretraining.modules as M
        orig = M.AttentionModule.forward
        def patched(self, h_crops):
            z, attn, y = orig(self, h_crops)
            captured.append(z.detach().cpu())
            return z, attn, y
        M.AttentionModule.forward = patched

    return cfg, captured


def extract_diffmic(ckpt_dir, test_pkl, batch_size, device):
    """DiffMIC — AttentionModule z_weighted_avg (B, 2048) 추출."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "DiffMIC-main"))
    sys.path.insert(0, root)
    os.chdir(root)
    from utils import get_dataset
    from pretraining.dcg import DCG as AuxCls

    cfg, captured = _diffmic_load_and_hook(root, ckpt_dir, "lesion_binary", test_pkl, device, is_v2=False)

    aux = AuxCls(cfg).to(device)
    aux_state = torch.load(os.path.join(ckpt_dir, "logs", "lesion_binary", "split_0", "aux_ckpt_best.pth"),
                           map_location=device, weights_only=False)
    aux.load_state_dict(aux_state[0], strict=False)
    aux.eval()

    class _A: pass
    _a = _A(); _a.dataroot = None
    _, _, test_ds = get_dataset(_a, cfg)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    labels = []
    with torch.no_grad():
        for x, y in loader:
            _ = aux(x.to(device))
            labels.append(y.numpy())
    feats = torch.cat(captured, dim=0).numpy()
    labs = np.concatenate(labels, axis=0)
    return feats, labs


def extract_diffmicv2(cfg_yml, ckpt, test_pkl, batch_size, device):
    """DiffMICv2 — AttentionModule z_weighted_avg (B, 512) 추출."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "DiffMICv2-main"))
    sys.path.insert(0, root)
    os.chdir(root)
    from utils import get_dataset
    from pretraining.dcg import DCG as AuxCls

    import yaml, copy
    from easydict import EasyDict
    with open(cfg_yml) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.data.testdata = test_pkl

    import pretraining.modules as M
    captured = []
    _orig = M.AttentionModule.forward
    def _patched(self, h_crops):
        z, attn, y = _orig(self, h_crops)
        captured.append(z.detach().cpu())
        return z, attn, y
    M.AttentionModule.forward = _patched

    aux = AuxCls(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)
    aux_sd = {k.replace("aux_model.", ""): v for k, v in sd.items() if k.startswith("aux_model.")}
    aux.load_state_dict(aux_sd, strict=False)
    aux.eval()

    _, _, test_ds = get_dataset(cfg)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    labels = []
    with torch.no_grad():
        for x, y in loader:
            _ = aux(x.to(device))
            labels.append(y.numpy())
    feats = torch.cat(captured, dim=0).numpy()
    labs = np.concatenate(labels, axis=0)
    return feats, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["medvit", "diffmic", "diffmicv2"])
    ap.add_argument("--ckpt", required=True,
                    help="MedViT: <output_dir>; DiffMIC: <exp_dir>; DiffMICv2: <ckpt.ckpt>")
    ap.add_argument("--data", required=True,
                    help="MedViT: imagefolder/test 경로; DiffMIC/v2: lesion_test.pkl")
    ap.add_argument("--out", required=True, help="features.npz 출력 경로")
    ap.add_argument("--config", default=None, help="DiffMICv2 용 config.yml")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    if args.model == "medvit":
        feats, labs = extract_medvit(args.ckpt, args.data, "MedViT_small", 2, args.batch_size, device)
    elif args.model == "diffmic":
        feats, labs = extract_diffmic(args.ckpt, args.data, args.batch_size, device)
    elif args.model == "diffmicv2":
        if not args.config:
            raise SystemExit("--config 필요 (DiffMICv2 의 lesion_binary.yml)")
        feats, labs = extract_diffmicv2(args.config, args.ckpt, args.data, args.batch_size, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, features=feats, labels=labs.astype(int))
    print(f"[extract] wrote {args.out}  features={feats.shape}  labels={labs.shape}")


if __name__ == "__main__":
    main()
