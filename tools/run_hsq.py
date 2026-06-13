"""HSQ (LENet) runner — standard imagefolder interface.

Accepts the same --train-path / --val-path / --output-dir interface as other
model runners (run_medvitv2_bpr.py etc.) and delegates to HSQ's train.main()
with mode=standard (torchvision.ImageFolder).

Usage (called by tools/train_hsq.sh):
    python3 tools/run_hsq.py \\
        --train-path data/5fold/fold_0/imagefolder/train \\
        --val-path   data/5fold/fold_0/imagefolder/val \\
        --output-dir experiments/option_b_5fold/results/TAG/hsq/fold_0 \\
        --seed 42 --epochs 10 --batch-size 20 --fold-index 0
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HSQ_DIR = os.path.join(ROOT, "models", "HSQ")

if not os.path.isdir(HSQ_DIR):
    print(f"[run_hsq] ERROR: HSQ directory not found: {HSQ_DIR}", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="HSQ runner — standard imagefolder mode")
parser.add_argument("--train-path",  type=str, required=True,
                    help="imagefolder train 경로")
parser.add_argument("--val-path",    type=str, required=True,
                    help="imagefolder val 경로")
parser.add_argument("--test-path",   type=str, default=None,
                    help="imagefolder test 경로 (없으면 val-path 사용)")
parser.add_argument("--output-dir",  type=str, required=True,
                    help="checkpoints / logs 저장 루트")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--epochs",      type=int, default=10)
parser.add_argument("--batch-size",  type=int, default=20)
parser.add_argument("--fold-index",  type=int, default=0,
                    help="현재 fold 번호 (checkpoint 파일명 suffix 용)")
parser.add_argument("--base",        action="store_true", default=False,
                    help="LENet_base: BPR/adversarial 없이 CE만 학습")
parser.add_argument("--no-adv",      action="store_true", default=False,
                    help="adversarial perturbation 없이 BPR만 적용")
parser.add_argument("--patience",    type=int, default=40,
                    help="early stopping patience (0=disable)")
runner_args = parser.parse_args()

train_path = os.path.abspath(runner_args.train_path)
val_path   = os.path.abspath(runner_args.val_path)
test_path  = os.path.abspath(runner_args.test_path) if runner_args.test_path else val_path
out_dir    = os.path.abspath(runner_args.output_dir)

for p, label in [(train_path, "train-path"), (val_path, "val-path")]:
    if not os.path.isdir(p):
        print(f"[run_hsq] ERROR: {label} not found: {p}", file=sys.stderr)
        sys.exit(1)

os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "logs"),        exist_ok=True)

sys.path.insert(0, HSQ_DIR)
os.chdir(HSQ_DIR)

_pre_done = {"done": False}
_YEL = "\033[93m"; _RED = "\033[91m"; _RST = "\033[0m"

def _maybe_load_pretrained(model):
    """HSQ_PRETRAIN=1 일 때 backbone에 공식 ImageNet 가중치 주입.
    timm 대신 MS Swin / FB ConvNeXt 원본 ckpt를 사용해야 키가 맞음."""
    if _pre_done["done"]:
        return
    _pre_done["done"] = True
    _hsq_pretrain = os.environ.get("HSQ_PRETRAIN", "0") == "1"
    if not _hsq_pretrain:
        print(f"{_RED}[hsq-pretrain] HSQ_PRETRAIN=0 — pretrained weight 로드 안 됨 (random init){_RST}")
        return
    try:
        import timm as _timm
    except Exception as e:
        print(f"{_RED}[hsq-pretrain] timm import 실패 — scratch 유지: {e}{_RST}")
        return
    import torch as _torch
    m = model.module if hasattr(model, "module") else model
    _URLS = {
        "swin":    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "convnext":"https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    }
    def _load(dst, url, tag):
        if dst is None:
            print(f"{_RED}[hsq-pretrain] backbone 없음 (skip {tag}){_RST}"); return
        try:
            ck = _torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
        except Exception as e:
            print(f"{_RED}[hsq-pretrain] {tag} 다운로드 실패: {e}{_RST}"); return
        ref = ck.get("model", ck) if isinstance(ck, dict) else ck
        cur = dst.state_dict()
        match = {k: v for k, v in ref.items() if k in cur and v.shape == cur[k].shape}
        dst.load_state_dict(match, strict=False)
        print(f"{_YEL}[hsq-pretrain] {tag} matched {len(match)}/{len(cur)} keys (ref={len(ref)}){_RST}")
    _load(getattr(m, "swintransformer", None), _URLS["swin"],    "swin_small(MS)")
    _load(getattr(m, "convnext",        None), _URLS["convnext"],"convnext_small(FB)")


import argparse as _ap

hsq_args = _ap.Namespace(
    log_dir      = os.path.join(out_dir, "logs"),
    checkpoints  = os.path.join(out_dir, "checkpoints"),
    output_csv   = None,
    seed         = runner_args.seed,
    num_workers  = 0,
    epochs       = runner_args.epochs,
    batch_size   = runner_args.batch_size,
    lr           = 1e-4,
    weight_decay = 0.0001,
    smoothing    = 0.1,
    mixup        = False,
    amp          = False,
    model_name   = "hsq",
    model        = "LENet",
    proposed          = not runner_args.base,
    imagenet_pretrain  = os.environ.get('HSQ_PRETRAIN', '0') == '1',
    adv_bpr      = not runner_args.no_adv,
    patience     = runner_args.patience,
    bpr_aux      = os.environ.get('BPR_AUX', '0') == '1',
    bpr_bn_dim   = int(os.environ.get('BPR_BN_DIM', '128')),
    bpr_lambda   = float(os.environ.get('BPR_LAMBDA', '0.3')),
    serial_parallel    = "serial",
    sparse_dense       = "sparse_token",
    num_experts        = 4,
    top_k              = 1,
    head_type          = "linear",
    cat_moe_head       = False,
    q_former_depths    = [1, 1, 1, 1],
    stage_dims         = [96, 192, 384, 768],
    q_former_head_num  = [3, 6, 12, 24],
    num_query_tokens   = 200,
    query_dim          = 384,
    # EMA
    model_ema           = True,
    model_ema_decay     = 0.9998,
    model_ema_warmup    = False,
    model_ema_force_cpu = False,
    ema_updata_epoch    = 4,
    criterion       = "CrossEntropyLoss",
    lr_param_groups = True,
    weight_sampler  = True,
    aux_loss        = False,
    resume          = False,
    freeze_layers   = False,
    cam_visualization = False,
    visual_feature  = False,
    statistics      = False,
    k_fold          = True,
    test            = False,
    categories      = "binary",
    num_classes     = 2,
    dataset         = "liver",
    mode       = "standard",
    train_path = train_path,
    val_path   = val_path,
    test_path  = test_path,
)

import torch
from train import main as hsq_main   # HSQ/train.py

device = 0 if torch.cuda.is_available() else "cpu"
fold_idx = runner_args.fold_index

print(f"[run_hsq] ============================================")
print(f"[run_hsq] fold={fold_idx}  seed={hsq_args.seed}  epochs={hsq_args.epochs}  bs={hsq_args.batch_size}")
print(f"[run_hsq] train={train_path}")
print(f"[run_hsq] val  ={val_path}")
print(f"[run_hsq] test ={test_path}")
print(f"[run_hsq] out  ={out_dir}")
print(f"[run_hsq] device={device}")

hsq_main(device, fold_idx, hsq_args)

print(f"[run_hsq] DONE  fold={fold_idx}  ckpt={os.path.join(out_dir, 'checkpoints')}")

_ckpt_path = os.path.join(hsq_args.checkpoints, f'{hsq_args.model_name}_val{fold_idx}.pth')

if os.path.isfile(_ckpt_path):
    from models.LivNet import LENet, LENet_base
    from torchvision import datasets as _tds, transforms as _T
    import numpy as _np

    _MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
    _test_tf = _T.Compose([
        _T.Resize((224, 224)),
        _T.ToTensor(),
        _T.Normalize(_MEAN, _STD),
    ])

    class _IFP(_tds.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            return sample, target, path

    # proposed=True → LENet (forward requires mode arg),  False → LENet_base
    if hsq_args.proposed:
        _model = LENet(args=hsq_args)
    else:
        _model = LENet_base(args=hsq_args)
    _ckpt = torch.load(_ckpt_path, map_location='cpu', weights_only=False)
    _model.load_state_dict(_ckpt['state_dict'], strict=True)
    _model.to(device)
    _model.eval()

    def _infer_loader(data_path):
        """주어진 imagefolder 경로 → (y_true, y_score (N,2), paths)"""
        _ds     = _IFP(data_path, transform=_test_tf)
        _loader = torch.utils.data.DataLoader(
            _ds, batch_size=hsq_args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        _yt, _ys, _ps = [], [], []
        with torch.no_grad():
            for _img, _lbl, _p in _loader:
                _img = _img.to(device)
                _out = _model(_img, 'val') if hsq_args.proposed else _model(_img)
                _prob = torch.softmax(_out, 1).cpu().numpy()   # (B, 2)
                _yt.extend(_lbl.numpy())
                _ys.extend(_prob)
                _ps.extend(_p)
        return (_np.array(_yt, dtype=_np.int64),
                _np.array(_ys, dtype=_np.float32),
                _np.array(_ps))

    # val predictions
    _val_npz = os.path.join(out_dir, "predictions_val.npz")
    print(f"[run_hsq] val predictions → {_val_npz}")
    _yt, _ys, _ps = _infer_loader(val_path)
    _np.savez(_val_npz, y_true=_yt, y_score=_ys, paths=_ps, split="val")
    print(f"[run_hsq] val  saved  n={len(_yt)}")

    _model_tag  = "hsq" if hsq_args.proposed else "hsq_base"
    _test_npz   = os.path.join(out_dir, f"predictions_test_{_model_tag}.npz")
    if test_path != val_path:
        print(f"[run_hsq] test predictions → {_test_npz}")
        _yt, _ys, _ps = _infer_loader(test_path)
        _np.savez(_test_npz, y_true=_yt, y_score=_ys, paths=_ps, split="test")
        print(f"[run_hsq] test saved  n={len(_yt)}")
    else:
        print(f"[run_hsq] test_path == val_path — test npz 생략 (--test-path 를 별도로 지정하세요)")
else:
    print(f"[run_hsq] WARNING: best ckpt not found ({_ckpt_path}) — predictions 저장 건너뜀")
