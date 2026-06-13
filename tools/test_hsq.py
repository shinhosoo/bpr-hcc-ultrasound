"""HSQ (LENet/LENet_base) test set 추론 — predictions_test_hsq.npz 저장.

다른 모델의 eval_only.py / test_medvitv2.sh 와 동일한 역할.
학습 중 선택된 best ckpt 를 로드해 test set 을 평가하고
표준 포맷(y_true, y_score, split="test") 으로 저장합니다.

사용법 (test_hsq.sh 에서 호출):
    python3 tools/test_hsq.py \\
        --ckpt-dir  experiments/option_b_5fold/results/<tag>/hsq/fold_0 \\
        --test-path data/5fold/fold_0/imagefolder/test \\
        --out       experiments/option_b_5fold/results/<tag>/hsq/fold_0/predictions_test_hsq.npz \\
        --fold-index 0

env:
    HSQ_BASE=1      → LENet_base 사용 (default: LENet proposed)
    BS=32            → batch size (default: 20)
"""
import argparse
import os
import sys

import numpy as np
import torch
from torchvision import datasets as tds, transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HSQ_DIR = os.path.join(ROOT, "models", "HSQ")

if not os.path.isdir(HSQ_DIR):
    print(f"[test_hsq] ERROR: HSQ dir not found: {HSQ_DIR}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, HSQ_DIR)
os.chdir(HSQ_DIR)

parser = argparse.ArgumentParser(description="HSQ test set 평가 → predictions_test_hsq.npz")
parser.add_argument("--ckpt-dir",   required=True,
                    help="run_hsq.py 의 --output-dir 경로 (checkpoints/ 포함)")
parser.add_argument("--test-path",  required=True,
                    help="test imagefolder 경로")
parser.add_argument("--out",        required=True,
                    help="출력 npz 경로 (예: .../predictions_test_hsq.npz)")
parser.add_argument("--fold-index", type=int, default=0,
                    help="fold 번호 — ckpt 파일명 suffix (default: 0)")
parser.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("BS", "20")))
args = parser.parse_args()

ckpt_dir   = os.path.abspath(args.ckpt_dir)
test_path  = os.path.abspath(args.test_path)
out_path   = os.path.abspath(args.out)
fold_idx   = args.fold_index
use_base   = os.environ.get("HSQ_BASE", "0") == "1"

ckpt_path = os.path.join(ckpt_dir, "checkpoints", f"hsq_val{fold_idx}.pth")
if not os.path.isfile(ckpt_path):
    print(f"[test_hsq] ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(test_path):
    print(f"[test_hsq] ERROR: test-path not found: {test_path}", file=sys.stderr)
    sys.exit(1)

import argparse as _ap
hsq_args = _ap.Namespace(
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
    num_classes        = 2,
    aux_loss           = False,
    visual_feature     = False,
    proposed           = not use_base,
    imagenet_pretrain  = False,
)

from models.LivNet import LENet, LENet_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LENet_base(args=hsq_args) if use_base else LENet(args=hsq_args)
ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.to(device)
model.eval()
print(f"[test_hsq] loaded {'LENet_base' if use_base else 'LENet'}  from {ckpt_path}")

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_tf   = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

class _IFWithPath(tds.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target, path

dataset = _IFWithPath(test_path, transform=_tf)
loader  = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=0, pin_memory=True,
)
print(f"[test_hsq] test set: {len(dataset)} samples  ({test_path})")

y_true, y_score, paths = [], [], []

with torch.no_grad():
    for imgs, labels, img_paths in loader:
        imgs = imgs.to(device)
        out  = model(imgs, "val") if not use_base else model(imgs)
        prob = torch.softmax(out, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_score.extend(prob)
        paths.extend(img_paths)

y_true  = np.array(y_true,  dtype=np.int64)
y_score = np.array(y_score, dtype=np.float32)   # (N, 2)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
np.savez(out_path,
         y_true  = y_true,
         y_score = y_score,
         paths   = np.array(paths),
         split   = "test")

print(f"[test_hsq] saved → {out_path}  (N={len(y_true)})")
