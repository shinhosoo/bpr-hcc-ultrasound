"""Class-balanced batch sampler.

배치 내 각 클래스의 샘플 수가 정확히 동일하도록 인덱스를 묶어주는 BatchSampler.
- batch_size 는 num_classes 의 배수여야 함 (강제, 아닐 시 자동 보정 + 경고).
- 매 epoch 마다 각 클래스의 인덱스를 섞고, 작은 클래스는 wrap-around 로 재사용해
  epoch 길이는 (가장 많은 클래스 수 * num_classes / batch_size) 만큼 보장 (over-sampling).

PyTorch DataLoader 사용 예:
    sampler = BalancedBatchSampler(targets, batch_size=32, num_classes=2)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class BalancedBatchSampler(Sampler[list]):
    """배치 내 모든 클래스가 동일한 개수로 등장하는 BatchSampler.

    Args:
        labels (Sequence[int] | np.ndarray): 데이터셋 전체 라벨 (len = N).
        batch_size (int): 배치 크기. num_classes 의 배수여야 하며, 아닐 경우 가장 가까운
            num_classes 의 배수로 내림.
        num_classes (int | None): 클래스 수. None 이면 labels 에서 유추.
        seed (int): 셔플 seed.
        drop_last (bool): 마지막 배치가 정확히 채워지지 않으면 drop. 기본 True (균형 유지).
        over_sample (bool): True 면 epoch 길이를 가장 많은 클래스 기준으로 늘리고
            적은 클래스는 wrap-around 로 재사용. False 면 가장 적은 클래스 기준 (under-sample).
            기본 True (불균형 데이터에서 정보 손실 방지).
    """
    def __init__(self, labels, batch_size, num_classes=None, seed=42,
                 drop_last=True, over_sample=True):
        labels = np.asarray(labels).astype(int).ravel()
        if num_classes is None:
            num_classes = int(labels.max()) + 1
        if batch_size % num_classes != 0:
            new_bs = (batch_size // num_classes) * num_classes
            if new_bs == 0:
                raise ValueError(
                    f"batch_size({batch_size}) < num_classes({num_classes}); 늘려주세요."
                )
            print(f"[BalancedBatchSampler] batch_size {batch_size} → {new_bs} "
                  f"(num_classes={num_classes} 의 배수로 조정)")
            batch_size = new_bs

        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.per_class = batch_size // num_classes
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.over_sample = over_sample

        self.class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        counts = [len(ci) for ci in self.class_indices]
        if min(counts) == 0:
            raise ValueError(f"빈 클래스가 있습니다: counts={counts}")

        if over_sample:
            ref_count = max(counts)
        else:
            ref_count = min(counts)
        self.num_batches = ref_count // self.per_class
        if not drop_last and ref_count % self.per_class != 0:
            self.num_batches += 1

        print(f"[BalancedBatchSampler] class_counts={counts} "
              f"per_class_per_batch={self.per_class} batches/epoch={self.num_batches} "
              f"({'over-sample' if over_sample else 'under-sample'})")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        per_class_needed = self.num_batches * self.per_class
        shuffled = []
        for ci in self.class_indices:
            if self.over_sample and len(ci) < per_class_needed:
                tiles = []
                remaining = per_class_needed
                while remaining > 0:
                    perm = g.permutation(ci)
                    tiles.append(perm[:remaining])
                    remaining -= len(perm)
                shuffled.append(np.concatenate(tiles)[:per_class_needed])
            else:
                shuffled.append(g.permutation(ci)[:per_class_needed])

        for b in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                s = b * self.per_class
                e = s + self.per_class
                batch.extend(shuffled[c][s:e].tolist())
            g.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def make_balanced_sampler(dataset, batch_size, num_classes=None, seed=42,
                          drop_last=True, over_sample=True):
    """dataset.targets / dataset.labels 자동 추출 후 BalancedBatchSampler 생성."""
    targets = None
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            v = getattr(dataset, attr)
            if v is not None:
                targets = v
                break
    if targets is None:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return BalancedBatchSampler(targets, batch_size=batch_size,
                                num_classes=num_classes, seed=seed,
                                drop_last=drop_last, over_sample=over_sample)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=200)
    ap.add_argument("--n1", type=int, default=435)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    y = np.array([0]*args.n0 + [1]*args.n1)
    s = BalancedBatchSampler(y, batch_size=args.bs, num_classes=2, seed=0)
    print(f"len(sampler) = {len(s)}")
    seen = {0: 0, 1: 0}
    for i, batch in enumerate(s):
        cnt0 = int((y[batch] == 0).sum())
        cnt1 = int((y[batch] == 1).sum())
        seen[0] += cnt0
        seen[1] += cnt1
        if i < 3:
            print(f"  batch {i}: cls0={cnt0} cls1={cnt1}")
    print(f"epoch totals: {seen}")
