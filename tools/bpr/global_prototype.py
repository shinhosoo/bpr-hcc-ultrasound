"""Global class-prototype manager.

Computes per-class prototypes (mean / geomedian / sinkhorn) over the full
training set and caches them for use as external_prototypes in bpr_prototype_loss.
Prototypes are recomputed each epoch as features evolve during training.
"""
from __future__ import annotations
import os as _os
import torch

from bpr_loss import geometric_median, sinkhorn_centroid


_SINKHORN_EPS = float(_os.environ.get("BPR_SINKHORN_EPS", "0.1"))


def _aggregate(X, kind):
    if X.size(0) == 0:
        return None
    if X.size(0) == 1:
        return X[0]
    if kind == "geomedian":
        return geometric_median(X)
    if kind == "sinkhorn":
        return sinkhorn_centroid(X, eps_reg=_SINKHORN_EPS)
    return X.mean(0)


class GlobalPrototypeManager:
    """Computes and caches per-class prototypes from the full training set.

    Args:
        num_classes: number of classes.
        kind: 'mean' | 'geomedian' | 'sinkhorn'.
        max_per_class: max samples per class for memory efficiency (None = unlimited).
        device: device to store prototypes on. Inferred from first refresh if None.
        ema: EMA smoothing factor. 0.0 = recompute each time; (0, 1) = EMA blend.
    """

    def __init__(self, num_classes=2, kind="geomedian",
                 max_per_class=None, device=None, ema=0.0):
        self.num_classes = num_classes
        self.kind = kind
        self.max_per_class = max_per_class
        self.device = device
        self.ema = float(ema)
        self.prototypes = None
        self.refresh_count = 0

    @torch.no_grad()
    def refresh(self, feature_fn, dataloader, device=None, projection_fn=None,
                l2_normalize=True):
        """Recompute prototypes by running one forward pass over the dataloader.

        Args:
            feature_fn(x) -> (B, D): image -> feature vector (e.g., model.forward_features).
            dataloader: typically a sequential (non-shuffled) loader.
            device: tensor device; falls back to self.device if None.
            projection_fn(f) -> (B, D'): optional projection head.
            l2_normalize: if True, L2-normalize prototypes.
        """
        if device is None:
            device = self.device or torch.device("cpu")
        self.device = device

        per_class = {c: [] for c in range(self.num_classes)}
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get("image", batch.get("x"))
                y = batch.get("label", batch.get("y"))
            else:
                continue
            x = x.to(device, non_blocking=True)
            if torch.is_tensor(y):
                y = y.view(-1).cpu().long()
            else:
                y = torch.tensor(y, dtype=torch.long)
            f = feature_fn(x)
            if projection_fn is not None:
                f = projection_fn(f)
            if l2_normalize:
                f = torch.nn.functional.normalize(f, dim=-1)
            f = f.detach()
            for c in range(self.num_classes):
                m = (y == c)
                if m.any():
                    per_class[c].append(f[m])

        new_protos = []
        for c in range(self.num_classes):
            if not per_class[c]:
                if self.prototypes is not None:
                    new_protos.append(self.prototypes[c])
                else:
                    raise RuntimeError(f"class {c} has 0 samples — refresh failed")
                continue
            X = torch.cat(per_class[c], dim=0)
            if self.max_per_class is not None and X.size(0) > self.max_per_class:
                idx = torch.randperm(X.size(0), device=X.device)[:self.max_per_class]
                X = X[idx]
            proto = _aggregate(X, self.kind)
            if l2_normalize:
                proto = torch.nn.functional.normalize(proto, dim=-1)
            new_protos.append(proto)

        new_protos = torch.stack(new_protos, dim=0)
        if self.ema > 0 and self.prototypes is not None:
            new_protos = (1 - self.ema) * new_protos + self.ema * self.prototypes
            if l2_normalize:
                new_protos = torch.nn.functional.normalize(new_protos, dim=-1)
        self.prototypes = new_protos.to(device)
        self.refresh_count += 1
        return self.prototypes

    def get(self, c):
        if self.prototypes is None:
            raise RuntimeError("refresh() has not been called")
        return self.prototypes[c]

    def __repr__(self):
        d = "?" if self.prototypes is None else f"{self.prototypes.size(-1)}"
        return (f"GlobalPrototypeManager(C={self.num_classes}, kind='{self.kind}', "
                f"D={d}, ema={self.ema}, refresh#={self.refresh_count})")
