import os
import torch
import torch.nn.functional as F

_BPR_SAMPLE_WEIGHT = os.environ.get("BPR_SAMPLE_WEIGHT", "none").lower()
_BPR_FOCAL_GAMMA   = float(os.environ.get("BPR_FOCAL_GAMMA", "2.0"))


def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    per = F.softplus(-diff)
    if _BPR_SAMPLE_WEIGHT == "focal" and _BPR_FOCAL_GAMMA > 0 and per.numel() > 0:
        with torch.no_grad():
            p = torch.sigmoid(diff)
            w = (1.0 - p).clamp(min=1e-6) ** _BPR_FOCAL_GAMMA
            w = w / w.mean().clamp(min=1e-8)
        return (w * per).mean()
    return per.mean()


def geometric_median(X, eps=1e-5, max_iter=100):
    """Weiszfeld algorithm for geometric median of points X (N, D)."""
    if X.size(0) == 1:
        return X[0]
    y = X.mean(0)
    for _ in range(max_iter):
        d = torch.norm(X - y.unsqueeze(0), dim=1)
        d = torch.clamp(d, min=eps)
        w = 1.0 / d
        y_new = (w.unsqueeze(1) * X).sum(0) / w.sum()
        if torch.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


def sinkhorn_centroid(X, eps_reg=0.1, max_iter=50, tol=1e-5):
    """Entropy-regularized weighted centroid (Sinkhorn-flavored).

    Iteratively assigns weights w_i = softmax(-d_i / eps_reg) and
    computes center = sum(w_i * X_i). eps_reg controls smoothness:
    large values converge to arithmetic mean, small values toward geometric median.
    """
    if X.size(0) == 1:
        return X[0]
    y = X.mean(0)
    for _ in range(max_iter):
        d = torch.norm(X - y.unsqueeze(0), dim=1)
        w = torch.softmax(-d / max(eps_reg, 1e-8), dim=0)
        y_new = (w.unsqueeze(1) * X).sum(0)
        if torch.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


import os as _os
_SINKHORN_EPS = float(_os.environ.get("BPR_SINKHORN_EPS", "0.1"))


def _class_prototype(f_c, kind="mean"):
    """kind = 'mean' | 'geomedian' | 'sinkhorn'"""
    if kind == "geomedian":
        return geometric_median(f_c).unsqueeze(0)
    if kind == "sinkhorn":
        return sinkhorn_centroid(f_c, eps_reg=_SINKHORN_EPS).unsqueeze(0)
    # default
    return f_c.mean(0, keepdim=True)


def bpr_prototype_loss(features, labels, num_classes=2, prototype="mean",
                       external_prototypes=None):
    """Prototype-based BPR loss.

    Args:
        external_prototypes: (C, D) tensor of precomputed global prototypes.
                             If None, prototypes are computed from the current batch.
    """
    if features.dim() != 2:
        features = features.view(features.size(0), -1)
    labels = labels.view(-1).to(features.device)

    use_ext = external_prototypes is not None
    if use_ext:
        ext = external_prototypes.to(features.device)
        if ext.size(0) != num_classes:
            raise ValueError(
                f"external_prototypes first dim {ext.size(0)} != num_classes {num_classes}"
            )

    losses = []
    for c in range(num_classes):
        mask_c = (labels == c)
        mask_o = (labels != c)
        if mask_c.sum() < 1:
            continue
        f_c = features[mask_c]
        if use_ext:
            other_mask = torch.ones(num_classes, dtype=torch.bool, device=ext.device)
            other_mask[c] = False
            center_c = ext[c:c+1]
            center_o = ext[other_mask].mean(0, keepdim=True)
        else:
            if mask_c.sum() < 2 or mask_o.sum() < 1:
                continue
            f_o = features[mask_o]
            center_c = _class_prototype(f_c, kind=prototype)
            center_o = _class_prototype(f_o, kind=prototype)
        pos = (f_c * center_c).sum(-1)
        neg = (f_c * center_o).sum(-1)
        losses.append(bpr_loss(pos, neg))
    if not losses:
        return features.new_zeros(())
    return sum(losses) / len(losses)


def bpr_adversarial_perturb(features, labels, num_classes=2, max_eps_ratio=0.1,
                            prototype="mean", external_prototypes=None,
                            num_adv_steps=1):
    """Perturb latent features in the direction of BPR loss gradient (PGD-style)."""
    y = features.detach()
    for _ in range(max(int(num_adv_steps), 1)):
        y = y.detach().requires_grad_(True)
        loss = bpr_prototype_loss(y, labels, num_classes=num_classes,
                                  prototype=prototype,
                                  external_prototypes=external_prototypes)
        if not torch.is_tensor(loss) or loss.item() == 0:
            return y.detach()
        grad_y = torch.autograd.grad(loss, y)[0]
        eps_max = grad_y.std() * max_eps_ratio
        grad_norm = grad_y.norm(p=2, dim=1)
        eps_i = eps_max / (grad_norm + 1e-8)
        y = (y + eps_i.view(-1, 1) * grad_y.sign()).detach()
    return y


def total_bpr_loss(features, labels, num_classes=2, use_adversarial=True,
                   prototype="mean", external_prototypes=None,
                   num_adv_steps=1):
    """Total BPR loss: clean + adversarially perturbed features."""
    loss = bpr_prototype_loss(features, labels, num_classes=num_classes,
                              prototype=prototype, external_prototypes=external_prototypes)
    if use_adversarial:
        with torch.no_grad():
            r_adv = bpr_adversarial_perturb(features, labels, num_classes=num_classes,
                                            prototype=prototype,
                                            external_prototypes=external_prototypes,
                                            num_adv_steps=num_adv_steps) - features.detach()
        z_adv = features + r_adv
        loss = loss + 0.1 * bpr_prototype_loss(z_adv, labels, num_classes=num_classes,
                                         prototype=prototype,
                                         external_prototypes=external_prototypes)
    return loss
