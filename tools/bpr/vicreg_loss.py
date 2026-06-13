"""VICReg-style variance/covariance regularizers (Bardes et al., ICLR 2022)."""
import torch
import torch.nn.functional as F


def variance_loss(z, gamma=1.0, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)          # (D,)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    B, D = z.shape
    if B < 2:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (B - 1)                    # (D, D)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / D


def variance_loss_classwise(z, labels, num_classes=2, gamma=1.0, eps=1e-4):
    """Per-class variance hinge loss. labels: (N,) int."""
    z = z.float()
    labels = labels.view(-1)
    losses = []
    for c in range(num_classes):
        zc = z[labels == c]
        if zc.shape[0] < 2:
            continue
        std = torch.sqrt(zc.var(dim=0) + eps)
        losses.append(torch.mean(F.relu(gamma - std)))
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()
