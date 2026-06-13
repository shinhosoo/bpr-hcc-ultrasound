"""VICReg-style variance/covariance regularizers — diffusion conditioning richness 보호용.

배경: BPR(분산 축소)을 conditioning 에 걸면 denoiser 가 쓰는 instance richness 가 붕괴된다.
VICReg(Bardes et al., ICLR 2022)의 variance/covariance 항을 BPR 과 함께 걸면,
BPR 이 클래스를 분리(attraction)하는 동안 variance 항이 per-dim 표준편차를 임계 이상으로
강제 유지하고 covariance 항이 차원 간 상관을 제거 → 분산을 죽이지 않고 분리한다.
즉 "목적 분리": BPR=분리, VICReg=분산 보존.
"""
import torch
import torch.nn.functional as F


def variance_loss(z, gamma=1.0, eps=1e-4):
    """per-dim 표준편차가 gamma 미만이면 hinge 패널티 → 분산(=instance richness) 유지."""
    std = torch.sqrt(z.var(dim=0) + eps)          # (D,)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """off-diagonal covariance^2 합 / D → 차원 간 탈상관(중복 제거)."""
    B, D = z.shape
    if B < 2:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (B - 1)                    # (D, D)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / D
