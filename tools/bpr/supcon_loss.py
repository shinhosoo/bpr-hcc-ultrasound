"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

원논문: "Supervised Contrastive Learning" (https://arxiv.org/abs/2004.11362)

핵심 식 — anchor i 에 대해 P(i) = batch 안의 같은 클래스 sample 집합 (i 자신 제외):
    L_i = -1/|P(i)| · Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )

BPR 와의 차이:
  - BPR  : class prototype (centroid) 과의 거리 기반 — centroid-aware
  - SupCon: 모든 batch instance 쌍 기반 — pair-aware
  - 둘이 같은 projected feature 위에서 합산되면 보완적
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor,
                labels: torch.Tensor,
                temperature: float = 0.1,
                base_temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss.

    Args:
        features:  (B, D) tensor — 이미 L2 normalize 돼 있다고 가정 (호출 측에서 처리).
                   normalize 안 돼 있어도 작동하지만 dot product 가 norm 으로 흔들림.
        labels:    (B,) long tensor — class index.
        temperature: NT-Xent 의 τ. 작을수록 hard 한 contrastive, 클수록 soft.
                     0.07-0.5 사이가 보통.
        base_temperature: scale factor (원논문 detail). 기본은 τ 와 같게.

    Returns:
        scalar tensor — batch 안의 평균 loss.
        Batch 안에 같은 클래스 쌍이 하나도 없으면 0 반환.
    """
    if features.dim() != 2:
        features = features.view(features.size(0), -1)
    device = features.device
    B = features.size(0)
    if B < 2:
        return features.new_zeros(())

    labels = labels.contiguous().view(-1, 1)
    if labels.size(0) != B:
        raise ValueError(f"labels({labels.size(0)}) != features({B}) batch size")

    # mask[i,j] = 1 if labels[i]==labels[j] (same class)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / max(temperature, 1e-8)

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    self_mask = torch.eye(B, dtype=torch.float32, device=device)
    pos_mask = mask - self_mask
    non_self_mask = 1.0 - self_mask

    exp_logits = torch.exp(logits) * non_self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return features.new_zeros(())

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)
    mean_log_prob_pos = mean_log_prob_pos[valid] / pos_count[valid]

    loss = -(temperature / max(base_temperature, 1e-8)) * mean_log_prob_pos
    return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    z = F.normalize(torch.randn(8, 64), dim=-1)
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    l = supcon_loss(z, y, temperature=0.1)
    print(f"loss = {l.item():.4f}")
    assert torch.isfinite(l)
    y_same = torch.zeros(8, dtype=torch.long)
    l_same = supcon_loss(z, y_same, temperature=0.1)
    print(f"all same class loss = {l_same.item():.4f}")
    y_uniq = torch.arange(8)
    l_uniq = supcon_loss(z, y_uniq, temperature=0.1)
    print(f"no positive pairs loss = {l_uniq.item():.4f}  (should be 0)")
