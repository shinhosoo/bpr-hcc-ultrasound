"""MGDA (Multiple Gradient Descent Algorithm) — two-task closed-form solver.

Reference: Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization"
NeurIPS 2018. Section 3.2 — analytical solution for 2 tasks.

Idea:
  공유 parameters 에 대한 두 loss 의 gradient g1, g2 가 있을 때
  gradient = α·g1 + (1-α)·g2 의 norm 을 최소화하는 α* 를 찾고
  그 결합 gradient 로 update.

  α* 의 closed form (2 task):
       g2·(g2 - g1)
  α* = ───────────────         (clip to [0, 1])
       ‖g1 - g2‖²
"""
import torch


def _flatten_grads(grads):
    return torch.cat([g.contiguous().view(-1) for g in grads])


def mgda_two_task_alpha(g1_flat, g2_flat, eps=1e-12):
    """두 flat gradient 로부터 closed-form α 계산. clip to [0, 1]."""
    diff = g2_flat - g1_flat
    denom = (diff * diff).sum()
    if denom.item() < eps:
        return 0.5
    alpha = (g2_flat * diff).sum() / denom
    return float(alpha.clamp(0.0, 1.0).item())


def mgda_step(loss1, loss2, params, optimizer):
    """두 loss 에 대해 MGDA 가중치를 구해 optimizer.step 수행.

    loss1, loss2 : torch.Tensor scalar
    params       : update 대상 parameter list (보통 model.parameters())
    optimizer    : torch optimizer (zero_grad / step 호출 우리가 함)

    반환: dict {alpha, l1, l2, combined_norm}
    """
    params = [p for p in params if p.requires_grad]
    g1 = torch.autograd.grad(loss1, params, retain_graph=True, create_graph=False, allow_unused=True)
    g2 = torch.autograd.grad(loss2, params, retain_graph=False, create_graph=False, allow_unused=True)

    g1 = [torch.zeros_like(p) if g is None else g for p, g in zip(params, g1)]
    g2 = [torch.zeros_like(p) if g is None else g for p, g in zip(params, g2)]

    g1_flat = _flatten_grads(g1)
    g2_flat = _flatten_grads(g2)
    alpha = mgda_two_task_alpha(g1_flat, g2_flat)

    optimizer.zero_grad()
    for p, gi, gj in zip(params, g1, g2):
        p.grad = alpha * gi + (1.0 - alpha) * gj
    optimizer.step()
    return {
        "alpha": alpha,
        "l1": float(loss1.item()),
        "l2": float(loss2.item()),
    }
