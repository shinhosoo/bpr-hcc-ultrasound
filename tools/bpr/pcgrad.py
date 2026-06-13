"""PCGrad (Projecting Conflicting Gradients) — two-task gradient surgery.

Reference: Yu, Kumar, Gupta, Levine, Hausman, Finn,
"Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

핵심: 두 task gradient 가 서로 음의 cosine similarity 를 가지면 (충돌),
각각을 상대의 normal plane 으로 projection 해서 충돌 성분만 제거.
충돌 없으면 (g1·g2 ≥ 0) 그냥 합산.

두 task 의 경우:
    if g1·g2 < 0:
        g1_proj = g1 - (g1·g2 / ‖g2‖²) · g2
        g2_proj = g2 - (g2·g1 / ‖g1‖²) · g1
    else:
        g1_proj, g2_proj = g1, g2
    final = g1_proj + g2_proj
"""
import torch


def _flatten_grads(grads):
    return torch.cat([g.contiguous().view(-1) for g in grads])


def _unflatten_like(flat, like_list):
    out = []; idx = 0
    for t in like_list:
        n = t.numel()
        out.append(flat[idx:idx+n].view_as(t))
        idx += n
    return out


def pcgrad_step(loss1, loss2, params, optimizer):
    """두 loss 에 대해 PCGrad 결합 gradient 로 optimizer.step.

    loss1, loss2 : torch.Tensor scalar
    params       : update 대상 (보통 model.parameters())
    optimizer    : torch optimizer

    반환: dict { cos: cosine similarity, projected: 충돌 여부, l1, l2 }
    """
    params = [p for p in params if p.requires_grad]
    g1 = torch.autograd.grad(loss1, params, retain_graph=True, create_graph=False, allow_unused=True)
    g2 = torch.autograd.grad(loss2, params, retain_graph=False, create_graph=False, allow_unused=True)

    # None → 0
    g1 = [torch.zeros_like(p) if g is None else g for p, g in zip(params, g1)]
    g2 = [torch.zeros_like(p) if g is None else g for p, g in zip(params, g2)]

    g1_flat = _flatten_grads(g1)
    g2_flat = _flatten_grads(g2)

    dot = (g1_flat * g2_flat).sum()
    n1_sq = (g1_flat * g1_flat).sum().clamp(min=1e-12)
    n2_sq = (g2_flat * g2_flat).sum().clamp(min=1e-12)
    cos = (dot / (n1_sq.sqrt() * n2_sq.sqrt())).item()
    projected = bool(dot.item() < 0)

    if projected:
        g1p_flat = g1_flat - (dot / n2_sq) * g2_flat
        g2p_flat = g2_flat - (dot / n1_sq) * g1_flat
    else:
        g1p_flat = g1_flat
        g2p_flat = g2_flat

    final_flat = g1p_flat + g2p_flat
    final = _unflatten_like(final_flat, params)

    optimizer.zero_grad()
    for p, g in zip(params, final):
        p.grad = g.clone()
    optimizer.step()
    return {"cos": cos, "projected": projected,
            "l1": float(loss1.item()), "l2": float(loss2.item())}
