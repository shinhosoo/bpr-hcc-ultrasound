"""Pairwise AUC-surrogate — 최종 점수 랭킹을 직접 최적화 (BPR/VICReg 와 결이 다름).

AUC = P(score(pos) > score(neg)) 를 미분가능하게 근사한다(Wilcoxon-Mann-Whitney surrogate).
모든 (양성, 음성) 쌍에서 양성 점수가 음성보다 높도록 민다.
  - logistic(기본): softplus(-(s_pos - s_neg))  — smooth WMW, scale-robust
  - hinge       : relu(margin - (s_pos - s_neg))^2

scores: (N,) 양성클래스 점수/확률, labels: (N,) 0/1.
"""
import torch
import torch.nn.functional as F


def auc_surrogate(scores, labels, mode="logistic", margin=0.1):
    scores = scores.float().view(-1)
    labels = labels.view(-1)
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_tensor(0.0)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    if mode == "hinge":
        return F.relu(margin - diff).pow(2).mean()
    # logistic (default)
    return F.softplus(-diff).mean()
