"""원본 모델 학습 코드를 변경하지 않고 BPR 표현학습 효과를 확인하는 도구.

흐름:
  1) tools/extract_features.py 로 학습된 모델에서 train + test latent 추출
  2) 이 도구가 그 latent 위에 projection head + classifier 를 BPR loss + CE 로 학습
  3) test latent 에 적용해 predictions npz 저장 (baseline 과 동일 포맷)
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bpr_loss import total_bpr_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-train", required=True, help="train features npz")
    ap.add_argument("--features-test",  required=True, help="test features npz")
    ap.add_argument("--features-val",   default=None,
                    help="(optional) val features npz — best epoch 선택용. 없으면 train 내부 분할")
    ap.add_argument("--val-split-ratio", type=float, default=0.15,
                    help="--features-val 미지정 시 train 에서 holdout 비율 (기본 0.15)")
    ap.add_argument("--out",            required=True, help="predictions_test_<model>_bpr.npz")
    ap.add_argument("--lam", type=float, default=0.3, help="BPR loss weight")
    ap.add_argument("--use-adv", action="store_true", help="adversarial perturbation 도 사용")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--prototype", choices=["mean", "geomedian", "sinkhorn"], default="mean",
                    help="class prototype 추정 방식 (mean | geomedian)")
    ap.add_argument("--no-bpr", action="store_true",
                    help="BPR 끄기 — baseline 의 head refinement 만 (sanity check)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_tr = np.load(a.features_train, allow_pickle=True)
    d_te = np.load(a.features_test,  allow_pickle=True)
    X_tr_all = torch.from_numpy(d_tr["features"]).float().to(device)
    y_tr_all = torch.from_numpy(d_tr["labels"].astype(np.int64)).to(device)
    X_te = torch.from_numpy(d_te["features"]).float().to(device)
    y_te = torch.from_numpy(d_te["labels"].astype(np.int64)).to(device)

    if a.features_val is not None and os.path.exists(a.features_val):
        d_va = np.load(a.features_val, allow_pickle=True)
        X_va = torch.from_numpy(d_va["features"]).float().to(device)
        y_va = torch.from_numpy(d_va["labels"].astype(np.int64)).to(device)
        X_tr, y_tr = X_tr_all, y_tr_all
        print(f"[refine_head] using external val npz: {a.features_val}  N={len(y_va)}")
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(X_tr_all.size(0))
        idx_tr, idx_va = train_test_split(idx, test_size=a.val_split_ratio,
                                          random_state=a.seed,
                                          stratify=y_tr_all.cpu().numpy())
        X_tr = X_tr_all[idx_tr]; y_tr = y_tr_all[idx_tr]
        X_va = X_tr_all[idx_va]; y_va = y_tr_all[idx_va]
        print(f"[refine_head] internal train/val split — "
              f"train={len(y_tr)} val={len(y_va)} (val ratio={a.val_split_ratio})")

    D = X_tr.size(1); K = a.num_classes
    print(f"[refine_head] train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}  dim={D}")
    print(f"  lambda_bpr={a.lam}  adv={a.use_adv}  epochs={a.epochs}  no_bpr={a.no_bpr}")

    # projection head + linear classifier
    projector = nn.Sequential(
        nn.Linear(D, D // 2), nn.ReLU(),
        nn.Linear(D // 2, a.proj_dim),
    ).to(device)
    classifier = nn.Linear(a.proj_dim, K).to(device)

    opt = torch.optim.Adam(
        list(projector.parameters()) + list(classifier.parameters()),
        lr=a.lr, weight_decay=1e-4)

    bs = min(a.batch_size, X_tr.size(0))
    n_train = X_tr.size(0)

    best_val_acc = 0.0
    best_state = None

    for ep in range(a.epochs):
        projector.train(); classifier.train()
        perm = torch.randperm(n_train, device=device)
        ce_sum = 0.0; bpr_sum = 0.0; n_batch = 0

        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            x = X_tr[idx]; y = y_tr[idx]

            z = F.normalize(projector(x), dim=-1)
            logits = classifier(z)
            ce = F.cross_entropy(logits, y)
            if a.no_bpr:
                bpr = torch.tensor(0.0, device=device)
            else:
                bpr = total_bpr_loss(z, y, num_classes=K, use_adversarial=a.use_adv, prototype=a.prototype)
            loss = ce + a.lam * bpr

            opt.zero_grad(); loss.backward(); opt.step()
            ce_sum += ce.item(); bpr_sum += float(bpr.item()); n_batch += 1

        projector.eval(); classifier.eval()
        with torch.no_grad():
            z_va = F.normalize(projector(X_va), dim=-1)
            logits_va = classifier(z_va)
            pred_va = logits_va.argmax(dim=1)
            val_acc = (pred_va == y_va).float().mean().item()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}: ce={ce_sum/n_batch:.4f}  bpr={bpr_sum/n_batch:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (
                {k: v.detach().clone() for k, v in projector.state_dict().items()},
                {k: v.detach().clone() for k, v in classifier.state_dict().items()},
            )

    if best_state is not None:
        projector.load_state_dict(best_state[0])
        classifier.load_state_dict(best_state[1])

    projector.eval(); classifier.eval()
    with torch.no_grad():
        z_te = F.normalize(projector(X_te), dim=-1)
        logits_te = classifier(z_te)
        prob = F.softmax(logits_te, dim=1).cpu().numpy()
    y_true = y_te.cpu().numpy().astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, y_true=y_true, y_score=prob.astype(float))
    print(f"\n[refine_head] saved: {a.out}  N={len(y_true)}  best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
