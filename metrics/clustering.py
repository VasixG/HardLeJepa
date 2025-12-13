import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .hungarian import hungarian

@torch.no_grad()
def kmeans_torch(x: torch.Tensor, k: int, iters: int = 30, restarts: int = 3) -> Tuple[torch.Tensor, torch.Tensor, float]:
    x = x.float()
    N, D = x.shape
    best_inertia = float("inf")
    best_labels = None
    best_centers = None
    for _ in range(restarts):
        idx = torch.randperm(N, device=x.device)[:k]
        centers = x[idx].clone()
        for _it in range(iters):
            d2 = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
            labels = d2.argmin(dim=1)
            new_centers = []
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centers.append(x[mask].mean(dim=0))
                else:
                    new_centers.append(x[torch.randint(0, N, (1,), device=x.device)].squeeze(0))
            centers = torch.stack(new_centers, dim=0)
        d2 = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
        labels = d2.argmin(dim=1)
        inertia = d2.min(dim=1).values.sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centers = centers.clone()
    return best_labels, best_centers, best_inertia

def _contingency(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    C = num_classes
    K = num_classes
    M = torch.zeros((C, K), dtype=torch.long)
    for c in range(C):
        for k in range(K):
            M[c, k] = ((y_true == c) & (y_pred == k)).sum()
    return M

def clustering_purity(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    M = _contingency(y_true, y_pred, num_classes)
    return (M.max(dim=0).values.sum().float() / M.sum().float()).item()

def clustering_accuracy_hungarian(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    M = _contingency(y_true, y_pred, num_classes).cpu().numpy()
    cost = (-M).tolist()
    assign, _ = hungarian(cost)
    correct = 0
    for c, k in enumerate(assign):
        correct += M[c][k]
    return float(correct) / float(M.sum())

def nmi(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    M = _contingency(y_true, y_pred, num_classes).float()
    N = M.sum()
    if N.item() == 0:
        return 0.0
    p_y = M.sum(dim=1) / N
    p_c = M.sum(dim=0) / N
    p_yc = M / N
    eps = 1e-12
    denom = (p_y.unsqueeze(1) * p_c.unsqueeze(0)).clamp_min(eps)
    mi = (p_yc.clamp_min(eps) * (p_yc.clamp_min(eps) / denom).log()).sum()
    Hy = -(p_y.clamp_min(eps) * p_y.clamp_min(eps).log()).sum()
    Hc = -(p_c.clamp_min(eps) * p_c.clamp_min(eps).log()).sum()
    return (mi / (torch.sqrt(Hy * Hc) + eps)).item()

def ari(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    M = _contingency(y_true, y_pred, num_classes).float()
    n = M.sum()
    if n.item() <= 1:
        return 0.0
    def comb2(x):
        return x * (x - 1) / 2
    sum_comb = comb2(M).sum()
    a = M.sum(dim=1)
    b = M.sum(dim=0)
    sum_a = comb2(a).sum()
    sum_b = comb2(b).sum()
    total = comb2(n)
    expected = (sum_a * sum_b) / (total + 1e-12)
    max_index = 0.5 * (sum_a + sum_b)
    return ((sum_comb - expected) / (max_index - expected + 1e-12)).item()

@torch.no_grad()
def silhouette_score(x: torch.Tensor, labels: torch.Tensor, max_samples: int = 2000) -> float:
    x = F.normalize(x.float(), dim=-1)
    N = x.size(0)
    if N < 3:
        return 0.0
    S = min(max_samples, N)
    idx = torch.randperm(N, device=x.device)[:S]
    xs = x[idx]
    ls = labels[idx]
    cos = xs @ xs.T
    d2 = (2.0 - 2.0 * cos).clamp_min(0.0)
    sil = []
    uniq = ls.unique()
    for i in range(S):
        same = ls == ls[i]
        if same.sum() > 1:
            a = d2[i][same].sum() / (same.sum() - 1)
        else:
            a = torch.tensor(0.0, device=x.device)
        b = torch.tensor(float("inf"), device=x.device)
        for c in uniq:
            if c == ls[i]:
                continue
            mask = ls == c
            if mask.any():
                b = torch.minimum(b, d2[i][mask].mean())
        s = (b - a) / (torch.maximum(a, b) + 1e-12)
        sil.append(s)
    return torch.stack(sil).mean().item()

@torch.no_grad()
def clustering_metrics(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
    iters: int = 30,
    restarts: int = 3,
    silhouette_samples: int = 2000,
) -> Dict[str, float]:
    z = z.detach()
    y = y.detach().cpu()
    labels, _centers, inertia = kmeans_torch(z, k=num_classes, iters=iters, restarts=restarts)
    labels_cpu = labels.detach().cpu()
    return {
        "kmeans_inertia": float(inertia),
        "purity": clustering_purity(y, labels_cpu, num_classes),
        "cluster_acc": clustering_accuracy_hungarian(y, labels_cpu, num_classes),
        "nmi": nmi(y, labels_cpu, num_classes),
        "ari": ari(y, labels_cpu, num_classes),
        "silhouette": silhouette_score(z, labels, max_samples=silhouette_samples),
    }
