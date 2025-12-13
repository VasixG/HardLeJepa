from typing import Dict, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def knn_predict(
    z_bank: torch.Tensor,
    y_bank: torch.Tensor,
    z_query: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
    chunk_size: int = 2048,
) -> torch.Tensor:
    z_bank = F.normalize(z_bank, dim=-1)
    z_query = F.normalize(z_query, dim=-1)
    y_bank = y_bank.long()
    Nb = z_bank.size(0)
    preds = []
    for start in range(0, z_query.size(0), chunk_size):
        end = min(start + chunk_size, z_query.size(0))
        q = z_query[start:end]
        sim = q @ z_bank.T
        vals, idx = sim.topk(k=min(k, Nb), dim=1, largest=True, sorted=True)
        nn_labels = y_bank[idx]
        weights = torch.exp(vals / max(temperature, 1e-8))
        vote = torch.zeros((q.size(0), num_classes), device=z_bank.device)
        vote.scatter_add_(dim=1, index=nn_labels, src=weights)
        preds.append(vote.argmax(dim=1))
    return torch.cat(preds, dim=0)

@torch.no_grad()
def knn_metrics(
    z_bank: torch.Tensor,
    y_bank: torch.Tensor,
    z_query: torch.Tensor,
    y_query: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: int = 10,
    chunk_size: int = 2048,
) -> Tuple[Dict[str, float], torch.Tensor]:
    device = z_bank.device
    y_query = y_query.to(device).long()
    pred = knn_predict(z_bank, y_bank.to(device), z_query, k=k, temperature=temperature, num_classes=num_classes, chunk_size=chunk_size)
    acc1 = (pred == y_query).float().mean().item()
    pred5 = knn_predict(z_bank, y_bank.to(device), z_query, k=min(5, k), temperature=temperature, num_classes=num_classes, chunk_size=chunk_size)
    acc5 = (pred5 == y_query).float().mean().item()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for t, p in zip(y_query, pred):
        cm[t, p] += 1
    return {"knn_acc1": acc1, "knn_acc5": acc5}, cm.detach().cpu()
