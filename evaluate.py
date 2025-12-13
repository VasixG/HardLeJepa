from typing import Dict, Tuple
import torch
from tqdm import tqdm
from metrics.clustering import clustering_metrics
from metrics.knn import knn_metrics

@torch.no_grad()
def extract_embeddings(encoder: torch.nn.Module, loader, device: str, max_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    zs, ys = [], []
    seen = 0
    for x, y in tqdm(loader, desc="extract", leave=False):
        x = x.to(device)
        z = encoder(x)
        zs.append(z.detach().cpu())
        ys.append(y.detach().cpu())
        seen += x.size(0)
        if seen >= max_samples:
            break
    z = torch.cat(zs, dim=0)[:max_samples]
    y = torch.cat(ys, dim=0)[:max_samples]
    return z, y

@torch.no_grad()
def eval_clustering(z: torch.Tensor, y: torch.Tensor, device: str, kmeans_iters: int, kmeans_restarts: int, silhouette_samples: int) -> Dict[str, float]:
    return clustering_metrics(z.to(device), y, num_classes=10, iters=kmeans_iters, restarts=kmeans_restarts, silhouette_samples=silhouette_samples)

@torch.no_grad()
def eval_knn(z_bank: torch.Tensor, y_bank: torch.Tensor, z_query: torch.Tensor, y_query: torch.Tensor, device: str, k: int, temperature: float, chunk_size: int) -> Tuple[Dict[str, float], torch.Tensor]:
    return knn_metrics(z_bank.to(device), y_bank.to(device), z_query.to(device), y_query.to(device), k=k, temperature=temperature, num_classes=10, chunk_size=chunk_size)
