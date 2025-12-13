import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def grad_global_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.pow(2).sum().item()
    return math.sqrt(total)

@torch.no_grad()
def pairwise_cos_stats(z: torch.Tensor, max_pairs: int = 4096) -> Tuple[float, float]:
    B = z.size(0)
    if B < 2:
        return 0.0, 0.0
    z = F.normalize(z, dim=-1)
    n = min(max_pairs, B * (B - 1) // 2)
    i = torch.randint(0, B, (n,), device=z.device)
    j = torch.randint(0, B, (n,), device=z.device)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = (z[i] * z[j]).sum(dim=-1)
    return cos.mean().item(), cos.std(unbiased=False).item()

@torch.no_grad()
def cov_metrics(z: torch.Tensor) -> Dict[str, float]:
    B, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False) + 1e-12
    std = var.sqrt()
    cov = (z.T @ z) / max(B - 1, 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    return {
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "var_mean": var.mean().item(),
        "var_min": var.min().item(),
        "cov_diag_mean": torch.diag(cov).mean().item(),
        "cov_offdiag_abs_mean": offdiag.abs().mean().item(),
    }

@torch.no_grad()
def spectrum_metrics(z: torch.Tensor, max_dim: int = 512) -> Dict[str, float]:
    B, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    if D > max_dim:
        R = torch.randn(D, max_dim, device=z.device) / math.sqrt(max_dim)
        z = z @ R
    s = torch.linalg.svdvals(z).clamp_min(1e-12)
    p = s / s.sum()
    H = -(p * (p + 1e-12).log()).sum()
    eff_rank = H.exp().item()
    return {
        "effective_rank": eff_rank,
        "sv_ratio": (s[0] / s[-1]).item() if s.numel() > 1 else 1.0,
        "sv_max": s[0].item(),
        "sv_min": s[-1].item(),
    }

@torch.no_grad()
def alignment_uniformity(z1: torch.Tensor, z2: torch.Tensor = None, t: float = 2.0) -> Dict[str, float]:
    z1 = F.normalize(z1, dim=-1)
    out = {}
    if z2 is not None:
        z2 = F.normalize(z2, dim=-1)
        out["alignment"] = ((z1 - z2).pow(2).sum(dim=-1)).mean().item()
    B = z1.size(0)
    if B >= 2:
        n = min(4096, B * (B - 1) // 2)
        i = torch.randint(0, B, (n,), device=z1.device)
        j = torch.randint(0, B, (n,), device=z1.device)
        mask = i != j
        i, j = i[mask], j[mask]
        d2 = (z1[i] - z1[j]).pow(2).sum(dim=-1)
        out["uniformity"] = torch.log(torch.exp(-t * d2).mean() + 1e-12).item()
    else:
        out["uniformity"] = 0.0
    return out
