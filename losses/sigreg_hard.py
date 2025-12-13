import math
from typing import Dict, Tuple
import torch
import torch.nn as nn

def _sigreg_slice_losses(z: torch.Tensor, A: torch.Tensor, t: torch.Tensor, phi_gauss: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    u = z @ A
    u_exp = u.unsqueeze(-1) * t.view(1, 1, -1)
    ecf = torch.exp(1j * u_exp).mean(dim=0)
    phi = phi_gauss.view(1, -1)
    diff2 = (ecf.real - phi).pow(2) + (ecf.imag).pow(2)
    integrand = diff2 * phi
    return torch.trapezoid(integrand, t, dim=-1)

def _direction_metrics(A: torch.Tensor) -> Dict[str, float]:
    M = A.size(1)
    if M <= 1:
        return {"dir_cos_abs_mean": 0.0, "dir_cos_abs_max": 0.0, "dir_eff_rank": float(M), "dir_sv_ratio": 1.0}
    G = (A.t() @ A).clamp(-1.0, 1.0)
    off = G - torch.diag(torch.diag(G))
    abs_off = off.abs()
    mean_abs = abs_off.mean().item()
    max_abs = abs_off.max().item()
    s = torch.linalg.svdvals(A).clamp_min(1e-12)
    p = s / s.sum()
    H = -(p * (p + 1e-12).log()).sum()
    eff = H.exp().item()
    sv_ratio = (s[0] / s[-1]).item() if s.numel() > 1 else 1.0
    return {"dir_cos_abs_mean": mean_abs, "dir_cos_abs_max": max_abs, "dir_eff_rank": eff, "dir_sv_ratio": sv_ratio}

def _gs_select_from_ranked(A_pool: torch.Tensor, ranked_idx: torch.Tensor, m: int, eps: float = 1e-6) -> torch.Tensor:
    D, P = A_pool.shape
    selected = []
    for idx in ranked_idx.tolist():
        v = A_pool[:, idx]
        for u in selected:
            v = v - (u @ v) * u
        n = v.norm()
        if n > eps:
            selected.append(v / n)
        if len(selected) >= m:
            break
    if len(selected) == 0:
        v0 = A_pool[:, ranked_idx[0]].clone()
        selected = [v0 / (v0.norm() + 1e-12)]
    if len(selected) < m:
        g = torch.Generator(device=A_pool.device)
        g.manual_seed(int(ranked_idx[0].item()) + 1337)
        while len(selected) < m:
            v = torch.randn(D, generator=g, device=A_pool.device)
            for u in selected:
                v = v - (u @ v) * u
            n = v.norm()
            if n > eps:
                selected.append(v / n)
    return torch.stack(selected, dim=1)

def _orthogonalize_random(D: int, n: int, device, seed: int, against: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    A = torch.randn(D, n, generator=g, device=device)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
    if against is None or against.numel() == 0:
        return A
    U = against
    out = []
    for j in range(n):
        v = A[:, j]
        v = v - U @ (U.t() @ v)
        nn = v.norm()
        if nn > eps:
            out.append(v / nn)
        else:
            v2 = torch.randn(D, generator=g, device=device)
            v2 = v2 - U @ (U.t() @ v2)
            out.append(v2 / (v2.norm() + 1e-12))
    return torch.stack(out, dim=1)

class SIGRegHard(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        pool_size: int,
        hard_slices: int,
        random_extra: int,
        mode: str,
        t_min: float = -5.0,
        t_max: float = 5.0,
        t_steps: int = 17,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pool_size = pool_size
        self.hard_slices = hard_slices
        self.random_extra = random_extra
        self.mode = mode.lower()
        t = torch.linspace(t_min, t_max, t_steps)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi_gauss", torch.exp(-0.5 * (t ** 2)), persistent=False)

    def forward(self, z: torch.Tensor, seed: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, D = z.shape
        if D != self.emb_dim:
            raise ValueError("bad embedding dim")
        g = torch.Generator(device=z.device)
        g.manual_seed(int(seed))
        A_pool = torch.randn(D, self.pool_size, generator=g, device=z.device)
        A_pool = A_pool / (A_pool.norm(dim=0, keepdim=True) + 1e-12)
        slice_losses = _sigreg_slice_losses(z, A_pool, self.t, self.phi_gauss)
        ranked = torch.argsort(slice_losses, descending=True)
        hard_raw = A_pool[:, ranked[: self.hard_slices]]
        if self.mode == "hardmaxqr":
            hard = _gs_select_from_ranked(A_pool, ranked, self.hard_slices)
        elif self.mode == "hardqr":
            hard = hard_raw
        else:
            raise ValueError("unknown mode")
        metrics = {}
        metrics.update({f"pool_loss_mean": slice_losses.mean().item(), f"pool_loss_max": slice_losses.max().item()})
        metrics.update({f"hard_loss_mean": slice_losses[ranked[: self.hard_slices]].mean().item(), f"hard_loss_max": slice_losses[ranked[: self.hard_slices]].max().item()})
        metrics.update({f"hard_raw_dir_cos_abs_mean": _direction_metrics(hard_raw)["dir_cos_abs_mean"], f"hard_raw_dir_cos_abs_max": _direction_metrics(hard_raw)["dir_cos_abs_max"]})
        metrics.update({f"hard_dir_cos_abs_mean": _direction_metrics(hard)["dir_cos_abs_mean"], f"hard_dir_cos_abs_max": _direction_metrics(hard)["dir_cos_abs_max"], f"hard_dir_eff_rank": _direction_metrics(hard)["dir_eff_rank"], f"hard_dir_sv_ratio": _direction_metrics(hard)["dir_sv_ratio"]})
        A_final = hard
        if self.random_extra > 0:
            rand = _orthogonalize_random(D, self.random_extra, z.device, seed + 9991, against=hard if self.mode == "hardmaxqr" else None)
            A_final = torch.cat([hard, rand], dim=1)
            dm_all = _direction_metrics(A_final)
            metrics.update({f"all_dir_cos_abs_mean": dm_all["dir_cos_abs_mean"], f"all_dir_cos_abs_max": dm_all["dir_cos_abs_max"], f"all_dir_eff_rank": dm_all["dir_eff_rank"], f"all_dir_sv_ratio": dm_all["dir_sv_ratio"]})
        else:
            dm_all = _direction_metrics(A_final)
            metrics.update({f"all_dir_cos_abs_mean": dm_all["dir_cos_abs_mean"], f"all_dir_cos_abs_max": dm_all["dir_cos_abs_max"], f"all_dir_eff_rank": dm_all["dir_eff_rank"], f"all_dir_sv_ratio": dm_all["dir_sv_ratio"]})
        final_slice_losses = _sigreg_slice_losses(z, A_final, self.t, self.phi_gauss)
        loss = final_slice_losses.mean() * B
        metrics.update({f"sigreg_final_mean": final_slice_losses.mean().item(), f"sigreg_final_max": final_slice_losses.max().item()})
        return loss, metrics
