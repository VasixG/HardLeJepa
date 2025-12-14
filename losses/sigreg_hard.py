import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def collinearity_metrics(A: torch.Tensor) -> dict:
    A = F.normalize(A, dim=0, eps=1e-12)
    G = A.t() @ A
    K = G.size(0)
    I = torch.eye(K, device=G.device, dtype=G.dtype)
    off = (G - I).abs()
    off_flat = off[~I.bool()]

    if off_flat.numel() == 0:
        return {
            "dir_cos_abs_mean": 0.0,
            "dir_cos_abs_max": 0.0,
            "dir_cos_abs_p95": 0.0,
            "dir_cos_abs_frac_gt_0p9": 0.0,
            "dir_cos_abs_frac_gt_0p99": 0.0,
            "dir_gram_cond": 0.0,
            "dir_eff_rank": 0.0,
        }

    mean_abs = float(off_flat.mean().item())
    max_abs = float(off_flat.max().item())
    p95_abs = float(torch.quantile(off_flat, 0.95).item())
    frac_gt_0p9 = float((off_flat > 0.90).float().mean().item())
    frac_gt_0p99 = float((off_flat > 0.99).float().mean().item())

    evals = torch.linalg.eigvalsh(G).clamp_min(1e-12)
    cond = float((evals.max() / evals.min()).item())
    p = (evals / evals.sum()).clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())

    return {
        "dir_cos_abs_mean": mean_abs,
        "dir_cos_abs_max": max_abs,
        "dir_cos_abs_p95": p95_abs,
        "dir_cos_abs_frac_gt_0p9": frac_gt_0p9,
        "dir_cos_abs_frac_gt_0p99": frac_gt_0p99,
        "dir_gram_cond": cond,
        "dir_eff_rank": eff_rank,
    }


def orth_penalty(A: torch.Tensor) -> torch.Tensor:
    A = F.normalize(A, dim=0, eps=1e-12)
    K = A.size(1)
    G = A.t() @ A
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    return ((G - I) ** 2).mean()


class SIGRegHard(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hard_slices: int,
        t_min: float,
        t_max: float,
        t_steps: int,
        mode: str = "learnable",
        standardize_proj: bool = True,
        ortho_lambda: float = 0.05,
        init_ortho: bool = True,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hard_slices = int(hard_slices)
        self.mode = str(mode)
        self.standardize_proj = bool(standardize_proj)
        self.ortho_lambda = float(ortho_lambda)

        t = torch.linspace(float(t_min), float(t_max), int(t_steps), dtype=torch.float32)
        self.register_buffer("t", t)
        phi = torch.exp(-0.5 * t * t)
        self.register_buffer("phi", phi)

        dt = float(t_max - t_min) / float(max(1, t_steps - 1))
        w = torch.full((int(t_steps),), dt, dtype=torch.float32)
        w[0] *= 0.5
        w[-1] *= 0.5
        w = w * phi
        self.register_buffer("weights", w)

        A0 = torch.randn(self.emb_dim, self.hard_slices, dtype=torch.float32)
        A0 = A0 / (A0.norm(dim=0, keepdim=True) + 1e-12)
        if init_ortho and self.hard_slices <= self.emb_dim:
            q, _ = torch.linalg.qr(A0, mode="reduced")
            A0 = q[:, : self.hard_slices].contiguous()
        self.A_raw = nn.Parameter(A0)

    def dirs(self) -> torch.Tensor:
        return F.normalize(self.A_raw, dim=0, eps=1e-12)

    def _ep_stat_per_dir(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        u = z @ A
        if self.standardize_proj:
            u = u - u.mean(dim=0, keepdim=True)
            u = u / (u.std(dim=0, keepdim=True, unbiased=False) + 1e-6)

        x_t = u.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(dim=0)
        sin_mean = torch.sin(x_t).mean(dim=0)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        stat = (err @ self.weights) * u.size(0)
        return stat

    def forward(self, z: torch.Tensor, seed: int = 0, detach_z: bool = False, detach_A: bool = False):
        z = z.float()
        if detach_z:
            z = z.detach()

        A = self.dirs()
        if detach_A:
            A = A.detach()

        stat_per_dir = self._ep_stat_per_dir(z, A)
        loss = stat_per_dir.mean()

        met = {}
        with torch.no_grad():
            met["stat_dir_mean"] = float(stat_per_dir.mean().item())
            met["stat_dir_max"] = float(stat_per_dir.max().item())
            met["stat_dir_p95"] = float(torch.quantile(stat_per_dir, 0.95).item()) if stat_per_dir.numel() > 1 else float(stat_per_dir.item())
            met["dir_ortho_pen"] = float(orth_penalty(A).item())
            met.update(collinearity_metrics(A))

        return loss, met

    def dir_objective(self, z_detached: torch.Tensor):
        A = self.dirs()
        stat_per_dir = self._ep_stat_per_dir(z_detached.float(), A)
        stat = stat_per_dir.mean()
        pen = orth_penalty(A)
        return (-stat) + self.ortho_lambda * pen
