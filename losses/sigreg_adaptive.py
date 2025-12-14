import torch
import torch.nn as nn
import torch.nn.functional as F


def _trapz_weights(t: torch.Tensor) -> torch.Tensor:
    dt = (t[-1] - t[0]) / max(1, (t.numel() - 1))
    w = torch.full((t.numel(),), dt, dtype=t.dtype, device=t.device)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


class SIGRegAdaptive(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_slices: int,
        hard_slices: int,
        t_min: float = -5.0,
        t_max: float = 5.0,
        t_steps: int = 17,
        hard_start_step: int = 10**9,
        hard_ramp_steps: int = 0,
        hard_max_frac: float = 0.0,
        dir_ortho_init: bool = True,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_slices = int(num_slices)

        self.hard_slices = int(min(hard_slices, emb_dim))
        self.hard_start_step = int(hard_start_step)
        self.hard_ramp_steps = int(hard_ramp_steps)
        self.hard_max_frac = float(hard_max_frac)

        t = torch.linspace(float(t_min), float(t_max), int(t_steps), dtype=torch.float32)
        self.register_buffer("t", t, persistent=False)
        phi = torch.exp(-0.5 * (t ** 2))
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("w_trapz", _trapz_weights(t) * phi, persistent=False)

        if self.hard_slices > 0:
            A0 = torch.randn(self.emb_dim, self.hard_slices, dtype=torch.float32)
            A0 = A0 / (A0.norm(dim=0, keepdim=True) + 1e-12)
            if dir_ortho_init and self.hard_slices <= self.emb_dim:
                q, _ = torch.linalg.qr(A0, mode="reduced")
                A0 = q[:, : self.hard_slices].contiguous()
            self.A_raw = nn.Parameter(A0)
        else:
            self.A_raw = None

    @torch.no_grad()
    def qr_project_(self) -> float:
        if self.A_raw is None:
            return 0.0
        A_before = self.A_raw.data
        D, K = A_before.shape
        if K <= D:
            Q, _ = torch.linalg.qr(A_before, mode="reduced")
            A_after = Q[:, :K].contiguous()
        else:
            A_after = F.normalize(A_before, dim=0, eps=1e-12)
        delta = float((A_after - A_before).norm().item())
        self.A_raw.data.copy_(A_after)
        return delta

    def hard_frac(self, step: int) -> float:
        if self.hard_slices <= 0:
            return 0.0
        if step < self.hard_start_step:
            return 0.0
        if self.hard_ramp_steps <= 0:
            return float(self.hard_max_frac)
        x = (step - self.hard_start_step) / float(self.hard_ramp_steps)
        x = max(0.0, min(1.0, x))
        return float(self.hard_max_frac) * x

    def _sample_random_dirs(self, D: int, K: int, seed: int, device: torch.device) -> torch.Tensor:
        if K <= 0:
            return torch.empty(D, 0, device=device, dtype=torch.float32)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        A = torch.randn(D, K, generator=g, device=device, dtype=torch.float32)
        return A / (A.norm(dim=0, keepdim=True) + 1e-12)

    def _loss_from_u(self, u: torch.Tensor) -> torch.Tensor:
        u = u - u.mean(dim=0, keepdim=True)
        t = self.t
        u_exp = u.unsqueeze(-1) * t.view(1, 1, -1)

        cosm = torch.cos(u_exp).mean(dim=0)
        sinm = torch.sin(u_exp).mean(dim=0)

        phi = self.phi.view(1, -1)
        diff2 = (cosm - phi).pow(2) + sinm.pow(2)

        integrand = diff2 * phi
        w = self.w_trapz.view(1, -1)
        loss_per_dir = (integrand * w).sum(dim=-1)
        return loss_per_dir.mean()

    def forward(self, z: torch.Tensor, seed: int = 0, detach_A: bool = True) -> torch.Tensor:
        B, D = z.shape
        if D != self.emb_dim:
            raise ValueError("bad embedding dim")

        frac = self.hard_frac(int(seed))
        K_hard_use = int(round(frac * self.num_slices))
        K_hard_use = min(K_hard_use, self.hard_slices)
        K_rand = self.num_slices - K_hard_use

        parts = []
        if K_hard_use > 0:
            A_h = self.A_raw[:, :K_hard_use]
            A_h = F.normalize(A_h, dim=0, eps=1e-12)
            if detach_A:
                A_h = A_h.detach()
            parts.append(A_h)
        if K_rand > 0:
            A_r = self._sample_random_dirs(D, K_rand, seed=int(seed), device=z.device)
            parts.append(A_r)

        A = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        u = z @ A
        loss = self._loss_from_u(u)
        return loss * B

    def dir_objective(self, z_detached: torch.Tensor) -> torch.Tensor:
        if self.A_raw is None:
            return torch.tensor(0.0, device=z_detached.device)
        A = F.normalize(self.A_raw, dim=0, eps=1e-12)
        u = z_detached.float() @ A
        loss = self._loss_from_u(u)
        return -loss
