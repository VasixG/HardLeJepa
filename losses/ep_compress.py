import torch
import torch.nn as nn


def _trapz_weights(t: torch.Tensor) -> torch.Tensor:
    dt = (t[-1] - t[0]) / max(1, (t.numel() - 1))
    w = torch.full((t.numel(),), dt, dtype=t.dtype, device=t.device)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


class EPCompression(nn.Module):
    def __init__(self, t_min: float = -5.0, t_max: float = 5.0, t_steps: int = 17):
        super().__init__()
        t = torch.linspace(float(t_min), float(t_max), int(t_steps), dtype=torch.float32)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("w_base", _trapz_weights(t), persistent=False)

    def _phi_gaussian(self, sigma: torch.Tensor) -> torch.Tensor:
        t = self.t
        return torch.exp(-0.5 * (sigma * t) ** 2)

    def _phi_laplace(self, b: torch.Tensor) -> torch.Tensor:
        t = self.t
        return 1.0 / (1.0 + (b * t) ** 2)

    def forward(self, u: torch.Tensor, target: str, scale: float) -> torch.Tensor:
        if u.ndim != 2:
            raise ValueError(f"u must be [B,K], got {tuple(u.shape)}")
        B, K = u.shape
        if K == 0:
            return u.new_tensor(0.0)

        u = u.float()
        u = u - u.mean(dim=0, keepdim=True)

        s = u.new_tensor(float(scale))
        if target == "gaussian":
            phi = self._phi_gaussian(s)
        elif target == "laplace":
            phi = self._phi_laplace(s)
        else:
            raise ValueError(f"unknown target={target}")

        w = self.w_base * phi
        t = self.t.view(1, 1, -1)

        u_exp = u.unsqueeze(-1) * t
        cos_m = torch.cos(u_exp).mean(dim=0)
        sin_m = torch.sin(u_exp).mean(dim=0)

        phiKT = phi.view(1, -1)
        diff2 = (cos_m - phiKT).pow(2) + sin_m.pow(2)

        loss_per_dir = (diff2 * w.view(1, -1)).sum(dim=-1)
        return loss_per_dir.mean() * float(B)
