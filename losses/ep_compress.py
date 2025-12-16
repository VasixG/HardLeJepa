import torch
import torch.nn as nn


def _trapz_weights(t: torch.Tensor) -> torch.Tensor:
    dt = (t[-1] - t[0]) / max(1, (t.numel() - 1))
    w = torch.full((t.numel(),), dt, dtype=t.dtype, device=t.device)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


class CompressionEP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        t_min: float = -5.0,
        t_max: float = 5.0,
        t_steps: int = 17,
        target: str = "gauss",
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.target = str(target)

        t = torch.linspace(float(t_min), float(t_max), int(t_steps), dtype=torch.float32)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("w_trapz", _trapz_weights(t), persistent=False)

    def _phi(self, scale: float) -> torch.Tensor:
        t = self.t
        s = float(scale)
        if self.target == "gauss":
            return torch.exp(-0.5 * (s * s) * (t * t))
        if self.target == "laplace":
            return 1.0 / (1.0 + (s * s) * (t * t))
        raise ValueError(f"Unknown target={self.target}")

    def forward(self, z: torch.Tensor, dims: int, scale: float) -> torch.Tensor:
        B, D = z.shape
        if D != self.emb_dim:
            raise ValueError("bad embedding dim")
        K = int(min(max(0, dims), D))
        if K <= 0:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        u = z[:, :K]
        u = u - u.mean(dim=0, keepdim=True)

        t = self.t
        phi = self._phi(scale).view(1, 1, -1)

        u_exp = u.unsqueeze(-1) * t.view(1, 1, -1)
        cos_m = torch.cos(u_exp).mean(dim=0, keepdim=True)
        sin_m = torch.sin(u_exp).mean(dim=0, keepdim=True)

        diff2 = (cos_m - phi).square() + sin_m.square()

        w = self.w_trapz.view(1, 1, -1)
        loss_per_dim = (diff2 * w).sum(dim=-1).squeeze(0)
        return loss_per_dim.mean() * B
