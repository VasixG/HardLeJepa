import torch
import torch.nn as nn

class SIGReg(nn.Module):
    def __init__(self, emb_dim: int, num_slices: int = 64, t_min: float = -5.0, t_max: float = 5.0, t_steps: int = 17):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_slices = num_slices
        t = torch.linspace(t_min, t_max, t_steps)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi_gauss", torch.exp(-0.5 * (t ** 2)), persistent=False)

    def forward(self, z: torch.Tensor, seed: int = 0) -> torch.Tensor:
        B, D = z.shape
        if D != self.emb_dim:
            raise ValueError("bad embedding dim")
        z = z - z.mean(dim=0, keepdim=True)
        g = torch.Generator(device=z.device)
        g.manual_seed(int(seed))
        A = torch.randn(D, self.num_slices, generator=g, device=z.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
        u = z @ A
        t = self.t
        u_exp = u.unsqueeze(-1) * t.view(1, 1, -1)
        ecf = torch.exp(1j * u_exp).mean(dim=0)
        phi = self.phi_gauss.view(1, -1)
        diff2 = (ecf.real - phi).pow(2) + (ecf.imag).pow(2)
        integrand = diff2 * phi
        loss = torch.trapezoid(integrand, t, dim=-1).mean()
        return loss * B
