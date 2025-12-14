import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPAPredictor(nn.Module):
    def __init__(self, emb_dim: int = 256, pos_dim: int = 4, hidden: int = 512):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 128, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, z_ctx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        p = self.pos_mlp(pos)
        x = torch.cat([z_ctx, p], dim=-1)
        z = self.net(x)
        z = torch.nn.functional.layer_norm(z, (z.size(-1),))
        return z
