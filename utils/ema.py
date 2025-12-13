import torch
import torch.nn as nn

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
