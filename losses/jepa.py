from typing import List
import torch
import torch.nn.functional as F

def jepa_pred_loss(preds: List[torch.Tensor], tgts: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for p, t in zip(preds, tgts):
        loss = loss + F.mse_loss(p, t)
    return loss / len(preds)
