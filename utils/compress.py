from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CompressConfig:
    enabled: bool = True
    target: str = "gaussian"
    dims: int = 0
    start_step: int = 0
    warm_steps: int = 0
    ramp_steps: int = 2000
    max_lambda: float = 0.1
    scale_start: float = 1.0
    scale_end: float = 0.05
    scale_decay_steps: int = 5000
    abs_thr1: float = 1e-2
    abs_thr2: float = 1e-3


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compress_lambda(cc: CompressConfig, step: int) -> float:
    if (not cc.enabled) or cc.dims <= 0 or cc.max_lambda <= 0.0:
        return 0.0
    if step < cc.start_step:
        return 0.0
    s = step - cc.start_step
    if cc.warm_steps > 0 and s < cc.warm_steps:
        return 0.0
    s2 = max(0, s - cc.warm_steps)
    if cc.ramp_steps <= 0:
        return float(cc.max_lambda)
    frac = _clamp01(s2 / float(cc.ramp_steps))
    return float(cc.max_lambda) * frac


def compress_scale(cc: CompressConfig, step: int) -> float:
    if (not cc.enabled) or cc.dims <= 0:
        return float(cc.scale_start)
    if step < cc.start_step:
        return float(cc.scale_start)
    s = step - cc.start_step
    if cc.scale_decay_steps <= 0:
        return float(cc.scale_end)
    frac = _clamp01(s / float(cc.scale_decay_steps))
    return float(cc.scale_start) + (float(cc.scale_end) - float(cc.scale_start)) * frac


@torch.no_grad()
def compression_metrics(z: torch.Tensor, dims: int, thr1: float, thr2: float) -> Dict[str, float]:
    B, D = z.shape
    k = int(min(max(0, dims), D))
    if k == 0:
        return {
            "comp/k": 0.0,
            "comp/firstk_abs_mean": 0.0,
            "comp/firstk_std_mean": 0.0,
            "comp/firstk_var_mean": 0.0,
            "comp/firstk_frac_abs_lt_thr1": 0.0,
            "comp/firstk_frac_abs_lt_thr2": 0.0,
            "comp/firstk_energy_frac": 0.0,
            "comp/energy_removed_frac": 0.0,
        }

    u = z[:, :k]
    abs_mean = float(u.abs().mean().item())
    std_mean = float(u.std(dim=0, unbiased=False).mean().item())
    var_mean = float((u.var(dim=0, unbiased=False)).mean().item())
    frac1 = float((u.abs() < float(thr1)).float().mean().item())
    frac2 = float((u.abs() < float(thr2)).float().mean().item())

    e_first = u.pow(2).sum(dim=-1) + 1e-12
    e_all = z.pow(2).sum(dim=-1) + 1e-12
    energy_frac = float((e_first / e_all).mean().item())

    return {
        "comp/k": float(k),
        "comp/firstk_abs_mean": abs_mean,
        "comp/firstk_std_mean": std_mean,
        "comp/firstk_var_mean": var_mean,
        "comp/firstk_frac_abs_lt_thr1": frac1,
        "comp/firstk_frac_abs_lt_thr2": frac2,
        "comp/firstk_energy_frac": energy_frac,
        "comp/energy_removed_frac": float(1.0 - energy_frac),
    }


@torch.no_grad()
def drop_first_dims(z: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return z
    k = min(int(k), z.size(1))
    out = z.clone()
    out[:, :k] = 0.0
    return out
