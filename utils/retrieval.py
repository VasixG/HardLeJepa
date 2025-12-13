from pathlib import Path
from typing import Tuple
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def build_bank(encoder, loader, device: str, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.eval()
    zs, ys, xs = [], [], []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        z = encoder(x)
        zs.append(z)
        ys.append(y.to(device))
        xs.append(x.detach().cpu())
        seen += x.size(0)
        if seen >= max_samples:
            break
    bank_z = torch.cat(zs, dim=0)[:max_samples]
    bank_y = torch.cat(ys, dim=0)[:max_samples]
    bank_x = torch.cat(xs, dim=0)[:max_samples]
    return bank_z, bank_y, bank_x

@torch.no_grad()
def select_queries(loader, num_queries: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    got = 0
    for x, y in loader:
        xs.append(x.cpu())
        ys.append(y.cpu())
        got += x.size(0)
        if got >= num_queries:
            break
    return torch.cat(xs, dim=0)[:num_queries], torch.cat(ys, dim=0)[:num_queries]

@torch.no_grad()
def retrieval_topk(z_bank: torch.Tensor, z_query: torch.Tensor, topk: int, chunk_size: int = 256):
    z_bank = F.normalize(z_bank, dim=-1)
    z_query = F.normalize(z_query, dim=-1)
    all_idx, all_val = [], []
    for s in range(0, z_query.size(0), chunk_size):
        q = z_query[s:s + chunk_size]
        sim = q @ z_bank.T
        val, idx = sim.topk(k=min(topk, z_bank.size(0)), dim=1, largest=True, sorted=True)
        all_idx.append(idx)
        all_val.append(val)
    return torch.cat(all_idx, dim=0), torch.cat(all_val, dim=0)

def save_retrieval_collage(
    encoder,
    train_plain_loader,
    test_plain_loader,
    device: str,
    path: str,
    title: str,
    bank_max: int = 8000,
    query_num: int = 10,
    topk: int = 5,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bank_z, bank_y, bank_x = build_bank(encoder, train_plain_loader, device, max_samples=bank_max)
    xq_cpu, yq_cpu = select_queries(test_plain_loader, num_queries=query_num)
    xq = xq_cpu.to(device)
    encoder.eval()
    with torch.no_grad():
        zq = encoder(xq)
    idx, val = retrieval_topk(bank_z, zq, topk=topk)
    rows = query_num
    cols = 1 + topk
    plt.figure(figsize=(2.2 * cols, 2.2 * rows))
    plt.suptitle(title)
    for r in range(rows):
        ax = plt.subplot(rows, cols, r * cols + 1)
        ax.imshow(xq_cpu[r, 0].numpy(), cmap="gray")
        ax.set_title(f"Q:{int(yq_cpu[r])}")
        ax.axis("off")
        for c in range(topk):
            bi = int(idx[r, c].item())
            ax = plt.subplot(rows, cols, r * cols + 2 + c)
            ax.imshow(bank_x[bi, 0].numpy(), cmap="gray")
            ax.set_title(f"{int(bank_y[bi].cpu())}|{val[r, c].item():.2f}")
            ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path, dpi=200)
    plt.close()
