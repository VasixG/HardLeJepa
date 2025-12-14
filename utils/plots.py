from pathlib import Path
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def pca_2d(z: torch.Tensor, max_points: int = 3000) -> torch.Tensor:
    z = z.detach().cpu().float()
    n = z.size(0)
    if n > max_points:
        idx = torch.randperm(n)[:max_points]
        z = z[idx]
    z = z - z.mean(dim=0, keepdim=True)
    Vh = torch.linalg.svd(z, full_matrices=False).Vh
    return z @ Vh[:2].T


def save_pca_scatter(z: torch.Tensor, y: torch.Tensor, path: str, title: str, max_points: int = 3000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pc = pca_2d(z, max_points=max_points)
    y = y.detach().cpu()
    if pc.size(0) != y.size(0):
        y = y[: pc.size(0)]
    plt.figure(figsize=(7, 6))
    plt.scatter(pc[:, 0].numpy(), pc[:, 1].numpy(), c=y.numpy(), s=8, alpha=0.8, cmap="tab10")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_tsne_scatter(
    z: torch.Tensor,
    y: torch.Tensor,
    path: str,
    title: str,
    max_points: int = 2000,
    perplexity: float = 30.0,
    iters: int = 800,
    seed: int = 0,
    learning_rate: str = "auto",
):
    from sklearn.manifold import TSNE

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    z = z.detach().cpu().float()
    y = y.detach().cpu()

    n = z.size(0)
    if n > max_points:
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:max_points]
        z = z[idx]
        y = y[idx]

    if z.size(0) <= 3:
        return

    perp = float(perplexity)
    if z.size(0) - 1 < perp:
        perp = max(2.0, float(z.size(0) - 1) / 3.0)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=int(iters),
        init="pca",
        random_state=int(seed),
        learning_rate=learning_rate,
    )
    emb = tsne.fit_transform(z.numpy())

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=y.numpy(), s=8, alpha=0.8, cmap="tab10")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_matrix(cm: torch.Tensor, path: str, title: str, normalize: bool = True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmf = cm.float()
    if normalize:
        cmf = cmf / (cmf.sum(dim=1, keepdim=True) + 1e-12)
    plt.figure(figsize=(7, 6))
    plt.imshow(cmf.numpy(), interpolation="nearest")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
