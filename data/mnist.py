import gzip
import math
import os
import random
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

Box = Tuple[int, int, int, int]

_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(path)


def _read_idx_images(gz_path: Path) -> torch.Tensor:
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("bad idx image file")
        data = f.read(n * rows * cols)
    x = torch.frombuffer(data, dtype=torch.uint8).reshape(n, rows, cols)
    return x


def _read_idx_labels(gz_path: Path) -> torch.Tensor:
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("bad idx label file")
        data = f.read(n)
    y = torch.frombuffer(data, dtype=torch.uint8)
    return y


class MNISTRaw(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.root = Path(root)
        self.train = train
        self._ensure()
        if train:
            xi = _read_idx_images(self.root / "train-images-idx3-ubyte.gz")
            yi = _read_idx_labels(self.root / "train-labels-idx1-ubyte.gz")
        else:
            xi = _read_idx_images(self.root / "t10k-images-idx3-ubyte.gz")
            yi = _read_idx_labels(self.root / "t10k-labels-idx1-ubyte.gz")
        self.x = (xi.float() / 255.0).unsqueeze(1)
        self.y = yi.long()

    def _ensure(self):
        _download(_URLS["train_images"], self.root / "train-images-idx3-ubyte.gz")
        _download(_URLS["train_labels"], self.root / "train-labels-idx1-ubyte.gz")
        _download(_URLS["test_images"], self.root / "t10k-images-idx3-ubyte.gz")
        _download(_URLS["test_labels"], self.root / "t10k-labels-idx1-ubyte.gz")

    def __len__(self):
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def sample_box(H: int, W: int, area_range: Tuple[float, float], ratio_range: Tuple[float, float] = (0.75, 1.33)) -> Box:
    area = H * W
    for _ in range(10):
        target_area = random.uniform(*area_range) * area
        log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
        aspect = math.exp(random.uniform(*log_ratio))
        h = int(round(math.sqrt(target_area / aspect)))
        w = int(round(math.sqrt(target_area * aspect)))
        if 0 < h <= H and 0 < w <= W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            return top, left, h, w
    size = min(H, W)
    top = (H - size) // 2
    left = (W - size) // 2
    return top, left, size, size


def box_to_pos(box: Box, H: int, W: int) -> torch.Tensor:
    top, left, h, w = box
    cx = (left + 0.5 * w) / W
    cy = (top + 0.5 * h) / H
    return torch.tensor([cx, cy, w / W, h / H], dtype=torch.float32)


def resized_crop(x: torch.Tensor, top: int, left: int, h: int, w: int, out_size: int) -> torch.Tensor:
    patch = x[..., top:top + h, left:left + w]
    if patch.size(-1) != out_size or patch.size(-2) != out_size:
        patch = F.interpolate(patch.unsqueeze(0), size=(out_size, out_size), mode="bilinear", align_corners=False).squeeze(0)
    return patch


def _gaussian_blur(x: torch.Tensor, k: int = 3, sigma: float = 1.0) -> torch.Tensor:
    if k <= 1:
        return x
    device = x.device
    ax = torch.arange(k, device=device) - (k - 1) / 2
    kernel = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)
    pad = k // 2
    return F.conv2d(x, kernel2d, padding=pad)


def _random_erasing(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    if random.random() > p:
        return x
    _, H, W = x.shape
    area = H * W
    erase_area = random.uniform(0.02, 0.12) * area
    aspect = random.uniform(0.3, 3.3)
    h = int(round(math.sqrt(erase_area / aspect)))
    w = int(round(math.sqrt(erase_area * aspect)))
    h = max(1, min(h, H))
    w = max(1, min(w, W))
    top = random.randint(0, H - h)
    left = random.randint(0, W - w)
    x[:, top:top + h, left:left + w] = 0.0
    return x


def _random_affine(x: torch.Tensor) -> torch.Tensor:
    angle = math.radians(random.uniform(-18, 18))
    scale = random.uniform(0.85, 1.15)
    shear = math.radians(random.uniform(-8, 8))
    tx = random.uniform(-0.12, 0.12)
    ty = random.uniform(-0.12, 0.12)
    ca, sa = math.cos(angle), math.sin(angle)
    cs, ss = math.cos(shear), math.sin(shear)
    A = torch.tensor(
        [
            [scale * ca, -scale * sa + ss, tx],
            [scale * sa, scale * ca + cs, ty],
        ],
        dtype=torch.float32,
        device=x.device,
    )
    grid = F.affine_grid(A.unsqueeze(0), size=(1, 1, x.size(-2), x.size(-1)), align_corners=False)
    y = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    return y


def _jitter(x: torch.Tensor) -> torch.Tensor:
    c = random.uniform(0.90, 1.10)
    b = random.uniform(-0.10, 0.10)
    return (x * c + b).clamp(0.0, 1.0)


def strong_aug(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.95:
        x = _random_affine(x)
    x = _jitter(x)
    return x


@dataclass(frozen=True)
class JEPAViews:
    ctx: torch.Tensor
    tgt_imgs: List[torch.Tensor]
    tgt_pos: List[torch.Tensor]


class MNISTJEPACollate:
    def __init__(self, out_size: int, num_targets: int, ctx_area: Tuple[float, float], tgt_area: Tuple[float, float]):
        self.out_size = out_size
        self.num_targets = num_targets
        self.ctx_area = ctx_area
        self.tgt_area = tgt_area

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        B, _, H, W = x.shape
        ctx_imgs = []
        tgt_imgs = [[] for _ in range(self.num_targets)]
        tgt_pos = [[] for _ in range(self.num_targets)]
        for i in range(B):
            xi = x[i]
            ctx_box = sample_box(H, W, self.ctx_area)
            ctx = resized_crop(xi, *ctx_box, out_size=self.out_size)
            ctx = strong_aug(ctx)
            ctx_imgs.append(ctx)
            for t in range(self.num_targets):
                tb = sample_box(H, W, self.tgt_area)
                tgt = resized_crop(xi, *tb, out_size=self.out_size)
                tgt_imgs[t].append(tgt)
                tgt_pos[t].append(box_to_pos(tb, H, W))
        ctx = torch.stack(ctx_imgs, dim=0)
        tgt_imgs = [torch.stack(v, dim=0) for v in tgt_imgs]
        tgt_pos = [torch.stack(v, dim=0) for v in tgt_pos]
        return (JEPAViews(ctx=ctx, tgt_imgs=tgt_imgs, tgt_pos=tgt_pos), y)


@dataclass(frozen=True)
class MultiViews:
    views: List[torch.Tensor]


class MNISTMultiViewCollate:
    def __init__(self, out_size: int, area: Tuple[float, float], num_views: int):
        self.out_size = out_size
        self.area = area
        self.num_views = num_views

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        B, _, H, W = x.shape
        all_views = [[] for _ in range(self.num_views)]
        for i in range(B):
            xi = x[i]
            for v in range(self.num_views):
                b = sample_box(H, W, self.area)
                vv = resized_crop(xi, *b, out_size=self.out_size)
                vv = strong_aug(vv)
                all_views[v].append(vv)
        views = [torch.stack(vs, dim=0) for vs in all_views]
        return (MultiViews(views=views), y)


def make_plain_mnist_loaders(batch_size: int, num_workers: int, train_fraction: float = 1.0, test_fraction: float = 1.0, seed: int = 0):
    train_ds = MNISTRaw("./data/mnist", train=True)
    test_ds = MNISTRaw("./data/mnist", train=False)
    if train_fraction < 1.0:
        g = torch.Generator()
        g.manual_seed(int(seed))
        n = max(1, int(len(train_ds) * float(train_fraction)))
        idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
        train_ds = Subset(train_ds, idx)
    if test_fraction < 1.0:
        g = torch.Generator()
        g.manual_seed(int(seed) + 1)
        n = max(1, int(len(test_ds) * float(test_fraction)))
        idx = torch.randperm(len(test_ds), generator=g)[:n].tolist()
        test_ds = Subset(test_ds, idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


def make_mnist_ssl_and_plain_loaders(batch_size: int, num_workers: int, ssl_collate, train_fraction: float = 1.0, test_fraction: float = 1.0, seed: int = 0):
    train_ds = MNISTRaw("./data/mnist", train=True)
    test_ds = MNISTRaw("./data/mnist", train=False)
    if train_fraction < 1.0:
        g = torch.Generator()
        g.manual_seed(int(seed))
        n = max(1, int(len(train_ds) * float(train_fraction)))
        idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
        train_ds = Subset(train_ds, idx)
    if test_fraction < 1.0:
        g = torch.Generator()
        g.manual_seed(int(seed) + 1)
        n = max(1, int(len(test_ds) * float(test_fraction)))
        idx = torch.randperm(len(test_ds), generator=g)[:n].tolist()
        test_ds = Subset(test_ds, idx)
    train_ssl_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=ssl_collate)
    train_plain_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_plain_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_ssl_loader, train_plain_loader, test_plain_loader
