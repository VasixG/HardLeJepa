import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import TrainConfig
from data.mnist import MNISTMultiViewCollate, make_mnist_ssl_and_plain_loaders
from models.encoder import ResNetMNISTEncoder
from losses.sigreg import SIGReg
from losses.compress_ep import CompressionEP
from utils.seed import set_seed
from utils.logger import Logger
from utils.plots import save_pca_scatter, save_tsne_scatter, save_confusion_matrix
from utils.retrieval import save_retrieval_collage
from metrics.repr_stats import (
    grad_global_norm,
    cov_metrics,
    spectrum_metrics,
    pairwise_cos_stats,
    alignment_uniformity,
)
from evaluate import extract_embeddings, eval_clustering, eval_knn


def lam_schedule(cfg: TrainConfig, global_step: int) -> float:
    warm = 500
    ramp = 2000
    t = max(0, int(global_step) - warm)
    return float(cfg.sigreg_lambda) * min(1.0, t / max(1, ramp))


def progress_after_resume(epoch: int, start_epoch: int, end_epoch: int) -> float:
    denom = max(1, end_epoch - start_epoch)
    p = (epoch - start_epoch) / denom
    return float(max(0.0, min(1.0, p)))


def compress_scale_warm(
    epoch: int, start_epoch: int, end_epoch: int, s0: float, s1: float, warm: int
) -> float:
    if epoch <= start_epoch + int(warm):
        return float(s0)
    denom = max(1, end_epoch - (start_epoch + int(warm)))
    p = (epoch - (start_epoch + int(warm))) / denom
    p = float(max(0.0, min(1.0, p)))
    return float(s0 + (s1 - s0) * p)


def compress_lambda_warm(
    epoch: int, start_epoch: int, end_epoch: int, lam_max: float, warm: int
) -> float:
    if epoch <= start_epoch + int(warm):
        return 0.0
    denom = max(1, end_epoch - (start_epoch + int(warm)))
    p = (epoch - (start_epoch + int(warm))) / denom
    p = float(max(0.0, min(1.0, p)))
    return float(lam_max * p)


@torch.no_grad()
def dims_zero_metrics(z: torch.Tensor, dims: int, eps: float = 1e-3) -> Dict[str, float]:
    D = z.size(1)
    K = int(min(max(0, dims), D))
    if K <= 0:
        return {
            "compress/dims": 0.0,
            "compress/abs_mean": 0.0,
            "compress/std_mean": 0.0,
            "compress/frac_abs_lt_eps": 0.0,
            "compress/energy_frac": 0.0,
        }
    u = z[:, :K]
    abs_mean = float(u.abs().mean().item())
    std_mean = float(u.std(dim=0, unbiased=False).mean().item())
    frac = float((u.abs() < float(eps)).float().mean().item())

    e_all = float((z * z).mean().item())
    e_sel = float((u * u).mean().item())
    energy_frac = float(e_sel / max(1e-12, e_all))

    return {
        "compress/dims": float(K),
        "compress/abs_mean": abs_mean,
        "compress/std_mean": std_mean,
        "compress/frac_abs_lt_eps": frac,
        "compress/energy_frac": energy_frac,
    }


def eval_with_optional_drop(
    encoder: nn.Module,
    train_plain_loader,
    test_plain_loader,
    device: str,
    cfg: TrainConfig,
    drop_dims: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[torch.Tensor]]:
    z_test, y_test = extract_embeddings(
        encoder, test_plain_loader, device, max_samples=cfg.eval_max_samples
    )
    cm = eval_clustering(
        z_test,
        y_test,
        device=device,
        kmeans_iters=cfg.kmeans_iters,
        kmeans_restarts=cfg.kmeans_restarts,
        silhouette_samples=cfg.silhouette_samples,
    )

    z_bank, y_bank = extract_embeddings(
        encoder, train_plain_loader, device, max_samples=cfg.knn_max_train
    )
    qN = min(cfg.knn_max_test, y_test.size(0))
    z_query = z_test[:qN]
    y_query = y_test[:qN]
    knn, knn_cm = eval_knn(
        z_bank,
        y_bank,
        z_query,
        y_query,
        device=device,
        k=cfg.knn_k,
        temperature=cfg.knn_temperature,
        chunk_size=cfg.knn_chunk_size,
    )

    out = {}
    for k, v in cm.items():
        out[f"eval/{k}"] = v
    for k, v in knn.items():
        out[f"eval/{k}"] = v

    out_drop = {}
    if int(drop_dims) > 0:
        z_test_d = z_test.clone()
        z_bank_d = z_bank.clone()
        z_test_d[:, : int(drop_dims)] = 0.0
        z_bank_d[:, : int(drop_dims)] = 0.0

        cm_d = eval_clustering(
            z_test_d,
            y_test,
            device=device,
            kmeans_iters=cfg.kmeans_iters,
            kmeans_restarts=cfg.kmeans_restarts,
            silhouette_samples=cfg.silhouette_samples,
        )
        z_query_d = z_test_d[:qN]
        knn_d, _ = eval_knn(
            z_bank_d,
            y_bank,
            z_query_d,
            y_query,
            device=device,
            k=cfg.knn_k,
            temperature=cfg.knn_temperature,
            chunk_size=cfg.knn_chunk_size,
        )

        for k, v in cm_d.items():
            out_drop[f"eval_drop/{k}"] = v
        for k, v in knn_d.items():
            out_drop[f"eval_drop/{k}"] = v

    return out, out_drop, knn_cm


def train(cfg: TrainConfig, args) -> None:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg.logdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    collate = MNISTMultiViewCollate(
        cfg.out_size, (cfg.ctx_area_min, cfg.ctx_area_max), num_views=cfg.num_views
    )
    train_ssl_loader, train_plain_loader, test_plain_loader = make_mnist_ssl_and_plain_loaders(
        cfg.batch_size,
        cfg.num_workers,
        collate,
        train_fraction=cfg.train_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )

    encoder = ResNetMNISTEncoder(cfg.emb_dim).to(device)
    sigreg = SIGReg(
        cfg.emb_dim,
        num_slices=cfg.sigreg_num_slices,
        t_min=cfg.sigreg_t_min,
        t_max=cfg.sigreg_t_max,
        t_steps=cfg.sigreg_t_steps,
    ).to(device)

    compress = CompressionEP(
        emb_dim=cfg.emb_dim,
        t_min=cfg.sigreg_t_min,
        t_max=cfg.sigreg_t_max,
        t_steps=cfg.sigreg_t_steps,
        target=args.compress_target,
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    global_step = 0

    if args.init_from_ckpt is not None:
        ckpt = torch.load(args.init_from_ckpt, map_location=device)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))

    logger = Logger(cfg.logdir, str(Path(cfg.logdir) / "metrics.csv"), use_tb=True)

    end_epoch = int(args.epochs)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        encoder.train()
        pbar = tqdm(train_ssl_loader, desc=f"epoch {epoch}/{end_epoch}")
        for views, _y in pbar:
            views_list = [v.to(device, non_blocking=True) for v in views.views]
            zs = [encoder(v) for v in views_list]

            zs_n = [F.normalize(z, dim=-1) for z in zs]
            z_center_n = F.normalize(torch.stack(zs_n, dim=0).mean(dim=0), dim=-1).detach()
            loss_pull = sum(F.mse_loss(z, z_center_n) for z in zs_n) / len(zs_n)

            loss_sig = sum(sigreg(z, seed=global_step + i) for i, z in enumerate(zs)) / len(zs)

            lam_sig = lam_schedule(cfg, global_step)

            c_scale = compress_scale_warm(
                epoch=epoch,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                s0=float(args.compress_scale_start),
                s1=float(args.compress_scale_end),
                warm=int(args.compress_warm_epochs),
            )
            c_lam = compress_lambda_warm(
                epoch=epoch,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                lam_max=float(args.compress_lambda),
                warm=int(args.compress_warm_epochs),
            )

            loss_comp = sum(
                compress(z, dims=int(args.compress_dims), scale=c_scale) for z in zs
            ) / len(zs)

            loss_main = (1.0 - lam_sig) * loss_pull + lam_sig * loss_sig
            loss = loss_main + c_lam * loss_comp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            if global_step % cfg.log_every == 0:
                with torch.no_grad():
                    z0 = zs[0]
                    pc_mean, pc_std = pairwise_cos_stats(z0)
                    au = alignment_uniformity(zs[0], zs[1] if len(zs) > 1 else None)
                    z_norm = z0.norm(dim=-1)

                    m: Dict[str, float] = {
                        "train/loss_total": float(loss.item()),
                        "train/loss_main": float(loss_main.item()),
                        "train/loss_pull": float(loss_pull.item()),
                        "train/loss_sigreg": float(loss_sig.item()),
                        "train/loss_compress": float(loss_comp.item()),
                        "cfg/lambda_sigreg": float(lam_sig),
                        "cfg/compress_lambda": float(c_lam),
                        "cfg/compress_scale": float(c_scale),
                        "repr/z_norm_mean": float(z_norm.mean().item()),
                        "repr/z_norm_std": float(z_norm.std(unbiased=False).item()),
                        "repr/pairwise_cos_mean": pc_mean,
                        "repr/pairwise_cos_std": pc_std,
                        "repr/alignment": au.get("alignment", 0.0),
                        "repr/uniformity": au.get("uniformity", 0.0),
                        "opt/grad_norm_encoder": grad_global_norm(encoder.parameters()),
                    }

                    m.update(
                        dims_zero_metrics(
                            z0, dims=int(args.compress_dims), eps=float(args.compress_eps)
                        )
                    )

                    for k, v in cov_metrics(z0).items():
                        m[f"encoder/collapse_{k}"] = v
                    for k, v in spectrum_metrics(z0).items():
                        m[f"encoder/spectrum_{k}"] = v

                    logger.log(global_step, m)

            pbar.set_postfix(loss=float(loss.item()))
            global_step += 1

        if epoch % cfg.eval_every_epochs == 0:
            eval_m, eval_drop_m, knn_cm = eval_with_optional_drop(
                encoder=encoder,
                train_plain_loader=train_plain_loader,
                test_plain_loader=test_plain_loader,
                device=device,
                cfg=cfg,
                drop_dims=int(args.drop_dims),
            )

            metrics = {"epoch": float(epoch)}
            metrics.update(eval_m)
            metrics.update(eval_drop_m)
            logger.log(global_step, metrics)

            if epoch % cfg.report_every_epochs == 0:
                z_test, y_test = extract_embeddings(
                    encoder, test_plain_loader, device, max_samples=cfg.eval_max_samples
                )
                save_pca_scatter(
                    z_test,
                    y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_pca.png"),
                    title=f"compress_{args.compress_target} | epoch {epoch} | PCA (test)",
                    max_points=cfg.report_max_points,
                )
                save_tsne_scatter(
                    z_test,
                    y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_tsne.png"),
                    title=f"compress_{args.compress_target} | epoch {epoch} | t-SNE (test)",
                    max_points=min(cfg.report_max_points, 2000),
                    perplexity=cfg.tsne_perplexity,
                    iters=cfg.tsne_iters,
                    seed=cfg.seed + epoch,
                    learning_rate=cfg.tsne_lr,
                )
                if knn_cm is not None:
                    save_confusion_matrix(
                        knn_cm,
                        path=str(report_dir / f"epoch{epoch:03d}_knn_cm.png"),
                        title=f"compress_{args.compress_target} | epoch {epoch} | kNN confusion",
                        normalize=True,
                    )

                if epoch % cfg.retrieval_every_epochs == 0:
                    save_retrieval_collage(
                        encoder,
                        train_plain_loader,
                        test_plain_loader,
                        device=device,
                        path=str(report_dir / f"epoch{epoch:03d}_retrieval.png"),
                        title=f"compress_{args.compress_target} | epoch {epoch} | retrieval (test->train)",
                        bank_max=cfg.retrieval_bank_max,
                        query_num=cfg.retrieval_query_num,
                        topk=cfg.retrieval_topk,
                    )

        torch.save(
            {
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "global_step": global_step,
                "encoder": encoder.state_dict(),
                "opt": opt.state_dict(),
                "compress_target": str(args.compress_target),
                "compress_dims": int(args.compress_dims),
            },
            cfg.ckpt_path,
        )

    logger.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--init_from_ckpt", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=65)

    p.add_argument("--lambda_sigreg", type=float, default=0.005)
    p.add_argument("--num_views", type=int, default=6)

    p.add_argument("--compress_target", type=str, default="gauss", choices=["gauss", "laplace"])
    p.add_argument("--compress_dims", type=int, default=64)
    p.add_argument("--compress_lambda", type=float, default=0.10)
    p.add_argument("--compress_scale_start", type=float, default=1.0)
    p.add_argument("--compress_scale_end", type=float, default=0.05)
    p.add_argument("--compress_warm_epochs", type=int, default=3)
    p.add_argument("--compress_eps", type=float, default=1e-3)

    p.add_argument("--drop_dims", type=int, default=64)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tag = f"compress_{args.compress_target}_K{args.compress_dims}_lam{args.compress_lambda}_s{args.compress_scale_start}to{args.compress_scale_end}"
    cfg = TrainConfig(
        method=f"pure_sigreg_{tag}",
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        sigreg_lambda=args.lambda_sigreg,
        num_views=args.num_views,
        logdir=(args.logdir or f"runs/pure_sigreg_{tag}_seed{args.seed}"),
        ckpt_path=(args.ckpt or f"checkpoints/pure_sigreg_{tag}_seed{args.seed}.pt"),
    )
    train(cfg, args)
