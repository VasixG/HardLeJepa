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
from losses.ep_compress import EPCompression
from utils.compress import (
    CompressConfig,
    compress_lambda,
    compress_scale,
    compression_metrics,
    drop_first_dims,
)
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


def lam_schedule_sigreg(cfg: TrainConfig, global_step: int) -> float:
    warm = 500
    ramp = 2000
    t = max(0, int(global_step) - warm)
    return float(cfg.sigreg_lambda) * min(1.0, t / max(1, ramp))


def maybe_load_checkpoint(
    init_from_ckpt: Optional[str],
    device: str,
    encoder: nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, int]:
    if not init_from_ckpt:
        return 1, 0
    ckpt_path = Path(init_from_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--init_from_ckpt not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"], strict=True)
    elif "state_dict" in ckpt:
        encoder.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise KeyError("Checkpoint does not contain 'encoder' (or 'state_dict').")
    if opt is not None and "opt" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    return start_epoch, global_step


def train(cfg: TrainConfig, cc: CompressConfig, init_from_ckpt: Optional[str]) -> None:
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

    ep_comp = EPCompression(
        t_min=cfg.sigreg_t_min,
        t_max=cfg.sigreg_t_max,
        t_steps=cfg.sigreg_t_steps,
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, global_step = maybe_load_checkpoint(
        init_from_ckpt, device=device, encoder=encoder, opt=opt
    )

    logger = Logger(cfg.logdir, str(Path(cfg.logdir) / "metrics.csv"), use_tb=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        encoder.train()
        pbar = tqdm(train_ssl_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for views, _y in pbar:
            views_list = [v.to(device, non_blocking=True) for v in views.views]
            zs = [encoder(v) for v in views_list]

            zs_n = [F.normalize(z, dim=-1) for z in zs]
            z_center_n = F.normalize(torch.stack(zs_n, dim=0).mean(dim=0), dim=-1).detach()
            loss_pull = sum(F.mse_loss(z, z_center_n) for z in zs_n) / len(zs_n)

            loss_sig = sum(sigreg(z, seed=global_step + i) for i, z in enumerate(zs)) / len(zs)
            lam_sig = lam_schedule_sigreg(cfg, global_step)

            lam_comp = compress_lambda(cc, global_step)
            scale = compress_scale(cc, global_step)

            if lam_comp > 0.0 and cc.dims > 0:
                comp_losses = []
                for z in zs:
                    k = min(int(cc.dims), z.size(1))
                    u = z[:, :k]
                    comp_losses.append(ep_comp(u, target=cc.target, scale=scale))
                loss_comp = sum(comp_losses) / len(comp_losses)
            else:
                loss_comp = torch.tensor(0.0, device=device)

            loss = (1.0 - lam_sig) * loss_pull + lam_sig * loss_sig + lam_comp * loss_comp

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
                        "train/loss_pull": float(loss_pull.item()),
                        "train/loss_sigreg": float(loss_sig.item()),
                        "train/loss_comp": float(loss_comp.item()),
                        "cfg/lambda_sigreg": float(lam_sig),
                        "cfg/lambda_comp": float(lam_comp),
                        "cfg/comp_scale": float(scale),
                        "cfg/comp_target_is_laplace": float(1.0 if cc.target == "laplace" else 0.0),
                        "repr/z_norm_mean": float(z_norm.mean().item()),
                        "repr/z_norm_std": float(z_norm.std(unbiased=False).item()),
                        "repr/pairwise_cos_mean": pc_mean,
                        "repr/pairwise_cos_std": pc_std,
                        "repr/alignment": au.get("alignment", 0.0),
                        "repr/uniformity": au.get("uniformity", 0.0),
                        "opt/grad_norm_encoder": grad_global_norm(encoder.parameters()),
                    }

                    m.update(
                        compression_metrics(
                            z0, dims=int(cc.dims), thr1=cc.abs_thr1, thr2=cc.abs_thr2
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

            metrics = {"epoch": float(epoch)}
            for k, v in cm.items():
                metrics[f"eval/{k}"] = v
            for k, v in knn.items():
                metrics[f"eval/{k}"] = v

            kdrop = int(min(max(0, cc.dims), z_test.size(1))) if cc.enabled else 0
            if kdrop > 0:
                z_test_drop = drop_first_dims(z_test, kdrop)
                z_bank_drop = drop_first_dims(z_bank, kdrop)
                z_query_drop = drop_first_dims(z_query, kdrop)

                cm_drop = eval_clustering(
                    z_test_drop,
                    y_test,
                    device=device,
                    kmeans_iters=cfg.kmeans_iters,
                    kmeans_restarts=cfg.kmeans_restarts,
                    silhouette_samples=cfg.silhouette_samples,
                )

                knn_drop, _ = eval_knn(
                    z_bank_drop,
                    y_bank,
                    z_query_drop,
                    y_query,
                    device=device,
                    k=cfg.knn_k,
                    temperature=cfg.knn_temperature,
                    chunk_size=cfg.knn_chunk_size,
                )

                metrics["eval_drop/k"] = float(kdrop)
                for k, v in cm_drop.items():
                    metrics[f"eval_drop/{k}"] = v
                for k, v in knn_drop.items():
                    metrics[f"eval_drop/{k}"] = v

            logger.log(global_step, metrics)

            if epoch % cfg.report_every_epochs == 0:
                save_pca_scatter(
                    z_test,
                    y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_pca.png"),
                    title=f"pure_sigreg+compress | epoch {epoch} | PCA (test)",
                    max_points=cfg.report_max_points,
                )
                save_tsne_scatter(
                    z_test,
                    y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_tsne.png"),
                    title=f"pure_sigreg+compress | epoch {epoch} | t-SNE (test)",
                    max_points=min(cfg.report_max_points, 2000),
                    perplexity=cfg.tsne_perplexity,
                    iters=cfg.tsne_iters,
                    seed=cfg.seed + epoch,
                    learning_rate=cfg.tsne_lr,
                )
                save_confusion_matrix(
                    knn_cm,
                    path=str(report_dir / f"epoch{epoch:03d}_knn_cm.png"),
                    title=f"pure_sigreg+compress | epoch {epoch} | kNN confusion",
                    normalize=True,
                )
                if epoch % cfg.retrieval_every_epochs == 0:
                    save_retrieval_collage(
                        encoder,
                        train_plain_loader,
                        test_plain_loader,
                        device=device,
                        path=str(report_dir / f"epoch{epoch:03d}_retrieval.png"),
                        title=f"pure_sigreg+compress | epoch {epoch} | retrieval (test->train)",
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
                "compress": {
                    "enabled": cc.enabled,
                    "target": cc.target,
                    "dims": cc.dims,
                    "start_step": cc.start_step,
                    "warm_steps": cc.warm_steps,
                    "ramp_steps": cc.ramp_steps,
                    "max_lambda": cc.max_lambda,
                    "scale_start": cc.scale_start,
                    "scale_end": cc.scale_end,
                    "scale_decay_steps": cc.scale_decay_steps,
                },
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lambda_sigreg", type=float, default=0.005)
    p.add_argument("--num_views", type=int, default=6)

    p.add_argument(
        "--compress_target", type=str, default="gaussian", choices=["gaussian", "laplace"]
    )
    p.add_argument("--compress_dims", type=int, default=0)
    p.add_argument("--compress_start_step", type=int, default=0)
    p.add_argument("--compress_warm_steps", type=int, default=0)
    p.add_argument("--compress_ramp_steps", type=int, default=2000)
    p.add_argument("--compress_lambda", type=float, default=0.10)
    p.add_argument("--compress_scale_start", type=float, default=1.0)
    p.add_argument("--compress_scale_end", type=float, default=0.05)
    p.add_argument("--compress_scale_decay_steps", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        method="pure_sigreg_compress",
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        sigreg_lambda=args.lambda_sigreg,
        num_views=args.num_views,
        logdir=(args.logdir or f"runs/pure_sigreg_compress_{args.compress_target}_seed{args.seed}"),
        ckpt_path=(
            args.ckpt
            or f"checkpoints/pure_sigreg_compress_{args.compress_target}_seed{args.seed}.pt"
        ),
    )

    cc = CompressConfig(
        enabled=(args.compress_dims > 0),
        target=args.compress_target,
        dims=int(args.compress_dims),
        start_step=int(args.compress_start_step),
        warm_steps=int(args.compress_warm_steps),
        ramp_steps=int(args.compress_ramp_steps),
        max_lambda=float(args.compress_lambda),
        scale_start=float(args.compress_scale_start),
        scale_end=float(args.compress_scale_end),
        scale_decay_steps=int(args.compress_scale_decay_steps),
    )

    train(cfg, cc, init_from_ckpt=args.init_from_ckpt)
