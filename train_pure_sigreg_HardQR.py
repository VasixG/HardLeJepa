import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import TrainConfig
from data.mnist import MNISTMultiViewCollate, make_mnist_ssl_and_plain_loaders
from models.encoder import ResNetMNISTEncoder
from losses.sigreg_hard import SIGRegHard
from utils.seed import set_seed
from utils.logger import Logger
from utils.plots import save_pca_scatter, save_confusion_matrix
from utils.retrieval import save_retrieval_collage
from metrics.repr_stats import grad_global_norm, cov_metrics, spectrum_metrics, pairwise_cos_stats, alignment_uniformity
from evaluate import extract_embeddings, eval_clustering, eval_knn

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg.logdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    collate = MNISTMultiViewCollate(cfg.out_size, (cfg.ctx_area_min, cfg.ctx_area_max), num_views=cfg.num_views)
    train_ssl_loader, train_plain_loader, test_plain_loader = make_mnist_ssl_and_plain_loaders(cfg.batch_size, cfg.num_workers, collate)
    encoder = ResNetMNISTEncoder(cfg.emb_dim).to(device)
    sigreg = SIGRegHard(
        emb_dim=cfg.emb_dim,
        pool_size=cfg.hard_pool_size,
        hard_slices=cfg.hard_num_slices,
        random_extra=0,
        mode="hardqr",
        t_min=cfg.sigreg_t_min,
        t_max=cfg.sigreg_t_max,
        t_steps=cfg.sigreg_t_steps,
    ).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = Logger(cfg.logdir, str(Path(cfg.logdir) / "metrics.csv"), use_tb=True)
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        pbar = tqdm(train_ssl_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for (views, _y) in pbar:
            views_list = [v.to(device, non_blocking=True) for v in views.views]
            zs = [encoder(v) for v in views_list]
            z_center = torch.stack(zs, dim=0).mean(dim=0)
            loss_pull = sum(F.mse_loss(z, z_center) for z in zs) / len(zs)
            loss_sig_vals = []
            sig_metrics_acc = {}
            for i, z in enumerate(zs):
                ls, met = sigreg(z, seed=global_step + 17 * i)
                loss_sig_vals.append(ls)
                for k, v in met.items():
                    sig_metrics_acc[k] = sig_metrics_acc.get(k, 0.0) + float(v)
            loss_sig = sum(loss_sig_vals) / len(loss_sig_vals)
            for k in list(sig_metrics_acc.keys()):
                sig_metrics_acc[k] = sig_metrics_acc[k] / len(zs)
            lam = cfg.sigreg_lambda
            loss = (1.0 - lam) * loss_pull + lam * loss_sig
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            if global_step % cfg.log_every == 0:
                with torch.no_grad():
                    z0 = zs[0]
                    pc_mean, pc_std = pairwise_cos_stats(z0)
                    au = alignment_uniformity(zs[0], zs[1] if len(zs) > 1 else None)
                    m: Dict[str, float] = {
                        "train/loss_total": float(loss.item()),
                        "train/loss_pull": float(loss_pull.item()),
                        "train/loss_sigreg": float(loss_sig.item()),
                        "repr/pairwise_cos_mean": pc_mean,
                        "repr/pairwise_cos_std": pc_std,
                        "repr/alignment": au.get("alignment", 0.0),
                        "repr/uniformity": au.get("uniformity", 0.0),
                        "opt/grad_norm_encoder": grad_global_norm(encoder.parameters()),
                    }
                    for k, v in cov_metrics(z0).items():
                        m[f"encoder/collapse_{k}"] = v
                    for k, v in spectrum_metrics(z0).items():
                        m[f"encoder/spectrum_{k}"] = v
                    for k, v in sig_metrics_acc.items():
                        m[f"sigreg/{k}"] = v
                    logger.log(global_step, m)
            pbar.set_postfix(loss=float(loss.item()))
            global_step += 1
        if epoch % cfg.eval_every_epochs == 0:
            z_test, y_test = extract_embeddings(encoder, test_plain_loader, device, max_samples=cfg.eval_max_samples)
            cm = eval_clustering(z_test, y_test, device=device, kmeans_iters=cfg.kmeans_iters, kmeans_restarts=cfg.kmeans_restarts, silhouette_samples=cfg.silhouette_samples)
            z_bank, y_bank = extract_embeddings(encoder, train_plain_loader, device, max_samples=cfg.knn_max_train)
            z_query = z_test[: cfg.knn_max_test]
            y_query = y_test[: cfg.knn_max_test]
            knn, knn_cm = eval_knn(z_bank, y_bank, z_query, y_query, device=device, k=cfg.knn_k, temperature=cfg.knn_temperature, chunk_size=cfg.knn_chunk_size)
            metrics = {"epoch": float(epoch)}
            for k, v in cm.items():
                metrics[f"eval/{k}"] = v
            for k, v in knn.items():
                metrics[f"eval/{k}"] = v
            logger.log(global_step, metrics)
            if epoch % cfg.report_every_epochs == 0:
                save_pca_scatter(z_test, y_test, path=str(report_dir / f"epoch{epoch:03d}_pca.png"), title=f"hardqr | epoch {epoch} | PCA (test)", max_points=cfg.report_max_points)
                save_confusion_matrix(knn_cm, path=str(report_dir / f"epoch{epoch:03d}_knn_cm.png"), title=f"hardqr | epoch {epoch} | kNN confusion", normalize=True)
                if epoch % cfg.retrieval_every_epochs == 0:
                    save_retrieval_collage(encoder, train_plain_loader, test_plain_loader, device=device, path=str(report_dir / f"epoch{epoch:03d}_retrieval.png"), title=f"hardqr | epoch {epoch} | retrieval (test->train)", bank_max=cfg.retrieval_bank_max, query_num=cfg.retrieval_query_num, topk=cfg.retrieval_topk)
        torch.save({"cfg": cfg.__dict__, "epoch": epoch, "global_step": global_step, "encoder": encoder.state_dict(), "opt": opt.state_dict()}, cfg.ckpt_path)
    logger.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lambda_sigreg", type=float, default=0.25)
    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--pool", type=int, default=512)
    p.add_argument("--hard", type=int, default=64)
    p.add_argument("--rand", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        method="train_pure_sigreg_HardQR",
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        sigreg_lambda=args.lambda_sigreg,
        num_views=args.num_views,
        hard_pool_size=args.pool,
        hard_num_slices=args.hard,
        random_extra_slices=args.rand,
        logdir=(args.logdir or f"runs/train_pure_sigreg_HardQR_seed{args.seed}"),
        ckpt_path=(args.ckpt or f"checkpoints/train_pure_sigreg_HardQR_seed{args.seed}.pt"),
    )
    train(cfg)
