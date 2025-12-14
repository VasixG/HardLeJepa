import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import TrainConfig
from data.mnist import MNISTJEPACollate, make_mnist_ssl_and_plain_loaders
from models.encoder import ResNetMNISTEncoder
from models.predictor import JEPAPredictor
from losses.jepa import jepa_pred_loss
from losses.sigreg import SIGReg
from utils.seed import set_seed
from utils.ema import ema_update
from utils.logger import Logger
from utils.plots import save_pca_scatter, save_tsne_scatter, save_confusion_matrix
from utils.retrieval import save_retrieval_collage
from metrics.repr_stats import grad_global_norm, cov_metrics, spectrum_metrics, pairwise_cos_stats, alignment_uniformity
from evaluate import extract_embeddings, eval_clustering, eval_knn

def build_models(cfg: TrainConfig):
    student = ResNetMNISTEncoder(cfg.emb_dim)
    teacher = ResNetMNISTEncoder(cfg.emb_dim)
    predictor = JEPAPredictor(cfg.emb_dim)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    sigreg = None
    if cfg.method.lower() == "lejepa":
        sigreg = SIGReg(cfg.emb_dim, num_slices=cfg.sigreg_num_slices, t_min=cfg.sigreg_t_min, t_max=cfg.sigreg_t_max, t_steps=cfg.sigreg_t_steps)
    return student, teacher, predictor, sigreg

def train_one_run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg.logdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    collate = MNISTJEPACollate(cfg.out_size, cfg.num_targets, (cfg.ctx_area_min, cfg.ctx_area_max), (cfg.tgt_area_min, cfg.tgt_area_max))
    train_ssl_loader, train_plain_loader, test_plain_loader = make_mnist_ssl_and_plain_loaders(cfg.batch_size, cfg.num_workers, collate, train_fraction=cfg.train_fraction, test_fraction=cfg.test_fraction, seed=cfg.seed)
    student, teacher, predictor, sigreg = build_models(cfg)
    student.to(device); teacher.to(device); predictor.to(device)
    if sigreg is not None:
        sigreg.to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = Logger(cfg.logdir, str(Path(cfg.logdir) / "metrics.csv"), use_tb=True)
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        student.train(); predictor.train(); teacher.eval()
        pbar = tqdm(train_ssl_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for (views, _y) in pbar:
            ctx = views.ctx.to(device, non_blocking=True)
            tgt_imgs = [t.to(device, non_blocking=True) for t in views.tgt_imgs]
            tgt_pos = [p.to(device, non_blocking=True) for p in views.tgt_pos]
            z_ctx_s = student(ctx)
            with torch.no_grad():
                z_tgts_t = [teacher(ti) for ti in tgt_imgs]
                preds = [predictor(z_ctx_s, tgt_pos[i]) for i in range(cfg.num_targets)]

                preds_n = [F.normalize(p, dim=-1) for p in preds]
                z_tgts_t_n = [F.normalize(z, dim=-1) for z in z_tgts_t]

                loss_pred = jepa_pred_loss(preds_n, z_tgts_t_n)
            loss_sig = torch.tensor(0.0, device=device)
            if cfg.method.lower() == "lejepa":
                loss_sig = sigreg(z_ctx_s, seed=global_step)
                lam = cfg.sigreg_lambda
                loss = (1.0 - lam) * loss_pred + lam * loss_sig
            else:
                loss = loss_pred
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(student.parameters()) + list(predictor.parameters()), max_norm=cfg.grad_clip)
            opt.step()
            ema_update(teacher, student, cfg.ema_m)
            if global_step % cfg.log_every == 0:
                with torch.no_grad():
                    z_ref = z_tgts_t[0]
                    z_pred = preds[0]
                    cos = F.cosine_similarity(z_pred, z_ref, dim=-1)
                    pc_mean, pc_std = pairwise_cos_stats(z_ref)
                    m: Dict[str, float] = {
                        "train/loss_total": float(loss.item()),
                        "train/loss_pred": float(loss_pred.item()),
                        "train/loss_sigreg": float(loss_sig.item()),
                        "repr/cos_pred_tgt_mean": cos.mean().item(),
                        "repr/cos_pred_tgt_std": cos.std(unbiased=False).item(),
                        "repr/pairwise_cos_mean": pc_mean,
                        "repr/pairwise_cos_std": pc_std,
                        "opt/grad_norm_student": grad_global_norm(student.parameters()),
                        "opt/grad_norm_predictor": grad_global_norm(predictor.parameters()),
                    }
                    for k, v in cov_metrics(z_ref).items():
                        m[f"teacher_target/collapse_{k}"] = v
                    for k, v in spectrum_metrics(z_ref).items():
                        m[f"teacher_target/spectrum_{k}"] = v
                    for k, v in alignment_uniformity(z_ref).items():
                        m[f"teacher_target/{k}"] = v
                    for k, v in cov_metrics(z_ctx_s).items():
                        m[f"student_ctx/collapse_{k}"] = v
                    for k, v in spectrum_metrics(z_ctx_s).items():
                        m[f"student_ctx/spectrum_{k}"] = v
                    for k, v in alignment_uniformity(z_ctx_s).items():
                        m[f"student_ctx/{k}"] = v
                    logger.log(global_step, m)
            pbar.set_postfix(loss=float(loss.item()))
            global_step += 1
        if epoch % cfg.eval_every_epochs == 0:
            z_student_test, y_test = extract_embeddings(student, test_plain_loader, device, max_samples=cfg.eval_max_samples)
            z_teacher_test, y_test2 = extract_embeddings(teacher, test_plain_loader, device, max_samples=cfg.eval_max_samples)
            assert y_test.shape == y_test2.shape
            cm_s = eval_clustering(z_student_test, y_test, device=device, kmeans_iters=cfg.kmeans_iters, kmeans_restarts=cfg.kmeans_restarts, silhouette_samples=cfg.silhouette_samples)
            cm_t = eval_clustering(z_teacher_test, y_test, device=device, kmeans_iters=cfg.kmeans_iters, kmeans_restarts=cfg.kmeans_restarts, silhouette_samples=cfg.silhouette_samples)
            z_bank_s, y_bank = extract_embeddings(student, train_plain_loader, device, max_samples=cfg.knn_max_train)
            z_query_s = z_student_test[: cfg.knn_max_test]
            y_query = y_test[: cfg.knn_max_test]
            knn_s, knn_cm_s = eval_knn(z_bank_s, y_bank, z_query_s, y_query, device=device, k=cfg.knn_k, temperature=cfg.knn_temperature, chunk_size=cfg.knn_chunk_size)
            z_bank_t, y_bank_t = extract_embeddings(teacher, train_plain_loader, device, max_samples=cfg.knn_max_train)
            z_query_t = z_teacher_test[: cfg.knn_max_test]
            knn_t, knn_cm_t = eval_knn(z_bank_t, y_bank_t, z_query_t, y_query, device=device, k=cfg.knn_k, temperature=cfg.knn_temperature, chunk_size=cfg.knn_chunk_size)
            metrics = {"epoch": float(epoch)}
            for k, v in cm_s.items():
                metrics[f"eval_student/{k}"] = v
            for k, v in cm_t.items():
                metrics[f"eval_teacher/{k}"] = v
            for k, v in knn_s.items():
                metrics[f"eval_student/{k}"] = v
            for k, v in knn_t.items():
                metrics[f"eval_teacher/{k}"] = v
            logger.log(global_step, metrics)
            if epoch % cfg.report_every_epochs == 0:
                save_pca_scatter(
                    z_student_test, y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_student_pca.png"),
                    title=f"{cfg.method} | epoch {epoch} | student PCA (test)",
                    max_points=cfg.report_max_points,
                )
                save_tsne_scatter(
                    z_student_test, y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_student_tsne.png"),
                    title=f"{cfg.method} | epoch {epoch} | student t-SNE (test)",
                    max_points=min(cfg.report_max_points, 2000),
                    perplexity=cfg.tsne_perplexity,
                    iters=cfg.tsne_iters,
                    seed=cfg.seed + epoch,
                    learning_rate=cfg.tsne_lr,
                )

                save_pca_scatter(
                    z_teacher_test, y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_teacher_pca.png"),
                    title=f"{cfg.method} | epoch {epoch} | teacher PCA (test)",
                    max_points=cfg.report_max_points,
                )
                save_tsne_scatter(
                    z_teacher_test, y_test,
                    path=str(report_dir / f"epoch{epoch:03d}_teacher_tsne.png"),
                    title=f"{cfg.method} | epoch {epoch} | teacher t-SNE (test)",
                    max_points=min(cfg.report_max_points, 2000),
                    perplexity=cfg.tsne_perplexity,
                    iters=cfg.tsne_iters,
                    seed=cfg.seed + 999 + epoch,
                    learning_rate=cfg.tsne_lr,
                )

                save_confusion_matrix(knn_cm_s, path=str(report_dir / f"epoch{epoch:03d}_student_knn_cm.png"), title=f"{cfg.method} | epoch {epoch} | student kNN confusion", normalize=True)
                save_confusion_matrix(knn_cm_t, path=str(report_dir / f"epoch{epoch:03d}_teacher_knn_cm.png"), title=f"{cfg.method} | epoch {epoch} | teacher kNN confusion", normalize=True)
                if epoch % cfg.retrieval_every_epochs == 0:
                    save_retrieval_collage(student, train_plain_loader, test_plain_loader, device=device, path=str(report_dir / f"epoch{epoch:03d}_student_retrieval.png"), title=f"{cfg.method} | epoch {epoch} | student retrieval (test->train)", bank_max=cfg.retrieval_bank_max, query_num=cfg.retrieval_query_num, topk=cfg.retrieval_topk)
                    save_retrieval_collage(teacher, train_plain_loader, test_plain_loader, device=device, path=str(report_dir / f"epoch{epoch:03d}_teacher_retrieval.png"), title=f"{cfg.method} | epoch {epoch} | teacher retrieval (test->train)", bank_max=cfg.retrieval_bank_max, query_num=cfg.retrieval_query_num, topk=cfg.retrieval_topk)
        torch.save({"cfg": cfg.__dict__, "epoch": epoch, "global_step": global_step, "student": student.state_dict(), "teacher": teacher.state_dict(), "predictor": predictor.state_dict(), "opt": opt.state_dict()}, cfg.ckpt_path)
    logger.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="ijepa", choices=["ijepa", "lejepa"])
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lambda_sigreg", type=float, default=0.25)
    p.add_argument("--num_targets", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        method=args.method,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        sigreg_lambda=args.lambda_sigreg,
        logdir=(args.logdir or f"runs/{args.method}_seed{args.seed}"),
        ckpt_path=(args.ckpt or f"checkpoints/{args.method}_seed{args.seed}.pt"),
        num_targets=(args.num_targets if args.num_targets is not None else TrainConfig.num_targets),
    )
    train_one_run(cfg)
