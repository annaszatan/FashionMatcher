"""
Train retrieval model on precomputed frozen DINOv2 embeddings.
Model input: frozen embeddings (.npy), architecture: transformer or cnn.
"""
import os
import sys
import argparse
import json
import logging

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from ml.data.datasets.embedding_dataset import FrozenEmbeddingDataset
from ml.models.embedding_models import build_embedding_model
from ml.training.metric_losses import get_metric_loss
from ml.training import evaluate_retrieval as ev_retrieval


def _resolve(path: str, root: str) -> str:
    if not path or os.path.isabs(path):
        return path or ""
    return os.path.normpath(os.path.join(root, path))


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def forward_all(model, feats_np, device, batch_size=256):
    model.eval()
    outs = []
    n = feats_np.shape[0]
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        x = torch.from_numpy(feats_np[s:e]).to(device)
        y = model(x).cpu().numpy().astype(np.float32)
        outs.append(y)
    return np.concatenate(outs, axis=0)


def run_validation(model, gallery_feats, gallery_labels, query_feats, query_labels, device):
    g = forward_all(model, gallery_feats, device)
    q = forward_all(model, query_feats, device)
    index = faiss.IndexFlatIP(g.shape[1])
    index.add(g)
    return ev_retrieval.evaluate_retrieval_multi_topk(
        index, q, gallery_labels, query_labels, recall_ks=(1, 5, 10), map_ks=(10, 50)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg = load_config(_resolve(args.config, PROJECT_ROOT))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment_name = cfg.get("experiment_name", "embedding_exp")
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    features_dir = _resolve(paths.get("input_features_dir", "ml/features/A_frozen_baseline"), PROJECT_ROOT)
    results_dir = _resolve(paths.get("results_dir", "ml/results"), PROJECT_ROOT)
    ckpt_root = _resolve(paths.get("checkpoint_dir", "ml/checkpoints"), PROJECT_ROOT)
    log_dir = _resolve(paths.get("log_dir", "ml/logs"), PROJECT_ROOT)

    if not os.path.isdir(features_dir):
        raise FileNotFoundError(f"Frozen feature directory not found: {features_dir}")

    exp_dir = os.path.join(results_dir, "experiments", experiment_name)
    ckpt_dir = os.path.join(ckpt_root, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    device_name = model_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    train_ds = FrozenEmbeddingDataset(features_dir, split=training.get("train_split", "gallery"))
    val_gallery_ds = FrozenEmbeddingDataset(features_dir, split="gallery")
    val_query_ds = FrozenEmbeddingDataset(features_dir, split="query")

    input_dim = int(train_ds.features.shape[1])
    model = build_embedding_model(
        model_type=model_cfg.get("type", "transformer"),
        input_dim=input_dim,
        cfg=model_cfg,
    ).to(device)
    if args.resume:
        ckpt = torch.load(_resolve(args.resume, PROJECT_ROOT), map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    trainable = count_trainable_params(model)
    logger.info(f"Trainable parameters: {trainable}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    loss_fn = get_metric_loss(
        loss_type=training.get("loss_type", "supcon"),
        temperature=float(training.get("temperature", 0.07)),
        margin=float(training.get("triplet_margin", 0.2)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )

    g_feats = val_gallery_ds.features.astype(np.float32)
    q_feats = val_query_ds.features.astype(np.float32)
    g_labels = val_gallery_ds.item_ids
    q_labels = val_query_ds.item_ids

    epochs = int(training.get("epochs", 30))
    val_every = int(training.get("val_every", 2))
    best_recall10 = -1.0
    best_epoch = -1
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        batches = 0
        for feats, labels, _targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(feats)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        train_loss = running / max(1, batches)
        logger.info(f"Epoch {epoch+1} loss: {train_loss:.4f}")

        if (epoch + 1) % val_every == 0 or epoch == 0:
            metrics = run_validation(model, g_feats, g_labels, q_feats, q_labels, device)
            logger.info(f"Validation: {metrics}")
            recall10 = float(metrics.get("Recall@10", 0.0))
            with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                json.dump({**metrics, "epoch": epoch + 1, "loss": train_loss}, f, indent=2)
            if recall10 > best_recall10:
                best_recall10 = recall10
                best_epoch = epoch + 1
                best_metrics = {**metrics}
                payload = {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "metrics": metrics,
                    "config": cfg,
                    "trainable_params": trainable,
                    "input_dim": input_dim,
                }
                torch.save(payload, os.path.join(ckpt_dir, "best.pt"))
                logger.info(f"Saved best checkpoint (Recall@10={best_recall10:.4f})")

    final = {
        "experiment_name": experiment_name,
        "backbone": "dinov2_frozen_embedding",
        "training_strategy": model_cfg.get("type", "transformer"),
        "loss_type": training.get("loss_type", "supcon"),
        "trainable_params": trainable,
        "best_epoch": best_epoch,
        "Recall@1": best_metrics.get("Recall@1", 0.0),
        "Recall@5": best_metrics.get("Recall@5", 0.0),
        "Recall@10": best_metrics.get("Recall@10", 0.0),
        "mAP@10": best_metrics.get("mAP@10", 0.0),
        "mAP@50": best_metrics.get("mAP@50", 0.0),
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Done. Best Recall@10={final['Recall@10']:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
