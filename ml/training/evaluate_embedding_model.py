"""
Evaluate embedding-input retrieval model using frozen DINOv2 features.
"""
import os
import sys
import argparse
import json

import yaml
import numpy as np
import pandas as pd
import torch
import faiss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from ml.models.embedding_models import build_embedding_model
from ml.data.datasets.embedding_dataset import FrozenEmbeddingDataset
from ml.training import evaluate_retrieval as ev_retrieval


def _resolve(path: str, root: str) -> str:
    if not path or os.path.isabs(path):
        return path or ""
    return os.path.normpath(os.path.join(root, path))


@torch.no_grad()
def forward_all(model, feats_np, device, batch_size=256):
    model.eval()
    outs = []
    for s in range(0, feats_np.shape[0], batch_size):
        e = min(feats_np.shape[0], s + batch_size)
        x = torch.from_numpy(feats_np[s:e]).to(device)
        y = model(x).cpu().numpy().astype(np.float32)
        outs.append(y)
    return np.concatenate(outs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    with open(_resolve(args.config, PROJECT_ROOT), "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("experiment_name", "embedding_exp")
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    training = cfg.get("training", {})

    features_dir = _resolve(paths.get("input_features_dir", "ml/features/A_frozen_baseline"), PROJECT_ROOT)
    results_dir = _resolve(paths.get("results_dir", "ml/results"), PROJECT_ROOT)
    exp_dir = args.output_dir or os.path.join(results_dir, "experiments", experiment_name)
    exp_dir = _resolve(exp_dir, PROJECT_ROOT) if not os.path.isabs(exp_dir) else exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    device_name = model_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    gallery_ds = FrozenEmbeddingDataset(features_dir, split="gallery")
    query_ds = FrozenEmbeddingDataset(features_dir, split="query")

    feat_arr = gallery_ds.features
    if feat_arr.ndim == 3:
        sl, td = int(feat_arr.shape[1]), int(feat_arr.shape[2])
        model = build_embedding_model(
            model_type=model_cfg.get("type", "transformer"),
            cfg=model_cfg,
            seq_len=sl,
            token_dim=td,
        ).to(device)
    else:
        input_dim = int(feat_arr.shape[1])
        model = build_embedding_model(
            model_type=model_cfg.get("type", "transformer"),
            cfg=model_cfg,
            input_dim=input_dim,
        ).to(device)
    ckpt = torch.load(_resolve(args.checkpoint, PROJECT_ROOT), map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    g = forward_all(model, gallery_ds.features.astype(np.float32), device)
    q = forward_all(model, query_ds.features.astype(np.float32), device)

    index = faiss.IndexFlatIP(g.shape[1])
    index.add(g)
    metrics = ev_retrieval.evaluate_retrieval_multi_topk(
        index, q, gallery_ds.item_ids, query_ds.item_ids, recall_ks=(1, 5, 10), map_ks=(10, 50)
    )
    backbone_tag = paths.get("input_feature_backbone")
    if not backbone_tag:
        backbone_tag = (
            "dinov2_frozen_patch_tokens" if feat_arr.ndim == 3 else "dinov2_frozen_embedding"
        )
    metrics.update({
        "experiment_name": experiment_name,
        "backbone": backbone_tag,
        "training_strategy": model_cfg.get("type", "transformer"),
        "loss_type": training.get("loss_type", "supcon"),
        "best_epoch": ckpt.get("epoch", ""),
        "trainable_params": ckpt.get("trainable_params", ""),
    })

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
