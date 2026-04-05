"""
Build FAISS index from learned embedding model outputs.

Pipeline:
1) Load raw frozen DINOv2 features (gallery/query) from config paths.input_features_dir
2) Load trained checkpoint (best.pt) and project features through embedding model
3) Save projected features + ids/paths
4) Build and save FAISS IndexFlatIP from projected gallery features
"""
import os
import sys
import argparse
import shutil

import yaml
import numpy as np
import torch
import faiss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.data.datasets.embedding_dataset import FrozenEmbeddingDataset
from ml.models.embedding_models import build_embedding_model

# Allow for path resolution (absolute or relative to project root)
def resolve(path: str, root: str) -> str:
    if not path or os.path.isabs(path):
        return path or ""
    return os.path.normpath(os.path.join(root, path))


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


def copy_if_exists(src: str, dst: str):
    if os.path.isfile(src):
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Project frozen features with trained checkpoint and build FAISS index."
    )
    parser.add_argument("--config", type=str, required=True, help="Embedding experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument(
        "--output_features_dir",
        type=str,
        default=None,
        help="Directory to save projected features (default: features/<experiment_name>_projected)",
    )
    parser.add_argument(
        "--output_index",
        type=str,
        default=None,
        help="FAISS index output path (default: indices/<experiment_name>/faiss_gallery_index.ip)",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Projection batch size")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = resolve(args.config, PROJECT_ROOT)
    ckpt_path = resolve(args.checkpoint, PROJECT_ROOT)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("experiment_name", "embedding_exp")
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})
    input_features_dir = resolve(paths_cfg.get("input_features_dir", "ml/features/A_frozen_baseline"), PROJECT_ROOT)

    # Create output feature and index directories
    out_features_dir = args.output_features_dir or os.path.join("ml", "features", f"{experiment_name}_projected")
    out_features_dir = resolve(out_features_dir, PROJECT_ROOT)
    os.makedirs(out_features_dir, exist_ok=True)

    out_index = args.output_index or os.path.join("ml", "indices", experiment_name, "faiss_gallery_index.ip")
    out_index = resolve(out_index, PROJECT_ROOT)
    os.makedirs(os.path.dirname(out_index), exist_ok=True)

    device_name = model_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    # Create gallery dataset
    gallery_ds = FrozenEmbeddingDataset(input_features_dir, split="gallery")
    
    # Try to load query, but make it optional (only needed for evaluation)
    try:
        query_ds = FrozenEmbeddingDataset(input_features_dir, split="query")
    except FileNotFoundError:
        print(f"[WARNING] Query features not found in {input_features_dir}; skipping query projection")
        query_ds = None

    feat_arr = gallery_ds.features
    if feat_arr.ndim == 3:
        seq_len_i, token_dim_i = int(feat_arr.shape[1]), int(feat_arr.shape[2])
        model = build_embedding_model(
            model_type=model_cfg.get("type", "transformer"),
            cfg=model_cfg,
            seq_len=seq_len_i,
            token_dim=token_dim_i,
        ).to(device)
    else:
        input_dim = int(feat_arr.shape[1])
        model = build_embedding_model(
            model_type=model_cfg.get("type", "transformer"),
            cfg=model_cfg,
            input_dim=input_dim,
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    print(f"[INFO] Projecting gallery/query with checkpoint: {ckpt_path}")
    g_proj = forward_all(model, gallery_ds.features.astype(np.float32), device, batch_size=args.batch_size)
    print(f"[INFO] gallery projected shape: {g_proj.shape}")
    
    if query_ds is not None:
        q_proj = forward_all(model, query_ds.features.astype(np.float32), device, batch_size=args.batch_size)
        print(f"[INFO] query projected shape: {q_proj.shape}")
    else:
        q_proj = None

    np.save(os.path.join(out_features_dir, "gallery_feats.npy"), g_proj)
    np.save(os.path.join(out_features_dir, "gallery_item_ids.npy"), gallery_ds.item_ids)
    
    if q_proj is not None:
        np.save(os.path.join(out_features_dir, "query_feats.npy"), q_proj)
        np.save(os.path.join(out_features_dir, "query_item_ids.npy"), query_ds.item_ids)

    copy_if_exists(
        os.path.join(input_features_dir, "gallery_paths.npy"),
        os.path.join(out_features_dir, "gallery_paths.npy"),
    )
    if query_ds is not None:
        copy_if_exists(
            os.path.join(input_features_dir, "query_paths.npy"),
            os.path.join(out_features_dir, "query_paths.npy"),
        )

    print(f"[INFO] Saved projected features to: {out_features_dir}")

    index = faiss.IndexFlatIP(g_proj.shape[1])
    index.add(g_proj.astype(np.float32))
    faiss.write_index(index, out_index)
    print(f"[INFO] Saved FAISS index to: {out_index}")
    print(f"[INFO] Index ntotal={index.ntotal}, dim={index.d}")


if __name__ == "__main__":
    main()

