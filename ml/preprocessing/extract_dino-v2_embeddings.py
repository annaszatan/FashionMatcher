"""
This file extracts baseline DINOv2 embeddings for the gallery and query sets based on the provided config
"""

import os
import sys
import argparse
import glob

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

CURRENT_DIR = os.path.dirname(__file__)
ML_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.data.datasets.deepfashion2_dataset import ClothingRetrievalDataset


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_dino_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_dino_model(backbone_name: str, device: str):
    allowed = {"dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}
    if backbone_name not in allowed:
        raise ValueError(f"Unknown DINOv2 backbone: {backbone_name}")

    # Downloads once and then reuses the local torch hub cache.
    model = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)

    model.eval()
    model.to(device)
    return model


def extract_from_batch(images: torch.Tensor, model: torch.nn.Module, token_mode: str) -> np.ndarray:
    if token_mode == "cls":
        feats = model(images)  # (B, D)
        return feats.detach().float().cpu().numpy()
    if token_mode == "patch":
        out = model.forward_features(images)
        patch = out["x_norm_patchtokens"]  # (B, N_patches, D)
        return patch.detach().float().cpu().numpy()

    out = model.forward_features(images)
    cls_token = out["x_norm_clstoken"].unsqueeze(1)   # (B, 1, D)
    patch = out["x_norm_patchtokens"]                 # (B, N_patches, D)
    return torch.cat([cls_token, patch], dim=1).detach().float().cpu().numpy()


@torch.no_grad()
def extract_embeddings_for_image_paths(
    image_paths,
    cfg: dict,
    model: torch.nn.Module,
    transform,
):
    device = cfg["model"]["device"]
    normalize = cfg["model"].get("normalize_embeddings", True)
    token_mode = get_output_token_mode(cfg)

    feats_list = []
    kept_paths = []
    item_ids = []

    print(f"Extracting dynamic query embeddings (num_images={len(image_paths)}, output_tokens={token_mode})")
    for p in tqdm(image_paths, desc="dynamic_query"):
        if not os.path.isfile(p):
            print(f"[WARN] Query image not found, skipping: {p}")
            continue

        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device, non_blocking=True)
        feats = extract_from_batch(x, model, token_mode)  # (1, D) or (1, L, D)

        feats_list.append(feats)
        kept_paths.append(p)
        item_ids.append(os.path.splitext(os.path.basename(p))[0])

    if len(feats_list) == 0:
        raise ValueError("No valid query images found for dynamic extraction.")

    all_feats = np.concatenate(feats_list, axis=0)

    if normalize:
        if all_feats.ndim == 2:
            norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-10
            all_feats = all_feats / norms
        else:
            norms = np.linalg.norm(all_feats, axis=-1, keepdims=True) + 1e-10
            all_feats = all_feats / norms

    return all_feats, np.array(kept_paths), np.array(item_ids)


def get_output_token_mode(cfg: dict) -> str:
    """
    cls: model(images) head output (legacy, (N, D))
    patch: frozen patch token sequence ((N, N_patches, D))
    cls_patch: concatenate [CLS, patch tokens] ((N, 1 + N_patches, D))
    """
    mode = str(cfg.get("embedding", {}).get("output_tokens", "cls")).strip().lower()
    if mode not in ("cls", "patch", "cls_patch"):
        raise ValueError('embedding.output_tokens must be one of: "cls", "patch", "cls_patch"')
    return mode


@torch.no_grad()
def extract_embeddings_for_split(
    split: str,
    cfg: dict,
    model: torch.nn.Module,
    transform,
):
    metadata_path = cfg["paths"]["metadata"]
    image_root = cfg["paths"]["image_root"]
    batch_size = cfg["embedding"]["batch_size"]
    num_workers = cfg["embedding"]["num_workers"]
    device = cfg["model"]["device"]
    normalize = cfg["model"].get("normalize_embeddings", True)
    token_mode = get_output_token_mode(cfg)

    if ClothingRetrievalDataset is None:
        raise ImportError(
            "ClothingRetrievalDataset import failed. Install dataset dependencies or use dynamic query mode."
        )

    dataset = ClothingRetrievalDataset(
        metadata_csv=metadata_path,
        root_dir=image_root,
        split=split,
        transform=transform,
    )

    if len(dataset) == 0:
        print(f"[{split}] split has no samples. Skipping.")
        return None, None, None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_feats = []
    all_paths = []
    all_item_ids = []

    print(
        f"Extracting DINOv2 embeddings for split: {split} (num_samples={len(dataset)}, "
        f"output_tokens={token_mode})"
    )

    for images, targets in tqdm(dataloader, desc=f"{split}"):
        images = images.to(device, non_blocking=True)

        feats = extract_from_batch(images, model, token_mode)

        all_feats.append(feats)
        all_paths.extend(targets["image_path"])
        all_item_ids.extend(targets["item_id"])

    all_feats = np.concatenate(all_feats, axis=0)  # (N, D) or (N, L, D)

    if normalize:
        if all_feats.ndim == 2:
            norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-10
            all_feats = all_feats / norms
        else:
            norms = np.linalg.norm(all_feats, axis=-1, keepdims=True) + 1e-10
            all_feats = all_feats / norms

    all_paths = np.array(all_paths)
    all_item_ids = np.array(all_item_ids)

    return all_feats, all_paths, all_item_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("scripts", "config", "baseline.yaml"),
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--query_images",
        nargs="*",
        default=None,
        help="Optional list of query image paths (dynamic mode).",
    )
    parser.add_argument(
        "--query_glob",
        type=str,
        default=None,
        help="Optional glob for dynamic query images (e.g., uploaded_images/*.jpg).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Get feature directory
    features_dir_cfg = cfg["paths"].get("features_dir", "")
    if os.path.isabs(features_dir_cfg):
        features_dir = features_dir_cfg
    elif features_dir_cfg.startswith("ml/") or features_dir_cfg.startswith("ml\\"):
        features_dir = os.path.join(PROJECT_ROOT, features_dir_cfg)
    else:
        suffix = features_dir_cfg if features_dir_cfg else "default"
        features_dir = os.path.join(ML_ROOT, "features", suffix)
    features_dir = os.path.normpath(features_dir)

    os.makedirs(features_dir, exist_ok=True)

    device = cfg["model"]["device"]
    backbone = cfg["model"]["backbone"]

    # Load the DINOv2 model
    print(f"Loading DINOv2 backbone: {backbone} on device: {device}")
    model = load_dino_model(backbone, device)
    transform = build_dino_transform()
    token_mode = get_output_token_mode(cfg)
    print(f"embedding.output_tokens = {token_mode}")

    # Dynamic mode: extract query embeddings from arbitrary image paths.
    dynamic_paths = []
    if args.query_images:
        dynamic_paths.extend(args.query_images)
    if args.query_glob:
        dynamic_paths.extend(sorted(glob.glob(args.query_glob)))

    if len(dynamic_paths) > 0:
        feats, paths, item_ids = extract_embeddings_for_image_paths(
            image_paths=dynamic_paths,
            cfg=cfg,
            model=model,
            transform=transform,
        )

        feats_path = os.path.join(features_dir, "query_feats.npy")
        paths_path = os.path.join(features_dir, "query_paths.npy")
        item_ids_path = os.path.join(features_dir, "query_item_ids.npy")

        np.save(feats_path, feats)
        np.save(paths_path, paths)
        np.save(item_ids_path, item_ids)

        print(
            "[dynamic_query] Saved features to:\n"
            f"  {feats_path}\n  {paths_path}\n  {item_ids_path}"
        )
        return

    # Embedding extraction for gallery vs query split based on metadata label
    splits = ["gallery", "query"]
    # Extract data for each split and save to features directory
    for split in splits:
        feats, paths, item_ids = extract_embeddings_for_split(
            split=split,
            cfg=cfg,
            model=model,
            transform=transform,
        )
        # If nothing is returned skip saving
        if feats is None:
            continue

        feats_path = os.path.join(features_dir, f"{split}_feats.npy")
        paths_path = os.path.join(features_dir, f"{split}_paths.npy")
        item_ids_path = os.path.join(features_dir, f"{split}_item_ids.npy")

        np.save(feats_path, feats)
        np.save(paths_path, paths)
        np.save(item_ids_path, item_ids)

        print(
            f"[{split}] Saved features to:\n"
            f"  {feats_path}\n  {paths_path}\n  {item_ids_path}"
        )


if __name__ == "__main__":
    main()
