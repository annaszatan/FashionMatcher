"""
The purpose of this file is to extract DINOv2 features for the demo gallery images

Output:
- gallery_feats.npy: (N, D) array of DINOv2 features for N gallery items
- gallery_item_ids.npy: (N,) array of item IDs (string) for each gallery item
- gallery_paths.npy: (N,) array of image paths (string) for each gallery item   
- gallery_product_names.npy: (N,) array of product names (string) for each gallery item
- gallery_source_urls.npy: (N,) array of source URLs (string) for each gallery item (if available in annos)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ML_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Demo gallery is small - images taken from American Eagle
DEFAULT_IMAGES_ROOT = os.path.join(ML_ROOT, "demo_data", "gallery_data", "images")
DEFAULT_ANNOS_ROOT = os.path.join(ML_ROOT, "demo_data", "gallery_data", "annos")
DEFAULT_OUTPUT_BASE = os.path.join(ML_ROOT, "features", "demo_data")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_BASE, "american_eagle_gallery")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_dino_model(backbone_name: str, device: str) -> torch.nn.Module:
    allowed = {"dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}
    if backbone_name not in allowed:
        raise ValueError(f"Unknown DINOv2 backbone: {backbone_name}")

    # Downloads once, then reuses local torch hub cache.
    model = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)

    model.eval()
    model.to(device)
    return model


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_first_string(obj: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in obj and obj[k] is not None:
            val = str(obj[k]).strip()
            if val:
                return val

    for _, v in obj.items():
        if isinstance(v, dict):
            for k in keys:
                if k in v and v[k] is not None:
                    val = str(v[k]).strip()
                    if val:
                        return val
    return None


def _extract_item_id(obj: dict, fallback: str) -> str:
    # Prefer explicit identifiers; fallback to product-oriented fields; last resort is image stem.
    id_like_keys = ["item_id", "id", "pair_id", "style", "sku", "goods_id"]
    product_like_keys = ["product_name", "product", "name", "title", "link", "url", "website"]

    val = _extract_first_string(obj, id_like_keys)
    if val is not None:
        return val

    val = _extract_first_string(obj, product_like_keys)
    if val is not None:
        return val

    return fallback


def _extract_product_name(obj: dict) -> str:
    keys = ["product_name", "product", "name", "title"]
    val = _extract_first_string(obj, keys)
    return val if val is not None else ""


def _extract_source_url(obj: dict) -> str:
    keys = ["link", "url", "website", "source_url", "product_url"]
    val = _extract_first_string(obj, keys)
    return val if val is not None else ""


class DemoGalleryDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        annos_root: str,
        transform,
        require_annos: bool = True,
        save_paths_mode: str = "relative",
    ):
        self.images_root = images_root
        self.annos_root = annos_root
        self.transform = transform
        self.require_annos = require_annos
        self.save_paths_mode = save_paths_mode

        images_dir = images_root
        annos_dir = annos_root
        output_base_dir = images_root

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images folder not found: {images_dir}")
        if not os.path.isdir(annos_dir):
            raise FileNotFoundError(f"Annotation folder not found: {annos_dir}")

        self.images_dir = images_dir
        self.annos_dir = annos_dir
        self.output_base_dir = output_base_dir

        self.samples: List[Dict[str, str]] = []

        for root, _, files in os.walk(images_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue

                img_path = os.path.join(root, name)
                rel_to_subset = os.path.relpath(img_path, images_dir)
                rel_no_ext = os.path.splitext(rel_to_subset)[0]
                anno_path = os.path.join(annos_dir, rel_no_ext + ".json")

                if self.require_annos and (not os.path.isfile(anno_path)):
                    continue

                if self.save_paths_mode == "relative":
                    path_for_output = os.path.relpath(img_path, self.output_base_dir)
                else:
                    path_for_output = os.path.abspath(img_path)

                self.samples.append(
                    {
                        "img_path": img_path,
                        "anno_path": anno_path,
                        "output_path": path_for_output.replace("\\", "/"),
                    }
                )

        self.samples.sort(key=lambda x: x["output_path"])

        if len(self.samples) == 0:
            raise ValueError("No matched gallery samples found. Check folder layout and options.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        anno_path = sample["anno_path"]
        out_path = sample["output_path"]

        image = Image.open(img_path).convert("RGB")

        item_id = os.path.splitext(os.path.basename(img_path))[0]
        product_name = ""
        source_url = ""
        if os.path.isfile(anno_path):
            anno = _safe_read_json(anno_path)
            if isinstance(anno, dict):
                item_id = _extract_item_id(anno, item_id)
                product_name = _extract_product_name(anno)
                source_url = _extract_source_url(anno)

        tensor = self.transform(image)
        return tensor, str(item_id), out_path, product_name, source_url


def _extract_token_mode(model: torch.nn.Module, images: torch.Tensor, token_mode: str) -> torch.Tensor:
    if token_mode == "cls":
        return model(images)

    out = model.forward_features(images)
    if token_mode == "patch":
        return out["x_norm_patchtokens"]

    cls_token = out["x_norm_clstoken"].unsqueeze(1)
    patch = out["x_norm_patchtokens"]
    return torch.cat([cls_token, patch], dim=1)


@torch.no_grad()
def build_gallery_features(
    dataset: Dataset,
    model: torch.nn.Module,
    token_mode: str,
    device: str,
    batch_size: int,
    num_workers: int,
    normalize: bool,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
    )

    all_feats = []
    all_item_ids: List[str] = []
    all_paths: List[str] = []
    all_product_names: List[str] = []
    all_source_urls: List[str] = []

    for images, item_ids, paths, product_names, source_urls in tqdm(loader, desc="Extracting gallery"):
        images = images.to(device, non_blocking=True)
        feats = _extract_token_mode(model, images, token_mode)
        feats_np = feats.detach().float().cpu().numpy()
        all_feats.append(feats_np)
        all_item_ids.extend(item_ids)
        all_paths.extend(paths)
        all_product_names.extend(product_names)
        all_source_urls.extend(source_urls)

    gallery_feats = np.concatenate(all_feats, axis=0)

    if normalize:
        if gallery_feats.ndim == 2:
            norms = np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-10
            gallery_feats = gallery_feats / norms
        else:
            norms = np.linalg.norm(gallery_feats, axis=-1, keepdims=True) + 1e-10
            gallery_feats = gallery_feats / norms

    return (
        gallery_feats,
        np.array(all_item_ids),
        np.array(all_paths),
        np.array(all_product_names),
        np.array(all_source_urls),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract demo gallery features and save gallery_feats.npy"
        )
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=DEFAULT_IMAGES_ROOT,
        help="Path to images root folder",
    )
    parser.add_argument(
        "--annos_root",
        type=str,
        default=DEFAULT_ANNOS_ROOT,
        help="Path to annos root folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .npy files",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINOv2 backbone",
    )
    parser.add_argument(
        "--token_mode",
        type=str,
        default="cls",
        choices=["cls", "patch", "cls_patch"],
        help="Embedding output format",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize output features (recommended for cosine/IP retrieval)",
    )
    parser.add_argument(
        "--allow_missing_annos",
        action="store_true",
        help="If set, keep images even when matching json is missing",
    )
    parser.add_argument(
        "--save_paths_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="How to store gallery_paths.npy entries",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE, output_dir)
    output_dir = os.path.normpath(output_dir)

    device = args.device
    if device == "cuda" and (not torch.cuda.is_available()):
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)

    transform = build_transform(args.image_size)
    dataset = DemoGalleryDataset(
        images_root=args.images_root,
        annos_root=args.annos_root,
        transform=transform,
        require_annos=(not args.allow_missing_annos),
        save_paths_mode=args.save_paths_mode,
    )

    print(f"[INFO] Found {len(dataset)} gallery samples")
    print(f"[INFO] Loading model {args.backbone} on {device}")
    model = load_dino_model(args.backbone, device)

    feats, item_ids, paths, product_names, source_urls = build_gallery_features(
        dataset=dataset,
        model=model,
        token_mode=args.token_mode,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
    )

    feats_path = os.path.join(output_dir, "gallery_feats.npy")
    item_ids_path = os.path.join(output_dir, "gallery_item_ids.npy")
    paths_path = os.path.join(output_dir, "gallery_paths.npy")
    product_names_path = os.path.join(output_dir, "gallery_product_names.npy")
    source_urls_path = os.path.join(output_dir, "gallery_source_urls.npy")

    np.save(feats_path, feats.astype(np.float32))
    np.save(item_ids_path, item_ids)
    np.save(paths_path, paths)
    np.save(product_names_path, product_names)
    np.save(source_urls_path, source_urls)

    print("[INFO] Saved gallery files:")
    print(f"  - {feats_path} shape={feats.shape}")
    print(f"  - {item_ids_path} len={len(item_ids)}")
    print(f"  - {paths_path} len={len(paths)}")
    print(f"  - {product_names_path} len={len(product_names)}")
    print(f"  - {source_urls_path} len={len(source_urls)}")


if __name__ == "__main__":
    main()
