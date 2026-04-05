import os
import sys
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
_DINOV2_ROOT = os.path.join(PROJECT_ROOT, "models", "dinov2")
if os.path.isdir(_DINOV2_ROOT):
    sys.path.insert(0, _DINOV2_ROOT)

from data.datasets.deepfashion2_dataset import ClothingRetrievalDataset
from models.components.backbone import build_retrieval_model


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(path: str, root: str) -> str:
    if not path or os.path.isabs(path):
        return path or ""
    return os.path.normpath(os.path.join(root, path))


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def forward_batch_tokens(model, images, output_tokens: str, normalize: bool):
    if output_tokens == "projection":
        feats = model(images)
        x = feats.detach().float().cpu().numpy()
        if normalize:
            norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
            x = x / norms
        return x

    if output_tokens != "cls_patch":
        raise ValueError('output_tokens must be "cls_patch" or "projection"')

    out = model.backbone.forward_features(images)
    if not isinstance(out, dict) or "x_norm_clstoken" not in out or "x_norm_patchtokens" not in out:
        raise RuntimeError("backbone.forward_features did not return expected dict keys")
    cls_t = out["x_norm_clstoken"].unsqueeze(1)
    patch = out["x_norm_patchtokens"]
    feats = torch.cat([cls_t, patch], dim=1).detach().float().cpu().numpy()
    if normalize:
        norms = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-10
        feats = feats / norms
    return feats


@torch.no_grad()
def extract_split(model, loader, device, output_tokens: str, normalize: bool):
    model.eval()
    all_feats = []
    all_paths = []
    all_item_ids = []
    for images, targets in tqdm(loader, desc="Extract"):
        images = images.to(device, non_blocking=True)
        batch = forward_batch_tokens(model, images, output_tokens, normalize)
        all_feats.append(batch)
        all_paths.extend(targets["image_path"])
        all_item_ids.extend(targets["item_id"])
    feats = np.concatenate(all_feats, axis=0).astype(np.float32)
    return feats, np.array(all_paths), np.array(all_item_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config YAML used for the checkpoint (e.g. scripts/config/C_lora_peft.yaml)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Output directory for .npy (default: paths.features_dir in config, or features/<exp>_tokens)",
    )
    parser.add_argument(
        "--output_tokens",
        type=str,
        default="cls_patch",
        choices=("cls_patch", "projection"),
        help="cls_patch: (N,L,D) for TransformerEmbeddingModel+use_cls_token; projection: (N,D) flat head output",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = _resolve(args.config, PROJECT_ROOT)
    ckpt_path = _resolve(args.checkpoint, PROJECT_ROOT)
    cfg = load_config(cfg_path)

    device = cfg.get("model", {}).get("device", "cuda")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    strategy = cfg.get("training", {}).get("strategy", "linear_probe")
    if strategy != "lora_peft" and args.output_tokens == "cls_patch":
        print(
            f"[WARN] training.strategy={strategy} is not lora_peft; "
            "checkpoint must still be a RetrievalBackbone with ViT forward_features."
        )

    model = build_retrieval_model(
        backbone_name=cfg.get("model", {}).get("backbone", "dinov2_vitb14"),
        training_strategy=strategy,
        embedding_dim=cfg.get("model", {}).get("embedding_dim", 256),
        head_type=cfg.get("model", {}).get("head_type", "linear"),
        head_hidden_dim=cfg.get("model", {}).get("head_hidden_dim", 512),
        head_num_layers=cfg.get("model", {}).get("head_num_layers", 2),
        num_unfrozen_blocks=cfg.get("training", {}).get("num_unfrozen_blocks", 2),
        normalize_output=True,
        device=str(device),
        lora_r=cfg.get("model", {}).get("lora_r", 8),
        lora_alpha=cfg.get("model", {}).get("lora_alpha", 16.0),
        lora_dropout=cfg.get("model", {}).get("lora_dropout", 0.0),
        lora_target_modules=cfg.get("model", {}).get("lora_target_modules"),
        train_projection_head=cfg.get("model", {}).get("train_projection_head", True),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    normalize = bool(cfg.get("model", {}).get("normalize_embeddings", True))
    metadata_path = _resolve(cfg["paths"]["metadata"], PROJECT_ROOT)
    image_root = _resolve(cfg["paths"]["image_root"], PROJECT_ROOT)

    features_dir = args.features_dir or cfg.get("paths", {}).get("features_dir")
    if not features_dir:
        suffix = "cls_patch" if args.output_tokens == "cls_patch" else "projection"
        exp = cfg.get("experiment_name", "lora_tokens")
        features_dir = os.path.join("features", f"{exp}_{suffix}")
    features_dir = _resolve(features_dir, PROJECT_ROOT)
    os.makedirs(features_dir, exist_ok=True)

    batch_size = cfg.get("embedding", {}).get("batch_size", 64)
    num_workers = cfg.get("embedding", {}).get("num_workers", 4)
    transform = build_transform()

    print(f"[INFO] output_tokens={args.output_tokens}, features_dir={features_dir}, normalize={normalize}")

    for split in ["gallery", "query"]:
        dataset = ClothingRetrievalDataset(
            metadata_csv=metadata_path,
            root_dir=image_root,
            split=split,
            transform=transform,
        )
        if len(dataset) == 0:
            print(f"[{split}] no samples, skip.")
            continue
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        feats, paths, item_ids = extract_split(
            model, loader, device, args.output_tokens, normalize
        )
        np.save(os.path.join(features_dir, f"{split}_feats.npy"), feats)
        np.save(os.path.join(features_dir, f"{split}_paths.npy"), paths)
        np.save(os.path.join(features_dir, f"{split}_item_ids.npy"), item_ids)
        print(f"[{split}] saved shape={feats.shape} -> {features_dir}")


if __name__ == "__main__":
    main()
