import os
import random
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import Dataset


def _auto_split_records(records, split, train_ratio=0.8, seed=42):
    """
    Simple auto-split: 80/20 train/test by identity (image_name).
    For test split: query comes from source==0 (user), gallery from source==1 (shop).
    """
    # Group records by image name
    grouped = {}
    for record in records:
        img_name = record.get("image_name", "")
        grouped.setdefault(img_name, []).append(record)

    # Deterministic 80/20 split by identity
    group_keys = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_train = int(round(len(group_keys) * train_ratio))
    train_keys = set(group_keys[:n_train])

    if split == "train":
        result = []
        for key in group_keys:
            if key in train_keys:
                result.extend(grouped[key])
        return result

    # For query/gallery: split test set by source
    query_result = []
    gallery_result = []
    for key in group_keys:
        if key in train_keys:
            continue
        for record in grouped[key]:
            source = record.get("source")
            if source == 0:
                query_result.append(record)
            elif source == 1:
                gallery_result.append(record)

    return query_result if split == "query" else gallery_result


class PortableRetrievalDataset(Dataset):
    def __init__(self, pt_path, split="gallery", root_dir=".", transform=None):
        """
        Load portable .pt dataset with records from ClothingClassificationDataset.
        
        Records expected to have: image_name, source, box, category, style, pair_id, occlusion
        Auto-splits 80/20 train/test by image_name identity, then splits test:
          - query: source == 0 (user)
          - gallery: source == 1 (shop)
        """
        checkpoint = torch.load(pt_path, map_location="cpu")
        self.transform = transform
        self.root_dir = root_dir or "."

        # Load records list
        records = checkpoint.get("records")
        if not isinstance(records, list):
            raise KeyError(f"Expected 'records' key in checkpoint. Found keys: {list(checkpoint.keys())}")

        # Auto-split the data
        self.samples = _auto_split_records(records, split)
        if not self.samples:
            raise ValueError(f"No samples found for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        # Load image from embedded bytes (crop_bytes preferred, else full_bytes)
        crop_bytes = record.get("crop_bytes")
        full_bytes = record.get("full_bytes")
        box = record.get("box")

        if isinstance(crop_bytes, (bytes, bytearray)) and len(crop_bytes) > 0:
            image = Image.open(BytesIO(crop_bytes)).convert("RGB")
        elif isinstance(full_bytes, (bytes, bytearray)) and len(full_bytes) > 0:
            image = Image.open(BytesIO(full_bytes)).convert("RGB")
            if box is not None:
                image = image.crop(tuple(box))
        else:
            # Fallback: load from file path
            image_path = os.path.join(self.root_dir, record.get("image_name"))
            image = Image.open(image_path).convert("RGB")
            if box is not None:
                image = image.crop(tuple(box))

        if self.transform is not None:
            image = self.transform(image)

        # Return image and metadata
        item_id = record.get("item_id")
        if item_id is None:
            pair_id = record.get("pair_id")
            item_id = str(pair_id) if pair_id is not None else str(record.get("image_name"))
        target = {
            "image_name": record.get("image_name"),
            "image_path": record.get("image_name"),
            "source": record.get("source"),
            "category": record.get("category"),
            "style": record.get("style"),
            "pair_id": record.get("pair_id"),
            "occlusion": record.get("occlusion"),
            "item_id": item_id,
        }

        return image, target