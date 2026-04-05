import numpy as np
import torch
from torch.utils.data import Dataset


class FrozenEmbeddingDataset(Dataset):
    """
    Dataset for precomputed frozen embeddings.
    Expected files under features_dir:
      - <split>_feats.npy
      - <split>_item_ids.npy
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "gallery",
        label_mode: str = "item_id",
        style_suffix: str = "style_labels",
        category_suffix: str = "category_labels",
        style_category_suffix: str = "style_category_labels",
    ):
        feats_path = f"{features_dir}/{split}_feats.npy"
        item_ids_path = f"{features_dir}/{split}_item_ids.npy"
        style_path = f"{features_dir}/{split}_{style_suffix}.npy"
        category_path = f"{features_dir}/{split}_{category_suffix}.npy"
        style_category_path = f"{features_dir}/{split}_{style_category_suffix}.npy"

        self.features = np.load(feats_path).astype(np.float32)

        self.item_ids = np.load(item_ids_path, allow_pickle=True).astype(str)
        self.label_mode = str(label_mode).strip().lower()

        if self.label_mode == "item_id":
            raw_labels = self.item_ids
            self.label_key = "item_id"
            self.style_labels = None
            self.category_labels = None
        elif self.label_mode == "style_category":
            self.label_key = "style_category"
            if _exists(style_category_path):
                raw_labels = np.load(style_category_path, allow_pickle=True).astype(str)
                self.style_labels = None
                self.category_labels = None
            else:
                if not _exists(style_path) or not _exists(category_path):
                    raise FileNotFoundError(
                        "style_category mode requires either "
                        f"'{style_category_path}' or both '{style_path}' and '{category_path}'."
                    )
                self.style_labels = np.load(style_path, allow_pickle=True).astype(str)
                self.category_labels = np.load(category_path, allow_pickle=True).astype(str)
                if len(self.style_labels) != len(self.category_labels):
                    raise ValueError(
                        f"Style/category length mismatch: {len(self.style_labels)} vs {len(self.category_labels)}"
                    )
                raw_labels = np.array(
                    [f"{s}||{c}" for s, c in zip(self.style_labels, self.category_labels)],
                    dtype=object,
                ).astype(str)
        else:
            raise ValueError("label_mode must be one of: 'item_id', 'style_category'")

        if len(raw_labels) != len(self.features):
            raise ValueError(
                f"Feature/label length mismatch for split='{split}': {len(self.features)} vs {len(raw_labels)}"
            )

        unique_ids = sorted(np.unique(raw_labels).tolist())
        self.item_id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        self.labels = np.array([self.item_id_to_idx[sid] for sid in raw_labels], dtype=np.int64)
        self.label_values = raw_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        target = {
            "item_id": self.item_ids[idx],
            self.label_key: self.label_values[idx],
        }
        if self.style_labels is not None:
            target["style"] = self.style_labels[idx]
        if self.category_labels is not None:
            target["category"] = self.category_labels[idx]
        return feat, label, target


def _exists(path: str) -> bool:
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False
