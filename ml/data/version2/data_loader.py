from collections import defaultdict
from pathlib import Path

import json
import os

import pandas as pd

# This class assumes the data files have already been downloaded and unzipped
class DeepFashion2DataLoader:
    def __init__(self, annos_dir="ml/data/validation/annos", image_root="ml/data/validation/image", target_categories=(1, 8), metadata_csv=None):
        self.annos_dir = Path(annos_dir)
        self.image_root = Path(image_root)
        # DeepFashion folders may be named either "image" or "images".
        if not self.image_root.exists():
            alt = self.image_root.parent / ("image" if self.image_root.name == "images" else "images")
            if alt.exists():
                self.image_root = alt
        self.target_categories = set(target_categories)
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else Path(__file__).resolve().parent / "metadata.csv"
    
    @staticmethod
    def get_image_name(ann_file):
        return ann_file.replace(".json", ".jpg")

    @staticmethod
    def _item_index(item_key: str):
        # item1 -> 1, item12 -> 12
        digits = "".join(ch for ch in item_key if ch.isdigit())
        return int(digits) if digits else None

    @staticmethod
    def _source_code(source_value):
        # Match expected DeepFashion-style encoding: user/query=0, shop/gallery=1
        if source_value == "user":
            return 0
        if source_value == "shop":
            return 1
        return None

    # Create contents of metadata.csv file based on annotation files
    def build_metadata_rows(self):
        if not self.annos_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annos_dir}")

        # pair_id -> {"user": {item_idx: [records]}, "shop": {item_idx: [records]}}
        items_by_pair = defaultdict(lambda: {"user": defaultdict(list), "shop": defaultdict(list)})

        for ann_file in os.listdir(self.annos_dir):
            if not ann_file.endswith(".json"):
                continue

            ann_path = self.annos_dir / ann_file
            with open(ann_path, "r") as f:
                annotation = json.load(f)

            source = annotation.get("source")
            pair_id = annotation.get("pair_id")

            if source is None or pair_id is None:
                continue

            image_name = self.get_image_name(ann_file)
            image_path = self.image_root / image_name
            if not image_path.exists():
                continue

            item_keys = [key for key in annotation.keys() if key.startswith("item")]

            for item_key in item_keys:
                item = annotation[item_key]
                cat_id = item.get("category_id")
                bbox = item.get("bounding_box")
                item_idx = self._item_index(item_key)

                if cat_id not in self.target_categories:
                    continue
                if bbox is None or len(bbox) != 4:
                    continue
                if item_idx is None:
                    continue
                if source not in ("user", "shop"):
                    continue

                source_code = self._source_code(source)
                if source_code is None:
                    continue

                items_by_pair[pair_id][source][item_idx].append(
                    {
                        "image_name": image_name,
                        "category_id": int(cat_id),
                        "bbox": bbox,
                        "source_code": source_code,
                        "item_idx": int(item_idx),
                        "style": int(item.get("style", -1)),
                        "occlusion": int(item.get("occlusion", -1)),
                        "scale": int(item.get("scale", -1)),
                        "zoom_in": int(item.get("zoom_in", -1)),
                        "viewpoint": int(item.get("viewpoint", -1)),
                    }
                )

        rows = []

        for pair_id, grouped in items_by_pair.items():
            user_by_item = grouped["user"]
            shop_by_item = grouped["shop"]

            # Keep only items that appear in both user and shop so labels are retrievable.
            common_item_idxs = sorted(set(user_by_item.keys()).intersection(shop_by_item.keys()))
            if not common_item_idxs:
                continue

            for item_idx in common_item_idxs:
                shared_item_id = f"{pair_id}_{item_idx}"

                for item in user_by_item[item_idx]:
                    x1, y1, x2, y2 = map(int, item["bbox"])
                    rows.append(
                        {
                            "image_path": item["image_name"],
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "split": "query",
                            "source": item["source_code"],
                            "pair_id": int(pair_id),
                            "category": item["category_id"],
                            "style": item["style"],
                            "occlusion": item["occlusion"],
                            "scale": item["scale"],
                            "zoom_in": item["zoom_in"],
                            "viewpoint": item["viewpoint"],
                            "item_id": shared_item_id,
                        }
                    )

                for item in shop_by_item[item_idx]:
                    x1, y1, x2, y2 = map(int, item["bbox"])
                    rows.append(
                        {
                            "image_path": item["image_name"],
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "split": "gallery",
                            "source": item["source_code"],
                            "pair_id": int(pair_id),
                            "category": item["category_id"],
                            "style": item["style"],
                            "occlusion": item["occlusion"],
                            "scale": item["scale"],
                            "zoom_in": item["zoom_in"],
                            "viewpoint": item["viewpoint"],
                            "item_id": shared_item_id,
                        }
                    )

        return rows

    def load_data(self):
        rows = self.build_metadata_rows()
        metadata_df = pd.DataFrame(rows)
        self.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(self.metadata_csv, index=False)
        return metadata_df
