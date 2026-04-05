
import json
import os

from PIL import Image
from torch.utils.data import Dataset


def _first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default

class ClothingClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir,
        annos_dir,
        image_files,
        categories={1, 8},
        transform=None,
        return_uncropped=False
    ):
        self.img_dir = img_dir
        self.annos_dir = annos_dir
        self.transforms = transform
        self.image_files = image_files
        self.return_uncropped = return_uncropped
        self.samples = []
        self.source_key = {
            'user': 0,
            'shop': 1
        }
        self.categories = categories

        # Store metadata only (portable + lower RAM). Load image lazily in __getitem__.
        for img in image_files:
            annos_path = os.path.join(annos_dir, img.replace('.jpg', '.json'))

            with open(annos_path) as f:
                annotation = json.load(f)

            items = [k for k in annotation.keys() if 'item' in k]

            for i in items:
                item = annotation[i]
                category = _first_present(item, ["category_id", "category"])
                if category in self.categories:
                    source_raw = _first_present(annotation, ["source", "split"], default=-1)
                    if isinstance(source_raw, str):
                        source = self.source_key.get(source_raw.lower(), -1)
                    else:
                        source = source_raw

                    box = _first_present(item, ["bounding_box", "bbox", "box"])
                    if box is None:
                        continue

                    style = _first_present(
                        item,
                        ["style", "style_id", "styleId"],
                        default=_first_present(annotation, ["style", "style_id", "styleId"]),
                    )
                    pair_id = _first_present(
                        annotation,
                        ["pair_id", "pairId"],
                        default=_first_present(item, ["pair_id", "pairId"]),
                    )
                    occlusion = _first_present(item, ["occlusion"], default=None)

                    self.samples.append({
                        'image_name': img,
                        'source': source,
                        'box': box,
                        'category': category,
                        'style': style,
                        'pair_id': pair_id,
                        'occlusion': occlusion
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(self.img_dir, sample['image_name'])
        image = Image.open(image_path).convert("RGB")
        crop = image.crop(sample['box'])

        target = {
            "image_name": sample['image_name'],
            "bbox": sample['box'],
            "source": sample['source'],
            "category": sample['category'],
            "style": sample['style'],
            "pair_id": sample['pair_id'],
            "occlusion": sample['occlusion']
        }

        if self.transforms:
            crop = self.transforms(crop)

        if self.return_uncropped:
            full_image = self.transforms(image) if self.transforms else image
            return full_image, crop, target

        return crop, target
