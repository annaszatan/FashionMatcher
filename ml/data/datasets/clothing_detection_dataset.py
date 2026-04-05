import os
import json
import pandas as pd
from PIL import Image 
import torch 
from torch.utils.data import Dataset


class ClothingDetectionDataset(Dataset):
    def __init__(self, img_dir, annos_dir, image_names, categories={1, 8}, transform=None):
        self.img_dir = img_dir
        self.annos_dir = annos_dir
        self.transform = transform
        self.image_names = image_names
        self.source_key = {
            'user': 0,
            'shop': 1
        }
        self.categories = categories
        
        self.image_files = []
        self.annos_files = []
        
        # Process all the images and annotations at initialization  
        for img_name in self.image_names:
            img_path = os.path.join(self.img_dir, img_name)
            with open(img_path, "rb") as f:
                self.image_files.append(f.read())
            
            annos_path = os.path.join(self.annos_dir, img_name.replace('.jpg', '.json'))
            with open(annos_path) as f:
                self.annos_files.append(json.load(f))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        annotation = self.annos_files[idx]
        image_data = self.image_files[idx]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        sources = []
        boxes = []
        categories = []
        styles = []
        pair_ids = []
        occlusions = []

        items = [annotation[k] for k in annotation.keys() if 'item' in k]
        for item in items:
            if item['category_id'] in self.categories:
                sources.append(self.source_key[annotation['source']])
                boxes.append(item['bounding_box'])
                categories.append(item['category_id'])
                styles.append(item['style'])
                pair_ids.append(annotation['pair_id'])
                occlusions.append(item['occlusion'])
            else:
                continue

        sources = torch.tensor(sources, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        categories = torch.tensor(categories, dtype=torch.int64)
        styles = torch.tensor(styles, dtype=torch.int64)
        pair_ids = torch.tensor(pair_ids, dtype=torch.int64)
        occlusions = torch.tensor(occlusions, dtype=torch.int64)

        target = {
            "sources": sources,
            "boxes": boxes,
            "categories": categories,
            "styles": styles,
            "pair_ids": pair_ids,
            "occlusions": occlusions
        }

        if self.transform:
            image = self.transform(image)

        return image, target

