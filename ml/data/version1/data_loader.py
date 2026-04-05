import json
import pathlib
from pathlib import Path
from collections import Counter, defaultdict
import csv
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

from ml.data.datasets.clothing_classification_dataset import ClothingClassificationDataset
from ml.data.datasets.clothing_detection_dataset import ClothingDetectionDataset
from ml.data.datasets.portable_dataset import PortableRetrievalDataset
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import shutil
import PIL
from PIL import Image
import random

import io

import gdown

DATA_KEY = {
    "train": "1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK",
    "validation": "1O45YqhREBOoLudjA06HcTehcEebR0o9y",
    "test": "1hsa-UE-LX8sks8eAcGLL-9QDNyNt6VgP"

}

class DeepFashionDataLoader:

    def __init__(
        self,
        target_categories={1, 8},
        target_size=5000,
        annos_path=None,
        img_dir=None,
        transform=None,
        data_id="validation",
        data_root="ml/data",
    ):
        self.target_categories = target_categories
        self.transform = transform
        self.data_name = data_id
        self.data_id = DATA_KEY.get(data_id)
        self.target_size = target_size
        self.retrieval_dataset_name = "retrieval_dataset.pt"
        self.localization_dataset_name = "localization_dataset.pt"
        self.project_root = Path(__file__).resolve().parents[3]
        self.data_root = Path(data_root)
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root

        split_root = self.data_root / self.data_name
        self.annos_path = Path(annos_path) if annos_path else split_root / "annos"
        self.img_dir = Path(img_dir) if img_dir else split_root / "image"
        if not self.annos_path.is_absolute():
            self.annos_path = self.project_root / self.annos_path
        if not self.img_dir.is_absolute():
            self.img_dir = self.project_root / self.img_dir
        if not self.img_dir.exists():
            alt = self.img_dir.parent / ("image" if self.img_dir.name == "images" else "images")
            if alt.exists():
                self.img_dir = alt

        self.save_dir = self.data_root / "version1"

        # set random seed
        random.seed(42)

    def _has_local_split_data(self):
        split_root = self.data_root / self.data_name
        annos_dir = split_root / "annos"
        image_dir = split_root / "image"
        images_dir = split_root / "images"
        return annos_dir.exists() and (image_dir.exists() or images_dir.exists())

    def load_data(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Check ml/data/<split> first so we do not re-download if not necessary
        if not self._has_local_split_data():
            print("Data not found locally. Downloading from Google Drive...")
            self.download_drive_data()
            # Refresh image dir in case extracted folder name is image/images.
            if not self.img_dir.exists():
                alt = self.img_dir.parent / ("image" if self.img_dir.name == "images" else "images")
                if alt.exists():
                    self.img_dir = alt
        else:
            print("Data found locally. Loading...")

        # Check if pt files already exist
        retrieval_path = self.save_dir / self.retrieval_dataset_name
        localization_path = self.save_dir / self.localization_dataset_name

        if retrieval_path.exists() and localization_path.exists():
            print("Dataset files found locally. Loading...")
            retrieval_dataset = torch.load(retrieval_path)
            localization_dataset = torch.load(localization_path)
            return retrieval_dataset, localization_dataset
        
        print("Processing data to create datasets...")
        items, final_list = self.get_item_lists()

        cd_dataset = ClothingDetectionDataset(self.img_dir, self.annos_path, [item['image_name'] for item in final_list], categories=self.target_categories, transform=self.transform)
        cc_dataset = ClothingClassificationDataset(self.img_dir, self.annos_path, [item['image_name'] for item in final_list], categories=self.target_categories, transform=self.transform, return_uncropped=True)

        torch.save(cd_dataset, self.save_dir / self.localization_dataset_name)
        torch.save(cc_dataset, self.save_dir / self.retrieval_dataset_name)
        return cc_dataset, cd_dataset

    # Save data from drive to local directory
    def download_drive_data(self):
        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / f"{self.data_name}.zip"

        print(f"Checking for existing zip file at {zip_path}...")

        # Check if zip file is already downloaded
        if not zip_path.exists():
            # Replace FILE_ID with the actual ID
            file_id = self.data_id
            url = f"https://drive.google.com/uc?id={file_id}"
            output = str(zip_path)  # Desired filename

            gdown.download(url, output, quiet=False)

        print(f"Zip file downloaded to {zip_path}. Extracting...")
        # Unzip the downloaded files with password
        extract_to = self.data_root
        password = b"2019Deepfashion2**"

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=extract_to, pwd=password)
        
        print(f"Extraction complete. Data available at {extract_to}. Cleaning up zip file...")
        # Delete the zip file after extraction
        zip_path.unlink()
    
    @staticmethod
    def valid_img(img_details):
            return (img_details.get('pair_id') is not None) and (img_details.get('category_id') is not None)

    @staticmethod
    def get_image_dir(annos_path):

        annos_name = os.path.basename(annos_path)

        img_dir = annos_name.replace('.json', '.jpg')

        return img_dir
    
    @staticmethod
    def pil_to_jpeg_bytes(img, quality=95):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()

    # Isolate the items to keep in the final dataset 
    def get_item_lists(self):
        items_by_pair = defaultdict(list)

        for ann in os.listdir(self.annos_path):
            if not ann.lower().endswith('.json'):
                continue

            with open(os.path.join(self.annos_path, ann)) as f:
                annotation = json.load(f)
                
            source = annotation.get('source')
            pair_id = annotation.get('pair_id')

            # Check to make sure Source exists
            if source is None:
                print(f"Image {ann} is missing source information. Skipping.")
                continue

            img_name = self.get_image_dir(ann)

            # Check how many items exist
            item_keys = [k for k in annotation.keys() if 'item' in k]

            for i in item_keys:
                item = annotation[i]
                cat_id = item.get('category_id')
                bbox = item.get('bounding_box')
                
                # Confirm that the item is from the desired category and has bounding box + pair ID
                if cat_id not in self.target_categories or bbox is None or pair_id is None:
                    print(f"Image {ann} has item {i} that is missing required information or is not in target categories. Skipping.")
                    continue
                else:

                    items_by_pair[pair_id].append({
                        "image_name": img_name,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "source": source
                    })

        final_items = []

        # Process to create final item list
        for pair_id, group in items_by_pair.items():

            if len(group) < 1:
                continue

            sources = set(item['source'] for item in group)

            # Make sure its a user-shop pair
            if not {'shop', 'user'}.issubset(sources):
                continue

            final_items.extend(group)

        print(f"Total items after filtering: {len(final_items)}")
        
        valid_pairs = {
            pair_id
            for pair_id, group in items_by_pair.items()
            if {'shop', 'user'}.issubset(set(item['source'] for item in group))
        }

        all_pairs = list(valid_pairs)
        random.shuffle(all_pairs)

        final_pairs = []
        count_items = 0

        for pair_id in all_pairs:
            pair_size = len(items_by_pair[pair_id])
            if count_items + pair_size > self.target_size:
                continue
            final_pairs.append(pair_id)
            count_items += pair_size
            if count_items >= self.target_size:
                break

        final_list = []
        for pair_id in final_pairs:
            final_list.extend(items_by_pair[pair_id])

        return final_items, final_list
