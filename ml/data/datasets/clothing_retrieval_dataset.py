#Dataset

from PIL import Image
import torch
from torch.utils.data import Dataset

class ClothingRetrievalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")

        x1, y1, x2, y2 = row["bbox"]
        img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        label = int(row["item_id"])

        return img, label