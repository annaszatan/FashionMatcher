"""
The purpose of this script is to load the DeepFashion2 dataset and save the relevant metadata to a CSV file for the retrieval models

Terminal usage example:
```
python scripts/load_v2_data.py
"""

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.version2.data_loader import DeepFashion2DataLoader
from ml.data.version1.data_loader import DeepFashionDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data for retrieval models.")
    parser.add_argument(
        "--split_type",
        default="train",
        help="DeepFashion2 data split to load (e.g., 'train', 'validation', 'test')."
    )
    args = parser.parse_args()

    # Load the specified split of the DeepFashion2 dataset if not already done so that metadata CSV files can be created
    v1_loader = DeepFashionDataLoader(data_id=args.split_type)
    v1_loader.download_drive_data()

    # Create metadata CSV for the specified split
    loader = DeepFashion2DataLoader(
        annos_dir=f"ml/data/{args.split_type}/annos",
        image_root=f"ml/data/{args.split_type}/image",
        metadata_csv=f"ml/data/version2/{args.split_type}_metadata.csv",
    )
    metadata_df = loader.load_data()

    # Save the metadata to a CSV file for later use in training/evaluation
    metadata_df.to_csv(f"ml/data/version2/{args.split_type}_metadata.csv", index=False)