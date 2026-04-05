"""
The purpose of this script is to load the DeepFashion2 dataset for the localization model 
(includes retrieval as well but this is unused)

Terminal usage example:
```
python scripts/load_v1_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.version1.data_loader import DeepFashionDataLoader

# Call the dataloader to load the data and save the datasets
if __name__ == "__main__":
    data_loader = DeepFashionDataLoader()
    data_loader.load_data()