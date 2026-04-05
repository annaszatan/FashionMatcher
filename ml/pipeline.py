import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.localization_model import ClothingLocalization
from ml.models.retrieval_model import RetrievalModel

UPLOADED_IMG_DIR = os.path.join(PROJECT_ROOT, "uploaded_images")
CROPPED_IMG_DIR = os.path.join(PROJECT_ROOT, "cropped_images")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "ml", "features")
INDEX_DIR = os.path.join(PROJECT_ROOT, "ml", "indices")
DEMO_INDEX_DIR = os.path.join(INDEX_DIR, "G_finetuning_transformer_demo")
VALIDATION_INDEX_DIR = os.path.join(INDEX_DIR, "G_finetuning_plus_transfer-learning_ep50_validation")
DEMO_GALLERY_FEATURE_DIR = os.path.join(FEATURE_DIR, "american_eagle_gallery")
VALIDATION_FEATURE_DIR = os.path.join(FEATURE_DIR, "G_finetuning_plus_transfer-learning_projected")

class FashionRetrievalPipeline:

    def __init__(
            self, 
            localization_folder="yolo_model", 
            embedding_folder="G_finetuning_plus_transfer-learning_ep50",
            embedding_config="G_finetuning_plus_transfer-learning.yaml",
            is_demo=True):
        self.localization_checkpoint = os.path.join(
            PROJECT_ROOT,
            "ml",
            "checkpoints",
            localization_folder,
            "yolo_best.pt",
        )
        self.embedding_checkpoint = os.path.join(
            PROJECT_ROOT,
            "ml",
            "checkpoints",
            embedding_folder,
            "best.pt",
        )
        self.embedding_config = os.path.join(
            PROJECT_ROOT,
            "ml",
            "configs",
            embedding_config,
        )
        self.index_path = os.path.join(
            DEMO_INDEX_DIR if is_demo else VALIDATION_INDEX_DIR,
            "faiss_gallery_index.ip",
        )

        self.feature_gallery_dir = DEMO_GALLERY_FEATURE_DIR if is_demo else VALIDATION_FEATURE_DIR

        self.demo_feature_paths = [
            os.path.join(self.feature_gallery_dir, "gallery_feats.npy"),
            os.path.join(self.feature_gallery_dir, "gallery_paths.npy"),
            os.path.join(self.feature_gallery_dir, "gallery_item_ids.npy"),
            os.path.join(self.feature_gallery_dir, "gallery_product_names.npy"),
            os.path.join(self.feature_gallery_dir, "gallery_source_urls.npy"),
        ]

        # Make sure that all required files are present
        required_files = {
            "localization checkpoint": self.localization_checkpoint,
            "embedding checkpoint": self.embedding_checkpoint,
            "embedding config": self.embedding_config,
            "FAISS index": self.index_path,
            "gallery paths": self.demo_feature_paths[1],
            "gallery item ids": self.demo_feature_paths[2],
        }
        for label, path in required_files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing {label}: {path}")
        
        # Initialize models
        self.localization_model = ClothingLocalization(
            model_loc=self.localization_checkpoint,
            uploaded_img_dir=UPLOADED_IMG_DIR,
            save_dir=CROPPED_IMG_DIR,
        )
        self.retrieval_model = RetrievalModel(
            index_path=self.index_path,
            gallery_paths_path=os.path.join(self.feature_gallery_dir, "gallery_paths.npy"),
            gallery_item_ids_path=os.path.join(self.feature_gallery_dir, "gallery_item_ids.npy"),
            gallery_product_names_path=os.path.join(self.feature_gallery_dir, "gallery_product_names.npy") if is_demo else None,
            gallery_source_urls_path=os.path.join(self.feature_gallery_dir, "gallery_source_urls.npy") if is_demo else None,
        )

    def run(self, image):

        # SETUP
        # Load models
        cl = self.localization_model
        retrieval = self.retrieval_model

        # Run localization and save crops using hash-based filenames.
        if isinstance(image, str):

            with open(image, "rb") as f:
                cropped_img_paths = cl.upload_image(f)
        elif hasattr(image, "read") and hasattr(image, "seek"):
            cropped_img_paths = cl.upload_image(image)
        else:
            raise TypeError("run(image) expects an image file path or a file-like object with read/seek.")
        
        if len(cropped_img_paths) == 0:
            print("No clothing items detected in the image.")
            return []

        # Call retrieval machine on cropped images - returns topk results for each cropped image
        results = retrieval.search_from_images(
            cropped_img_paths,
            topk=10,
            backbone_name="dinov2_vitb14",
            device="cuda",
            token_mode="cls_patch",
            project_root=PROJECT_ROOT,
            embedding_config_path=self.embedding_config,
            embedding_checkpoint_path=self.embedding_checkpoint,
            embedding_device="cuda",
        )

        # Return images and their annotation json files
        return results

### DEMO: Uncomment the following lines to run a quick demo of the pipeline 
### Terminal command to run this file: `python src/pipeline.py`
# def run_demo():
#     pipeline = FashionRetrievalPipeline()
#     # Example usage with a test image path
#     test_image_path = os.path.join(PROJECT_ROOT, "random_test_image.jpg")
#     results = pipeline.run(test_image_path)
#     print(results)

# run_demo()