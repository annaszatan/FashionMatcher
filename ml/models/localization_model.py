import hashlib
import cv2
import numpy as np

from ultralytics import YOLO
import os

import PIL
from PIL import Image
import yaml

class ClothingLocalization:
    def __init__(self, model_loc="/content/best.pt", uploaded_img_dir="/content/uploaded_images", save_dir="/content/cropped_data", verbose=False):

        # Load the desired trained YOLO model
        self.model = YOLO(model_loc)

        # Uploaded images will be stored in this directory
        self.uploaded_img_dir = uploaded_img_dir
        os.makedirs(self.uploaded_img_dir, exist_ok=True)

        # Directory to save cropped images
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Create image cache
        self.img_cache = {}
        self.verbose = verbose

    def hash_img(self, img_object):
        return hashlib.sha256(img_object).hexdigest()

    def preprocess_image(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        scale = min(224/h, 224/w)
        new_w = int(w*scale)
        new_h = int(h*scale)

        resized_img = cv2.resize(image, (new_w, new_h))

        # Add padding around the image
        canvas = np.full((224, 224, 3), 114, dtype=np.uint8)
        x_offset = (224 - new_w) // 2
        y_offset = (224 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        return canvas

    def process_image(self, img_path):

        # Load image and make sure it is correct dimensions for model (224x224)
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess_image(img)

        # Run the model on the image and get bounding box predictions
        results = self.model(img, verbose=self.verbose)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()


        h,w = img.shape[:2]
        cropped_images = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w, x2))
            y2 = int(min(h, y2))
            if x2 > x1 and y2 > y1:
                cropped_img = img[y1:y2, x1:x2]
                cropped_images.append(cropped_img)

        return cropped_images

    def upload_image(self, img_object):
        img_object.seek(0)
        img = img_object.read()
        img_hash = self.hash_img(img)

        # Check if image has already been processed and return cached cropped paths if so
        if img_hash in self.img_cache:
            if self.verbose:
                print("Image already processed. Retrieving from cache.")
            return self.img_cache[img_hash]
        else:
            # Save the uploaded image to the directory
            img_name = f"{img_hash}.jpg"
            img_path = os.path.join(self.uploaded_img_dir, img_name)
            with open(img_path, "wb") as f:
                f.write(img)

            # Process the image and get cropped results
            cropped_images = self.process_image(img_path)

            # Save cropped images and update cache
            cropped_paths = []
            for i, cropped_img in enumerate(cropped_images):
                cropped_name = f"{img_hash}_crop_{i}.jpg"
                cropped_path = os.path.join(self.save_dir, cropped_name)
                cv2.imwrite(cropped_path, cropped_img)
                cropped_paths.append(cropped_path)

            self.img_cache[img_hash] = cropped_paths

            return cropped_paths