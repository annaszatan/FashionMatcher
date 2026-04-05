import base64
import os
import tempfile
from io import BytesIO
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from ml.pipeline import FashionRetrievalPipeline

FRONTEND_DIR = os.path.join(CURRENT_DIR, "app")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
GALLERY_IMAGES_ROOT = os.path.join(CURRENT_DIR, "ml", "demo_data", "gallery_data", "images")
MIN_MATCH_SCORE = 0.0

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/find-similar", methods=["POST"])
def find_similar():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    temp_upload_path = None
    try:
        uploaded = Image.open(image_file.stream)
        image_format = (uploaded.format or "").upper()
        
        if image_format not in ALLOWED_IMAGE_FORMATS:
            return jsonify({"error": "Only .jpg, .jpeg, .png, and .webp are supported."}), 400

        suffix = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }.get(image_format, ".img")
        
        # Save the file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_upload_path = tmp.name
            image_file.stream.seek(0)
            image_file.save(temp_upload_path)

        pipeline = FashionRetrievalPipeline()
        results = pipeline.run(temp_upload_path)
        if not results or not results[0]:
            return jsonify({"error": "No clothing items detected in the image."}), 200
        
        top_image = results[0][0]
        raw_score = top_image.get("score")
        score = float(raw_score) if raw_score is not None else None
        score_pct = (score * 100.0) if score is not None else 0.0
        threshold_pct = MIN_MATCH_SCORE * 100.0
        if score is None or score < MIN_MATCH_SCORE:
            return jsonify(
                {
                    "error": f"No match found (Highest Confidence Score: {score_pct:.2f}%). Minimum required: {threshold_pct:.0f}%.",
                    "match_score_pct": round(score_pct, 2),
                }
            ), 200

        matched_image_path = os.path.join(GALLERY_IMAGES_ROOT, top_image.get("path", ""))
        matched_image_link = top_image.get("product_url", None)
        matched_image_name = top_image.get("product_name", "Unknown Product")

        matched_image = Image.open(matched_image_path)
        buffer = BytesIO()
        matched_image.save(buffer, format="JPEG", quality=92)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify(
            {
                "matched_image": f"data:image/jpeg;base64,{encoded_image}",
                "product_link": matched_image_link,
                "product_name": matched_image_name,
                "match_score_pct": round(score_pct, 2),
            }
        )
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400
    finally:
        # Delete the temporary uploaded file if present from search
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)

if __name__ == "__main__":
    app.run(debug=True)
