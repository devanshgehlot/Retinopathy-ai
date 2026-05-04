"""
app.py — RetinaScan AI | Flask API Server
Endpoints:
  GET  /health   → server health check
  POST /predict  → DR grade prediction + Grad-CAM heatmap
"""

import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from model import load_model
from inference import predict_and_explain


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_model.pth")
MAX_FILE_SIZE   = 10 * 1024 * 1024          # 10 MB
ALLOWED_TYPES   = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTS    = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────
# App initialization
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow frontend on a different port to call this API

# ── Load model ONCE at startup (never inside a request handler) ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = load_model(MODEL_PATH, device)  # global — reused for every request


# ─────────────────────────────────────────────
# Helper: validate uploaded file
# ─────────────────────────────────────────────

def validate_image_file(file) -> str | None:
    """
    Validate the uploaded file.
    Returns an error message string if invalid, else None.
    """
    if file is None or file.filename == "":
        return "No file uploaded."

    # Extension check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return f"Invalid file type '{ext}'. Only .jpg, .jpeg, .png are accepted."

    # Size check — read content and check length
    file_bytes = file.read()
    if len(file_bytes) == 0:
        return "Uploaded file is empty."
    if len(file_bytes) > MAX_FILE_SIZE:
        return f"File too large ({len(file_bytes)//1024}KB). Maximum allowed size is 10MB."

    # Rewind so PIL can read it later
    file.seek(0)
    return None  # All good


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health_check():
    """GET / or GET /health — verify server is running."""
    return jsonify({"status": "ok", "message": "RetinaScan AI backend is running."}), 200


@app.post("/predict")
def predict():
    """
    POST /predict
    Input  : multipart/form-data with key "image" (.jpg / .jpeg / .png)
    Output : JSON with grade, confidence, probabilities, heatmap_base64, disclaimer
    """
    # ── 1. File presence check ──────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field found in request."}), 400

    file = request.files["image"]

    # ── 2. Validation ───────────────────────────────────────────────────────
    error_msg = validate_image_file(file)
    if error_msg:
        return jsonify({"success": False, "error": error_msg}), 400

    # ── 3. Open image with PIL ──────────────────────────────────────────────
    try:
        pil_image = Image.open(file.stream)
        pil_image.verify()          # Detect corrupted files early
        file.stream.seek(0)         # Reset after verify (verify consumes the stream)
        pil_image = Image.open(file.stream)
    except UnidentifiedImageError:
        return jsonify({"success": False, "error": "Cannot identify image file. It may be corrupted."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to open image: {str(e)}"}), 400

    # ── 4. Run inference pipeline ───────────────────────────────────────────
    try:
        result = predict_and_explain(model, pil_image, device)
    except Exception as e:
        # Model or Grad-CAM failure — return a clean error, not a stack trace
        app.logger.exception("Inference error")
        return jsonify({"success": False, "error": f"Model inference failed: {str(e)}"}), 500

    # ── 5. Return result ────────────────────────────────────────────────────
    return jsonify(result), 200


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Use host="0.0.0.0" so the frontend (on localhost) can reach it
    app.run(host="0.0.0.0", port=5000, debug=False)