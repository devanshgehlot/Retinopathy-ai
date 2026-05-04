"""
inference.py — Preprocessing, prediction, and Grad-CAM heatmap for RetinaScan AI.

Preprocessing pipeline (MUST match training exactly):
  1. Resize to 224×224
  2. CLAHE on L channel (LAB color space) — enhances blood vessel visibility
  3. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""

import io
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

INPUT_SIZE = 224  # Must match training

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

DISCLAIMERS = {
    0: "No signs of Diabetic Retinopathy detected. Continue regular eye check-ups as recommended.",
    1: "Mild Diabetic Retinopathy detected. Please consult an ophthalmologist for a follow-up.",
    2: "Moderate Diabetic Retinopathy detected. Please consult an ophthalmologist immediately.",
    3: "Severe Diabetic Retinopathy detected. Urgent ophthalmological evaluation is required.",
    4: "Proliferative Diabetic Retinopathy detected. Seek immediate medical attention.",
}


# ─────────────────────────────────────────────
# Step 1: CLAHE enhancement
# ─────────────────────────────────────────────

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L channel of the LAB color space.
    Enhances contrast of blood vessels in retinal fundus images.

    Args:
        img_rgb: uint8 NumPy array, shape (H, W, 3), RGB.

    Returns:
        CLAHE-enhanced uint8 NumPy array, shape (H, W, 3), RGB.
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    img_lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    img_rgb_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2RGB)
    return img_rgb_enhanced


# ─────────────────────────────────────────────
# Step 2: Full preprocessing pipeline
# ─────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image):
    """
    Full inference preprocessing:
      1. Convert to RGB
      2. Resize to 224×224
      3. Apply CLAHE
      4. Normalize with ImageNet stats
      5. Add batch dimension → shape (1, 3, 224, 224)

    Returns:
        input_tensor : torch.Tensor, shape (1, 3, 224, 224)  — for model inference
        rgb_float    : np.ndarray,   shape (224, 224, 3), float32 in [0,1]  — for Grad-CAM overlay
    """
    # Convert to RGB numpy array
    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)

    # Resize
    img_rgb = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    # CLAHE enhancement
    img_rgb = apply_clahe(img_rgb)

    # Float copy for Grad-CAM overlay (values in [0,1])
    rgb_float = img_rgb.astype(np.float32) / 255.0

    # ImageNet normalization
    img_normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD  # (224, 224, 3)

    # HWC → CHW → batch
    img_tensor = np.transpose(img_normalized, (2, 0, 1))          # (3, 224, 224)
    input_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0)  # (1, 3, 224, 224)

    return input_tensor, rgb_float


# ─────────────────────────────────────────────
# Step 3: Prediction
# ─────────────────────────────────────────────

def run_prediction(model, input_tensor: torch.Tensor, device: torch.device) -> dict:
    """
    Run forward pass and return predicted grade + softmax probabilities.

    Args:
        model       : Loaded EfficientNet-B0 in eval mode.
        input_tensor: Preprocessed tensor, shape (1, 3, 224, 224).
        device      : torch.device.

    Returns:
        dict with keys: grade (int), confidence (float), probabilities (dict[str, float])
    """
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)            # (1, 5)
        probs  = F.softmax(logits, dim=1)       # (1, 5)

    probs_np      = probs.squeeze(0).cpu().numpy()      # (5,)
    predicted_idx = int(np.argmax(probs_np))
    confidence    = float(probs_np[predicted_idx]) * 100.0

    probabilities = {
        str(i): round(float(probs_np[i]) * 100.0, 2)
        for i in range(5)
    }

    return {
        "grade":         predicted_idx,
        "confidence":    round(confidence, 2),
        "probabilities": probabilities,
    }


# ─────────────────────────────────────────────
# Step 4: Grad-CAM heatmap
# ─────────────────────────────────────────────

def generate_gradcam(model, input_tensor: torch.Tensor, rgb_float: np.ndarray, target_class: int) -> str:
    """
    Generate a Grad-CAM heatmap overlaid on the original image and encode as base64.

    Target layer: last convolutional block of EfficientNet-B0 → model.features[-1]

    Args:
        model        : EfficientNet-B0 model.
        input_tensor : Preprocessed tensor, shape (1, 3, 224, 224).
        rgb_float    : Original image as float32 [0,1] numpy array, (224,224,3).
        target_class : Predicted class index (used to compute gradients).

    Returns:
        base64-encoded JPEG string of the heatmap overlay.
    """
    # EfficientNet-B0 via timm: last conv block
    target_layers = [model.blocks[-1]]  # timm EfficientNet uses .blocks

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    # Grad-CAM needs gradients — temporarily allow them
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (1, 224, 224)
    grayscale_cam = grayscale_cam[0]                                  # (224, 224)

    # Overlay heatmap on original image
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # uint8 (224, 224, 3)

    # Encode to base64 JPEG
    pil_heatmap = Image.fromarray(visualization)
    buffer = io.BytesIO()
    pil_heatmap.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return heatmap_b64


# ─────────────────────────────────────────────
# Step 5: Full pipeline (called by app.py)
# ─────────────────────────────────────────────

def predict_and_explain(model, pil_image: Image.Image, device: torch.device) -> dict:
    """
    Complete inference pipeline: preprocess → predict → Grad-CAM → build response.

    Args:
        model     : Loaded EfficientNet-B0 (eval mode).
        pil_image : PIL Image from the uploaded file.
        device    : torch.device.

    Returns:
        Full response dict matching the project brief JSON schema.
    """
    # Preprocess
    input_tensor, rgb_float = preprocess_image(pil_image)

    # Predict
    result = run_prediction(model, input_tensor, device)
    grade  = result["grade"]

    # Grad-CAM
    heatmap_b64 = generate_gradcam(model, input_tensor, rgb_float, target_class=grade)

    # Build final response
    response = {
        "success":       True,
        "grade":         grade,
        "grade_label":   GRADE_LABELS[grade],
        "confidence":    result["confidence"],
        "probabilities": result["probabilities"],
        "heatmap_base64": heatmap_b64,
        "is_serious":    grade >= 2,
        "disclaimer":    DISCLAIMERS[grade],
    }
    return response