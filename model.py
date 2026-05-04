"""
model.py — EfficientNet-B0 model definition for RetinaScan AI
Matches the architecture used during training (timm-based EfficientNet-B0).
"""

import torch
import torch.nn as nn
import timm


def build_model(num_classes: int = 5) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classifier head for 5-class DR grading.
    This MUST match the architecture used during training exactly.
    """
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    return model


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """
    Load trained weights into the model and set it to eval mode.
    Called ONCE at server startup — never inside the prediction function.

    Args:
        weights_path: Path to the saved .pth file (state_dict).
        device: torch.device (cpu or cuda).

    Returns:
        model ready for inference.
    """
    model = build_model(num_classes=5)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from '{weights_path}' on {device}")
    return model