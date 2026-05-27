"""
services/image_service.py
Logic inference cho ViolenceViT image violence detection (notebook vit-1.ipynb).

Thay đổi so với phiên bản cũ:
  - Label check dùng "violent" thay vì "Violence"
  - Fallback label dùng "non-violent" thay vì "NonViolence"
  - Threshold mặc định lấy từ model registry (recommended_threshold trong checkpoint)
"""
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image as PILImage

from backend.config             import DEVICE, VIT_CLASSES, VIT_THRESHOLD_FALLBACK
from backend.models.image_model import VIT_TRANSFORM
from backend.schemas            import ImagePrediction, ImagesResponse


def _decode_image(b64: str) -> PILImage.Image | None:
    """
    Decode base64 JPEG → PIL Image RGB.
    Ảnh được fetch qua background script trong image_scanner.js.
    """
    try:
        data = b64.split(",", 1)[1] if "," in b64 else b64
        return PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception:
        return None


def _predict_single(
    img: PILImage.Image,
    model,
    threshold: float,
) -> ImagePrediction:
    """
    Inference ViolenceViT cho một ảnh PIL.
    Labels từ notebook: "non-violent" (0) / "violent" (1).
    Uncertain nếu label "violent" nhưng confidence < threshold.
    """
    tensor = VIT_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    nonviol_p = float(probs[0])   # "non-violent"
    viol_p    = float(probs[1])   # "violent"
    pred_idx  = int(np.argmax(probs))
    label     = VIT_CLASSES[pred_idx]   # "non-violent" hoặc "violent"
    conf      = float(probs[pred_idx])

    # Áp threshold: violent nhưng confidence thấp → Uncertain
    if label == "violent" and conf < threshold:
        label = "Uncertain"

    return ImagePrediction(
        label=label,
        confidence=conf,
        violence_prob=viol_p,
        nonviolence_prob=nonviol_p,
    )


# Fallback cho ảnh không decode được hoặc inference lỗi
_FALLBACK = ImagePrediction(
    label="non-violent",
    confidence=1.0,
    violence_prob=0.0,
    nonviolence_prob=1.0,
)


def predict_images(
    raw_images: list[str],
    model,
    threshold: float,
) -> ImagesResponse:
    """
    Entry point cho router /predict_images.
    Nhận list base64 JPEG → decode → ViolenceViT inference → ImagesResponse.
    Ảnh lỗi decode trả về fallback non-violent, không crash toàn batch.
    """
    predictions: list[ImagePrediction] = []

    for b64 in raw_images:
        img = _decode_image(b64)
        if img is None:
            predictions.append(_FALLBACK)
            continue
        try:
            predictions.append(_predict_single(img, model, threshold))
        except Exception:
            predictions.append(_FALLBACK)

    # Đếm theo label "violent" (notebook mới dùng lowercase)
    v_count = sum(1 for p in predictions if p.label == "violent")

    return ImagesResponse(
        predictions=predictions,
        total=len(predictions),
        violence_count=v_count,
        safe_count=len(predictions) - v_count,
    )
