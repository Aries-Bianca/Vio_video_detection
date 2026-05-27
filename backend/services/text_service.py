"""
services/text_service.py
Logic inference cho PhoBERT hate speech detection.
Tách hoàn toàn khỏi FastAPI route.
"""
import torch
import numpy as np
from underthesea import word_tokenize

from backend.config  import DEVICE, TEXT_MAX_LEN, TEXT_LABEL_MAP
from backend.schemas import TextResponse


def _word_segment(text: str) -> str:
    """Tách từ tiếng Việt bằng underthesea (thay thế VnCoreNLP)."""
    try:
        return word_tokenize(str(text).strip(), format="text")
    except Exception:
        return str(text)


def predict_text(text: str, model, tokenizer) -> TextResponse:
    """
    Nhận raw text → tách từ → tokenize → PhoBERT → softmax.
    Trả về TextResponse với label, confidence và full probability vector.
    """
    segmented = _word_segment(text)
    encoding  = tokenizer(
        segmented,
        max_length=TEXT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(DEVICE),
            encoding["attention_mask"].to(DEVICE),
        )

    probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))

    return TextResponse(
        text=text,
        label=TEXT_LABEL_MAP[pred_id],
        label_id=pred_id,
        confidence=float(probs[pred_id]),
        probs=probs.tolist(),
    )
