"""
services/video_service.py
Logic inference cho MoBiLSTM video violence detection.
Nhận mảng base64 JPEG frames từ video_scanner.js → sliding window → kết quả.
"""
import base64
from io import BytesIO

import numpy as np
from PIL import Image as PILImage

from backend.config  import VIDEO_IMAGE_H, VIDEO_IMAGE_W, VIDEO_SEQ_LEN, VIDEO_CLASSES
from backend.schemas import VideoSegment, VideoResponse


def _decode_frame(b64: str) -> np.ndarray | None:
    """
    Decode một base64 JPEG string → numpy array (H, W, 3) float32 [0,1].
    Frame được capture từ Canvas API trong video_scanner.js.
    """
    try:
        data = b64.split(",", 1)[1] if "," in b64 else b64
        img  = PILImage.open(BytesIO(base64.b64decode(data))).convert("RGB")
        img  = img.resize((VIDEO_IMAGE_W, VIDEO_IMAGE_H), PILImage.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _sliding_window_inference(
    frames_np: list[np.ndarray],
    model,
    threshold: float,
) -> VideoResponse:
    """
    Chia frames thành các cửa sổ VIDEO_SEQ_LEN (overlap 50%),
    dự đoán từng cửa sổ bằng MoBiLSTM.
    Đây là cùng logic với predict_video() trong notebook.
    """
    # Pad nếu chưa đủ VIDEO_SEQ_LEN frame (10).
    # Với 10 frame capture mỗi 120ms ≈ 1.2 giây video thực tế.
    while len(frames_np) < VIDEO_SEQ_LEN:
        frames_np.append(frames_np[-1])

    segments: list[VideoSegment] = []
    step = max(1, VIDEO_SEQ_LEN // 2)   # overlap 50%

    for seg_idx, start in enumerate(range(0, len(frames_np) - VIDEO_SEQ_LEN + 1, step)):
        window = np.array(frames_np[start : start + VIDEO_SEQ_LEN])
        probs  = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
        pred   = int(np.argmax(probs))
        conf   = float(probs[pred])
        label  = VIDEO_CLASSES[pred]

        # Confidence thấp → đánh dấu Uncertain thay vì Violence
        if label == "Violence" and conf < threshold:
            label = "Uncertain"

        segments.append(VideoSegment(
            segment_idx=seg_idx,
            label=label,
            confidence=conf,
            start_frame=start,
            end_frame=start + VIDEO_SEQ_LEN,
        ))

    violence_segs  = [s for s in segments if s.label == "Violence"]
    violence_ratio = len(violence_segs) / len(segments) if segments else 0.0
    best_seg       = max(segments, key=lambda s: s.confidence) if segments else None

    return VideoResponse(
        has_violence=len(violence_segs) > 0,
        violence_ratio=round(violence_ratio, 4),
        overall_label=best_seg.label if best_seg else "NonViolence",
        overall_confidence=round(best_seg.confidence, 4) if best_seg else 1.0,
        segments=segments,
        total_frames_analyzed=len(frames_np),
    )


def predict_from_frames(
    raw_frames: list[str],
    model,
    threshold: float = 0.70,
) -> VideoResponse:
    """
    Entry point cho router /predict_frames.
    Nhận list base64 JPEG → decode → sliding window → VideoResponse.
    """
    frames_np = [f for b64 in raw_frames if (f := _decode_frame(b64)) is not None]
    if len(frames_np) < 4:
        raise ValueError(f"Chỉ decode được {len(frames_np)} frame hợp lệ (tối thiểu 4)")

    return _sliding_window_inference(frames_np, model, threshold)
