"""
schemas.py – Toàn bộ request/response schema dùng chung cho các router.
"""
from pydantic import BaseModel, Field


# ── Text (PhoBERT) ────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class TextResponse(BaseModel):
    text:       str
    label:      str           # "Clean" | "Offensive" | "Hate"
    label_id:   int           # 0 | 1 | 2
    confidence: float
    probs:      list[float]   # [p_clean, p_offensive, p_hate]


# ── Video (MoBiLSTM) ──────────────────────────────────────────────────────────
class VideoSegment(BaseModel):
    segment_idx: int
    label:       str          # "NonViolence" | "Violence" | "Uncertain"
    confidence:  float
    start_frame: int
    end_frame:   int

class FramesRequest(BaseModel):
    """
    Gửi từ video_scanner.js: mảng base64 JPEG captured từ <video> element.
    Mỗi string là một frame đã resize về FRAME_WIDTH × FRAME_HEIGHT.
    """
    frames:    list[str] = Field(..., min_length=1, max_length=60)
    threshold: float     = Field(default=0.70, ge=0.0, le=1.0)

class VideoResponse(BaseModel):
    has_violence:        bool
    violence_ratio:      float
    overall_label:       str
    overall_confidence:  float
    segments:            list[VideoSegment]
    total_frames_analyzed: int


# ── Image (ViT) ───────────────────────────────────────────────────────────────
class ImagePrediction(BaseModel):
    label:           str    # "NonViolence" | "Violence" | "Uncertain"
    confidence:      float
    violence_prob:   float
    nonviolence_prob: float

class ImagesRequest(BaseModel):
    """
    Gửi từ image_scanner.js: mảng base64 JPEG fetched qua background script.
    Mỗi string là một ảnh đã được fetch và encode.
    """
    images:    list[str] = Field(..., min_length=1, max_length=20)
    threshold: float     = Field(default=0.70, ge=0.0, le=1.0)

class ImagesResponse(BaseModel):
    predictions:    list[ImagePrediction]
    total:          int
    violence_count: int
    safe_count:     int
