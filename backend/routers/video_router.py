"""
routers/video_router.py
Route POST /predict_frames – MoBiLSTM video violence detection.
Nhận base64 frames được capture từ <video> element trên trang.
"""
from fastapi import APIRouter, HTTPException

from backend.schemas           import FramesRequest, VideoResponse
from backend.models            import get_video_model
from backend.services.video_service import predict_from_frames

router = APIRouter(prefix="/predict_frames", tags=["Video – Violence Detection"])


@router.post("", response_model=VideoResponse, summary="Phân tích frames video từ trang web")
def predict_frames_route(req: FramesRequest):
    """
    Nhận mảng base64 JPEG frames được capture từ video_scanner.js.
    Dùng sliding window (step = SEQ_LEN // 2) để phân tích nhiều đoạn.
    Trả về timeline Violence/NonViolence/Uncertain theo từng segment.
    """
    model = get_video_model()
    if model is None:
        raise HTTPException(503, "MoBiLSTM chưa được load")
    try:
        return predict_from_frames(req.frames, model, req.threshold)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Lỗi inference: {e}")
