"""
routers/text_router.py
Route POST /predict – PhoBERT hate speech detection.
"""
from fastapi import APIRouter, HTTPException

from backend.schemas          import TextRequest, TextResponse
from backend.models           import get_text_model
from backend.services.text_service import predict_text

router = APIRouter(prefix="/predict", tags=["Text – Hate Speech"])


@router.post("", response_model=TextResponse, summary="Phân loại văn bản thù ghét")
def predict_text_route(req: TextRequest):
    """
    Nhận một đoạn văn bản tiếng Việt.
    Trả về nhãn Clean / Offensive / Hate kèm confidence và probability vector.
    """
    model, tokenizer = get_text_model()
    if model is None:
        raise HTTPException(503, "PhoBERT chưa được load")
    try:
        return predict_text(req.text, model, tokenizer)
    except Exception as e:
        raise HTTPException(500, f"Lỗi inference: {e}")
