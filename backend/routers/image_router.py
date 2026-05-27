"""
routers/image_router.py
Route POST /predict_images – ViolenceViT image violence detection.

Thay đổi: threshold mặc định lấy từ get_vit_threshold()
(recommended_threshold được tuning trong notebook vit-1.ipynb).
Nếu client truyền threshold riêng thì dùng của client.
"""
from fastapi import APIRouter, HTTPException

from backend.schemas                import ImagesRequest, ImagesResponse
from backend.models                 import get_image_model, get_vit_threshold
from backend.services.image_service import predict_images

router = APIRouter(prefix="/predict_images", tags=["Image – Violence Detection"])


@router.post("", response_model=ImagesResponse, summary="Phân loại ảnh bạo lực từ trang web")
def predict_images_route(req: ImagesRequest):
    """
    Nhận mảng base64 JPEG ảnh từ image_scanner.js.
    Mỗi ảnh được phân loại độc lập bởi ViolenceViT.

    Threshold:
      - Dùng req.threshold nếu client truyền vào (khác 0.70 mặc định schema).
      - Fallback về recommended_threshold từ checkpoint nếu client không truyền.
    """
    model = get_image_model()
    if model is None:
        raise HTTPException(503, "ViolenceViT chưa được load")

    # Nếu client dùng default (0.70), ưu tiên threshold từ notebook
    threshold = req.threshold
    model_threshold = get_vit_threshold()
    if threshold == 0.70 and model_threshold is not None:
        threshold = model_threshold

    try:
        return predict_images(req.images, model, threshold)
    except Exception as e:
        raise HTTPException(500, f"Lỗi inference: {e}")
