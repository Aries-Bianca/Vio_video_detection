"""
app.py – FastAPI application entry point.
Chỉ chứa: tạo app, CORS, lifespan (load models), đăng ký routers, /health.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import models as model_registry
from backend.routers import text_router, video_router, image_router
from backend.config  import DEVICE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tất cả models trước khi nhận request."""
    model_registry.startup()
    yield
    # Cleanup nếu cần


app = FastAPI(
    title="ViDetect API",
    version="3.1.0",
    description="Hate Speech (PhoBERT) · Video Violence (MoBiLSTM) · Image Violence (ViT)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(text_router.router)
app.include_router(video_router.router)
app.include_router(image_router.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    text_m, _   = model_registry.get_text_model()
    video_m      = model_registry.get_video_model()
    image_m      = model_registry.get_image_model()
    vit_thresh = model_registry.get_vit_threshold()
    return {
        "status":  "ok",
        "version": "3.1.0",
        "device":  str(DEVICE),
        "models": {
            "phobert":      text_m  is not None,
            "mobilstm":     video_m is not None,
            "violence_vit": image_m is not None,
        },
        "vit_threshold": vit_thresh,
        "endpoints": ["/predict", "/predict_frames", "/predict_images"],
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
