"""
config.py – Toàn bộ hằng số của dự án ở một chỗ.
Thay đổi checkpoint path, threshold, hay model name → chỉ sửa file này.
"""
import os
import torch

# ── Base directory (backend/) ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model 1: PhoBERT (Hate Speech) ────────────────────────────────────────────
TEXT_MODEL_NAME  = "vinai/phobert-base"
TEXT_CHECKPOINT  = os.path.join(BASE_DIR, "checkpoints", "phobert_hate_full.pt")
TEXT_MAX_LEN     = 128
TEXT_NUM_CLASSES = 3
TEXT_LABEL_MAP   = {0: "Clean", 1: "Offensive", 2: "Hate"}

# ── Model 2: MoBiLSTM (Video Violence) ────────────────────────────────────────
VIDEO_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "violence_model_v2.keras")
VIDEO_IMAGE_H    = 112
VIDEO_IMAGE_W    = 112
VIDEO_SEQ_LEN    = 10          # Giảm từ 20 → 10: RAM working set −50%, inference ~2x nhanh hơn
VIDEO_CLASSES    = ["NonViolence", "Violence"]
VIDEO_THRESHOLD  = 0.70

# ── TensorFlow memory config ───────────────────────────────────────────────────
# TF mặc định chiếm toàn bộ VRAM lúc khởi động.
# TF_MEMORY_GROWTH = True  → chỉ cấp thêm VRAM khi thực sự cần (khuyến nghị).
# TF_MEMORY_LIMIT_MB       → giới hạn cứng VRAM cho TF (MB). None = không giới hạn.
TF_MEMORY_GROWTH   = True
TF_MEMORY_LIMIT_MB = None     # Ví dụ: 1024 để giới hạn 1 GB

# ── Model 3: ViolenceViT (Image Violence) — vit-1.ipynb ───────────────────────
VIT_CHECKPOINT         = os.path.join(BASE_DIR, "checkpoints", "violence_vit_final.pt")
VIT_MODEL_NAME         = "google/vit-base-patch16-224-in21k"
VIT_IMG_SIZE           = 224
VIT_CLASSES            = ["non-violent", "violent"]
VIT_ID2LABEL           = {0: "non-violent", 1: "violent"}
VIT_LABEL2ID           = {"non-violent": 0, "violent": 1}
IMAGENET_MEAN          = [0.485, 0.456, 0.406]
IMAGENET_STD           = [0.229, 0.224, 0.225]
VIT_THRESHOLD_FALLBACK = 0.50
