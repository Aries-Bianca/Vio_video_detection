"""
models/__init__.py
Model registry – load một lần lúc startup, dùng lại cho mọi request.
"""
from backend.models.text_model  import load_text_model
from backend.models.video_model import configure_tf_memory, load_video_model
from backend.models.image_model import load_image_model

_text_model      = None
_text_tokenizer  = None
_video_model     = None
_image_model     = None
_vit_threshold   = None   # threshold đọc từ checkpoint (notebook mới)


def startup():
    """Gọi một lần khi FastAPI khởi động."""
    global _text_model, _text_tokenizer, _video_model, _image_model, _vit_threshold

    print("\n[ViDetect] Đang load models...")

    print("[0/3] Cấu hình TensorFlow memory...")
    configure_tf_memory()

    print("[1/3] PhoBERT – Hate Speech")
    _text_model, _text_tokenizer = load_text_model()

    print("[2/3] MoBiLSTM – Video Violence")
    _video_model = load_video_model()

    print("[3/3] ViolenceViT – Image Violence")
    # load_image_model() trả về (model, threshold) từ notebook mới
    _image_model, _vit_threshold = load_image_model()

    print("[ViDetect] Tất cả models đã sẵn sàng ✓\n")


def get_text_model():
    return _text_model, _text_tokenizer

def get_video_model():
    return _video_model

def get_image_model():
    return _image_model

def get_vit_threshold() -> float:
    """Trả về threshold được tuning từ notebook, dùng khi client không truyền threshold."""
    return _vit_threshold
