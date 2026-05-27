"""
models/image_model.py
Load ViolenceViT từ notebook vit-1.ipynb.

Thay đổi so với notebook cũ (viiolence_v3):
  - Backbone: google/vit-base-patch16-224-in21k  (thay vì patch16-224)
  - Checkpoint key: "model_state"                (thay vì "model_state_dict")
  - Custom head: Dropout→Linear→GELU→LayerNorm→Dropout→Linear
  - Labels: "non-violent" / "violent"            (thay vì NonViolence/Violence)
  - Threshold: đọc từ checkpoint["config"]["recommended_threshold"]
  - Val transform: Resize(224)→ToTensor→Normalize (giống cũ, không đổi)
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTForImageClassification

from backend.config import (
    VIT_CHECKPOINT, VIT_MODEL_NAME, VIT_IMG_SIZE,
    VIT_CLASSES, VIT_ID2LABEL, VIT_LABEL2ID,
    IMAGENET_MEAN, IMAGENET_STD,
    VIT_THRESHOLD_FALLBACK, DEVICE,
)

# Val transform khớp get_transforms('val') trong notebook:
#   T.Resize((IMG_SIZE, IMG_SIZE)) → T.ToTensor() → T.Normalize(mean, std)
VIT_TRANSFORM = T.Compose([
    T.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ViolenceViT(nn.Module):
    """
    Khớp đúng kiến trúc ViolenceViT trong notebook vit-1.ipynb:
      backbone: ViTForImageClassification (frozen ban đầu, unfreeze epoch 5)
      head: Dropout(0.3) → Linear(768, 256) → GELU → LayerNorm(256)
                         → Dropout(0.2) → Linear(256, 2)
    """
    def __init__(self):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME,
            num_labels=len(VIT_CLASSES),
            id2label=VIT_ID2LABEL,
            label2id=VIT_LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        hidden_size = self.backbone.config.hidden_size  # 768 cho ViT-B
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(VIT_CLASSES)),
        )

    def forward(self, pixel_values):
        return self.backbone(pixel_values=pixel_values).logits


def load_image_model() -> tuple[ViolenceViT, float]:
    """
    Load ViolenceViT từ checkpoint violence_vit_final.pt.

    Checkpoint format (từ notebook):
        {
            'model_state': model.state_dict(),
            'config': {
                'recommended_threshold': float,
                'test_auroc': float,
                ...
            }
        }

    Returns: (model, threshold)
    """
    print(f"  Đang tải ViolenceViT ({VIT_MODEL_NAME})...")

    ckpt = torch.load(VIT_CHECKPOINT, map_location=DEVICE, weights_only=False)

    # Đọc threshold được tuning từ notebook (threshold analysis cell)
    threshold = VIT_THRESHOLD_FALLBACK
    if isinstance(ckpt, dict):
        if "config" in ckpt and "recommended_threshold" in ckpt["config"]:
            threshold = float(ckpt["config"]["recommended_threshold"])
            print(f"  Threshold từ checkpoint: {threshold:.2f}")
        else:
            print(f"  Threshold fallback: {threshold:.2f}")

    # Build model và load weights
    model = ViolenceViT()

    # Key "model_state" (notebook mới) — khác với "model_state_dict" (notebook cũ)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # Log metrics nếu có trong checkpoint
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        print(f"  Test AUROC    : {cfg.get('test_auroc', 'N/A')}")
        print(f"  Test Accuracy : {cfg.get('test_accuracy', 'N/A')}")

    print("  ViolenceViT ✓")
    return model, threshold
