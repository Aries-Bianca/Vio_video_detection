"""
models/text_model.py
Định nghĩa PhoBERTClassifier và hàm load checkpoint.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from backend.config import (
    TEXT_MODEL_NAME, TEXT_CHECKPOINT,
    TEXT_NUM_CLASSES, DEVICE
)


class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out     = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]          # lấy token [CLS]
        return self.classifier(self.dropout(cls_vec))


def load_text_model() -> tuple[PhoBERTClassifier, AutoTokenizer]:
    """Load PhoBERT model + tokenizer từ checkpoint. Trả về (model, tokenizer)."""
    print("  Đang tải PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    print("  Đang tải PhoBERT weights...")
    model = PhoBERTClassifier(TEXT_MODEL_NAME, TEXT_NUM_CLASSES)
    ckpt  = torch.load(TEXT_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    print("  PhoBERT ✓")
    return model, tokenizer
