"""
models/video_model.py
Định nghĩa MobileNetV2 + BiLSTM + TemporalAttention và hàm load checkpoint.
Kiến trúc khớp hoàn toàn với violence-detection-v4.ipynb.
"""
import tensorflow as tf
import keras
from keras.layers import (
    Dense, Dropout, LSTM, Bidirectional,
    TimeDistributed, BatchNormalization,
    GlobalAveragePooling2D, Input,
)
from keras.models import Model
from keras.regularizers import l2
from keras.applications.mobilenet_v2 import MobileNetV2

from backend.config import (
    VIDEO_CHECKPOINT, VIDEO_IMAGE_H, VIDEO_IMAGE_W,
    VIDEO_SEQ_LEN, VIDEO_CLASSES,
    TF_MEMORY_GROWTH, TF_MEMORY_LIMIT_MB,
)


def configure_tf_memory() -> None:
    """
    Giới hạn mức sử dụng VRAM của TensorFlow.

    Mặc định TF chiếm toàn bộ VRAM ngay lúc khởi động (pre-allocation),
    dẫn đến xung đột với PyTorch (PhoBERT, ViT) đang cùng dùng GPU.

    Hai chế độ (cấu hình trong config.py):
      TF_MEMORY_GROWTH = True   → cấp thêm VRAM theo nhu cầu thực tế (khuyến nghị).
      TF_MEMORY_LIMIT_MB = N    → giới hạn cứng N MB, kết hợp hoặc thay thế growth.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("  TensorFlow: không có GPU, chạy trên CPU.")
        return

    for gpu in gpus:
        try:
            if TF_MEMORY_GROWTH:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  TF memory growth = True trên {gpu.name}")

            if TF_MEMORY_LIMIT_MB is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=TF_MEMORY_LIMIT_MB
                    )],
                )
                print(f"  TF memory limit  = {TF_MEMORY_LIMIT_MB} MB trên {gpu.name}")

        except RuntimeError as e:
            # Phải gọi trước khi TF khởi tạo bất kỳ tensor nào.
            # Nếu đã muộn → log cảnh báo, không crash server.
            print(f"  CẢNH BÁO: Không set TF memory config được ({e}). "
                  f"Gọi configure_tf_memory() trước khi import keras layers.")


class TemporalAttention(keras.layers.Layer):
    """Self-attention theo chiều thời gian giữa các LSTM hidden states."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, use_bias=False)
        self.V = Dense(1)

    def call(self, hidden_states):
        score   = tf.nn.tanh(self.W(hidden_states))
        weights = tf.nn.softmax(self.V(score), axis=1)
        return weights * hidden_states

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


def build_video_model() -> Model:
    """
    Xây dựng kiến trúc MobileNetV2 → BiLSTM → TemporalAttention → classifier.
    Khớp đúng với notebook violence-detection-v4.
    """
    # Backbone: MobileNetV2 không có top, weights ImageNet
    mobilenet = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(VIDEO_IMAGE_H, VIDEO_IMAGE_W, 3),
    )
    mobilenet.trainable = False
    backbone = Model(
        inputs=mobilenet.input,
        outputs=GlobalAveragePooling2D()(mobilenet.output),
        name="mobilenet_backbone",
    )

    # Temporal head
    seq_input = Input(shape=(VIDEO_SEQ_LEN, VIDEO_IMAGE_H, VIDEO_IMAGE_W, 3), name="frame_sequence")
    x = TimeDistributed(backbone, name="feature_extraction")(seq_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    x = TemporalAttention(units=256, name="temporal_attention")(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    output = Dense(len(VIDEO_CLASSES), activation="softmax", name="classifier")(x)

    return Model(inputs=seq_input, outputs=output, name="violence_detector")


def load_video_model() -> Model:
    """Load MoBiLSTM từ checkpoint Keras."""
    print("  Đang build MoBiLSTM architecture...")
    model = build_video_model()

    print("  Đang load weights MoBiLSTM...")
    model.load_weights(VIDEO_CHECKPOINT)
    model.trainable = False

    print("  MoBiLSTM ✓")
    return model
