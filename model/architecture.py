"""
UPGRADED: MobileNetV2 + Temporal Transformer architecture.
Replaces CNN+LSTM with CNN+Transformer for better accuracy
on sign language recognition.

Why Transformer over LSTM:
- Attention mechanisms capture which frames matter most
- No vanishing gradient problem
- Better parallelization → faster training
- State-of-the-art on video classification tasks
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import (
    IMG_HEIGHT, IMG_WIDTH, CHANNELS, SEQUENCE_LENGTH,
    GPU_MEMORY_LIMIT, MIXED_PRECISION, L2_REG, DENSE_DROPOUT
)


# ─────────────────────────────────────────────
# GPU CONFIGURATION
# ─────────────────────────────────────────────

def configure_gpu():
    """Configure GPU for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=GPU_MEMORY_LIMIT
                )]
            )
            print(f"[GPU] Configured: {gpus[0].name} | "
                  f"Limit: {GPU_MEMORY_LIMIT}MB")
        except RuntimeError as e:
            print(f"[GPU] Config error: {e}")
    else:
        print("[CPU] No GPU found — running on CPU")

    if MIXED_PRECISION and gpus:
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')
        print("[GPU] Mixed precision (float16) enabled")


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding.
    Tells the Transformer the temporal order of frames.
    Without this, attention treats frames as a bag (no order).
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Compute fixed sinusoidal encodings
        positions = np.arange(max_len)[:, np.newaxis]       # (max_len, 1)
        dims      = np.arange(d_model)[np.newaxis, :]        # (1, d_model)
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.encoding = tf.cast(
            angles[np.newaxis, :, :], dtype=tf.float32
        )  # (1, max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# ─────────────────────────────────────────────
# TRANSFORMER ENCODER BLOCK
# ─────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer encoder block.
    Multi-Head Self-Attention + Feed Forward + Residual + LayerNorm.

    d_model:   feature dimension (must match CNN output projection)
    num_heads: number of attention heads
    ff_dim:    feed-forward inner dimension
    dropout:   dropout rate
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout_rate = dropout

        self.attention   = layers.MultiHeadAttention(
            num_heads  = num_heads,
            key_dim    = d_model // num_heads,
            dropout    = dropout,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu',
                          kernel_regularizer=regularizers.l2(L2_REG)),
            layers.Dropout(dropout),
            layers.Dense(d_model,
                          kernel_regularizer=regularizers.l2(L2_REG)),
        ])
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head self-attention with residual
        attn_out = self.attention(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        # Feed-forward with residual
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model":   self.d_model,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout_rate,
        })
        return config


# ─────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────

def build_model(
    num_classes,
    d_model      = 256,    # Transformer hidden dimension
    num_heads    = 4,      # Attention heads
    ff_dim       = 512,    # Feed-forward dimension
    num_layers   = 2,      # Transformer encoder layers
    dropout      = 0.1,
    cnn_trainable = False,
):
    """
    MobileNetV2 + Temporal Transformer for sign language recognition.

    Architecture:
      Input(30, 224, 224, 3)
      → TimeDistributed(MobileNetV2)  → (30, 1280)
      → Dense projection              → (30, d_model)
      → PositionalEncoding            → (30, d_model)
      → TransformerEncoder × N        → (30, d_model)
      → GlobalAveragePooling1D        → (d_model,)
      → Dense(256) → Dropout
      → Dense(num_classes) → Softmax

    Args:
        num_classes:   number of sign language classes
        d_model:       transformer hidden dimension
        num_heads:     multi-head attention heads
        ff_dim:        feed-forward layer size
        num_layers:    number of transformer blocks
        dropout:       dropout rate throughout
        cnn_trainable: whether CNN backbone is trainable
    """

    # ── CNN Backbone ──
    cnn_backbone = MobileNetV2(
        input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        include_top = False,
        weights     = 'imagenet',
        pooling     = 'avg',
    )
    cnn_backbone.trainable = cnn_trainable

    # ── Input ──
    sequence_input = tf.keras.Input(
        shape = (SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        name  = "sequence_input"
    )

    # ── Feature extraction per frame ──
    # MobileNetV2 output: (batch, 30, 1280)
    x = layers.TimeDistributed(cnn_backbone, name="td_mobilenet")(sequence_input)

    # ── Project to d_model dimensions ──
    # (batch, 30, 1280) → (batch, 30, d_model)
    x = layers.TimeDistributed(
        layers.Dense(
            d_model,
            activation = 'relu',
            kernel_regularizer = regularizers.l2(L2_REG),
        ),
        name = "feature_projection"
    )(x)
    x = layers.TimeDistributed(
        layers.LayerNormalization(epsilon=1e-6),
        name = "projection_norm"
    )(x)
    x = layers.Dropout(dropout, name="projection_dropout")(x)

    # ── Positional encoding ──
    x = PositionalEncoding(
        max_len = SEQUENCE_LENGTH,
        d_model = d_model,
        name    = "positional_encoding"
    )(x)

    # ── Transformer encoder blocks ──
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model   = d_model,
            num_heads = num_heads,
            ff_dim    = ff_dim,
            dropout   = dropout,
            name      = f"transformer_block_{i}"
        )(x)

    # ── Aggregate temporal dimension ──
    # Mean pooling across all 30 frames
    x = layers.GlobalAveragePooling1D(name="temporal_pooling")(x)

    # ── Classification head ──
    x = layers.Dense(
        256,
        activation = 'relu',
        kernel_regularizer = regularizers.l2(L2_REG),
        name = "head_dense_1"
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(DENSE_DROPOUT, name="head_dropout")(x)

    x = layers.Dense(
        128,
        activation = 'relu',
        kernel_regularizer = regularizers.l2(L2_REG),
        name = "head_dense_2"
    )(x)
    x = layers.Dropout(DENSE_DROPOUT * 0.5, name="head_dropout_2")(x)

    # ── Output ──
    output = layers.Dense(num_classes, name="logits")(x)
    output = layers.Activation(
        'softmax', dtype='float32', name="output"
    )(output)

    model = Model(
        inputs  = sequence_input,
        outputs = output,
        name    = "SignLang_MobileNetV2_Transformer"
    )
    return model


# ─────────────────────────────────────────────
# FINE-TUNING HELPERS
# ─────────────────────────────────────────────

def unfreeze_cnn_top_layers(model, n_layers=30):
    """
    Unfreeze the top N layers of the MobileNetV2 backbone.
    Used in Stage 2 (fine-tuning) to adapt pretrained features
    to sign language domain.
    """
    td_layer = None
    for layer in model.layers:
        if layer.name == "td_mobilenet":
            td_layer = layer
            break

    if td_layer is None:
        print("[WARN] td_mobilenet layer not found")
        return

    cnn    = td_layer.layer
    total  = len(cnn.layers)
    cnn.trainable = True

    for layer in cnn.layers[:total - n_layers]:
        layer.trainable = False
    for layer in cnn.layers[total - n_layers:]:
        layer.trainable = True

    trainable = sum(1 for l in cnn.layers if l.trainable)
    print(f"[FINETUNE] CNN: {trainable}/{total} layers trainable "
          f"(unfroze last {n_layers})")


def get_model_summary(model):
    total      = model.count_params()
    trainable  = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_train  = total - trainable
    print(f"\n{'='*55}")
    print(f"  Model : {model.name}")
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}")
    print(f"  Frozen params    : {non_train:>12,}")
    print(f"  Input shape      : {model.input_shape}")
    print(f"  Output shape     : {model.output_shape}")
    print(f"{'='*55}\n")