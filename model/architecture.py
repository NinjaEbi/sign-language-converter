"""
CNN + LSTM architecture for sign language recognition.
Supports MobileNetV2 and EfficientNetB0 backbones.
TimeDistributed CNN extracts spatial features per frame,
stacked Bidirectional LSTM models temporal dynamics.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0
)
from tensorflow.keras.mixed_precision import set_global_policy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import *


def configure_gpu():
    """Configure GPU for optimal RTX 4050 performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit VRAM usage
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=GPU_MEMORY_LIMIT
                )]
            )
            print(f"[GPU] Configured: {gpus[0].name} | "
                  f"Memory limit: {GPU_MEMORY_LIMIT} MB")
        except RuntimeError as e:
            print(f"[GPU] Config error: {e}")

    if MIXED_PRECISION:
        set_global_policy('mixed_float16')
        print("[GPU] Mixed precision (float16) enabled")


def build_cnn_backbone(input_shape, backbone_name, pretrained, trainable):
    """
    Build CNN feature extractor.
    input_shape: (H, W, C) — single frame
    Returns: (base_model, feature_dim)
    """
    weights = 'imagenet' if pretrained else None

    if backbone_name == "mobilenetv2":
        base = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling='avg'
        )
        feature_dim = 1280

    elif backbone_name == "efficientnetb0":
        base = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling='avg'
        )
        feature_dim = 1280

    elif backbone_name == "custom":
        # Lightweight custom CNN for low-resource deployment
        inp = tf.keras.Input(shape=input_shape)
        x   = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
        x   = layers.BatchNormalization()(x)
        x   = layers.MaxPooling2D(2)(x)

        x   = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.MaxPooling2D(2)(x)

        x   = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.MaxPooling2D(2)(x)

        x   = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.GlobalAveragePooling2D()(x)
        base = tf.keras.Model(inputs=inp, outputs=x, name="custom_cnn")
        feature_dim = 256

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = trainable
    return base, feature_dim


def build_model(num_classes,
                backbone=CNN_BACKBONE,
                pretrained=CNN_PRETRAINED,
                cnn_trainable=CNN_TRAINABLE,
                lstm_units=None,
                dense_units=None):
    """
    Full CNN + LSTM model.

    Architecture:
      Input(seq_len, H, W, C)
      → TimeDistributed(MobileNetV2) → (seq_len, feature_dim)
      → Bidirectional LSTM × N
      → Dense → Dropout
      → Softmax output

    Args:
        num_classes: number of sign classes
    Returns:
        Keras Model
    """
    if lstm_units  is None: lstm_units  = LSTM_UNITS
    if dense_units is None: dense_units = DENSE_UNITS

    frame_shape  = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    cnn, feat_dim = build_cnn_backbone(frame_shape, backbone,
                                        pretrained, cnn_trainable)

    # Input
    sequence_input = tf.keras.Input(
        shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        name="sequence_input"
    )

    # TimeDistributed CNN — apply same CNN to each frame
    x = layers.TimeDistributed(cnn, name="td_cnn")(sequence_input)
    # x shape: (batch, seq_len, feature_dim)

    # Optional: lightweight temporal attention on CNN features
    # (helps model focus on key frames)
    x = frame_attention(x, feat_dim)

    # Stacked Bidirectional LSTM
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)  # return sequences for all but last
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=LSTM_DROPOUT,
                recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                kernel_regularizer=regularizers.l2(L2_REG),
            ),
            name=f"bilstm_{i}"
        )(x)
        if return_seq:
            x = layers.LayerNormalization(name=f"layernorm_{i}")(x)

    # Dense classification head
    for j, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f"dense_{j}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_dense_{j}")(x)
        x = layers.Dropout(DENSE_DROPOUT, name=f"dropout_dense_{j}")(x)

    # Output — cast to float32 for numerical stability with mixed precision
    output = layers.Dense(num_classes, name="logits")(x)
    output = layers.Activation('softmax', dtype='float32', name="output")(output)

    model = Model(inputs=sequence_input, outputs=output, name="SignLangCNN_LSTM")
    return model


def frame_attention(x, feat_dim):
    """
    Soft attention over temporal sequence.
    Learns which frames are most discriminative.
    """
    # x: (batch, seq_len, feat_dim)
    score = layers.Dense(1, name="attn_score")(x)           # (batch, seq_len, 1)
    weight = layers.Softmax(axis=1, name="attn_weight")(score)
    attended = layers.Multiply(name="attn_apply")([x, weight])  # (batch, seq_len, feat_dim)
    # Add residual connection
    x = layers.Add(name="attn_residual")([x, attended])
    x = layers.LayerNormalization(name="attn_norm")(x)
    return x


def unfreeze_cnn_top_layers(model, n_layers):
    """Unfreeze top N layers of the CNN backbone for fine-tuning."""
    td_cnn = None
    for layer in model.layers:
        if layer.name == "td_cnn":
            td_cnn = layer
            break

    if td_cnn is None:
        print("[WARN] Could not find td_cnn layer for unfreezing")
        return

    cnn = td_cnn.layer
    cnn.trainable = True

    # Freeze all but last n_layers
    total = len(cnn.layers)
    for layer in cnn.layers[:total - n_layers]:
        layer.trainable = False
    for layer in cnn.layers[total - n_layers:]:
        layer.trainable = True

    trainable = sum(1 for l in cnn.layers if l.trainable)
    print(f"[FINETUNE] CNN: {trainable}/{total} layers trainable "
          f"(unfroze last {n_layers})")


def get_model_summary(model):
    """Print model summary with parameter count."""
    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n{'='*50}")
    print(f"  Model: {model.name}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Input shape     : {model.input_shape}")
    print(f"  Output shape    : {model.output_shape}")
    print(f"{'='*50}\n")