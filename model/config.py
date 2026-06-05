"""
Central configuration for model training, inference, and deployment.
All hyperparameters and paths live here.
"""

import os
import json

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR      = os.path.join(BASE_DIR, "dataset", "final")
MODEL_DIR        = os.path.join(BASE_DIR, "model", "saved")
LOG_DIR          = os.path.join(BASE_DIR, "model", "logs")
TFLITE_DIR       = os.path.join(BASE_DIR, "model", "tflite")

LABEL_MAP_PATH   = os.path.join(DATASET_DIR, "label_map.json")
TRAIN_MANIFEST   = os.path.join(DATASET_DIR, "train.json")
VAL_MANIFEST     = os.path.join(DATASET_DIR, "val.json")
TEST_MANIFEST    = os.path.join(DATASET_DIR, "test.json")

PRETRAIN_MODEL_PATH = os.path.join(MODEL_DIR, "pretrained_model.h5")
FINETUNE_MODEL_PATH = os.path.join(MODEL_DIR, "finetuned_model.h5")
BEST_MODEL_PATH     = os.path.join(MODEL_DIR, "best_model.h5")
TFLITE_MODEL_PATH   = os.path.join(TFLITE_DIR, "sign_model.tflite")

# ─────────────────────────────────────────────
# INPUT SHAPE
# ─────────────────────────────────────────────
SEQUENCE_LENGTH  = 30
IMG_HEIGHT       = 224
IMG_WIDTH        = 224
CHANNELS         = 3
INPUT_SHAPE      = (SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# ─────────────────────────────────────────────
# CNN BACKBONE CONFIG
# ─────────────────────────────────────────────
CNN_BACKBONE     = "mobilenetv2"   # "mobilenetv2" | "efficientnetb0" | "custom"
CNN_PRETRAINED   = True             # Use ImageNet weights
CNN_TRAINABLE    = False            # Freeze CNN during LSTM pretraining
CNN_OUTPUT_DIM   = 1280             # MobileNetV2 output feature size

# ─────────────────────────────────────────────
# LSTM CONFIG
# ─────────────────────────────────────────────
LSTM_UNITS       = [256, 128]       # Stacked LSTM units
LSTM_DROPOUT     = 0.3
LSTM_RECURRENT_DROPOUT = 0.2

# ─────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────
DENSE_UNITS      = [256, 128]
DENSE_DROPOUT    = 0.4
L2_REG           = 1e-4

# ─────────────────────────────────────────────
# STAGE 1: PRETRAINING (on online/static dataset)
# ─────────────────────────────────────────────
PRETRAIN_EPOCHS     = 30
PRETRAIN_BATCH_SIZE = 8       # Keep small for GPU memory (RTX 4050 = 6GB)
PRETRAIN_LR         = 1e-3
PRETRAIN_LR_DECAY   = 0.5
PRETRAIN_PATIENCE   = 5       # Early stopping patience

# ─────────────────────────────────────────────
# STAGE 2: FINE-TUNING (on custom dataset)
# ─────────────────────────────────────────────
FINETUNE_EPOCHS      = 20
FINETUNE_BATCH_SIZE  = 4      # Smaller batch for fine-tuning
FINETUNE_LR          = 1e-4   # Lower LR for fine-tuning
FINETUNE_LR_DECAY    = 0.3
FINETUNE_PATIENCE    = 7
FINETUNE_CNN_UNFREEZE = True  # Unfreeze top CNN layers during fine-tuning
FINETUNE_UNFREEZE_LAYERS = 20 # Last N layers of CNN to unfreeze

# ─────────────────────────────────────────────
# GPU OPTIMIZATION
# ─────────────────────────────────────────────
MIXED_PRECISION      = True   # Use float16 for faster GPU training
GPU_MEMORY_LIMIT     = 5120   # MB — limit for RTX 4050 (6GB VRAM, leave headroom)
PREFETCH_BUFFER      = 4
NUM_PARALLEL_CALLS   = 4

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65   # Below this → "uncertain"
SMOOTHING_WINDOW     = 5       # Frames for prediction smoothing
TOP_K_PREDICTIONS    = 3       # Return top-K predictions

# ─────────────────────────────────────────────
# LABEL MAP LOADER
# ─────────────────────────────────────────────
def load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map not found at {LABEL_MAP_PATH}. "
            "Run scripts/preprocess_dataset.py first."
        )
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    return label_map


def get_num_classes():
    return len(load_label_map())


def get_index_to_label():
    label_map = load_label_map()
    return {v: k for k, v in label_map.items()}


# Create output directories
for d in [MODEL_DIR, LOG_DIR, TFLITE_DIR]:
    os.makedirs(d, exist_ok=True)