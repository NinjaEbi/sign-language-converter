"""
Two-stage training pipeline for public dataset → custom fine-tuning.

Stage 1 — Pretraining:
  Dataset: Public (ASL Alphabet, Kaggle)
  CNN:     FROZEN (ImageNet weights preserved)
  Trains:  Transformer + projection + head
  Goal:    Learn sign language temporal patterns
  LR:      1e-3

Stage 2 — Fine-tuning:
  Dataset: Custom collected OR same dataset
  CNN:     TOP 30 LAYERS UNFROZEN
  Goal:    Adapt CNN features to your camera/lighting
  LR:      1e-4 (10x smaller)

Output: model/saved/best_model.h5
"""

import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config as cfg
from model.architecture import (
    build_model, configure_gpu,
    unfreeze_cnn_top_layers, get_model_summary
)
from model.data_loader import (
    SequenceDataGenerator, get_class_weights
)


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def build_callbacks(stage, model_save_path, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath        = model_save_path,
            monitor         = 'val_accuracy',
            save_best_only  = True,
            verbose         = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.4,
            patience = 4,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = cfg.PRETRAIN_PATIENCE if stage == 1
                                   else cfg.FINETUNE_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir        = os.path.join(log_dir, f"stage{stage}"),
            histogram_freq = 0,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f"stage{stage}_log.csv")
        ),
    ]


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_history(history, stage, log_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],
                  label='Train', lw=2, color='#6366f1')
    axes[0].plot(history.history['val_accuracy'],
                  label='Val',   lw=2, color='#10b981')
    axes[0].set_title(f'Stage {stage} — Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    axes[1].plot(history.history['loss'],
                  label='Train', lw=2, color='#6366f1')
    axes[1].plot(history.history['val_loss'],
                  label='Val',   lw=2, color='#10b981')
    axes[1].set_title(f'Stage {stage} — Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(log_dir, f"stage{stage}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {path}")


def save_summary(history, stage, log_dir):
    summary = {
        "stage":           stage,
        "epochs_run":      len(history.history['accuracy']),
        "best_val_acc":    max(history.history['val_accuracy']),
        "best_val_loss":   min(history.history['val_loss']),
        "final_train_acc": history.history['accuracy'][-1],
    }
    path = os.path.join(log_dir, f"stage{stage}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ─────────────────────────────────────────────
# STAGE 1: PRETRAINING
# ─────────────────────────────────────────────

def pretrain(label_map, num_classes):
    print("\n" + "="*60)
    print("  STAGE 1 — PRETRAINING (CNN frozen)")
    print("  Dataset: Public / Converted sequences")
    print("="*60)

    model = build_model(
        num_classes   = num_classes,
        cnn_trainable = False,    # CNN frozen
        dropout       = 0.1,
        num_layers    = 2,
    )
    get_model_summary(model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate = cfg.PRETRAIN_LR,
        clipnorm      = 1.0,       # gradient clipping
    )
    model.compile(
        optimizer = optimizer,
        loss      = 'categorical_crossentropy',
        metrics   = [
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=3, name='top3_acc'
            ),
        ]
    )

    train_gen = SequenceDataGenerator(
        cfg.TRAIN_MANIFEST, label_map,
        batch_size = cfg.PRETRAIN_BATCH_SIZE,
        shuffle    = True,
        augment    = True,
    )
    val_gen = SequenceDataGenerator(
        cfg.VAL_MANIFEST, label_map,
        batch_size = cfg.PRETRAIN_BATCH_SIZE,
        shuffle    = False,
        augment    = False,
    )

    class_weights = get_class_weights(cfg.TRAIN_MANIFEST, label_map)
    callbacks     = build_callbacks(1, cfg.PRETRAIN_MODEL_PATH, cfg.LOG_DIR)

    print(f"\n[TRAIN] Epochs: {cfg.PRETRAIN_EPOCHS} | "
          f"Batch: {cfg.PRETRAIN_BATCH_SIZE} | "
          f"LR: {cfg.PRETRAIN_LR}")

    history = model.fit(
        train_gen,
        validation_data     = val_gen,
        epochs              = cfg.PRETRAIN_EPOCHS,
        callbacks           = callbacks,
        class_weight        = class_weights,
    )

    plot_history(history, 1, cfg.LOG_DIR)
    summary = save_summary(history, 1, cfg.LOG_DIR)

    print(f"\n[STAGE 1 DONE] Best val_acc: {summary['best_val_acc']:.4f}")
    print(f"[SAVED] {cfg.PRETRAIN_MODEL_PATH}")
    return model, history


# ─────────────────────────────────────────────
# STAGE 2: FINE-TUNING
# ─────────────────────────────────────────────

def finetune(label_map, num_classes, pretrained_model=None):
    print("\n" + "="*60)
    print("  STAGE 2 — FINE-TUNING (CNN partially unfrozen)")
    print("  Dataset: Custom collected data")
    print("="*60)

    if pretrained_model is None:
        if not os.path.exists(cfg.PRETRAIN_MODEL_PATH):
            raise FileNotFoundError(
                f"Pretrained model not found: {cfg.PRETRAIN_MODEL_PATH}\n"
                "Run Stage 1 first: python model/train.py --stage 1"
            )
        print(f"[LOAD] {cfg.PRETRAIN_MODEL_PATH}")
        model = tf.keras.models.load_model(
            cfg.PRETRAIN_MODEL_PATH,
            custom_objects={
                "PositionalEncoding":    __import__(
                    'model.architecture',
                    fromlist=['PositionalEncoding']
                ).PositionalEncoding,
                "TransformerEncoderBlock": __import__(
                    'model.architecture',
                    fromlist=['TransformerEncoderBlock']
                ).TransformerEncoderBlock,
            }
        )
    else:
        model = pretrained_model

    # Unfreeze top CNN layers
    unfreeze_cnn_top_layers(model, n_layers=30)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate = cfg.FINETUNE_LR,
        clipnorm      = 1.0,
    )
    model.compile(
        optimizer = optimizer,
        loss      = 'categorical_crossentropy',
        metrics   = [
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=3, name='top3_acc'
            ),
        ]
    )

    # Use custom manifest if it exists, else full train set
    custom_manifest = os.path.join(
        cfg.DATASET_DIR, "custom_train.json"
    )
    train_manifest = (
        custom_manifest
        if os.path.exists(custom_manifest)
        else cfg.TRAIN_MANIFEST
    )

    train_gen = SequenceDataGenerator(
        train_manifest, label_map,
        batch_size = cfg.FINETUNE_BATCH_SIZE,
        shuffle    = True,
        augment    = True,
    )
    val_gen = SequenceDataGenerator(
        cfg.VAL_MANIFEST, label_map,
        batch_size = cfg.FINETUNE_BATCH_SIZE,
        shuffle    = False,
        augment    = False,
    )

    class_weights = get_class_weights(train_manifest, label_map)
    callbacks     = build_callbacks(2, cfg.FINETUNE_MODEL_PATH, cfg.LOG_DIR)

    # Also save as best_model.h5
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = cfg.BEST_MODEL_PATH,
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 0,
        )
    )

    print(f"\n[TRAIN] Epochs: {cfg.FINETUNE_EPOCHS} | "
          f"Batch: {cfg.FINETUNE_BATCH_SIZE} | "
          f"LR: {cfg.FINETUNE_LR}")

    history = model.fit(
        train_gen,
        validation_data     = val_gen,
        epochs              = cfg.FINETUNE_EPOCHS,
        callbacks           = callbacks,
        class_weight        = class_weights,
    
    )

    plot_history(history, 2, cfg.LOG_DIR)
    summary = save_summary(history, 2, cfg.LOG_DIR)

    print(f"\n[STAGE 2 DONE] Best val_acc: {summary['best_val_acc']:.4f}")
    print(f"[SAVED] {cfg.BEST_MODEL_PATH}")
    return model, history


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("  Architecture: MobileNetV2 + Transformer")
    print("="*60)

    configure_gpu()

    label_map   = cfg.load_label_map()
    num_classes = len(label_map)
    print(f"\n[INFO] Classes: {num_classes}")
    print(f"[INFO] Labels : {list(label_map.keys())}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=int, default=0,
        help="0=both, 1=pretrain only, 2=finetune only"
    )
    args = parser.parse_args()

    pretrained = None

    if args.stage in (0, 1):
        pretrained, _ = pretrain(label_map, num_classes)

    if args.stage in (0, 2):
        finetune(label_map, num_classes, pretrained)

    print("\n[DONE] Training complete.")
    print(f"[MODEL] {cfg.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()