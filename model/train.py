"""
Two-stage training pipeline:
  Stage 1 (Pretraining)  — Train on large online/converted dataset
                           CNN frozen, LSTM + head learns temporal dynamics
  Stage 2 (Fine-tuning)  — Fine-tune on custom webcam-collected data
                           Top CNN layers unfrozen + lower LR

Outputs: pretrained_model.h5, finetuned_model.h5, best_model.h5
"""

import os
import json
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config as cfg
from model.architecture import (
    build_model, configure_gpu, unfreeze_cnn_top_layers, get_model_summary
)
from model.data_loader import (
    SequenceDataGenerator, get_class_weights, build_tf_dataset_from_generator
)


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def build_callbacks(stage, model_save_path, log_dir):
    callbacks = [
        # Save best model by val_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath   = model_save_path,
            monitor    = 'val_accuracy',
            save_best_only = True,
            save_weights_only = False,
            verbose    = 1
        ),

        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = cfg.PRETRAIN_LR_DECAY if stage == 1 else cfg.FINETUNE_LR_DECAY,
            patience  = 3,
            min_lr    = 1e-7,
            verbose   = 1
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = cfg.PRETRAIN_PATIENCE if stage == 1 else cfg.FINETUNE_PATIENCE,
            restore_best_weights = True,
            verbose              = 1
        ),

        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir        = os.path.join(log_dir, f"stage{stage}"),
            histogram_freq = 1,
            update_freq    = 'epoch'
        ),

        # CSV logging
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f"stage{stage}_training_log.csv")
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

def plot_training_history(history, stage, log_dir):
    """Save accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   lw=2)
    axes[0].set_title(f'Stage {stage} — Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   lw=2)
    axes[1].set_title(f'Stage {stage} — Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(log_dir, f"stage{stage}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def save_training_summary(history, stage, log_dir):
    summary = {
        "stage":           stage,
        "total_epochs":    len(history.history['accuracy']),
        "best_val_acc":    max(history.history['val_accuracy']),
        "best_val_loss":   min(history.history['val_loss']),
        "final_train_acc": history.history['accuracy'][-1],
        "final_val_acc":   history.history['val_accuracy'][-1],
    }
    path = os.path.join(log_dir, f"stage{stage}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[LOG] Summary saved: {path}")
    return summary


# ─────────────────────────────────────────────
# STAGE 1: PRETRAINING
# ─────────────────────────────────────────────

def pretrain(label_map, num_classes):
    print("\n" + "=" * 60)
    print("  STAGE 1: PRETRAINING")
    print("=" * 60)

    # Build model with frozen CNN
    model = build_model(
        num_classes   = num_classes,
        cnn_trainable = False
    )
    get_model_summary(model)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.PRETRAIN_LR)
    model.compile(
        optimizer = optimizer,
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy',
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    # Data generators
    print("\n[DATA] Building training data generator...")
    train_gen = SequenceDataGenerator(
        cfg.TRAIN_MANIFEST, label_map,
        batch_size = cfg.PRETRAIN_BATCH_SIZE,
        shuffle    = True,
        augment    = True
    )
    val_gen = SequenceDataGenerator(
        cfg.VAL_MANIFEST, label_map,
        batch_size = cfg.PRETRAIN_BATCH_SIZE,
        shuffle    = False,
        augment    = False
    )

    # Class weights
    class_weights = get_class_weights(cfg.TRAIN_MANIFEST, label_map)

    # Callbacks
    callbacks = build_callbacks(
        stage          = 1,
        model_save_path = cfg.PRETRAIN_MODEL_PATH,
        log_dir        = cfg.LOG_DIR
    )

    print(f"\n[TRAIN] Starting Stage 1 | Epochs: {cfg.PRETRAIN_EPOCHS} | "
          f"Batch: {cfg.PRETRAIN_BATCH_SIZE}")

    history = model.fit(
        train_gen,
        validation_data  = val_gen,
        epochs           = cfg.PRETRAIN_EPOCHS,
        callbacks        = callbacks,
        class_weight     = class_weights,
        workers          = 4,
        use_multiprocessing = False
    )

    plot_training_history(history, stage=1, log_dir=cfg.LOG_DIR)
    summary = save_training_summary(history, stage=1, log_dir=cfg.LOG_DIR)

    print(f"\n[STAGE 1 COMPLETE]")
    print(f"  Best val accuracy : {summary['best_val_acc']:.4f}")
    print(f"  Model saved to    : {cfg.PRETRAIN_MODEL_PATH}")

    return model, history


# ─────────────────────────────────────────────
# STAGE 2: FINE-TUNING
# ─────────────────────────────────────────────

def finetune(label_map, num_classes, pretrained_model=None):
    print("\n" + "=" * 60)
    print("  STAGE 2: FINE-TUNING")
    print("=" * 60)

    # Load pretrained model
    if pretrained_model is None:
        if not os.path.exists(cfg.PRETRAIN_MODEL_PATH):
            raise FileNotFoundError(
                f"Pretrained model not found: {cfg.PRETRAIN_MODEL_PATH}\n"
                "Run Stage 1 (pretrain) first."
            )
        print(f"[LOAD] Loading pretrained model: {cfg.PRETRAIN_MODEL_PATH}")
        model = tf.keras.models.load_model(cfg.PRETRAIN_MODEL_PATH)
    else:
        model = pretrained_model

    # Unfreeze top CNN layers
    if cfg.FINETUNE_CNN_UNFREEZE:
        unfreeze_cnn_top_layers(model, cfg.FINETUNE_UNFREEZE_LAYERS)

    # Recompile with lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.FINETUNE_LR)
    model.compile(
        optimizer = optimizer,
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy',
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    # Data — use custom dataset if it exists, else fall back to full dataset
    custom_train = os.path.join(cfg.DATASET_DIR, "custom_train.json")
    train_manifest = custom_train if os.path.exists(custom_train) else cfg.TRAIN_MANIFEST

    train_gen = SequenceDataGenerator(
        train_manifest, label_map,
        batch_size = cfg.FINETUNE_BATCH_SIZE,
        shuffle    = True,
        augment    = True
    )
    val_gen = SequenceDataGenerator(
        cfg.VAL_MANIFEST, label_map,
        batch_size = cfg.FINETUNE_BATCH_SIZE,
        shuffle    = False,
        augment    = False
    )

    class_weights = get_class_weights(train_manifest, label_map)

    callbacks = build_callbacks(
        stage           = 2,
        model_save_path = cfg.FINETUNE_MODEL_PATH,
        log_dir         = cfg.LOG_DIR
    )

    # Save best as best_model.h5 too
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath      = cfg.BEST_MODEL_PATH,
            monitor       = 'val_accuracy',
            save_best_only = True,
            verbose       = 0
        )
    )

    print(f"\n[TRAIN] Starting Stage 2 | Epochs: {cfg.FINETUNE_EPOCHS} | "
          f"Batch: {cfg.FINETUNE_BATCH_SIZE} | LR: {cfg.FINETUNE_LR}")

    history = model.fit(
        train_gen,
        validation_data  = val_gen,
        epochs           = cfg.FINETUNE_EPOCHS,
        callbacks        = callbacks,
        class_weight     = class_weights,
        workers          = 4,
        use_multiprocessing = False
    )

    plot_training_history(history, stage=2, log_dir=cfg.LOG_DIR)
    summary = save_training_summary(history, stage=2, log_dir=cfg.LOG_DIR)

    print(f"\n[STAGE 2 COMPLETE]")
    print(f"  Best val accuracy : {summary['best_val_acc']:.4f}")
    print(f"  Finetuned model   : {cfg.FINETUNE_MODEL_PATH}")
    print(f"  Best model        : {cfg.BEST_MODEL_PATH}")

    return model, history


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("=" * 60)

    # GPU setup
    configure_gpu()

    # Load label map
    label_map   = cfg.load_label_map()
    num_classes = len(label_map)
    print(f"\n[INFO] Classes: {num_classes} | Labels: {list(label_map.keys())[:10]}...")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="0=both stages, 1=pretrain only, 2=finetune only")
    args = parser.parse_args()

    if args.stage in (0, 1):
        pretrained_model, _ = pretrain(label_map, num_classes)
    else:
        pretrained_model = None

    if args.stage in (0, 2):
        finetune(label_map, num_classes, pretrained_model)

    print("\n[DONE] Training pipeline complete.")
    print(f"  Best model: {cfg.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()