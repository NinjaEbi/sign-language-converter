"""
Retraining script for continuous learning.
Merges new user feedback data with existing training set
and fine-tunes the model incrementally.

Usage:
  python scripts/retrain.py
  python scripts/retrain.py --min-samples 5 --epochs 10
"""

import os
import sys
import json
import shutil
import argparse
import logging
import datetime

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config as cfg
from model.architecture import build_model, configure_gpu, unfreeze_cnn_top_layers
from model.data_loader  import SequenceDataGenerator, get_class_weights
from model.train        import build_callbacks, plot_training_history

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("retrain")


def count_new_samples(new_data_dir):
    """Count new feedback samples per class."""
    if not os.path.exists(new_data_dir):
        return {}
    counts = {}
    for label in os.listdir(new_data_dir):
        label_path = os.path.join(new_data_dir, label)
        if not os.path.isdir(label_path):
            continue
        seqs = [d for d in os.listdir(label_path)
                if os.path.isdir(os.path.join(label_path, d))]
        if seqs:
            counts[label] = len(seqs)
    return counts


def build_retrain_manifest(new_data_dir, existing_manifest_path,
                             min_samples_per_class, output_path, label_map):
    """
    Build a retraining manifest that:
    1. Includes all new samples from new_data_dir
    2. Mixes in existing training data (up to 3× new samples per class)
    """
    # Load existing manifest
    with open(existing_manifest_path) as f:
        existing_manifest = json.load(f)

    existing_by_class = {}
    for item in existing_manifest:
        label = item["label"]
        existing_by_class.setdefault(label, []).append(item)

    new_items = []
    for label in os.listdir(new_data_dir):
        label_path = os.path.join(new_data_dir, label)
        if not os.path.isdir(label_path) or label not in label_map:
            continue
        seqs = [d for d in os.listdir(label_path)
                if os.path.isdir(os.path.join(label_path, d))]
        for seq in seqs:
            new_items.append({
                "path":  os.path.join(label_path, seq),
                "label": label
            })

    if not new_items:
        logger.warning("No new samples found in new_data_dir")
        return 0

    # Group new items by class
    new_by_class = {}
    for item in new_items:
        new_by_class.setdefault(item["label"], []).append(item)

    # Filter classes with enough samples
    qualified = {l: items for l, items in new_by_class.items()
                 if len(items) >= min_samples_per_class}

    if not qualified:
        logger.warning(f"No class has >= {min_samples_per_class} new samples")
        return 0

    retrain_manifest = []
    for label, new_class_items in qualified.items():
        retrain_manifest.extend(new_class_items)
        # Mix in existing samples (3× new count)
        existing = existing_by_class.get(label, [])
        mix_count = min(len(existing), len(new_class_items) * 3)
        if mix_count > 0:
            import random
            mixed = random.sample(existing, mix_count)
            retrain_manifest.extend(mixed)
        logger.info(f"  [{label}] New: {len(new_class_items)} + "
                    f"Existing: {mix_count}")

    with open(output_path, "w") as f:
        json.dump(retrain_manifest, f, indent=2)

    logger.info(f"Retrain manifest: {len(retrain_manifest)} sequences → {output_path}")
    return len(qualified)


def mark_samples_used(new_data_dir, used_dir):
    """Move processed new_data samples to used/ archive."""
    os.makedirs(used_dir, exist_ok=True)
    for label in os.listdir(new_data_dir):
        src = os.path.join(new_data_dir, label)
        dst = os.path.join(used_dir, label)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for seq in os.listdir(src):
                seq_src = os.path.join(src, seq)
                seq_dst = os.path.join(dst, seq)
                if os.path.isdir(seq_src) and not os.path.exists(seq_dst):
                    shutil.move(seq_src, seq_dst)
    logger.info(f"Moved processed samples to: {used_dir}")


def run_retraining(args):
    print("=" * 60)
    print("  CONTINUOUS LEARNING — RETRAIN PIPELINE")
    print("=" * 60)

    configure_gpu()

    new_data_dir = os.path.join(os.path.dirname(cfg.BASE_DIR),
                                 "new_data") if not hasattr(args, 'new_data_dir') \
                   else args.new_data_dir

    # Check new samples
    counts = count_new_samples(new_data_dir)
    if not counts:
        logger.info("No new samples found. Nothing to retrain.")
        return

    logger.info(f"New samples found:")
    for label, count in counts.items():
        logger.info(f"  {label}: {count}")

    # Load label map
    label_map   = cfg.load_label_map()
    num_classes  = len(label_map)

    # Build retrain manifest
    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    retrain_manifest = os.path.join(cfg.DATASET_DIR, f"retrain_{timestamp}.json")
    n_qualified      = build_retrain_manifest(
        new_data_dir,
        cfg.TRAIN_MANIFEST,
        args.min_samples,
        retrain_manifest,
        label_map
    )

    if n_qualified == 0:
        logger.info("Insufficient samples for any class. Skipping retraining.")
        return

    # Load current best model
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        logger.error(f"No base model found: {cfg.BEST_MODEL_PATH}")
        return

    logger.info(f"Loading base model: {cfg.BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(cfg.BEST_MODEL_PATH)

    # Unfreeze top CNN layers for adaptation
    unfreeze_cnn_top_layers(model, n_layers=10)

    # Compile with low LR
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer = optimizer,
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    # Data generators
    train_gen = SequenceDataGenerator(
        retrain_manifest, label_map,
        batch_size = args.batch_size,
        shuffle    = True,
        augment    = True
    )
    val_gen = SequenceDataGenerator(
        cfg.VAL_MANIFEST, label_map,
        batch_size = args.batch_size,
        shuffle    = False,
        augment    = False
    )

    class_weights = get_class_weights(retrain_manifest, label_map)

    # Save path
    retrain_model_path = os.path.join(
        cfg.MODEL_DIR, f"retrained_{timestamp}.h5"
    )

    callbacks = build_callbacks(
        stage          = 3,  # retrain
        model_save_path = retrain_model_path,
        log_dir        = cfg.LOG_DIR
    )

    logger.info(f"Starting retraining: {args.epochs} epochs")
    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = args.epochs,
        callbacks       = callbacks,
        class_weight    = class_weights,
    )

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    logger.info(f"Retraining complete. Best val accuracy: {best_val_acc:.4f}")

    # If improved, promote to best_model.h5
    if best_val_acc > args.min_improvement:
        shutil.copy(retrain_model_path, cfg.BEST_MODEL_PATH)
        logger.info(f"✓ Promoted retrained model to best_model.h5")

        # Archive used samples
        used_dir = os.path.join(os.path.dirname(new_data_dir), "new_data_used")
        mark_samples_used(new_data_dir, used_dir)
    else:
        logger.warning(f"Retrained model did not improve sufficiently "
                       f"({best_val_acc:.4f} < {args.min_improvement}). "
                       f"Not promoting.")

    plot_training_history(history, stage=3, log_dir=cfg.LOG_DIR)

    print("\n" + "=" * 60)
    print(f"  RETRAIN COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Model saved       : {retrain_model_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain sign language model with new feedback data"
    )
    parser.add_argument("--min-samples",    type=int,   default=5,
                        help="Minimum new samples per class to trigger retraining")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=5e-5)
    parser.add_argument("--min-improvement",type=float, default=0.5,
                        help="Minimum val_accuracy to promote model")
    parser.add_argument("--new-data-dir",   type=str,
                        default=os.path.join(cfg.BASE_DIR, "..", "new_data"))
    args = parser.parse_args()

    run_retraining(args)


if __name__ == "__main__":
    main()