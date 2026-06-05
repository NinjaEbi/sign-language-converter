"""
TensorFlow data pipeline for sequence-based sign language dataset.
Reads manifests (train.json / val.json / test.json),
loads frame sequences, normalizes, and builds tf.data pipelines
with prefetching and parallel loading for GPU-optimized training.
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import (
    IMG_HEIGHT, IMG_WIDTH, CHANNELS, SEQUENCE_LENGTH,
    PRETRAIN_BATCH_SIZE, NUM_PARALLEL_CALLS, PREFETCH_BUFFER
)


# ─────────────────────────────────────────────
# NUMPY-BASED LOADER (for small datasets)
# ─────────────────────────────────────────────

def load_sequence_numpy(seq_path, seq_len=SEQUENCE_LENGTH,
                         h=IMG_HEIGHT, w=IMG_WIDTH):
    """
    Load all frames in a sequence directory.
    Returns: np.array of shape (seq_len, H, W, 3), normalized to [0,1]
    """
    frames = sorted([f for f in os.listdir(seq_path)
                     if f.startswith("frame_") and f.endswith(".jpg")])

    sequence = np.zeros((seq_len, h, w, CHANNELS), dtype=np.float32)

    for i, fname in enumerate(frames[:seq_len]):
        img_path = os.path.join(seq_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sequence[i] = img.astype(np.float32) / 255.0

    # If fewer frames than expected, repeat last frame
    if len(frames) < seq_len:
        last_valid = min(len(frames) - 1, seq_len - 1)
        for i in range(len(frames), seq_len):
            sequence[i] = sequence[last_valid]

    return sequence


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


def build_numpy_dataset(manifest_path, label_map, batch_size,
                         shuffle=True, augment=False):
    """
    Load the entire dataset into memory as numpy arrays.
    Suitable for small datasets (<10k sequences).
    """
    manifest  = load_manifest(manifest_path)
    num_classes = len(label_map)

    X_list, y_list = [], []

    for item in manifest:
        seq_path = item["path"]
        label    = item["label"]

        if label not in label_map:
            continue
        if not os.path.exists(seq_path):
            continue

        seq = load_sequence_numpy(seq_path)
        X_list.append(seq)
        y_list.append(label_map[label])

    X = np.array(X_list, dtype=np.float32)   # (N, seq_len, H, W, C)
    y = to_categorical(y_list, num_classes=num_classes)

    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
    dataset = dataset.batch(batch_size).prefetch(PREFETCH_BUFFER)
    return dataset, len(X)


# ─────────────────────────────────────────────
# TF.DATA PIPELINE (for large datasets)
# ─────────────────────────────────────────────

class SequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for large datasets that don't fit in memory.
    Loads batches on-demand with optional augmentation.
    """
    def __init__(self, manifest_path, label_map, batch_size,
                  shuffle=True, augment=False,
                  seq_len=SEQUENCE_LENGTH,
                  img_h=IMG_HEIGHT, img_w=IMG_WIDTH):

        self.manifest   = load_manifest(manifest_path)
        self.label_map  = label_map
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment
        self.seq_len    = seq_len
        self.img_h      = img_h
        self.img_w      = img_w
        self.num_classes = len(label_map)

        # Filter valid entries
        self.items = [item for item in self.manifest
                      if item["label"] in label_map
                      and os.path.exists(item["path"])]

        self.indices = np.arange(len(self.items))
        if shuffle:
            np.random.shuffle(self.indices)

        print(f"  DataGenerator: {len(self.items)} sequences | "
              f"batch={batch_size} | augment={augment}")

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:
                                  (idx + 1) * self.batch_size]
        X = np.zeros((len(batch_idx), self.seq_len,
                       self.img_h, self.img_w, CHANNELS), dtype=np.float32)
        y = np.zeros((len(batch_idx), self.num_classes), dtype=np.float32)

        for i, data_idx in enumerate(batch_idx):
            item     = self.items[data_idx]
            seq_path = item["path"]
            label    = item["label"]

            seq = load_sequence_numpy(seq_path, self.seq_len,
                                       self.img_h, self.img_w)

            if self.augment:
                seq = self._augment_sequence(seq)

            X[i] = seq
            y[i] = to_categorical(self.label_map[label], self.num_classes)

        return X, y

    def _augment_sequence(self, seq):
        """Apply consistent augmentation to entire sequence."""
        seed = np.random.randint(0, 10000)
        rng  = np.random.RandomState(seed)

        # Random horizontal flip
        do_flip = rng.random() > 0.5
        # Random brightness
        brightness = rng.uniform(0.8, 1.2)
        # Random rotation
        angle = rng.uniform(-10, 10)
        # Consistent across all frames:
        augmented = np.zeros_like(seq)
        h, w = seq.shape[1], seq.shape[2]

        for i, frame in enumerate(seq):
            img = (frame * 255).astype(np.uint8)

            # Flip
            if do_flip:
                img = cv2.flip(img, 1)

            # Brightness
            img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Rotation
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            augmented[i] = img.astype(np.float32) / 255.0

        return augmented

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_tf_dataset_from_generator(manifest_path, label_map, batch_size,
                                     shuffle=True, augment=False):
    """
    Build tf.data.Dataset using from_generator for memory-efficient loading.
    """
    manifest    = load_manifest(manifest_path)
    num_classes = len(label_map)

    valid_items = [(item["path"], item["label"])
                   for item in manifest
                   if item["label"] in label_map
                   and os.path.exists(item["path"])]

    if shuffle:
        import random
        random.shuffle(valid_items)

    paths  = [x[0] for x in valid_items]
    labels = [label_map[x[1]] for x in valid_items]

    def generator():
        for path, label_idx in zip(paths, labels):
            seq = load_sequence_numpy(path)
            lbl = to_categorical(label_idx, num_classes).astype(np.float32)
            yield seq, lbl

    output_sig = (
        tf.TensorSpec(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS),
                      dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(500, len(valid_items)))

    dataset = (dataset
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return dataset, len(valid_items)


def get_class_weights(manifest_path, label_map):
    """
    Compute class weights to handle class imbalance.
    Returns dict: {class_idx: weight}
    """
    manifest = load_manifest(manifest_path)
    counts   = defaultdict(int)

    for item in manifest:
        if item["label"] in label_map:
            counts[label_map[item["label"]]] += 1

    total = sum(counts.values())
    n_cls = len(label_map)
    weights = {}
    for idx in range(n_cls):
        if counts[idx] > 0:
            weights[idx] = total / (n_cls * counts[idx])
        else:
            weights[idx] = 1.0

    return weights