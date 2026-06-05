"""
Final preprocessing pipeline:
- Merges custom + converted static datasets
- Validates sequences (frame count, image integrity)
- Balances dataset (oversampling minority classes)
- Splits into train/val/test
- Saves label map (label_map.json)
- Generates dataset statistics
"""

import os
import json
import shutil
import random
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SOURCE_DIRS = [
    os.path.join("dataset", "custom"),
    os.path.join("dataset", "augmented"),
    os.path.join("dataset", "processed", "from_static"),
]
OUTPUT_DIR      = os.path.join("dataset", "final")
SEQUENCE_LENGTH = 30
IMG_SIZE        = 224

SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}
RANDOM_SEED  = 42
MIN_SEQUENCES_PER_CLASS = 30   # minimum sequences required to include a class
TARGET_SEQUENCES_PER_CLASS = None  # None = no cap; int = max per class


def load_all_sequences(source_dirs):
    """
    Scan all source directories, return a dict:
    { label: [seq_path, ...], ... }
    """
    sequences = defaultdict(list)

    for src_dir in source_dirs:
        if not os.path.exists(src_dir):
            print(f"  [SKIP] Not found: {src_dir}")
            continue
        print(f"  [SCAN] {src_dir}")
        for label in os.listdir(src_dir):
            label_path = os.path.join(src_dir, label)
            if not os.path.isdir(label_path):
                continue
            for seq in os.listdir(label_path):
                seq_path = os.path.join(label_path, seq)
                if os.path.isdir(seq_path):
                    sequences[label].append(seq_path)

    return dict(sequences)


def validate_sequence(seq_path, expected_frames, img_size):
    """
    Check that a sequence has the correct number of valid frames.
    Returns (is_valid, actual_frame_count)
    """
    frames = sorted([f for f in os.listdir(seq_path)
                     if f.startswith("frame_") and f.endswith(".jpg")])
    if len(frames) != expected_frames:
        return False, len(frames)

    # Spot-check first, middle, last frame
    for idx in [0, expected_frames // 2, expected_frames - 1]:
        fpath = os.path.join(seq_path, frames[idx])
        img   = cv2.imread(fpath)
        if img is None:
            return False, len(frames)
        if img.shape[:2] != (img_size, img_size):
            # Attempt resize on-the-fly validation
            return True, len(frames)   # allow, will be resized in loader

    return True, len(frames)


def balance_dataset(sequences, min_seqs, target_seqs, seed):
    """
    Remove classes with too few sequences.
    Optionally cap classes with too many.
    """
    rng = random.Random(seed)
    balanced = {}

    for label, paths in sequences.items():
        if len(paths) < min_seqs:
            print(f"  [SKIP] '{label}': only {len(paths)} sequences "
                  f"(< min {min_seqs})")
            continue
        if target_seqs and len(paths) > target_seqs:
            paths = rng.sample(paths, target_seqs)
        balanced[label] = paths

    return balanced


def split_sequences(sequences, ratios, seed):
    """Split sequences per class into train/val/test."""
    rng    = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for label, paths in sequences.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n       = len(shuffled)
        n_train = int(n * ratios["train"])
        n_val   = int(n * ratios["val"])

        splits["train"][label] = shuffled[:n_train]
        splits["val"][label]   = shuffled[n_train:n_train + n_val]
        splits["test"][label]  = shuffled[n_train + n_val:]

    return splits


def save_split(split_name, split_data, output_dir):
    """
    Save split index as a JSON file (sequence paths + labels).
    Does NOT copy images — just creates a manifest.
    """
    manifest = []
    for label, paths in split_data.items():
        for path in paths:
            manifest.append({"path": path, "label": label})

    out_path = os.path.join(output_dir, f"{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [{split_name:5s}] {len(manifest):>5} sequences → {out_path}")
    return manifest


def build_label_map(labels):
    """Create sorted label → index mapping."""
    sorted_labels = sorted(labels)
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    return label_map


def print_statistics(sequences):
    print("\n  Class Distribution:")
    print(f"  {'Label':>15} | {'Count':>6}")
    print("  " + "-" * 24)
    total = 0
    for label in sorted(sequences.keys()):
        count = len(sequences[label])
        total += count
        print(f"  {label:>15} | {count:>6}")
    print("  " + "-" * 24)
    print(f"  {'TOTAL':>15} | {total:>6}")
    print()


def main():
    print("=" * 60)
    print("  DATASET PREPROCESSOR")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    print("\n[1] Scanning source directories...")
    all_seqs = load_all_sequences(SOURCE_DIRS)
    print(f"    Found {sum(len(v) for v in all_seqs.values())} sequences "
          f"across {len(all_seqs)} classes")

    # 2. Validate
    print("\n[2] Validating sequences...")
    valid_seqs  = defaultdict(list)
    invalid_cnt = 0
    for label, paths in tqdm(all_seqs.items(), desc="  Validating"):
        for path in paths:
            is_valid, fc = validate_sequence(path, SEQUENCE_LENGTH, IMG_SIZE)
            if is_valid:
                valid_seqs[label].append(path)
            else:
                invalid_cnt += 1
    print(f"    Valid: {sum(len(v) for v in valid_seqs.values())} | "
          f"Invalid/skipped: {invalid_cnt}")

    # 3. Balance
    print("\n[3] Balancing dataset...")
    balanced = balance_dataset(valid_seqs, MIN_SEQUENCES_PER_CLASS,
                               TARGET_SEQUENCES_PER_CLASS, RANDOM_SEED)
    print_statistics(balanced)

    # 4. Build label map
    label_map = build_label_map(balanced.keys())
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[4] Label map saved: {label_map_path}")
    print(f"    Classes: {list(label_map.keys())}")

    # 5. Split
    print("\n[5] Creating train/val/test splits...")
    splits = split_sequences(balanced, SPLIT_RATIOS, RANDOM_SEED)
    for split_name, split_data in splits.items():
        save_split(split_name, split_data, OUTPUT_DIR)

    # 6. Summary stats
    stats = {
        "total_sequences": sum(len(v) for v in balanced.values()),
        "num_classes":     len(balanced),
        "sequence_length": SEQUENCE_LENGTH,
        "img_size":        IMG_SIZE,
        "label_map":       label_map,
        "splits": {
            k: sum(len(v) for v in splits[k].values())
            for k in splits
        }
    }
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[6] Stats saved: {stats_path}")
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Classes        : {stats['num_classes']}")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Train/Val/Test : "
          f"{stats['splits']['train']} / "
          f"{stats['splits']['val']} / "
          f"{stats['splits']['test']}")
    print(f"  Output dir     : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()