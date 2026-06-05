"""
Converts a static image dataset (e.g., ASL Kaggle) into
sequence format for CNN+LSTM training.
Each static image → 30 slightly augmented frames simulating temporal variation.
"""

import cv2
import os
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STATIC_DATASET_PATH  = os.path.join("dataset", "raw", "asl_alphabet_train")
OUTPUT_PATH          = os.path.join("dataset", "processed", "from_static")
IMG_SIZE             = 224
SEQUENCE_LENGTH      = 30
SEQUENCES_PER_IMAGE  = 2      # generate 2 sequences per static image

# Subtle augmentation to simulate temporal variation across frames
TEMPORAL_AUG = {
    "rotation_jitter":  3.0,   # ±3 degrees per frame
    "translate_jitter": 0.03,  # ±3% translation per frame
    "brightness_jitter":0.08,  # ±8% brightness per frame
    "noise_sigma":      5.0,
}


def subtle_jitter(img, frame_idx, n_frames, cfg, seed_base):
    """
    Apply subtle, smoothly varying augmentation to simulate motion.
    Uses sinusoidal variation to create natural-looking sequences.
    """
    rng = np.random.RandomState(seed_base + frame_idx)
    t   = frame_idx / n_frames  # normalized time 0→1

    # Smooth sinusoidal rotation
    angle = cfg["rotation_jitter"] * np.sin(2 * np.pi * t + rng.uniform(0, np.pi))
    # Add small random jitter
    angle += rng.uniform(-0.5, 0.5)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Smooth translation
    tx = int(cfg["translate_jitter"] * w * np.sin(3 * np.pi * t))
    ty = int(cfg["translate_jitter"] * h * np.cos(2 * np.pi * t))
    tx += rng.randint(-2, 3)
    ty += rng.randint(-2, 3)
    T  = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, T, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness variation
    br = 1.0 + cfg["brightness_jitter"] * np.sin(np.pi * t) + rng.uniform(-0.02, 0.02)
    img = np.clip(img.astype(np.float32) * br, 0, 255).astype(np.uint8)

    # Noise
    noise = rng.randn(*img.shape).astype(np.float32) * cfg["noise_sigma"] * 0.5
    img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def image_to_sequence(img_path, seq_out_dir, seq_id, label,
                       seq_length, img_size, aug_cfg, aug_seed):
    """Convert one static image into a frame sequence."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    img = cv2.resize(img, (img_size, img_size))
    os.makedirs(seq_out_dir, exist_ok=True)

    for frame_idx in range(seq_length):
        frame = subtle_jitter(img.copy(), frame_idx, seq_length,
                               aug_cfg, aug_seed)
        frame_path = os.path.join(seq_out_dir,
                                  f"frame_{frame_idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)

    meta = {
        "label":         label,
        "sequence_id":   seq_id,
        "source_image":  str(img_path),
        "frame_count":   seq_length,
        "synthetic":     True,
        "img_size":      img_size
    }
    with open(os.path.join(seq_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return True


def convert_dataset(input_dir, output_dir, seq_per_img, seq_len, img_size, cfg):
    print("=" * 60)
    print("  STATIC → SEQUENCE CONVERTER")
    print("=" * 60)
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Seqs/img: {seq_per_img} | Length: {seq_len} | Size: {img_size}")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"[ERROR] Input path not found: {input_dir}")
        print("[INFO] Download ASL dataset from Kaggle and place in:")
        print(f"       {input_dir}")
        return

    labels = sorted([d for d in os.listdir(input_dir)
                     if os.path.isdir(os.path.join(input_dir, d))])
    print(f"[INFO] Found {len(labels)} classes: {labels}")

    total_seqs   = 0
    total_images = 0

    for label in labels:
        label_in  = os.path.join(input_dir,  label)
        label_out = os.path.join(output_dir, label)
        os.makedirs(label_out, exist_ok=True)

        images = [f for f in os.listdir(label_in)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        pbar = tqdm(images, desc=f"  [{label:>3}]", ncols=70)

        for img_file in pbar:
            img_path = os.path.join(label_in, img_file)

            for aug_i in range(seq_per_img):
                existing = [d for d in os.listdir(label_out)
                            if os.path.isdir(os.path.join(label_out, d))]
                seq_id      = len(existing)
                seq_name    = f"{Path(img_file).stem}_seq_{aug_i}"
                seq_out_dir = os.path.join(label_out, seq_name)

                if os.path.exists(seq_out_dir):
                    continue

                seed = hash(img_file + str(aug_i)) % 100000
                success = image_to_sequence(
                    img_path, seq_out_dir, seq_id,
                    label, seq_len, img_size, cfg, seed
                )
                if success:
                    total_seqs += 1

            total_images += 1

    print(f"\n[INFO] Conversion complete.")
    print(f"[INFO] Images processed : {total_images}")
    print(f"[INFO] Sequences created: {total_seqs}")
    print(f"[INFO] Output saved to  : {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    convert_dataset(
        input_dir  = STATIC_DATASET_PATH,
        output_dir = OUTPUT_PATH,
        seq_per_img = SEQUENCES_PER_IMAGE,
        seq_len    = SEQUENCE_LENGTH,
        img_size   = IMG_SIZE,
        cfg        = TEMPORAL_AUG
    )