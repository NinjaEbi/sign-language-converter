"""
Data augmentation pipeline for sign language sequences.
Applies rotation, zoom, brightness, flip, and noise augmentation
to artificially expand the dataset.
Works on both custom and converted static->sequence datasets.
"""

import cv2
import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_DIR  = os.path.join("dataset", "custom")
OUTPUT_DIR = os.path.join("dataset", "augmented")
IMG_SIZE   = 224
AUGMENTATIONS_PER_SEQUENCE = 4   # how many augmented copies per original

# Augmentation parameters
AUG_CONFIG = {
    "rotation_range":      15,      # ±degrees
    "zoom_range":          0.15,    # ±fraction
    "brightness_range":    (0.6, 1.4),
    "horizontal_flip":     True,
    "noise_sigma":         10,      # Gaussian noise std
    "contrast_range":      (0.8, 1.2),
    "translate_range":     0.1,     # ±fraction of image size
}


# ─────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────

def random_rotation(img, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def random_zoom(img, zoom_range):
    zoom  = 1.0 + random.uniform(-zoom_range, zoom_range)
    h, w  = img.shape[:2]
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    resized = cv2.resize(img, (new_w, new_h))
    if zoom > 1.0:
        # Crop center
        y1 = (new_h - h) // 2
        x1 = (new_w - w) // 2
        result = resized[y1:y1+h, x1:x1+w]
    else:
        # Pad
        result = np.zeros_like(img)
        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        result[y1:y1+new_h, x1:x1+new_w] = resized
    return result


def random_brightness(img, brightness_range):
    factor = random.uniform(*brightness_range)
    img_float = img.astype(np.float32) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)


def random_contrast(img, contrast_range):
    factor = random.uniform(*contrast_range)
    mean   = np.mean(img)
    img_float = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(img_float, 0, 255).astype(np.uint8)


def random_noise(img, sigma):
    noise  = np.random.randn(*img.shape).astype(np.float32) * sigma
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def random_translate(img, translate_range):
    h, w   = img.shape[:2]
    tx     = int(random.uniform(-translate_range, translate_range) * w)
    ty     = int(random.uniform(-translate_range, translate_range) * h)
    M      = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def horizontal_flip(img):
    return cv2.flip(img, 1)


def augment_frame(img, cfg, aug_seed):
    """Apply a consistent random augmentation to a single frame."""
    rng = random.Random(aug_seed)
    np.random.seed(aug_seed)

    # Always apply a subset of augmentations
    ops = []
    if rng.random() > 0.3:
        ops.append(('rotation',   cfg["rotation_range"]))
    if rng.random() > 0.3:
        ops.append(('zoom',       cfg["zoom_range"]))
    if rng.random() > 0.3:
        ops.append(('brightness', cfg["brightness_range"]))
    if rng.random() > 0.3:
        ops.append(('contrast',   cfg["contrast_range"]))
    if rng.random() > 0.4:
        ops.append(('noise',      cfg["noise_sigma"]))
    if rng.random() > 0.4:
        ops.append(('translate',  cfg["translate_range"]))
    if cfg["horizontal_flip"] and rng.random() > 0.5:
        ops.append(('flip', None))

    for op, param in ops:
        if op == 'rotation':
            img = random_rotation(img, param)
        elif op == 'zoom':
            img = random_zoom(img, param)
        elif op == 'brightness':
            img = random_brightness(img, param)
        elif op == 'contrast':
            img = random_contrast(img, param)
        elif op == 'noise':
            img = random_noise(img, param)
        elif op == 'translate':
            img = random_translate(img, param)
        elif op == 'flip':
            img = horizontal_flip(img)

    return img


def augment_sequence(seq_path, out_path, aug_idx, cfg):
    """
    Load all frames in a sequence, apply the SAME augmentation
    (with consistent seed) to each frame to preserve temporal coherence.
    """
    frames = sorted([f for f in os.listdir(seq_path)
                     if f.endswith('.jpg') and f.startswith('frame_')])
    if not frames:
        return False

    aug_seed = aug_idx * 1000 + random.randint(0, 999)
    os.makedirs(out_path, exist_ok=True)

    for frame_file in frames:
        img = cv2.imread(os.path.join(seq_path, frame_file))
        if img is None:
            continue
        aug_img = augment_frame(img, cfg, aug_seed)
        cv2.imwrite(os.path.join(out_path, frame_file), aug_img)

    # Copy metadata
    meta_src = os.path.join(seq_path, "meta.json")
    if os.path.exists(meta_src):
        with open(meta_src) as f:
            meta = json.load(f)
        meta["augmented"]   = True
        meta["aug_idx"]     = aug_idx
        meta["original_path"] = seq_path
        with open(os.path.join(out_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return True


def run_augmentation(input_dir, output_dir, n_augmentations, cfg):
    print("=" * 60)
    print("  SIGN LANGUAGE DATA AUGMENTATION")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    labels = [d for d in os.listdir(input_dir)
              if os.path.isdir(os.path.join(input_dir, d))]
    print(f"[INFO] Found {len(labels)} classes in {input_dir}")

    total_orig = 0
    total_aug  = 0

    for label in labels:
        label_in  = os.path.join(input_dir,  label)
        label_out = os.path.join(output_dir, label)
        os.makedirs(label_out, exist_ok=True)

        sequences = [d for d in os.listdir(label_in)
                     if os.path.isdir(os.path.join(label_in, d))]

        pbar = tqdm(sequences, desc=f"  [{label}]", ncols=70)
        class_aug = 0

        for seq in pbar:
            seq_in  = os.path.join(label_in,  seq)
            seq_out = os.path.join(label_out, seq)

            # Copy original sequence
            if not os.path.exists(seq_out):
                import shutil
                shutil.copytree(seq_in, seq_out)

            # Generate augmented versions
            existing_seqs = [d for d in os.listdir(label_out)
                             if os.path.isdir(os.path.join(label_out, d))]
            base_id = len(existing_seqs)

            for aug_i in range(n_augmentations):
                aug_seq_id  = f"{seq}_aug_{aug_i}"
                aug_out_path = os.path.join(label_out, aug_seq_id)
                if not os.path.exists(aug_out_path):
                    success = augment_sequence(
                        seq_in, aug_out_path, base_id + aug_i, cfg
                    )
                    if success:
                        class_aug += 1

            total_orig += 1
            total_aug  += n_augmentations

    print(f"\n[INFO] Augmentation complete.")
    print(f"[INFO] Original sequences : {total_orig}")
    print(f"[INFO] Augmented sequences: {total_aug}")
    print(f"[INFO] Output directory   : {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    run_augmentation(
        input_dir        = INPUT_DIR,
        output_dir       = OUTPUT_DIR,
        n_augmentations  = AUGMENTATIONS_PER_SEQUENCE,
        cfg              = AUG_CONFIG
    )