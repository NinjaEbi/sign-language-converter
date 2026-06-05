"""
Webcam-based dataset collection script.
Captures sequences of 30 frames per sample using MediaPipe for ROI extraction.
Saves structured folders: dataset/custom/<class_label>/<sequence_id>/frame_xxx.jpg
"""

import cv2
import os
import time
import json
import mediapipe as mp
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH    = os.path.join("dataset", "custom")
SEQUENCE_LENGTH = 30       # frames per sample
IMG_SIZE        = 224      # resize to 224x224
SEQUENCES_PER_CLASS = 50   # how many sequences to collect per class
DELAY_BETWEEN_SEQUENCES = 2  # seconds between recordings

# Adjust this list to your target classes
CLASS_LABELS = [
    "hello", "yes", "no", "thanks", "iloveyou",
    "please", "sorry", "help", "good", "bad",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_hand_roi(frame, results, padding=40):
    """
    Extracts a tight bounding box (ROI) around detected hand(s).
    Falls back to full frame if no hand detected.
    Returns: cropped ROI (BGR), bool hand_detected
    """
    h, w, _ = frame.shape
    if not results.multi_hand_landmarks:
        return None, False

    all_x, all_y = [], []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(int(lm.x * w))
            all_y.append(int(lm.y * h))

    x_min = max(0,     min(all_x) - padding)
    y_min = max(0,     min(all_y) - padding)
    x_max = min(w - 1, max(all_x) + padding)
    y_max = min(h - 1, max(all_y) + padding)

    roi = frame[y_min:y_max, x_min:x_max]
    return roi, True


def preprocess_frame(frame):
    """Resize and normalize a single frame for storage."""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return resized


def ensure_dirs(label):
    label_path = os.path.join(DATASET_PATH, label)
    os.makedirs(label_path, exist_ok=True)
    return label_path


def get_next_sequence_id(label_path):
    existing = [d for d in os.listdir(label_path)
                if os.path.isdir(os.path.join(label_path, d))]
    return len(existing)


def save_sequence_metadata(seq_path, label, seq_id, frame_count, timestamp):
    meta = {
        "label":       label,
        "sequence_id": seq_id,
        "frame_count": frame_count,
        "timestamp":   timestamp,
        "img_size":    IMG_SIZE
    }
    with open(os.path.join(seq_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def draw_ui(frame, label, seq_id, frame_idx, total_sequences, status, fps=0):
    """Overlay collection stats on the live feed."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Class: {label}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Sequence: {seq_id}/{total_sequences}  Frame: {frame_idx}/{SEQUENCE_LENGTH}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Status: {status}  FPS: {fps:.1f}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if status == "RECORDING" else (0, 165, 255), 2)
    return frame


def collect_class(label, cap):
    label_path = ensure_dirs(label)
    start_seq  = get_next_sequence_id(label_path)

    print(f"\n[INFO] Collecting class: '{label}' | "
          f"Starting at sequence {start_seq}")
    print("[INFO] Press SPACE to start, Q to quit, S to skip class")

    seq_id = start_seq

    while seq_id < start_seq + SEQUENCES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = draw_ui(frame.copy(), label, seq_id - start_seq + 1,
                          0, SEQUENCES_PER_CLASS, "READY")
        cv2.imshow("Sign Language Data Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return False        # quit entire collection
        if key == ord('s'):
            return True         # skip to next class
        if key != ord(' '):
            continue

        # ── START RECORDING ──
        seq_path = os.path.join(label_path, str(seq_id))
        os.makedirs(seq_path, exist_ok=True)

        frames_saved = 0
        frame_buffer = []
        start_time   = time.time()

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, str(countdown),
                        (frame.shape[1]//2 - 30, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
            cv2.imshow("Sign Language Data Collector", frame)
            cv2.waitKey(700)

        # Record frames
        while frames_saved < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            elapsed  = time.time() - start_time
            fps_val  = frames_saved / elapsed if elapsed > 0 else 0

            roi, hand_found = get_hand_roi(frame, results)

            if hand_found and roi.size > 0:
                processed = preprocess_frame(roi)
            else:
                # Use full frame if no hand detected — still save for robustness
                processed = preprocess_frame(frame)

            frame_path = os.path.join(seq_path, f"frame_{frames_saved:03d}.jpg")
            cv2.imwrite(frame_path, processed)
            frame_buffer.append(frame_path)
            frames_saved += 1

            display = draw_ui(frame.copy(), label, seq_id - start_seq + 1,
                              frames_saved, SEQUENCES_PER_CLASS,
                              "RECORDING", fps_val)

            # Draw hand landmarks on display (not on saved image)
            if hand_found:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display, hand_lm,
                                              mp_hands.HAND_CONNECTIONS)

            # Progress bar
            progress = int((frames_saved / SEQUENCE_LENGTH) * 400)
            cv2.rectangle(display,
                          (10, display.shape[0] - 20),
                          (10 + progress, display.shape[0] - 5),
                          (0, 255, 0), -1)
            cv2.imshow("Sign Language Data Collector", display)
            cv2.waitKey(1)

        save_sequence_metadata(seq_path, label, seq_id,
                               frames_saved, datetime.now().isoformat())
        print(f"  ✓ Saved sequence {seq_id} ({frames_saved} frames)")
        seq_id += 1

        # Pause between sequences
        time.sleep(DELAY_BETWEEN_SEQUENCES)

    return True


def main():
    print("=" * 60)
    print("  SIGN LANGUAGE DATASET COLLECTOR")
    print("=" * 60)
    print(f"  Dataset path  : {DATASET_PATH}")
    print(f"  Sequence len  : {SEQUENCE_LENGTH} frames")
    print(f"  Image size    : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Per class     : {SEQUENCES_PER_CLASS} sequences")
    print(f"  Classes       : {len(CLASS_LABELS)}")
    print("=" * 60)
    print("\nControls: SPACE = start recording | Q = quit | S = skip class\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        return

    try:
        for label in CLASS_LABELS:
            should_continue = collect_class(label, cap)
            if not should_continue:
                print("[INFO] Collection stopped by user.")
                break
            print(f"[INFO] Completed class: {label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n[INFO] Dataset collection complete.")
        print(f"[INFO] Saved to: {os.path.abspath(DATASET_PATH)}")


if __name__ == "__main__":
    main()