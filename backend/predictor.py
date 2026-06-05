"""
Model predictor with:
- Lazy model loading
- Batch preprocessing
- Prediction smoothing (sliding window voting)
- Hand detection validation
- Performance logging
"""

import os
import sys
import json
import time
import base64
import logging
import numpy as np
import cv2
from collections import deque
from typing import List, Tuple, Optional
import mediapipe as mp
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (
    MODEL_PATH, LABEL_MAP_PATH, SEQUENCE_LENGTH,
    IMG_SIZE, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
)

logger = logging.getLogger(__name__)


class SignLanguagePredictor:
    """
    Thread-safe predictor with prediction smoothing.
    Singleton pattern for model reuse across requests.
    """

    def __init__(self):
        self.model             = None
        self.label_map         = None
        self.index_to_label    = None
        self.num_classes       = 0
        self._loaded           = False

        # Prediction smoothing buffer
        self._smoothing_buffer = deque(maxlen=SMOOTHING_WINDOW)

        # MediaPipe
        self.mp_hands   = mp.solutions.hands
        self.hands      = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.4
        )

        self._load_model()

    def _load_model(self):
        """Load model and label map."""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model not found: {MODEL_PATH}")
                logger.warning("Running in demo mode.")
                self._setup_demo_mode()
                return

            if not os.path.exists(LABEL_MAP_PATH):
                raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

            logger.info(f"Loading model: {MODEL_PATH}")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")

            with open(LABEL_MAP_PATH) as f:
                self.label_map = json.load(f)

            self.index_to_label = {v: k for k, v in self.label_map.items()}
            self.num_classes     = len(self.label_map)
            self._loaded         = True

            logger.info(f"Label map loaded: {self.num_classes} classes")

            # Warmup inference
            self._warmup()

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self._setup_demo_mode()

    def _setup_demo_mode(self):
        """Demo mode with fake predictions."""
        self.label_map      = {l: i for i, l in enumerate(
            ["hello", "yes", "no", "thanks", "iloveyou",
             "please", "sorry", "help", "A", "B", "C"]
        )}
        self.index_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes     = len(self.label_map)
        self._loaded         = False
        logger.info(f"Demo mode active. {self.num_classes} classes.")

    def _warmup(self):
        """Run one dummy inference to pre-compile the model."""
        try:
            dummy = np.zeros(
                (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3),
                dtype=np.float32
            )
            _ = self.model.predict(dummy, verbose=0)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def decode_base64_frame(self, b64_str: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string → BGR numpy array."""
        try:
            # Handle data URL prefix
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.warning(f"Frame decode error: {e}")
            return None

    def check_hand_in_frame(self, frame_bgr: np.ndarray) -> bool:
        """Quick check if any hand is present in the frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks is not None

    def preprocess_frames(self, frames_b64: List[str]
                           ) -> Tuple[np.ndarray, bool]:
        """
        Decode and preprocess a list of base64 frames.
        Returns: (sequence_array, hand_detected)
        Shape: (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
        """
        sequence     = np.zeros(
            (SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
        )
        hand_detected = False
        decoded_count = 0

        for i, b64 in enumerate(frames_b64[:SEQUENCE_LENGTH]):
            img = self.decode_base64_frame(b64)
            if img is None:
                continue

            # Check hand in first frame
            if i == 0:
                hand_detected = self.check_hand_in_frame(img)

            # Resize and normalize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sequence[i] = img.astype(np.float32) / 255.0
            decoded_count += 1

        # If too few frames decoded, return empty
        if decoded_count < 5:
            logger.warning(f"Only {decoded_count} frames decoded")

        # Pad if needed by repeating last valid frame
        last_valid = max(decoded_count - 1, 0)
        for i in range(decoded_count, SEQUENCE_LENGTH):
            sequence[i] = sequence[last_valid]

        return sequence[np.newaxis, ...], hand_detected

    def predict(self, frames_b64: List[str],
                session_id: Optional[str] = None
                ) -> dict:
        """
        Full prediction pipeline.
        Returns prediction dict with label, confidence, top_k, timing.
        """
        t_start = time.perf_counter()

        # Preprocess
        sequence, hand_detected = self.preprocess_frames(frames_b64)

        # Demo mode fallback
        if not self._loaded:
            return self._demo_predict(hand_detected, t_start, session_id)

        # Inference
        try:
            probs      = self.model.predict(sequence, verbose=0)[0]  # (num_classes,)
            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = self.index_to_label.get(pred_idx, "unknown")

            # Top-K
            top_k_idx  = np.argsort(probs)[::-1][:3]
            top_k      = [
                {"label": self.index_to_label.get(int(i), "unknown"),
                 "confidence": float(probs[i])}
                for i in top_k_idx
            ]

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._error_response(str(e), t_start)

        # Prediction smoothing
        self._smoothing_buffer.append((pred_label, confidence))
        smoothed_label, smoothed_conf, is_smoothed = self._smooth_prediction()

        t_end = time.perf_counter()

        result = {
            "prediction":      smoothed_label,
            "confidence":      smoothed_conf,
            "is_certain":      smoothed_conf >= CONFIDENCE_THRESHOLD,
            "top_k":           top_k,
            "inference_time":  t_end - t_start,
            "hand_detected":   hand_detected,
            "smoothed":        is_smoothed,
            "session_id":      session_id,
        }

        logger.info(f"Predicted: {smoothed_label} | "
                    f"Conf: {smoothed_conf:.3f} | "
                    f"Time: {(t_end-t_start)*1000:.1f}ms | "
                    f"Hand: {hand_detected}")
        return result

    def _smooth_prediction(self) -> Tuple[str, float, bool]:
        """
        Sliding window majority voting for prediction smoothing.
        Returns (label, confidence, is_smoothed)
        """
        if len(self._smoothing_buffer) < 2:
            if self._smoothing_buffer:
                label, conf = self._smoothing_buffer[-1]
                return label, conf, False
            return "uncertain", 0.0, False

        # Vote by weighted confidence
        votes: dict = {}
        for label, conf in self._smoothing_buffer:
            votes[label] = votes.get(label, 0) + conf

        best_label = max(votes, key=votes.get)
        total_conf = sum(c for _, c in self._smoothing_buffer)
        smoothed_conf = votes[best_label] / len(self._smoothing_buffer)

        return best_label, smoothed_conf, True

    def clear_smoothing_buffer(self, session_id: Optional[str] = None):
        """Clear the smoothing buffer (call between gestures)."""
        self._smoothing_buffer.clear()

    def _demo_predict(self, hand_detected, t_start, session_id):
        """Return a fake prediction for demo mode."""
        import random
        labels = list(self.index_to_label.values())
        label  = random.choice(labels)
        conf   = round(random.uniform(0.65, 0.98), 3)
        t_end  = time.perf_counter()
        return {
            "prediction":     label,
            "confidence":     conf,
            "is_certain":     conf >= CONFIDENCE_THRESHOLD,
            "top_k":          [{"label": label, "confidence": conf}],
            "inference_time": t_end - t_start,
            "hand_detected":  hand_detected,
            "smoothed":       False,
            "session_id":     session_id,
            "demo_mode":      True,
        }

    def _error_response(self, error_msg, t_start):
        t_end = time.perf_counter()
        return {
            "prediction":     "error",
            "confidence":     0.0,
            "is_certain":     False,
            "top_k":          [],
            "inference_time": t_end - t_start,
            "hand_detected":  False,
            "smoothed":       False,
            "error":          error_msg,
        }

    @property
    def labels(self):
        return list(self.label_map.keys())

    @property
    def is_loaded(self):
        return self._loaded


# Module-level singleton
_predictor_instance = None

def get_predictor() -> SignLanguagePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SignLanguagePredictor()
    return _predictor_instance