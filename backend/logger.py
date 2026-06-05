"""
Logging utilities:
- Async prediction logging to MySQL
- Feedback data saving (frames to disk + DB entry)
- Structured JSON logging
"""

import os
import json
import uuid
import logging
import base64
import numpy as np
import cv2
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models import PredictionLog, FeedbackData
from backend.config import NEW_DATA_DIR

logger = logging.getLogger(__name__)


def log_prediction(db: Session, prediction_result: dict) -> int:
    """
    Write a prediction to the database.
    Returns the new record ID.
    """
    try:
        top_k      = prediction_result.get("top_k", [])
        top_labels = json.dumps([x["label"]      for x in top_k])
        top_scores = json.dumps([x["confidence"] for x in top_k])

        record = PredictionLog(
            prediction     = prediction_result["prediction"],
            confidence     = prediction_result["confidence"],
            top3_labels    = top_labels,
            top3_scores    = top_scores,
            inference_time = prediction_result["inference_time"],
            session_id     = prediction_result.get("session_id"),
            hand_detected  = prediction_result.get("hand_detected", True),
            frame_count    = len(prediction_result.get("frames", [])),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        db.rollback()
        return -1


def save_feedback(db: Session, feedback_data: dict) -> dict:
    """
    Save corrected prediction feedback:
    1. Save frame sequence to disk (new_data/<label>/<uuid>/)
    2. Write DB record
    Returns: {success, data_path, feedback_id}
    """
    try:
        corrected_label = feedback_data["corrected_label"]
        frames_b64      = feedback_data["frames"]
        session_id      = feedback_data.get("session_id")
        sample_id       = str(uuid.uuid4())[:8]

        # ── Save frames to disk ──
        label_dir = os.path.join(NEW_DATA_DIR, corrected_label)
        seq_dir   = os.path.join(label_dir, sample_id)
        os.makedirs(seq_dir, exist_ok=True)

        saved_frames = 0
        for i, b64 in enumerate(frames_b64):
            try:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    frame_path = os.path.join(seq_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, cv2.resize(img, (224, 224)))
                    saved_frames += 1
            except Exception as fe:
                logger.warning(f"Frame {i} save error: {fe}")

        # Save metadata
        meta = {
            "corrected_label":  corrected_label,
            "predicted_label":  feedback_data.get("predicted_label"),
            "session_id":       session_id,
            "timestamp":        datetime.now().isoformat(),
            "frame_count":      saved_frames,
        }
        with open(os.path.join(seq_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # ── DB record ──
        record = FeedbackData(
            predicted_label = feedback_data.get("predicted_label", ""),
            corrected_label = corrected_label,
            confidence      = feedback_data.get("confidence"),
            data_path       = seq_dir,
            session_id      = session_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(f"Feedback saved: {corrected_label} | "
                    f"Frames: {saved_frames} | Path: {seq_dir}")
        return {
            "success":     True,
            "data_path":   seq_dir,
            "feedback_id": record.id,
            "frames_saved": saved_frames,
        }

    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        db.rollback()
        return {"success": False, "error": str(e)}


def get_prediction_stats(db: Session) -> dict:
    """Aggregate statistics from the database."""
    from sqlalchemy import func
    from backend.models import PredictionLog, FeedbackData

    try:
        total_preds = db.query(func.count(PredictionLog.id)).scalar() or 0
        total_fb    = db.query(func.count(FeedbackData.id)).scalar() or 0

        avg_conf    = db.query(
            func.avg(PredictionLog.confidence)
        ).scalar() or 0.0

        avg_inf     = db.query(
            func.avg(PredictionLog.inference_time)
        ).scalar() or 0.0

        # Top 5 predicted labels
        top_preds = (
            db.query(PredictionLog.prediction,
                     func.count(PredictionLog.id).label("count"))
            .group_by(PredictionLog.prediction)
            .order_by(func.count(PredictionLog.id).desc())
            .limit(5)
            .all()
        )
        top_preds_dict = {row.prediction: row.count for row in top_preds}

        return {
            "total_predictions": total_preds,
            "total_feedback":    total_fb,
            "avg_confidence":    round(float(avg_conf), 4),
            "avg_inference_ms":  round(float(avg_inf) * 1000, 2),
            "top_predicted":     top_preds_dict,
        }
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        return {}