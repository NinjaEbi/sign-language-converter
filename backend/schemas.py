"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Prediction request.
    frames: list of 30 base64-encoded JPEG images (one per frame)
    """
    frames:     List[str]  = Field(..., description="List of base64-encoded frames")
    session_id: Optional[str] = Field(None, description="Client session identifier")

    @validator('frames')
    def validate_frames(cls, frames):
        if len(frames) == 0:
            raise ValueError("frames list cannot be empty")
        if len(frames) > 60:
            raise ValueError("Maximum 60 frames per request")
        return frames


class FeedbackRequest(BaseModel):
    """Correction feedback from user."""
    predicted_label: str = Field(..., min_length=1, max_length=100)
    corrected_label: str = Field(..., min_length=1, max_length=100)
    frames:          List[str]  = Field(..., description="Original frame sequence")
    confidence:      Optional[float] = None
    session_id:      Optional[str]   = None


# ─────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Single prediction with confidence."""
    label:      str
    confidence: float
    is_certain: bool   # True if confidence > threshold


class TopKPrediction(BaseModel):
    label:      str
    confidence: float


class PredictResponse(BaseModel):
    """Full prediction response."""
    prediction:      str
    confidence:      float
    is_certain:      bool
    top_k:           List[TopKPrediction]
    inference_time:  float             # seconds
    hand_detected:   bool
    smoothed:        Optional[bool] = None
    session_id:      Optional[str]  = None


class FeedbackResponse(BaseModel):
    success:    bool
    message:    str
    data_path:  Optional[str] = None
    feedback_id: Optional[int] = None


# ─────────────────────────────────────────────
# ANALYTICS SCHEMAS
# ─────────────────────────────────────────────

class PredictionLogSchema(BaseModel):
    id:             int
    prediction:     str
    confidence:     float
    inference_time: float
    session_id:     Optional[str]
    timestamp:      datetime

    class Config:
        from_attributes = True


class FeedbackDataSchema(BaseModel):
    id:              int
    predicted_label: str
    corrected_label: str
    data_path:       str
    timestamp:       datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    db_connected:  bool
    num_classes:   int
    labels:        List[str]
    version:       str = "1.0.0"


class StatsResponse(BaseModel):
    total_predictions: int
    total_feedback:    int
    avg_confidence:    float
    avg_inference_ms:  float
    top_predicted:     Dict[str, int]