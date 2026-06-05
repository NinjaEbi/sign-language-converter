"""
SQLAlchemy ORM models:
- PredictionLog: every prediction made by the system
- FeedbackData: user corrections for continuous learning
"""

from sqlalchemy import (
    Column, Integer, String, Float,
    DateTime, Text, Boolean
)
from sqlalchemy.sql import func
from backend.database import Base


class PredictionLog(Base):
    """
    Stores every prediction made.
    Used for analytics, debugging, and performance monitoring.
    """
    __tablename__ = "prediction_logs"

    id             = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction     = Column(String(100), nullable=False, index=True)
    confidence     = Column(Float,       nullable=False)
    top3_labels    = Column(Text,        nullable=True)    # JSON string
    top3_scores    = Column(Text,        nullable=True)    # JSON string
    inference_time = Column(Float,       nullable=False)   # seconds
    session_id     = Column(String(50),  nullable=True,  index=True)
    frame_count    = Column(Integer,     nullable=True)
    hand_detected  = Column(Boolean,     nullable=True)
    timestamp      = Column(DateTime,    server_default=func.now(), index=True)

    def __repr__(self):
        return (f"<PredictionLog id={self.id} "
                f"pred='{self.prediction}' "
                f"conf={self.confidence:.3f}>")


class FeedbackData(Base):
    """
    Stores user-corrected predictions for continuous learning.
    New samples are saved to disk and logged here.
    """
    __tablename__ = "feedback_data"

    id              = Column(Integer, primary_key=True, index=True, autoincrement=True)
    predicted_label = Column(String(100), nullable=False, index=True)
    corrected_label = Column(String(100), nullable=False, index=True)
    confidence      = Column(Float,       nullable=True)
    data_path       = Column(String(500), nullable=False)   # path to saved frames
    session_id      = Column(String(50),  nullable=True)
    verified        = Column(Boolean,     default=False)    # manually verified
    used_in_training = Column(Boolean,    default=False)    # used in retrain
    timestamp       = Column(DateTime,    server_default=func.now(), index=True)

    def __repr__(self):
        return (f"<FeedbackData id={self.id} "
                f"pred='{self.predicted_label}' → "
                f"correct='{self.corrected_label}'>")