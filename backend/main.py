"""
FastAPI application entry point.
Provides:
  POST /predict          — Predict sign from frame sequence
  POST /feedback         — Submit corrected prediction
  GET  /health           — System health check
  GET  /labels           — Available sign labels
  GET  /stats            — Prediction statistics
  DELETE /session/clear  — Clear smoothing buffer for session
"""

import os
import sys
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Add parent to path for shared config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config      import CORS_ORIGINS, API_HOST, API_PORT, NEW_DATA_DIR
from backend.database    import get_db, create_tables, check_database_connection
from backend.predictor   import get_predictor
from backend.schemas     import (
    PredictRequest, PredictResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, StatsResponse
)
from backend.logger      import log_prediction, save_feedback, get_prediction_stats

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("backend/logs/app.log", mode='a')
    ]
)
logger = logging.getLogger("sign_lang_api")
os.makedirs("backend/logs", exist_ok=True)
os.makedirs(NEW_DATA_DIR,   exist_ok=True)


# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Sign Language API...")

    # DB
    db_ok = check_database_connection()
    if db_ok:
        create_tables()
    else:
        logger.warning("DB unavailable — prediction logging disabled")

    # Model
    predictor = get_predictor()
    logger.info(f"Predictor ready | Classes: {predictor.num_classes} | "
                f"Model loaded: {predictor.is_loaded}")

    yield

    # Shutdown
    logger.info("Shutting down Sign Language API")


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Sign Language Recognition API",
    description = "Real-time CNN+LSTM sign language to text converter",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    request:          PredictRequest,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db)
):
    """
    Predict sign language from a sequence of frames.
    Accepts 30 base64-encoded JPEG frames.
    Returns prediction, confidence, top-K alternatives.
    """
    predictor = get_predictor()

    # Run prediction
    result = predictor.predict(
        frames_b64 = request.frames,
        session_id = request.session_id
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Log asynchronously (don't block response)
    def log_task():
        try:
            result_with_meta = {**result, "frames": request.frames}
            log_prediction(db, result_with_meta)
        except Exception as e:
            logger.error(f"Background log failed: {e}")

    background_tasks.add_task(log_task)

    return PredictResponse(
        prediction     = result["prediction"],
        confidence     = result["confidence"],
        is_certain     = result["is_certain"],
        top_k          = result["top_k"],
        inference_time = result["inference_time"],
        hand_detected  = result["hand_detected"],
        smoothed       = result.get("smoothed"),
        session_id     = request.session_id,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    db:      Session = Depends(get_db)
):
    """
    Submit a corrected prediction for continuous learning.
    Saves frames to disk and logs to database.
    """
    predictor = get_predictor()
    labels    = predictor.labels

    if request.corrected_label not in labels:
        # Allow new labels for extensibility
        logger.info(f"New label submitted: {request.corrected_label}")

    result = save_feedback(db, {
        "predicted_label": request.predicted_label,
        "corrected_label": request.corrected_label,
        "frames":          request.frames,
        "confidence":      request.confidence,
        "session_id":      request.session_id,
    })

    if not result["success"]:
        raise HTTPException(status_code=500,
                            detail=result.get("error", "Feedback save failed"))

    return FeedbackResponse(
        success     = True,
        message     = f"Feedback saved for label '{request.corrected_label}'",
        data_path   = result.get("data_path"),
        feedback_id = result.get("feedback_id"),
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — model status, DB connectivity, available labels."""
    predictor   = get_predictor()
    db_ok       = check_database_connection()

    return HealthResponse(
        status       = "ok" if predictor.is_loaded else "demo_mode",
        model_loaded = predictor.is_loaded,
        db_connected = db_ok,
        num_classes  = predictor.num_classes,
        labels       = predictor.labels,
        version      = "1.0.0",
    )


@app.get("/labels", tags=["System"])
async def get_labels():
    """Return all available sign language labels."""
    predictor = get_predictor()
    return {
        "labels":      predictor.labels,
        "num_classes": predictor.num_classes,
    }


@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats(db: Session = Depends(get_db)):
    """Prediction analytics: counts, averages, top predictions."""
    stats = get_prediction_stats(db)
    return StatsResponse(**stats)


@app.delete("/session/clear", tags=["Session"])
async def clear_session(session_id: Optional[str] = Query(None)):
    """
    Clear prediction smoothing buffer for a session.
    Call this between gesture captures.
    """
    predictor = get_predictor()
    predictor.clear_smoothing_buffer(session_id)
    return {"success": True, "message": "Smoothing buffer cleared"}


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Sign Language Recognition API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }


# ─────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = False,
        workers = 1,
        log_level = "info",
    )