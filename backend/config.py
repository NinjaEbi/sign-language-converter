"""
Backend configuration — reads from .env file.
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "sign_language_db")

# Encode special characters in password (@, #, %, etc.)
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/saved/best_model.h5"
)

LABEL_MAP_PATH = os.getenv(
    "LABEL_MAP_PATH",
    "dataset/final/label_map.json"
)

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "30"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.65")
)
SMOOTHING_WINDOW = int(
    os.getenv("SMOOTHING_WINDOW", "5")
)

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")

# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────
NEW_DATA_DIR = os.getenv(
    "NEW_DATA_DIR",
    "new_data"
)