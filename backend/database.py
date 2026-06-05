"""
SQLAlchemy database setup and session management.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging

from backend.config import DATABASE_URL, DB_NAME

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    poolclass          = QueuePool,
    pool_size          = 10,
    max_overflow       = 20,
    pool_pre_ping      = True,    # reconnect on stale connections
    pool_recycle       = 3600,    # recycle connections every hour
    echo               = False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified.")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def check_database_connection():
    """Verify database connectivity on startup."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connected: {DB_NAME}")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False