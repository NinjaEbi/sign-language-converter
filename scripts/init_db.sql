-- ─────────────────────────────────────────────────────
-- Sign Language Recognition System — MySQL Schema
-- Run: mysql -u root -p < scripts/init_db.sql
-- ─────────────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS sign_language_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE sign_language_db;

-- ─────────────────────────────────────────────────────
-- TABLE: prediction_logs
-- Every prediction made by the system
-- ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              INT          UNSIGNED NOT NULL AUTO_INCREMENT,
    prediction      VARCHAR(100) NOT NULL,
    confidence      FLOAT        NOT NULL,
    top3_labels     TEXT,
    top3_scores     TEXT,
    inference_time  FLOAT        NOT NULL COMMENT 'seconds',
    session_id      VARCHAR(50),
    frame_count     INT,
    hand_detected   TINYINT(1),
    timestamp       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_prediction (prediction),
    INDEX idx_session    (session_id),
    INDEX idx_timestamp  (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ─────────────────────────────────────────────────────
-- TABLE: feedback_data
-- User-corrected predictions for continuous learning
-- ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback_data (
    id               INT          UNSIGNED NOT NULL AUTO_INCREMENT,
    predicted_label  VARCHAR(100) NOT NULL,
    corrected_label  VARCHAR(100) NOT NULL,
    confidence       FLOAT,
    data_path        VARCHAR(500) NOT NULL COMMENT 'path to saved frame sequence',
    session_id       VARCHAR(50),
    verified         TINYINT(1)   NOT NULL DEFAULT 0,
    used_in_training TINYINT(1)   NOT NULL DEFAULT 0,
    timestamp        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_corrected   (corrected_label),
    INDEX idx_predicted   (predicted_label),
    INDEX idx_session_fb  (session_id),
    INDEX idx_timestamp_fb (timestamp),
    INDEX idx_verified    (verified),
    INDEX idx_used        (used_in_training)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ─────────────────────────────────────────────────────
-- VIEWS (for analytics dashboard)
-- ─────────────────────────────────────────────────────

-- Prediction summary by label
CREATE OR REPLACE VIEW v_prediction_summary AS
SELECT
    prediction,
    COUNT(*)                        AS total_count,
    AVG(confidence)                 AS avg_confidence,
    AVG(inference_time) * 1000      AS avg_inference_ms,
    MIN(timestamp)                  AS first_seen,
    MAX(timestamp)                  AS last_seen
FROM prediction_logs
GROUP BY prediction
ORDER BY total_count DESC;

-- Feedback correction analysis
CREATE OR REPLACE VIEW v_feedback_analysis AS
SELECT
    predicted_label,
    corrected_label,
    COUNT(*) AS count,
    AVG(confidence) AS avg_confidence,
    MIN(timestamp) AS first_reported
FROM feedback_data
GROUP BY predicted_label, corrected_label
ORDER BY count DESC;

-- Unverified feedback pending review
CREATE OR REPLACE VIEW v_pending_feedback AS
SELECT *
FROM feedback_data
WHERE verified = 0 AND used_in_training = 0
ORDER BY timestamp DESC;

-- Daily prediction volume
CREATE OR REPLACE VIEW v_daily_volume AS
SELECT
    DATE(timestamp)             AS date,
    COUNT(*)                    AS total_predictions,
    COUNT(DISTINCT session_id)  AS unique_sessions,
    AVG(confidence)             AS avg_confidence,
    AVG(inference_time) * 1000  AS avg_inference_ms
FROM prediction_logs
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- ─────────────────────────────────────────────────────
-- STORED PROCEDURE: Mark feedback as used after retraining
-- ─────────────────────────────────────────────────────
DELIMITER $$
CREATE PROCEDURE IF NOT EXISTS mark_feedback_used(IN label_name VARCHAR(100))
BEGIN
    UPDATE feedback_data
    SET used_in_training = 1
    WHERE corrected_label = label_name
      AND used_in_training = 0;
    SELECT ROW_COUNT() AS rows_updated;
END$$
DELIMITER ;

SHOW TABLES;
SELECT 'Database initialized successfully.' AS status;