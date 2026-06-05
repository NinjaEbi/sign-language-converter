/**
 * API service layer — all backend communication.
 * Handles base64 encoding, error handling, retry logic.
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor — log in dev
api.interceptors.request.use((config) => {
  if (import.meta.env.DEV) {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
  }
  return config
})

// Response interceptor — normalize errors
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail
      || error.message
      || 'Unknown API error'
    console.error('[API Error]', message)
    return Promise.reject(new Error(message))
  }
)


/**
 * Convert a HTMLVideoElement frame to base64 JPEG.
 */
export function captureFrameFromVideo(videoEl, canvasEl, quality = 0.7) {
  const ctx = canvasEl.getContext('2d')
  canvasEl.width  = videoEl.videoWidth  || 640
  canvasEl.height = videoEl.videoHeight || 480
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height)
  return canvasEl.toDataURL('image/jpeg', quality)
}


/**
 * Collect N frames from webcam at the target interval.
 * Returns array of base64 strings.
 */
export async function collectFrames(videoEl, canvasEl, n = 30, intervalMs = 100) {
  return new Promise((resolve, reject) => {
    const frames = []
    let count = 0

    const interval = setInterval(() => {
      try {
        const frame = captureFrameFromVideo(videoEl, canvasEl)
        frames.push(frame)
        count++
        if (count >= n) {
          clearInterval(interval)
          resolve(frames)
        }
      } catch (err) {
        clearInterval(interval)
        reject(err)
      }
    }, intervalMs)
  })
}


/**
 * Send frames to /predict endpoint.
 * @param {string[]} frames  - Array of base64 frames
 * @param {string}   sessionId
 * @returns {Promise<PredictResponse>}
 */
export async function predict(frames, sessionId) {
  return api.post('/predict', { frames, session_id: sessionId })
}


/**
 * Submit a corrected label for continuous learning.
 */
export async function submitFeedback(predictedLabel, correctedLabel, frames, sessionId, confidence) {
  return api.post('/feedback', {
    predicted_label: predictedLabel,
    corrected_label: correctedLabel,
    frames,
    session_id: sessionId,
    confidence,
  })
}


/**
 * Health check.
 */
export async function healthCheck() {
  return api.get('/health')
}


/**
 * Get available sign labels.
 */
export async function getLabels() {
  return api.get('/labels')
}


/**
 * Get prediction statistics.
 */
export async function getStats() {
  return api.get('/stats')
}


/**
 * Clear session smoothing buffer.
 */
export async function clearSession(sessionId) {
  return api.delete(`/session/clear?session_id=${sessionId}`)
}