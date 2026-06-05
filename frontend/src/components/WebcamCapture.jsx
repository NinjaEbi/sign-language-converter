/**
 * Webcam live feed component with:
 * - Mirror display
 * - Hand detection indicator
 * - Recording progress overlay
 * - Frame capture controls
 */

import React, { useEffect, useRef } from 'react'
import { FiCamera, FiCameraOff, FiRefreshCw } from 'react-icons/fi'
import { RecordingState } from '../hooks/usePrediction'

export default function WebcamCapture({
  videoRef,
  canvasRef,
  isReady,
  isLoading,
  error,
  cameraLabel,
  recordingState,
  progress,
  onStartCamera,
  onCapture,
}) {
  const FRAME_COUNT = 30
  const DURATION_S  = 3   // seconds to capture

  const getStatusColor = () => {
    switch (recordingState) {
      case RecordingState.RECORDING:  return 'border-red-500'
      case RecordingState.PROCESSING: return 'border-yellow-400'
      case RecordingState.DONE:       return 'border-secondary'
      case RecordingState.ERROR:      return 'border-red-700'
      default:                        return 'border-surface-border'
    }
  }

  const getStatusLabel = () => {
    switch (recordingState) {
      case RecordingState.COUNTDOWN:  return 'Get Ready...'
      case RecordingState.RECORDING:  return 'Capturing...'
      case RecordingState.PROCESSING: return 'Analyzing...'
      case RecordingState.DONE:       return 'Done!'
      case RecordingState.ERROR:      return 'Error'
      default:                        return isReady ? 'Ready' : 'Camera Off'
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Webcam frame */}
      <div className={`relative rounded-2xl overflow-hidden border-2 transition-all duration-300 ${getStatusColor()} glass-card`}>
        {/* Video element */}
        <video
          ref={videoRef}
          className="webcam-mirror w-full aspect-video object-cover bg-black"
          autoPlay
          muted
          playsInline
        />

        {/* Hidden capture canvas */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-text-muted text-sm">Starting camera...</span>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {error && !isLoading && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center p-6">
            <div className="text-center">
              <FiCameraOff className="mx-auto text-red-400 text-4xl mb-3" />
              <p className="text-red-400 font-medium text-sm mb-4">{error}</p>
              <button
                onClick={onStartCamera}
                className="px-4 py-2 bg-primary rounded-lg text-white text-sm hover:bg-primary-dark transition-colors flex items-center gap-2 mx-auto"
              >
                <FiRefreshCw size={14} /> Retry
              </button>
            </div>
          </div>
        )}

        {/* Recording overlay */}
        {recordingState === RecordingState.RECORDING && (
          <div className="absolute inset-0 pointer-events-none">
            {/* Red recording indicator */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-full">
              <div className="w-3 h-3 bg-red-500 rounded-full recording-ring" />
              <span className="text-white text-xs font-medium">REC</span>
            </div>

            {/* Pulsing border */}
            <div className="absolute inset-0 rounded-2xl border-2 border-red-500 animate-pulse-fast" />
          </div>
        )}

        {/* Processing overlay */}
        {recordingState === RecordingState.PROCESSING && (
          <div className="absolute inset-0 bg-black/60 flex items-center justify-center pointer-events-none">
            <div className="flex items-center gap-3 bg-surface-card px-5 py-3 rounded-xl">
              <div className="w-5 h-5 border-3 border-yellow-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-yellow-400 font-medium text-sm">Analyzing gesture...</span>
            </div>
          </div>
        )}

        {/* Countdown overlay */}
        {recordingState === RecordingState.COUNTDOWN && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <div className="text-7xl font-bold text-white animate-pulse">
                🤚
              </div>
              <p className="text-white text-lg mt-2 font-medium">Show your sign</p>
            </div>
          </div>
        )}

        {/* Progress bar */}
        {(recordingState === RecordingState.RECORDING ||
          recordingState === RecordingState.PROCESSING) && (
          <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-black/50">
            <div
              className="h-full bg-gradient-to-r from-primary to-secondary confidence-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Status badge */}
        <div className="absolute bottom-3 right-3">
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${
            isReady
              ? 'bg-secondary/20 text-secondary border border-secondary/30'
              : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}>
            {getStatusLabel()}
          </span>
        </div>
      </div>

      {/* Camera label */}
      {cameraLabel && (
        <p className="text-text-muted text-xs text-center flex items-center justify-center gap-1.5">
          <FiCamera size={11} />
          {cameraLabel}
        </p>
      )}

      {/* Capture button */}
      <button
        onClick={onCapture}
        disabled={!isReady || recordingState !== RecordingState.IDLE}
        className={`
          w-full py-4 rounded-xl font-semibold text-base
          flex items-center justify-center gap-3
          transition-all duration-200 select-none
          ${isReady && recordingState === RecordingState.IDLE
            ? 'bg-primary hover:bg-primary-dark active:scale-95 text-white shadow-glow-primary cursor-pointer'
            : 'bg-surface-card text-text-muted cursor-not-allowed opacity-50'
          }
        `}
      >
        {recordingState === RecordingState.IDLE ? (
          <>
            <span className="text-xl">🤟</span>
            Capture Sign ({DURATION_S}s / {FRAME_COUNT} frames)
          </>
        ) : recordingState === RecordingState.COUNTDOWN ? (
          <span>Get ready...</span>
        ) : recordingState === RecordingState.RECORDING ? (
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 bg-red-400 rounded-full animate-pulse" />
            Recording... {progress}%
          </span>
        ) : recordingState === RecordingState.PROCESSING ? (
          <span>Processing...</span>
        ) : null}
      </button>
    </div>
  )
}