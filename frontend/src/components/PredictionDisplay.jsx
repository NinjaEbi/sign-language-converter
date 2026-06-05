/**
 * Prediction results display:
 * - Main predicted label (large, animated)
 * - Confidence score with color-coded bar
 * - Top-K alternatives
 * - Hand detection status
 * - Feedback correction widget
 */

import React, { useState } from 'react'
import { FiCheck, FiAlertCircle, FiEdit2, FiX } from 'react-icons/fi'

const CONFIDENCE_COLORS = {
  high:   { bar: 'bg-secondary',     text: 'text-secondary',      label: 'High'   },
  medium: { bar: 'bg-yellow-400',    text: 'text-yellow-400',     label: 'Medium' },
  low:    { bar: 'bg-red-400',       text: 'text-red-400',        label: 'Low'    },
}

function getConfidenceLevel(conf) {
  if (conf >= 0.80) return 'high'
  if (conf >= 0.55) return 'medium'
  return 'low'
}

export default function PredictionDisplay({
  prediction,
  onAddToSentence,
  onCorrect,
  availableLabels = [],
}) {
  const [showCorrection, setShowCorrection]     = useState(false)
  const [correctionLabel, setCorrectionLabel]   = useState('')
  const [correctionSubmitted, setCorrectionSubmitted] = useState(false)

  if (!prediction) {
    return (
      <div className="glass-card rounded-2xl p-8 flex flex-col items-center justify-center min-h-48 text-center">
        <div className="text-5xl mb-4 opacity-30">🤙</div>
        <p className="text-text-muted text-sm">
          Capture a sign to see the prediction here
        </p>
      </div>
    )
  }

  const confLevel  = getConfidenceLevel(prediction.confidence)
  const colors     = CONFIDENCE_COLORS[confLevel]
  const confPct    = Math.round(prediction.confidence * 100)

  const handleCorrect = async (label) => {
    setCorrectionSubmitted(false)
    const success = await onCorrect(label)
    if (success) {
      setCorrectionSubmitted(true)
      setShowCorrection(false)
      setTimeout(() => setCorrectionSubmitted(false), 3000)
    }
  }

  return (
    <div className="glass-card rounded-2xl p-5 flex flex-col gap-4 animate-fade-in">

      {/* Main prediction */}
      <div className="text-center py-3">
        <div className={`
          text-6xl font-bold tracking-tight mb-1 transition-all duration-300
          ${prediction.is_certain ? 'text-white' : 'text-text-muted'}
        `}>
          {prediction.prediction}
        </div>

        {prediction.demo_mode && (
          <span className="text-xs bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-2 py-0.5 rounded-full">
            Demo Mode
          </span>
        )}

        {!prediction.hand_detected && (
          <div className="flex items-center justify-center gap-1.5 mt-1 text-yellow-400 text-xs">
            <FiAlertCircle size={12} />
            No hand detected — showing best guess
          </div>
        )}
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs mb-1.5">
          <span className="text-text-muted">Confidence</span>
          <span className={`font-semibold ${colors.text}`}>
            {confPct}% — {colors.label}
          </span>
        </div>
        <div className="h-2 bg-surface rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full confidence-bar-fill ${colors.bar}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Top-K alternatives */}
      {prediction.top_k?.length > 1 && (
        <div>
          <p className="text-xs text-text-muted mb-2">Alternatives</p>
          <div className="flex flex-wrap gap-2">
            {prediction.top_k.slice(1).map((alt, i) => (
              <button
                key={i}
                onClick={() => onAddToSentence(alt.label)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-surface rounded-lg text-xs
                           text-text-muted hover:text-white hover:bg-primary/20
                           border border-surface-border hover:border-primary/40
                           transition-all duration-150"
              >
                <span className="font-medium">{alt.label}</span>
                <span className="text-text-muted opacity-70">{Math.round(alt.confidence * 100)}%</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Inference info */}
      <div className="flex items-center justify-between text-xs text-text-muted border-t border-surface-border pt-3">
        <span>⏱ {Math.round(prediction.inference_time * 1000)} ms</span>
        {prediction.smoothed && (
          <span className="bg-primary/10 text-primary border border-primary/20 px-2 py-0.5 rounded-full">
            Smoothed
          </span>
        )}
        <span>{prediction.hand_detected ? '✋ Hand detected' : '❓ No hand'}</span>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => onAddToSentence(prediction.prediction)}
          disabled={!prediction.is_certain}
          className={`
            flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all duration-150
            flex items-center justify-center gap-2
            ${prediction.is_certain
              ? 'bg-secondary hover:bg-secondary-dark text-white active:scale-95 shadow-glow-secondary'
              : 'bg-surface-card text-text-muted cursor-not-allowed opacity-50'}
          `}
        >
          <FiCheck size={14} /> Add to Sentence
        </button>

        <button
          onClick={() => setShowCorrection(!showCorrection)}
          className="px-4 py-2.5 rounded-xl text-sm font-medium bg-surface-card
                     text-text-muted hover:text-white hover:bg-surface-border
                     transition-colors flex items-center gap-2"
        >
          <FiEdit2 size={14} /> Correct
        </button>
      </div>

      {/* Correction submitted */}
      {correctionSubmitted && (
        <div className="text-center text-xs text-secondary animate-fade-in">
          ✓ Correction saved — thank you!
        </div>
      )}

      {/* Correction dropdown */}
      {showCorrection && (
        <div className="border border-surface-border rounded-xl p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium">What was the correct sign?</p>
            <button
              onClick={() => setShowCorrection(false)}
              className="text-text-muted hover:text-white"
            >
              <FiX size={16} />
            </button>
          </div>

          {/* Quick labels */}
          {availableLabels.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3 max-h-28 overflow-y-auto">
              {availableLabels.slice(0, 40).map((label) => (
                <button
                  key={label}
                  onClick={() => handleCorrect(label)}
                  className="px-2.5 py-1 bg-surface rounded-lg text-xs
                             hover:bg-primary/20 hover:text-primary
                             border border-surface-border hover:border-primary/40
                             transition-all duration-100"
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          {/* Manual input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={correctionLabel}
              onChange={(e) => setCorrectionLabel(e.target.value)}
              placeholder="Type correct label..."
              className="flex-1 bg-surface border border-surface-border rounded-lg px-3 py-2
                         text-sm text-white placeholder-text-muted focus:outline-none
                         focus:border-primary transition-colors"
            />
            <button
              onClick={() => correctionLabel.trim() && handleCorrect(correctionLabel.trim())}
              disabled={!correctionLabel.trim()}
              className="px-4 py-2 bg-primary rounded-lg text-sm text-white
                         hover:bg-primary-dark disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors"
            >
              Submit
            </button>
          </div>
        </div>
      )}
    </div>
  )
}