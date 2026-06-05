/**
 * Main application component.
 * Layout: Header → StatusBar → [Webcam | Prediction + Sentence] → History
 */

import React, { useState, useCallback } from 'react'
import { FiSettings, FiInfo } from 'react-icons/fi'

import { useWebcam }      from './hooks/useWebcam'
import { useSpeech }      from './hooks/useSpeech'
import { usePrediction, RecordingState } from './hooks/usePrediction'

import WebcamCapture    from './components/WebcamCapture'
import PredictionDisplay from './components/PredictionDisplay'
import SentenceBuilder  from './components/SentenceBuilder'
import HistoryPanel     from './components/HistoryPanel'
import StatusBar        from './components/StatusBar'

export default function App() {
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true)
  const [availableLabels, setAvailableLabels]  = useState([])

  // Speech
  const {
    isSupported: speechSupported,
    isSpeaking,
    speak,
    speakSentence,
  } = useSpeech()

  // Webcam
  const {
    videoRef, canvasRef,
    isReady, isLoading, error,
    cameraLabel,
    startCamera,
    captureFrameSequence,
  } = useWebcam()

  // Handle new prediction → optionally auto-speak
  const handleNewPrediction = useCallback((result) => {
    if (isSpeechEnabled && speechSupported && result.is_certain) {
      speak(result.prediction, { interrupt: false, rate: 1.1 })
    }
    // Update labels from prediction if not yet loaded
    if (result.top_k) {
      setAvailableLabels((prev) => {
        const incoming = result.top_k.map((x) => x.label)
        const merged   = Array.from(new Set([...prev, ...incoming]))
        return merged
      })
    }
  }, [isSpeechEnabled, speechSupported, speak])

  // Prediction
  const {
    state: recordingState,
    prediction,
    history,
    sentence,
    sentenceText,
    progress,
    runPrediction,
    addToSentence,
    removeLastWord,
    clearSentence,
    submitCorrection,
  } = usePrediction(handleNewPrediction)

  const handleCapture = () => {
    runPrediction(captureFrameSequence)
  }

  const handleSpeakSentence = (text) => {
    if (isSpeechEnabled && text) {
      speakSentence(text)
    }
  }

  return (
    <div className="min-h-screen bg-surface flex flex-col">

      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between
                         bg-surface-card border-b border-surface-border">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🤟</span>
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">
              SignBridge
            </h1>
            <p className="text-xs text-text-muted">
              Real-Time Sign Language Converter
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-lg text-text-muted hover:text-white
                       hover:bg-surface-border transition-colors"
            title="API Docs"
          >
            <FiInfo size={18} />
          </a>
          <button
            className="p-2 rounded-lg text-text-muted hover:text-white
                       hover:bg-surface-border transition-colors"
            title="Settings"
          >
            <FiSettings size={18} />
          </button>
        </div>
      </header>

      {/* Status bar */}
      <StatusBar />

      {/* Main content */}
      <main className="flex-1 p-4 md:p-6 max-w-6xl mx-auto w-full">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Left column: Webcam */}
          <div className="flex flex-col gap-5">
            <WebcamCapture
              videoRef       = {videoRef}
              canvasRef      = {canvasRef}
              isReady        = {isReady}
              isLoading      = {isLoading}
              error          = {error}
              cameraLabel    = {cameraLabel}
              recordingState = {recordingState}
              progress       = {progress}
              onStartCamera  = {() => startCamera()}
              onCapture      = {handleCapture}
            />

            <HistoryPanel history={history} />
          </div>

          {/* Right column: Prediction + Sentence */}
          <div className="flex flex-col gap-5">
            <PredictionDisplay
              prediction      = {prediction}
              availableLabels = {availableLabels}
              onAddToSentence = {(label) => addToSentence(label)}
              onCorrect       = {submitCorrection}
            />

            <SentenceBuilder
              sentence         = {sentence}
              sentenceText     = {sentenceText}
              isSpeaking       = {isSpeaking}
              isSpeechEnabled  = {isSpeechEnabled && speechSupported}
              onToggleSpeech   = {() => setIsSpeechEnabled((v) => !v)}
              onSpeak          = {handleSpeakSentence}
              onAddWord        = {addToSentence}
              onRemoveLast     = {removeLastWord}
              onClear          = {clearSentence}
            />

            {/* Quick tips */}
            <div className="glass-card rounded-2xl p-4 text-xs text-text-muted">
              <p className="font-semibold text-white mb-2">💡 How to use</p>
              <ol className="list-decimal list-inside space-y-1">
                <li>Hold your sign clearly in front of the camera</li>
                <li>Click <span className="text-primary font-medium">Capture Sign</span> and hold for 3 seconds</li>
                <li>Click <span className="text-secondary font-medium">Add to Sentence</span> to build a phrase</li>
                <li>Use <span className="text-secondary font-medium">Speak</span> to play aloud</li>
                <li>Use <span className="text-text-primary font-medium">Correct</span> to improve the model</li>
              </ol>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-3 text-center text-xs text-text-muted
                         border-t border-surface-border bg-surface-card">
        SignBridge — CNN + LSTM Sign Language Recognition •{' '}
        <span className="text-primary">Final Year Project</span>
      </footer>
    </div>
  )
}