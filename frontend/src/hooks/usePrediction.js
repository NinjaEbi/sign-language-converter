/**
 * Prediction state management hook.
 * Handles: recording state machine, API calls, smoothing,
 * sentence building, and feedback submission.
 */

import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { predict, submitFeedback, clearSession } from '../services/api'

const SESSION_ID = uuidv4()

export const RecordingState = {
  IDLE:       'IDLE',
  COUNTDOWN:  'COUNTDOWN',
  RECORDING:  'RECORDING',
  PROCESSING: 'PROCESSING',
  DONE:       'DONE',
  ERROR:      'ERROR',
}

export function usePrediction(onNewPrediction = null) {
  const [state,         setState]         = useState(RecordingState.IDLE)
  const [prediction,    setPrediction]    = useState(null)
  const [history,       setHistory]       = useState([])
  const [sentence,      setSentence]      = useState([])
  const [error,         setError]         = useState(null)
  const [progress,      setProgress]      = useState(0)   // 0-100
  const [capturedFrames, setCapturedFrames] = useState([])

  const isRunning = useRef(false)


  const runPrediction = useCallback(async (captureFrameSequence) => {
    if (isRunning.current) return
    isRunning.current = true

    try {
      // Countdown
      setState(RecordingState.COUNTDOWN)
      setError(null)
      setProgress(0)

      await new Promise((r) => setTimeout(r, 1000)) // 1s countdown display

      // Recording
      setState(RecordingState.RECORDING)

      let frames
      try {
        frames = await captureFrameSequence(
          30,   // n frames
          100,  // 100ms interval = 10fps capture
          (captured, total) => {
            setProgress(Math.round((captured / total) * 100))
          }
        )
        setCapturedFrames(frames)
      } catch (captureErr) {
        throw new Error(`Frame capture failed: ${captureErr.message}`)
      }

      // Processing
      setState(RecordingState.PROCESSING)
      setProgress(100)

      const result = await predict(frames, SESSION_ID)

      // Update state
      setPrediction(result)
      setHistory((prev) => [
        { ...result, timestamp: new Date(), id: uuidv4() },
        ...prev.slice(0, 49)  // keep last 50
      ])

      if (onNewPrediction) {
        onNewPrediction(result)
      }

      setState(RecordingState.DONE)

    } catch (err) {
      console.error('[Prediction]', err)
      setError(err.message || 'Prediction failed')
      setState(RecordingState.ERROR)
    } finally {
      isRunning.current = false
      setProgress(0)
      setTimeout(() => setState(RecordingState.IDLE), 1500)
    }
  }, [onNewPrediction])


  const addToSentence = useCallback((label = null) => {
    const word = label || prediction?.prediction
    if (!word || word === 'error' || word === 'uncertain') return
    setSentence((prev) => [...prev, word])
  }, [prediction])


  const removeLastWord = useCallback(() => {
    setSentence((prev) => prev.slice(0, -1))
  }, [])


  const clearSentence = useCallback(() => {
    setSentence([])
  }, [])


  const submitCorrection = useCallback(async (correctedLabel) => {
    if (!capturedFrames.length || !prediction) return false
    try {
      await submitFeedback(
        prediction.prediction,
        correctedLabel,
        capturedFrames,
        SESSION_ID,
        prediction.confidence
      )
      return true
    } catch (err) {
      console.error('[Feedback]', err)
      return false
    }
  }, [capturedFrames, prediction])


  const resetSession = useCallback(async () => {
    setState(RecordingState.IDLE)
    setPrediction(null)
    setError(null)
    setProgress(0)
    setCapturedFrames([])
    isRunning.current = false
    await clearSession(SESSION_ID).catch(() => {})
  }, [])


  return {
    state,
    prediction,
    history,
    sentence,
    sentenceText: sentence.join(' '),
    error,
    progress,
    capturedFrames,
    sessionId: SESSION_ID,
    isIdle:        state === RecordingState.IDLE,
    isRecording:   state === RecordingState.RECORDING,
    isProcessing:  state === RecordingState.PROCESSING,
    isCountdown:   state === RecordingState.COUNTDOWN,
    runPrediction,
    addToSentence,
    removeLastWord,
    clearSentence,
    submitCorrection,
    resetSession,
  }
}