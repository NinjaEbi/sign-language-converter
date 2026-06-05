/**
 * Web Speech API hook for text-to-speech output.
 * Handles browser compatibility, queue management, and settings.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

export function useSpeech() {
  const synthRef  = useRef(null)
  const queueRef  = useRef([])

  const [isSupported,  setIsSupported]  = useState(false)
  const [isSpeaking,   setIsSpeaking]   = useState(false)
  const [voices,       setVoices]       = useState([])
  const [selectedVoice, setSelectedVoice] = useState(null)
  const [settings,     setSettings]     = useState({
    rate:   1.0,
    pitch:  1.0,
    volume: 1.0,
  })


  useEffect(() => {
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis
      setIsSupported(true)

      const loadVoices = () => {
        const available = synthRef.current.getVoices()
        setVoices(available)
        // Prefer English voice
        const english = available.find((v) =>
          v.lang.startsWith('en') && v.localService
        ) || available[0]
        if (english) setSelectedVoice(english)
      }

      loadVoices()
      synthRef.current.addEventListener('voiceschanged', loadVoices)

      return () => {
        synthRef.current?.removeEventListener('voiceschanged', loadVoices)
      }
    }
  }, [])


  const speak = useCallback((text, options = {}) => {
    if (!isSupported || !text?.trim()) return

    // Cancel current speech for immediate words
    if (options.interrupt) {
      synthRef.current.cancel()
      queueRef.current = []
      setIsSpeaking(false)
    }

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate   = options.rate   ?? settings.rate
    utterance.pitch  = options.pitch  ?? settings.pitch
    utterance.volume = options.volume ?? settings.volume

    if (selectedVoice) {
      utterance.voice = selectedVoice
    }

    utterance.onstart = () => setIsSpeaking(true)
    utterance.onend   = () => {
      setIsSpeaking(false)
      if (queueRef.current.length > 0) {
        const next = queueRef.current.shift()
        synthRef.current.speak(next)
      }
    }
    utterance.onerror = (e) => {
      console.error('[Speech] Error:', e)
      setIsSpeaking(false)
    }

    synthRef.current.speak(utterance)
  }, [isSupported, selectedVoice, settings])


  const speakSentence = useCallback((sentence) => {
    speak(sentence, { interrupt: true, rate: 0.9 })
  }, [speak])


  const cancel = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel()
      queueRef.current = []
    }
    setIsSpeaking(false)
  }, [])


  const updateSettings = useCallback((newSettings) => {
    setSettings((prev) => ({ ...prev, ...newSettings }))
  }, [])


  return {
    isSupported,
    isSpeaking,
    voices,
    selectedVoice,
    setSelectedVoice,
    settings,
    updateSettings,
    speak,
    speakSentence,
    cancel,
  }
}