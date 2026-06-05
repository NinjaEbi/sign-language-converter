/**
 * Webcam management hook.
 * Handles: stream setup, permissions, constraints, frame capture,
 * and graceful teardown.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const WEBCAM_CONSTRAINTS = {
  video: {
    width:      { ideal: 640,  min: 320 },
    height:     { ideal: 480,  min: 240 },
    frameRate:  { ideal: 30,   min: 15 },
    facingMode: 'user',
  },
  audio: false,
}

export function useWebcam() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  const [isReady,     setIsReady]     = useState(false)
  const [isLoading,   setIsLoading]   = useState(false)
  const [error,       setError]       = useState(null)
  const [cameraLabel, setCameraLabel] = useState('')
  const [devices,     setDevices]     = useState([])


  const startCamera = useCallback(async (deviceId = null) => {
    setIsLoading(true)
    setError(null)

    // Stop any existing stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
    }

    try {
      const constraints = {
        ...WEBCAM_CONSTRAINTS,
        video: deviceId
          ? { ...WEBCAM_CONSTRAINTS.video, deviceId: { exact: deviceId } }
          : WEBCAM_CONSTRAINTS.video,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // Get camera label
      const track = stream.getVideoTracks()[0]
      setCameraLabel(track.label || 'Camera')

      setIsReady(true)
    } catch (err) {
      let msg = 'Camera access failed.'
      if (err.name === 'NotAllowedError') {
        msg = 'Camera permission denied. Please allow access in browser settings.'
      } else if (err.name === 'NotFoundError') {
        msg = 'No camera device found.'
      } else if (err.name === 'NotReadableError') {
        msg = 'Camera is in use by another application.'
      }
      setError(msg)
      setIsReady(false)
      console.error('[Webcam]', err)
    } finally {
      setIsLoading(false)
    }
  }, [])


  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsReady(false)
  }, [])


  const enumerateDevices = useCallback(async () => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices()
      const cameras    = allDevices.filter((d) => d.kind === 'videoinput')
      setDevices(cameras)
      return cameras
    } catch (err) {
      console.error('[Webcam] enumerateDevices:', err)
      return []
    }
  }, [])


  const captureFrame = useCallback((quality = 0.8) => {
    if (!videoRef.current || !canvasRef.current || !isReady) return null

    const video  = videoRef.current
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')

    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480

    // Mirror the frame (since webcam is mirrored visually)
    ctx.save()
    ctx.scale(-1, 1)
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
    ctx.restore()

    return canvas.toDataURL('image/jpeg', quality)
  }, [isReady])


  const captureFrameSequence = useCallback(
    (n = 30, intervalMs = 100, onProgress = null) => {
      return new Promise((resolve, reject) => {
        if (!isReady) {
          reject(new Error('Camera not ready'))
          return
        }

        const frames = []
        let count = 0

        const interval = setInterval(() => {
          const frame = captureFrame()
          if (frame) {
            frames.push(frame)
            count++
            if (onProgress) onProgress(count, n)
            if (count >= n) {
              clearInterval(interval)
              resolve(frames)
            }
          }
        }, intervalMs)

        // Safety timeout
        setTimeout(() => {
          if (frames.length < n) {
            clearInterval(interval)
            if (frames.length > 5) {
              resolve(frames)
            } else {
              reject(new Error('Not enough frames captured'))
            }
          }
        }, n * intervalMs + 3000)
      })
    },
    [isReady, captureFrame]
  )


  // Auto-start camera on mount
  useEffect(() => {
    enumerateDevices().then(() => startCamera())
    return () => stopCamera()
  }, [])


  return {
    videoRef,
    canvasRef,
    isReady,
    isLoading,
    error,
    cameraLabel,
    devices,
    startCamera,
    stopCamera,
    captureFrame,
    captureFrameSequence,
    enumerateDevices,
  }
}