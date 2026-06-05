/**
 * Top status bar — API health, model status, demo mode indicator.
 */

import React, { useEffect, useState } from 'react'
import { FiWifi, FiWifiOff, FiCpu } from 'react-icons/fi'
import { healthCheck } from '../services/api'

export default function StatusBar() {
  const [health,    setHealth]    = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [lastPing,  setLastPing]  = useState(null)

  const fetchHealth = async () => {
    try {
      const data = await healthCheck()
      setHealth(data)
      setLastPing(new Date())
    } catch (err) {
      setHealth(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30000) // poll every 30s
    return () => clearInterval(interval)
  }, [])

  const isConnected = !!health

  return (
    <div className="flex items-center justify-between px-4 py-2
                    bg-surface-card border-b border-surface-border
                    text-xs text-text-muted">
      {/* Left: API status */}
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-1.5 ${isConnected ? 'text-secondary' : 'text-red-400'}`}>
          {isConnected ? <FiWifi size={12} /> : <FiWifiOff size={12} />}
          <span>{isConnected ? 'API Connected' : 'API Offline'}</span>
        </div>

        {health && (
          <>
            <div className="w-px h-3 bg-surface-border" />
            <div className="flex items-center gap-1.5">
              <FiCpu size={12} />
              <span>{health.model_loaded ? 'Model Ready' : 'Demo Mode'}</span>
            </div>
            <div className="w-px h-3 bg-surface-border" />
            <span>{health.num_classes} classes</span>
          </>
        )}
      </div>

      {/* Right: last ping */}
      {lastPing && (
        <span>Last check: {lastPing.toLocaleTimeString()}</span>
      )}
    </div>
  )
}