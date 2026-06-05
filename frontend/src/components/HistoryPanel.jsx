/**
 * Prediction history panel — last 20 predictions with timestamps.
 */

import React from 'react'

export default function HistoryPanel({ history }) {
  if (!history.length) return null

  return (
    <div className="glass-card rounded-2xl p-5">
      <h2 className="font-semibold text-white text-sm mb-3">
        Prediction History
        <span className="ml-2 text-xs text-text-muted font-normal">
          ({history.length})
        </span>
      </h2>

      <div className="flex flex-col gap-1.5 max-h-52 overflow-y-auto pr-1">
        {history.slice(0, 20).map((item) => {
          const conf  = Math.round(item.confidence * 100)
          const time  = new Date(item.timestamp).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
          })

          return (
            <div
              key={item.id}
              className="flex items-center justify-between px-3 py-2
                         bg-surface rounded-lg border border-surface-border
                         hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="font-semibold text-white text-sm min-w-12">
                  {item.prediction}
                </span>
                <div className="h-1.5 w-16 bg-surface-border rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      conf >= 80 ? 'bg-secondary' :
                      conf >= 55 ? 'bg-yellow-400' : 'bg-red-400'
                    }`}
                    style={{ width: `${conf}%` }}
                  />
                </div>
                <span className="text-xs text-text-muted">{conf}%</span>
              </div>
              <span className="text-xs text-text-muted font-mono">{time}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}