/**
 * Sentence builder component:
 * - Displays accumulated words
 * - Edit (remove last word / clear all)
 * - Text-to-speech read aloud
 * - Copy to clipboard
 */

import React, { useState } from 'react'
import {
  FiTrash2, FiDelete, FiVolume2, FiVolumeX,
  FiCopy, FiCheck
} from 'react-icons/fi'

export default function SentenceBuilder({
  sentence,
  sentenceText,
  isSpeaking,
  isSpeechEnabled,
  onToggleSpeech,
  onSpeak,
  onAddWord,
  onRemoveLast,
  onClear,
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (!sentenceText) return
    try {
      await navigator.clipboard.writeText(sentenceText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Copy failed:', err)
    }
  }

  return (
    <div className="glass-card rounded-2xl p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-white text-sm">Sentence Builder</h2>
        <div className="flex items-center gap-2">
          {/* Speech toggle */}
          <button
            onClick={onToggleSpeech}
            title={isSpeechEnabled ? 'Disable speech' : 'Enable speech'}
            className={`p-2 rounded-lg transition-all ${
              isSpeechEnabled
                ? 'bg-secondary/20 text-secondary border border-secondary/30'
                : 'bg-surface text-text-muted border border-surface-border'
            }`}
          >
            {isSpeechEnabled
              ? <FiVolume2 size={14} />
              : <FiVolumeX size={14} />
            }
          </button>

          {/* Copy */}
          <button
            onClick={handleCopy}
            disabled={!sentenceText}
            title="Copy to clipboard"
            className="p-2 rounded-lg bg-surface border border-surface-border
                       text-text-muted hover:text-white hover:bg-surface-border
                       disabled:opacity-40 transition-all"
          >
            {copied ? <FiCheck size={14} className="text-secondary" /> : <FiCopy size={14} />}
          </button>
        </div>
      </div>

      {/* Words display */}
      <div className={`
        min-h-16 p-4 rounded-xl flex flex-wrap gap-2 items-start
        border transition-all
        ${sentence.length > 0
          ? 'bg-surface border-primary/20'
          : 'bg-surface/50 border-surface-border border-dashed'
        }
      `}>
        {sentence.length === 0 ? (
          <span className="text-text-muted text-sm self-center w-full text-center">
            Signs will appear here...
          </span>
        ) : (
          sentence.map((word, i) => (
            <span
              key={`${word}-${i}`}
              className="px-3 py-1.5 bg-primary/15 border border-primary/25
                         text-primary-light rounded-lg text-sm font-medium
                         animate-fade-in cursor-default select-none"
            >
              {word}
            </span>
          ))
        )}
      </div>

      {/* Full text display */}
      {sentenceText && (
        <div className="bg-surface rounded-xl p-3 border border-surface-border">
          <p className="text-text-muted text-xs mb-1">Full text:</p>
          <p className="text-white text-base font-medium leading-relaxed">
            "{sentenceText}"
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="grid grid-cols-3 gap-2">
        {/* Speak */}
        <button
          onClick={() => onSpeak(sentenceText)}
          disabled={!sentenceText || !isSpeechEnabled}
          className={`
            py-2.5 rounded-xl text-xs font-semibold
            flex items-center justify-center gap-1.5 transition-all
            ${sentenceText && isSpeechEnabled
              ? isSpeaking
                  ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 cursor-wait'
                  : 'bg-secondary/20 text-secondary border border-secondary/30 hover:bg-secondary/30 active:scale-95'
              : 'bg-surface text-text-muted border border-surface-border opacity-40 cursor-not-allowed'
            }
          `}
        >
          <FiVolume2 size={13} />
          {isSpeaking ? 'Speaking...' : 'Speak'}
        </button>

        {/* Remove last */}
        <button
          onClick={onRemoveLast}
          disabled={sentence.length === 0}
          className="py-2.5 rounded-xl text-xs font-semibold
                     flex items-center justify-center gap-1.5
                     bg-surface border border-surface-border text-text-muted
                     hover:text-white hover:bg-surface-border
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all active:scale-95"
        >
          <FiDelete size={13} /> Undo
        </button>

        {/* Clear all */}
        <button
          onClick={onClear}
          disabled={sentence.length === 0}
          className="py-2.5 rounded-xl text-xs font-semibold
                     flex items-center justify-center gap-1.5
                     bg-red-500/10 border border-red-500/20 text-red-400
                     hover:bg-red-500/20
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all active:scale-95"
        >
          <FiTrash2 size={13} /> Clear
        </button>
      </div>
    </div>
  )
}