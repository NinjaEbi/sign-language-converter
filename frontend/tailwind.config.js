/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary:    { DEFAULT: '#6366f1', dark: '#4f46e5', light: '#a5b4fc' },
        secondary:  { DEFAULT: '#10b981', dark: '#059669', light: '#6ee7b7' },
        surface:    { DEFAULT: '#1e1e2e', card: '#27273a', border: '#3b3b52' },
        text:       { primary: '#e2e8f0', muted: '#94a3b8'  },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 0.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':    'fadeIn 0.3s ease-in-out',
        'slide-up':   'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn:  { '0%': { opacity: 0 },             '100%': { opacity: 1 } },
        slideUp: { '0%': { transform: 'translateY(20px)', opacity: 0 },
                   '100%': { transform: 'translateY(0)', opacity: 1 } },
      },
      boxShadow: {
        'glow-primary':   '0 0 20px rgba(99, 102, 241, 0.4)',
        'glow-secondary': '0 0 20px rgba(16, 185, 129, 0.4)',
      },
    },
  },
  plugins: [],
}