/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        forge: {
          bg: '#0f172a',
          card: '#1e293b',
          accent: '#38bdf8',
          success: '#4ade80',
          muted: '#94a3b8',
        },
      },
    },
  },
  plugins: [],
}
