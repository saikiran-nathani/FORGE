/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        forge: {
          bg: '#0a0a0b',      // forged charcoal
          bg2: '#0f0e11',
          card: '#141318',     // steel panel
          line: '#26242c',     // hairline borders
          accent: '#ff5a1e',   // molten ember
          amber: '#ffa23c',
          hot: '#ffe6c2',      // white-hot
          steel: '#8b8f99',    // cold steel grey
          muted: '#8b8f99',
          success: '#3ddc97',
        },
      },
      fontFamily: {
        display: ['"Chakra Petch"', 'ui-sans-serif', 'sans-serif'],
        sans: ['Sora', 'ui-sans-serif', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        ember: '0 0 0 1px rgba(255,90,30,0.35), 0 8px 40px -8px rgba(255,90,30,0.35)',
      },
    },
  },
  plugins: [],
}
