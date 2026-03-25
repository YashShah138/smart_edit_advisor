/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          50:  '#faf9f7',
          100: '#f5f2ee',
          200: '#ede9e4',
          300: '#e4dfd9',
          400: '#d6d0ca',
          500: '#c8c0b8',
        },
        accent: {
          DEFAULT: '#c2410c',
          light:   '#ea580c',
          dark:    '#9a3412',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
