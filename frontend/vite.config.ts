import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/enhance': 'http://localhost:8000',
      '/profiles': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
    },
  },
})
