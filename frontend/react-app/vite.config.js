import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/upload':       'http://localhost:8000',
      '/ask':          'http://localhost:8000',
      '/extract':      'http://localhost:8000',
      '/extract-file': 'http://localhost:8000',
      '/docs':         'http://localhost:8000',
      '/auth':         'http://localhost:8000',
    },
  },
})
