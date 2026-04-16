import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-mode proxy to the Python gpt_terminal server
    proxy: {
      '/api': 'http://127.0.0.1:3000',
      '/db': 'http://127.0.0.1:3000',
      '/rag': 'http://127.0.0.1:3000',
      '/auth': 'http://127.0.0.1:3000',
      '/ws': {
        target: 'ws://127.0.0.1:3000',
        ws: true,
      },
    },
  },
})
