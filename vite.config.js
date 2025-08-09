import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Minimal Vite config; no aliasing needed
export default defineConfig({
  plugins: [react()],
})