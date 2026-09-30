import { defineConfig, splitVendorChunkPlugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    splitVendorChunkPlugin(),   // Auto-splits node_modules into vendor chunk
  ],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  // ── Development Server ──────────────────────────────────────
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },

  // ── Production Build Optimisation ──────────────────────────
  build: {
    // Target modern browsers (ES2020) for smaller output
    target: 'es2020',

    // Warn when individual chunks exceed 500 KB
    chunkSizeWarningLimit: 500,

    rollupOptions: {
      output: {
        // Split vendor libraries into named chunks for optimal caching.
        // When only app code changes, browsers serve vendor chunks from cache.
        manualChunks: {
          // React core runtime — changes rarely
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // Charting library — large, changes rarely
          'vendor-charts': ['recharts'],
          // Animation library — large, changes rarely
          'vendor-motion': ['framer-motion'],
          // Icon library
          'vendor-icons': ['lucide-react'],
          // HTTP client
          'vendor-axios': ['axios'],
        },

        // Deterministic file naming with content hashes
        chunkFileNames:  'assets/js/[name]-[hash].js',
        entryFileNames:  'assets/js/[name]-[hash].js',
        assetFileNames:  'assets/[ext]/[name]-[hash].[ext]',
      },
    },
  },
})
