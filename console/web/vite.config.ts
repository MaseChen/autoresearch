import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../../kernel_research/console/static',
    emptyOutDir: true,
    manifest: true,
    sourcemap: false,
    target: 'es2022',
    assetsInlineLimit: 0,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
  test: {
    include: ['tests/**/*.test.{ts,tsx}'],
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    coverage: {
      provider: 'v8',
      include: [
        'src/api.ts',
        'src/chartAccessibility.ts',
        'src/components/EChart.tsx',
        'src/components/CodeEditor.tsx',
        'src/components/DeepEvidencePanel.tsx',
        'src/operationDrafts.ts',
        'src/components/StatusBadge.tsx',
        'src/deepEvidence.ts',
        'src/presentation.ts',
      ],
      reporter: ['text', 'json-summary'],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
  },
})
