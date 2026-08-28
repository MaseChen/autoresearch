import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: 0,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], browserName: 'chromium', viewport: { width: 1440, height: 1000 } } },
    {
      name: 'desktop-textarea-fallback',
      use: {
        ...devices['Desktop Chrome'],
        browserName: 'chromium',
        viewport: { width: 1440, height: 1000 },
        launchOptions: { args: ['--disable-blink-features=EditContext'] },
      },
    },
    {
      name: 'desktop-edge',
      use: {
        ...devices['Desktop Chrome'],
        browserName: 'chromium',
        channel: 'msedge',
        viewport: { width: 1440, height: 1000 },
      },
    },
    { name: 'tablet', use: { ...devices['Desktop Chrome'], browserName: 'chromium', viewport: { width: 1024, height: 768 }, hasTouch: true } },
    { name: 'mobile', use: { ...devices['Desktop Chrome'], browserName: 'chromium', viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true } },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
