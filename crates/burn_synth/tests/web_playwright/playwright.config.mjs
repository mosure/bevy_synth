import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  globalSetup: './webgpu-lock.global-setup.mjs',
  timeout: 1800000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--use-angle=vulkan',
        '--enable-features=Vulkan,UseSkiaRenderer,UnsafeWebGPU',
        '--ignore-gpu-blocklist',
        '--enable-gpu-rasterization',
        '--disable-vulkan-surface',
      ],
    },
  },
  webServer: {
    command: 'python3 -m http.server 4173 -d ../../../../',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 120000,
  },
});
