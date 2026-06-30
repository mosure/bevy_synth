import { test, expect } from '@playwright/test';

test('bevy_synth wasm launcher keeps model selection inside the app', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/www/index.html?model_source=local&sw=off&weights_precision=f32&synthesis_model=triposplat&triposplat_profile=low', {
    waitUntil: 'domcontentloaded',
  });

  await expect(page.locator('#boot-text')).toHaveText('click start to initialize wasm');
  await expect(page.locator('#boot-start')).toBeVisible();
  await expect(page.locator('#boot-quality')).toHaveCount(0);
  await expect(page.locator('#boot-synthesis-model')).toHaveCount(0);
  await expect(page.locator('#boot-rmbg-model')).toHaveCount(0);

  const urlState = await page.evaluate(() => {
    const url = new URL(window.location.href);
    return {
      synthesisModel: url.searchParams.get('synthesis_model'),
      triposplatProfile: url.searchParams.get('triposplat_profile'),
      weightsPrecision: url.searchParams.get('weights_precision'),
    };
  });
  expect(urlState).toEqual({
    synthesisModel: 'triposplat',
    triposplatProfile: 'low',
    weightsPrecision: 'f32',
  });
});

test('bevy_synth wasm page starts the app before loading models', async ({ page }) => {
  test.setTimeout(300000);

  const pageErrors = [];
  const consoleErrors = [];
  const modelRequests = [];
  const normalizeModelUrl = (url) => {
    if (url.includes('/www/assets/models/')) {
      return url.replace('/www/assets/models/', '/assets/models/');
    }
    if (url.includes('/www/assets/')) {
      return url.replace('/www/assets/', '/assets/models/');
    }
    return url;
  };

  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('requestfinished', (request) => {
    const url = normalizeModelUrl(request.url());
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
    }
  });
  page.on('console', (msg) => {
    const text = msg.text();
    if (msg.type() === 'error') {
      consoleErrors.push(text);
    }
  });

  await page.goto('http://127.0.0.1:4173/www/index.html?model_source=local&sw=off&weights_precision=f32', {
    waitUntil: 'domcontentloaded',
  });
  await page.click('#boot-start');

  await expect
    .poll(
      async () =>
        page.evaluate(
          () => window.__bevySynthWarmupState ?? window.__burnSynthWasmWarmupState ?? 'unknown',
        ),
      {
        timeout: 300000,
        message: 'expected wasm app ready signal before any model warmup',
      },
    )
    .toBe('ready');

  const ready = await page.evaluate(() => window.__burnSynthWasmReady === true);
  expect(ready).toBe(true);
  await expect(page.locator('canvas')).toHaveCount(1, { timeout: 60000 });

  const burnpackRequests = modelRequests.filter((url) =>
    url.endsWith('.bpk') || url.endsWith('.bpk.parts.json') || url.includes('.bpk.part-'),
  );
  expect(
    burnpackRequests,
    `startup should not eagerly load model burnpacks before inference: ${burnpackRequests.join(' | ')}`,
  ).toEqual([]);

  expect(pageErrors, `page errors: ${pageErrors.join(' | ')}`).toEqual([]);
  const nonBenignConsoleErrors = consoleErrors.filter(
    (entry) =>
      !entry.includes('favicon.ico') &&
      !entry.includes(
        'Failed to load resource: the server responded with a status of 404 (File not found)',
      ),
  );
  expect(
    nonBenignConsoleErrors,
    `console errors: ${nonBenignConsoleErrors.join(' | ')}`,
  ).toEqual([]);
});
