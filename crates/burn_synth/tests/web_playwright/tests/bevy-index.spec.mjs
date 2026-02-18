import { test, expect } from '@playwright/test';

test('bevy_synth wasm page eagerly warms models during startup', async ({ page }) => {
  test.setTimeout(600000);

  const pageErrors = [];
  const consoleErrors = [];
  const eagerModelLoadLogs = [];
  const modelRequests = [];
  let warmupState = 'unknown';

  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('requestfinished', (request) => {
    const url = request.url();
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
    }
  });
  page.on('console', (msg) => {
    const text = msg.text();
    if (msg.type() === 'error') {
      consoleErrors.push(text);
    }
    if (
      text.includes('bevy_synth wasm warmup:') ||
      text.includes('Loading model weights...') ||
      text.includes('TripoSG weight precision policy:') ||
      text.includes('Model weights ready.')
    ) {
      eagerModelLoadLogs.push(text);
    }
    if (text.includes('bevy_synth wasm warmup: ready')) {
      warmupState = 'ready';
    } else if (text.includes('bevy_synth wasm warmup: failed')) {
      warmupState = 'failed';
    }
  });

  await page.goto('http://127.0.0.1:4173/www/index.html', {
    waitUntil: 'domcontentloaded',
  });
  await page.click('#boot-start');

  await expect(page.locator('#boot-text')).toHaveText(/module ready/i, {
    timeout: 600000,
  });

  await expect
    .poll(
      async () =>
        page.evaluate(
          () => window.__bevySynthWarmupState ?? window.__burnSynthWasmWarmupState ?? 'unknown',
        ),
      {
        timeout: 600000,
        message: `expected wasm warmup ready signal, saw logs: ${eagerModelLoadLogs.join(' | ')}`,
      },
    )
    .toBe('ready');

  const ready = await page.evaluate(() => window.__burnSynthWasmReady === true);
  expect(ready).toBe(true);
  await expect(page.locator('canvas')).toHaveCount(1, { timeout: 60000 });

  const sawTransformerPartsManifest = modelRequests.some((url) =>
    /\/transformer\/diffusion_pytorch_model(_f16)?\.bpk\.parts\.json$/.test(url),
  );
  expect(
    sawTransformerPartsManifest,
    `expected transformer parts manifest request, saw: ${modelRequests.join(' | ')}`,
  ).toBe(true);

  const sawTransformerMonolith = modelRequests.some((url) =>
    /\/transformer\/diffusion_pytorch_model(_f16)?\.bpk$/.test(url),
  );
  expect(
    sawTransformerMonolith,
    `unexpected monolithic transformer burnpack request: ${modelRequests.join(' | ')}`,
  ).toBe(false);

  const sawDinoPartsManifest = modelRequests.some((url) =>
    /\/image_encoder_dinov2\/model(_f16)?\.bpk\.parts\.json$/.test(url),
  );
  expect(
    sawDinoPartsManifest,
    `expected DINO parts manifest request, saw: ${modelRequests.join(' | ')}`,
  ).toBe(true);

  const sawDinoMonolith = modelRequests.some((url) =>
    /\/image_encoder_dinov2\/model(_f16)?\.bpk$/.test(url),
  );
  expect(
    sawDinoMonolith,
    `unexpected monolithic DINO burnpack request: ${modelRequests.join(' | ')}`,
  ).toBe(false);

  const sawVaePartsManifest = modelRequests.some((url) =>
    /\/vae\/diffusion_pytorch_model(_f16)?\.bpk\.parts\.json$/.test(url),
  );
  expect(
    sawVaePartsManifest,
    `expected VAE parts manifest request, saw: ${modelRequests.join(' | ')}`,
  ).toBe(true);

  const sawRmbgPartsManifest = modelRequests.some((url) =>
    /\/RMBG-1\.4\/model(_f16)?\.bpk\.parts\.json$/.test(url),
  );
  expect(
    sawRmbgPartsManifest,
    `expected RMBG parts manifest request, saw: ${modelRequests.join(' | ')}`,
  ).toBe(true);

  const shardManifestRequests = modelRequests.filter((url) => url.endsWith('.bpk.shards.json'));
  expect(
    shardManifestRequests,
    `unexpected shard manifest requests in parts-first wasm loader: ${shardManifestRequests.join(' | ')}`,
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
