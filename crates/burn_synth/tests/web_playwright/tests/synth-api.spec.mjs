import { test, expect } from '@playwright/test';

test('burn_synth wasm sharded web inference produces a GLB', async ({ page }) => {
  test.setTimeout(1800000);

  const pageErrors = [];
  const consoleErrors = [];
  const modelRequests = [];
  const shardRequests = new Set();
  const failedModelResponses = [];

  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('request', (request) => {
    const url = request.url();
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
      if (url.includes('.bpk.shards/') || url.endsWith('.bpk.shards.json')) {
        shardRequests.add(url);
      }
    }
  });
  page.on('response', (response) => {
    const url = response.url();
    if (url.includes('/assets/models/') && response.status() >= 400) {
      failedModelResponses.push(`${response.status()} ${url}`);
    }
  });

  await page.goto('http://127.0.0.1:4173/www/synth_api.html', {
    waitUntil: 'domcontentloaded',
  });
  await page.click('#boot-start');
  await expect(page.locator('#boot-text')).toHaveText(/module ready/i, {
    timeout: 600000,
  });

  const webgpu = await page.evaluate(async () => {
    const hasGpu = !!navigator.gpu;
    if (!hasGpu) {
      return { hasGpu, adapter: false, shaderF16: false };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { hasGpu, adapter: false, shaderF16: false };
    }
    return {
      hasGpu,
      adapter: true,
      shaderF16: adapter.features.has('shader-f16'),
    };
  });
  expect(webgpu.hasGpu).toBe(true);
  expect(webgpu.adapter).toBe(true);

  const inference = await page.evaluate(async () => {
    const started = performance.now();
    try {
      const imageResp = await fetch('/docs/input_chair.jpg');
      if (!imageResp.ok) {
        throw new Error(`failed to fetch docs/input_chair.jpg: ${imageResp.status}`);
      }
      const imageBytes = new Uint8Array(await imageResp.arrayBuffer());
      const options = new window.__burnSynthWasm.WasmInferOptions();
      options.set_num_steps(2);
      options.set_num_tokens(512);
      options.set_resolution(15);
      options.set_faces(1000);
      options.set_backend('wgpu');
      options.set_dino_backend('auto');
      const inferPromise = window.__burnSynthWasm
        .infer_glb_from_image_bytes_with_options(imageBytes, 'input_chair.jpg', options)
        .then((glb) => ({
          ok: true,
          glbBytes: glb.byteLength,
          elapsedMs: performance.now() - started,
        }))
        .catch((error) => ({
          ok: false,
          error: String(error),
          elapsedMs: performance.now() - started,
        }));
      const timeoutPromise = new Promise((resolve) =>
        setTimeout(
          () =>
            resolve({
              ok: false,
              timeout: true,
              elapsedMs: performance.now() - started,
            }),
          600000,
        ),
      );
      return await Promise.race([inferPromise, timeoutPromise]);
    } catch (error) {
      return {
        ok: false,
        error: String(error),
        elapsedMs: performance.now() - started,
      };
    }
  });

  expect(inference.ok, `inference failed: ${JSON.stringify(inference)}`).toBe(true);
  expect(inference.glbBytes).toBeGreaterThan(0);
  expect(modelRequests.length).toBeGreaterThan(0);
  expect(shardRequests.size).toBeGreaterThan(0);
  expect(
    modelRequests.some((url) => url.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json')),
    'expected dedicated DINOv2 config request',
  ).toBe(true);
  expect(
    modelRequests.some((url) => url.includes('/assets/models/MIDI-3D/feature_extractor_dinov2/preprocessor_config.json')),
    'expected dedicated DINOv2 preprocessor request',
  ).toBe(true);
  const missingDedicatedDinoConfig = failedModelResponses.some(
    (entry) =>
      entry.includes('404 ') && entry.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json'),
  );
  if (!missingDedicatedDinoConfig) {
    expect(
      modelRequests.some((url) => url.includes('/assets/models/MIDI-3D/image_encoder_2/config.json')),
      'legacy image_encoder_2 config should not be used when dedicated DINOv2 config exists',
    ).toBe(false);
  }
  expect(
    modelRequests.some((url) =>
      url.includes('/assets/models/MIDI-3D/image_encoder_dinov2/model_f16.bpk'),
    ),
    'wasm path should prefer DINOv2 f16 burnpack assets for stable model load',
  ).toBe(true);
  const nonBenignFailedModelResponses = failedModelResponses.filter(
    (entry) =>
      !(
        entry.includes('404 ') &&
        (
          entry.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model.bpk.parts.json') ||
          entry.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json') ||
          entry.includes('/assets/models/MIDI-3D/feature_extractor_dinov2/preprocessor_config.json')
        )
      ),
  );
  expect(
    nonBenignFailedModelResponses,
    `failed model responses: ${nonBenignFailedModelResponses.join(' | ')}`,
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
