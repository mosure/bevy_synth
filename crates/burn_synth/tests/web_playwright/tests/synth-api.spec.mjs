import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

function parseGlbStats(glbBytesLike) {
  const glbBytes = glbBytesLike instanceof Uint8Array ? glbBytesLike : new Uint8Array(glbBytesLike);
  const view = new DataView(glbBytes.buffer, glbBytes.byteOffset, glbBytes.byteLength);
  if (view.getUint32(0, true) !== 0x46546c67) {
    throw new Error('invalid GLB header');
  }

  let offset = 12;
  let jsonChunk = null;
  let binChunk = null;
  while (offset + 8 <= glbBytes.byteLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    offset += 8;
    const chunk = glbBytes.subarray(offset, offset + chunkLength);
    if (chunkType === 0x4e4f534a) {
      jsonChunk = chunk;
    } else if (chunkType === 0x004e4942) {
      binChunk = chunk;
    }
    offset += chunkLength;
  }
  if (!jsonChunk || !binChunk) {
    throw new Error('missing GLB JSON/BIN chunk');
  }

  const gltf = JSON.parse(new TextDecoder().decode(jsonChunk));
  const mesh = gltf.meshes?.[0];
  const primitive = mesh?.primitives?.[0];
  if (!primitive) {
    throw new Error('GLB has no mesh primitive');
  }
  const positionAccessor = gltf.accessors?.[primitive.attributes?.POSITION];
  if (!positionAccessor) {
    throw new Error('GLB has no POSITION accessor');
  }
  const indexAccessor = primitive.indices !== undefined ? gltf.accessors?.[primitive.indices] : null;

  const positions = readVec3F32Accessor(gltf, binChunk, positionAccessor);
  const mins = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const maxs = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i + 0];
    const y = positions[i + 1];
    const z = positions[i + 2];
    mins[0] = Math.min(mins[0], x);
    mins[1] = Math.min(mins[1], y);
    mins[2] = Math.min(mins[2], z);
    maxs[0] = Math.max(maxs[0], x);
    maxs[1] = Math.max(maxs[1], y);
    maxs[2] = Math.max(maxs[2], z);
  }

  const vertexCount = positionAccessor.count;
  const faceCount = indexAccessor ? Math.floor(indexAccessor.count / 3) : Math.floor(vertexCount / 3);
  return {
    vertexCount,
    faceCount,
    boundsMin: mins,
    boundsMax: maxs,
  };
}

function readVec3F32Accessor(gltf, binChunk, accessor) {
  if (accessor.componentType !== 5126 || accessor.type !== 'VEC3') {
    throw new Error('POSITION accessor is not float32 vec3');
  }
  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    throw new Error('missing buffer view for POSITION accessor');
  }
  const count = accessor.count;
  const stride = bufferView.byteStride ?? 12;
  const baseOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const view = new DataView(binChunk.buffer, binChunk.byteOffset, binChunk.byteLength);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const off = baseOffset + i * stride;
    out[i * 3 + 0] = view.getFloat32(off + 0, true);
    out[i * 3 + 1] = view.getFloat32(off + 4, true);
    out[i * 3 + 2] = view.getFloat32(off + 8, true);
  }
  return out;
}

function parseSplatStats(splatBytesLike) {
  const splatBytes =
    splatBytesLike instanceof Uint8Array ? splatBytesLike : new Uint8Array(splatBytesLike);
  const recordBytes = 32;
  if (splatBytes.byteLength === 0 || splatBytes.byteLength % recordBytes !== 0) {
    throw new Error(`invalid .splat byte length ${splatBytes.byteLength}`);
  }

  const view = new DataView(splatBytes.buffer, splatBytes.byteOffset, splatBytes.byteLength);
  const count = splatBytes.byteLength / recordBytes;
  const mins = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const maxs = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  let nonFinite = 0;
  let positiveAlpha = 0;
  let positiveScale = 0;
  for (let index = 0; index < count; index += 1) {
    const base = index * recordBytes;
    for (let axis = 0; axis < 3; axis += 1) {
      const position = view.getFloat32(base + axis * 4, true);
      const scale = view.getFloat32(base + 12 + axis * 4, true);
      if (!Number.isFinite(position) || !Number.isFinite(scale)) {
        nonFinite += 1;
        continue;
      }
      mins[axis] = Math.min(mins[axis], position);
      maxs[axis] = Math.max(maxs[axis], position);
      if (scale > 0) {
        positiveScale += 1;
      }
    }
    if (view.getUint8(base + 27) > 0) {
      positiveAlpha += 1;
    }
  }

  return {
    count,
    byteLength: splatBytes.byteLength,
    boundsMin: mins,
    boundsMax: maxs,
    nonFinite,
    positiveAlpha,
    positiveScale,
  };
}

function relDiff(a, b) {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom;
}

test('burn_synth wasm API page exposes asset-specific TripoSplat controls', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/www/synth_api.html?model_source=local&sw=off', {
    waitUntil: 'domcontentloaded',
  });

  await expect(page.locator('#synthesis-model')).toHaveValue('triposg');
  await expect(page.locator('#synthesis-model option')).toHaveText(['triposg', 'triposplat']);
  await expect(page.locator('#asset-format')).toHaveValue('splat');
  await expect(page.locator('#asset-format')).toBeDisabled();
  await expect(page.locator('#run-infer')).toHaveText('Infer GLB');
  await expect(page.locator('#download')).toHaveAttribute('download', 'burn_synth_output.glb');

  await page.selectOption('#synthesis-model', 'triposplat');

  await expect(page.locator('#asset-format')).toBeEnabled();
  await expect(page.locator('#run-infer')).toHaveText('Infer SPLAT');
  await expect(page.locator('#download')).toHaveAttribute('download', 'burn_synth_output.splat');
  await expect
    .poll(
      async () => {
        const url = new URL(page.url());
        return {
          synthesisModel: url.searchParams.get('synthesis_model'),
          synthesis: url.searchParams.get('synthesis'),
          assetFormat: url.searchParams.get('asset_format'),
        };
      },
      { timeout: 10000 },
    )
    .toEqual({
      synthesisModel: 'triposplat',
      synthesis: null,
      assetFormat: null,
    });

  await page.selectOption('#asset-format', 'ply');

  await expect(page.locator('#run-infer')).toHaveText('Infer PLY');
  await expect(page.locator('#download')).toHaveAttribute('download', 'burn_synth_output.ply');
  await expect
    .poll(
      async () => {
        const url = new URL(page.url());
        return {
          synthesisModel: url.searchParams.get('synthesis_model'),
          assetFormat: url.searchParams.get('asset_format'),
        };
      },
      { timeout: 10000 },
    )
    .toEqual({
      synthesisModel: 'triposplat',
      assetFormat: 'ply',
    });

  await page.selectOption('#synthesis-model', 'triposg');

  await expect(page.locator('#asset-format')).toBeDisabled();
  await expect(page.locator('#run-infer')).toHaveText('Infer GLB');
  await expect(page.locator('#download')).toHaveAttribute('download', 'burn_synth_output.glb');
  await expect
    .poll(
      async () => {
        const url = new URL(page.url());
        return {
          synthesisModel: url.searchParams.get('synthesis_model'),
          assetFormat: url.searchParams.get('asset_format'),
        };
      },
      { timeout: 10000 },
    )
    .toEqual({
      synthesisModel: null,
      assetFormat: null,
    });
});

test('burn_synth wasm parts-based web inference produces a GLB', async ({ page }) => {
  test.setTimeout(1800000);
  test.skip(
    process.env.BURN_SYNTH_WEB_TRIPOSG_SMOKE !== '1',
    'set BURN_SYNTH_WEB_TRIPOSG_SMOKE=1 to enable TripoSG wasm GLB smoke',
  );

  const pageErrors = [];
  const consoleErrors = [];
  const modelRequests = [];
  const partRequests = new Set();
  const legacyShardManifestRequests = new Set();
  const failedModelResponses = [];
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
  page.on('crash', () => pageErrors.push('page crashed'));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('request', (request) => {
    const url = normalizeModelUrl(request.url());
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
      if (url.endsWith('.bpk.parts.json') || url.includes('.bpk.part-')) {
        partRequests.add(url);
      }
      if (url.endsWith('.bpk.shards.json')) {
        legacyShardManifestRequests.add(url);
      }
    }
  });
  page.on('response', (response) => {
    const url = normalizeModelUrl(response.url());
    if (url.includes('/assets/models/') && response.status() >= 400) {
      failedModelResponses.push(`${response.status()} ${url}`);
    }
  });

  await page.goto('http://127.0.0.1:4173/www/synth_api.html?model_source=local&sw=off', {
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
      const imageResp = await fetch('/docs/output_chair_bg_removed.png');
      if (!imageResp.ok) {
        throw new Error(`failed to fetch docs/output_chair_bg_removed.png: ${imageResp.status}`);
      }
      const imageBytes = new Uint8Array(await imageResp.arrayBuffer());
      const options = new window.__burnSynthWasm.WasmInferOptions();
      options.set_num_steps(2);
      options.set_num_tokens(512);
      options.set_resolution(15);
      options.set_faces(1000);
      options.set_seed(42n);
      options.set_backend('wgpu');
      options.set_dino_backend('auto');
      options.set_rmbg_model('none');
      options.set_weights_precision('f32');
      const inferPromise = window.__burnSynthWasm
        .infer_glb_from_image_bytes_with_options(imageBytes, 'output_chair_bg_removed.png', options)
        .then((glb) => ({
          ok: true,
          glbBytes: glb.byteLength,
          glbData: Array.from(glb),
          elapsedMs: performance.now() - started,
        }))
        .catch((error) => ({
          ok: false,
          error: error?.stack ?? String(error),
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
        error: error?.stack ?? String(error),
        elapsedMs: performance.now() - started,
      };
    }
  });

  expect(inference.ok, `inference failed: ${JSON.stringify(inference)}`).toBe(true);
  expect(inference.glbBytes).toBeGreaterThan(0);
  const wasmGlbBytes = Uint8Array.from(inference.glbData);
  const tmpDir = process.env.BURN_SYNTH_WEB_TMP_DIR;
  if (tmpDir) {
    fs.mkdirSync(tmpDir, { recursive: true });
    fs.writeFileSync(path.join(tmpDir, 'wasm_output.glb'), Buffer.from(wasmGlbBytes));
  }
  const wasmStats = parseGlbStats(wasmGlbBytes);
  expect(wasmStats.vertexCount).toBeGreaterThan(0);
  expect(wasmStats.faceCount).toBeGreaterThan(0);

  const cacheSummary = await page.evaluate(async () => {
    const cacheNames = await caches.keys();
    const cacheName = cacheNames.find((name) => name.startsWith('burn-synth-bpk-')) ?? null;
    if (!cacheName) {
      return {
        cacheName: null,
        cachedModelEntries: 0,
      };
    }
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    const cachedModelEntries = requests.filter((request) => {
      const pathname = new URL(request.url).pathname;
      return (
        pathname.endsWith('.bpk') ||
        pathname.endsWith('.bpk.parts.json') ||
        pathname.includes('.bpk.part-')
      );
    }).length;
    return { cacheName, cachedModelEntries };
  });
  expect(cacheSummary.cacheName, 'expected burn_synth model cache storage to be created').not.toBeNull();
  expect(
    cacheSummary.cachedModelEntries,
    'expected cached .bpk model entries after wasm inference',
  ).toBeGreaterThan(0);

  const nativeRefGlb = process.env.BURN_SYNTH_NATIVE_REF_GLB;
  if (nativeRefGlb && fs.existsSync(nativeRefGlb)) {
    const nativeStats = parseGlbStats(fs.readFileSync(nativeRefGlb));
    if (tmpDir) {
      fs.writeFileSync(
        path.join(tmpDir, 'web_parity_stats.json'),
        JSON.stringify({ wasm: wasmStats, native: nativeStats }, null, 2),
      );
    }
    expect(
      relDiff(wasmStats.vertexCount, nativeStats.vertexCount),
      `vertex count drift (wasm=${wasmStats.vertexCount}, native=${nativeStats.vertexCount})`,
    ).toBeLessThan(0.15);
    expect(
      relDiff(wasmStats.faceCount, nativeStats.faceCount),
      `face count drift (wasm=${wasmStats.faceCount}, native=${nativeStats.faceCount})`,
    ).toBeLessThan(0.20);
    for (let axis = 0; axis < 3; axis += 1) {
      const wasmExtent = wasmStats.boundsMax[axis] - wasmStats.boundsMin[axis];
      const nativeExtent = nativeStats.boundsMax[axis] - nativeStats.boundsMin[axis];
      expect(
        relDiff(wasmExtent, nativeExtent),
        `extent axis ${axis} drift (wasm=${wasmExtent}, native=${nativeExtent})`,
      ).toBeLessThan(0.45);
    }
  }
  expect(modelRequests.length).toBeGreaterThan(0);
  expect(partRequests.size).toBeGreaterThan(0);
  expect(
    Array.from(legacyShardManifestRequests),
    `unexpected legacy shard manifest requests in parts-first loader: ${Array.from(legacyShardManifestRequests).join(' | ')}`,
  ).toEqual([]);
  expect(
    modelRequests.some((url) => url.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json')),
    'expected dedicated DINOv2 config request',
  ).toBe(true);
  const missingDedicatedDinoConfig = failedModelResponses.some(
    (entry) =>
      (entry.includes('404 ') || entry.includes('403 ')) &&
      entry.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json'),
  );
  expect(
    missingDedicatedDinoConfig,
    'dedicated DINOv2 config must be bundled and fetchable',
  ).toBe(false);
  expect(
    modelRequests.some((url) => url.includes('/assets/models/MIDI-3D/image_encoder_2/config.json')),
    'legacy image_encoder_2 config should not be used when dedicated DINOv2 config exists',
  ).toBe(false);
  const requestedDinoF16 = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/image_encoder_dinov2/model_f16.bpk.parts.json'),
  );
  const requestedDinoF32 = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/image_encoder_dinov2/model.bpk.parts.json'),
  );
  expect(
    requestedDinoF16 || requestedDinoF32,
    'expected DINOv2 burnpack requests (f32 primary, f16 fallback)',
  ).toBe(true);
  const requestedTransformerF32PartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model.bpk.parts.json'),
  );
  const requestedTransformerF16PartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model_f16.bpk.parts.json'),
  );
  expect(
    requestedTransformerF32PartsManifest || requestedTransformerF16PartsManifest,
    'expected transformer parts manifest request (f32 or f16)',
  ).toBe(true);
  const requestedVaeF32PartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/vae/diffusion_pytorch_model.bpk.parts.json'),
  );
  const requestedVaeF16PartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/vae/diffusion_pytorch_model_f16.bpk.parts.json'),
  );
  expect(
    requestedVaeF32PartsManifest || requestedVaeF16PartsManifest,
    'expected VAE parts manifest request (f32 or f16)',
  ).toBe(true);
  const rmbgModelRequests = modelRequests.filter((url) =>
    url.includes('/assets/models/RMBG-1.4/'),
  );
  expect(
    rmbgModelRequests,
    `rmbg_model=none GLB smoke should not request RMBG artifacts: ${rmbgModelRequests.join(' | ')}`,
  ).toEqual([]);
  const nonBenignFailedModelResponses = failedModelResponses.filter(
    (entry) =>
      !(
        entry.includes('404 ') &&
        (
          entry.includes('.bpk.parts.json') ||
          entry.includes('.bpk.part-') ||
          entry.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model.bpk') ||
          entry.includes('/assets/models/MIDI-3D/vae/diffusion_pytorch_model.bpk') ||
          entry.includes('/assets/models/MIDI-3D/image_encoder_dinov2/model.bpk') ||
          entry.includes('/assets/models/RMBG-1.4/model.bpk') ||
          entry.includes('/assets/models/RMBG-1.4/model_f16.bpk') ||
          entry.includes('/assets/models/MIDI-3D/image_encoder_dinov2/config.json')
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

test('burn_synth wasm TripoSplat inference produces valid splat output', async ({ page }) => {
  test.setTimeout(1800000);
  test.skip(
    process.env.BURN_SYNTH_WEB_TRIPOSPLAT_SMOKE !== '1',
    'set BURN_SYNTH_WEB_TRIPOSPLAT_SMOKE=1 to enable TripoSplat wasm smoke',
  );

  const expectedSplats = 32768;
  const weightsPrecision = process.env.BURN_SYNTH_WEB_TRIPOSPLAT_PRECISION || 'auto';
  const inferenceTimeoutMs = Number.parseInt(
    process.env.BURN_SYNTH_WEB_TRIPOSPLAT_TIMEOUT_MS || '240000',
    10,
  );
  const pageErrors = [];
  const consoleMessages = [];
  const consoleErrors = [];
  const modelRequests = [];
  const failedModelResponses = [];
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
  page.on('crash', () => pageErrors.push('page crashed'));
  page.on('console', (msg) => {
    consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  page.on('request', (request) => {
    const url = normalizeModelUrl(request.url());
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
    }
  });
  page.on('response', (response) => {
    const url = normalizeModelUrl(response.url());
    if (url.includes('/assets/models/') && response.status() >= 400) {
      failedModelResponses.push(`${response.status()} ${url}`);
    }
  });

  await page.goto('http://127.0.0.1:4173/www/synth_api.html?model_source=local&sw=off', {
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
  if (!webgpu.shaderF16) {
    const tmpDir = process.env.BURN_SYNTH_WEB_TMP_DIR;
    if (tmpDir) {
      fs.mkdirSync(tmpDir, { recursive: true });
      fs.writeFileSync(
        path.join(tmpDir, 'wasm_triposplat_result.json'),
        JSON.stringify(
          {
            inference: {
              ok: false,
              skipped: true,
              reason:
                'TripoSplat wasm requires WebGPU shader-f16; f32 browser decode exceeds WebGPU memory limits.',
            },
            weightsPrecision,
            webgpu,
            modelRequests,
            failedModelResponses,
            pageErrors,
          },
          null,
          2,
        ),
      );
      fs.writeFileSync(
        path.join(tmpDir, 'wasm_triposplat_console.log'),
        `${consoleMessages.join('\n')}\n`,
      );
    }
    test.skip(
      true,
      'TripoSplat wasm requires WebGPU shader-f16; this adapter only exposes f32.',
    );
  }

  const inferenceStartedAt = Date.now();
  const inferencePromise = page.evaluate(async ({ expectedSplats, inferenceTimeoutMs, weightsPrecision }) => {
    const started = performance.now();
    try {
      const imageResp = await fetch('/docs/output_chair_bg_removed.png');
      if (!imageResp.ok) {
        throw new Error(`failed to fetch docs/output_chair_bg_removed.png: ${imageResp.status}`);
      }
      const imageBytes = new Uint8Array(await imageResp.arrayBuffer());
      const options = new window.__burnSynthWasm.WasmInferOptions();
      options.set_synthesis_model('triposplat');
      options.set_rmbg_model('none');
      options.set_backend('wgpu');
      options.set_dino_backend('auto');
      options.set_weights_precision(weightsPrecision);
      options.set_num_steps(5);
      options.set_guidance_scale(3.0);
      options.set_triposplat_num_gaussians(expectedSplats);
      options.set_triposplat_shift(3.0);
      options.set_triposplat_erode_radius(1);
      options.set_seed(42n);

      const inferPromise = window.__burnSynthWasm
        .infer_splat_from_image_bytes_with_options(imageBytes, 'output_chair_bg_removed.png', options)
        .then((splat) => ({
          ok: true,
          splatBytes: splat.byteLength,
          splatData: Array.from(splat),
          elapsedMs: performance.now() - started,
        }))
        .catch((error) => ({
          ok: false,
          error: error?.stack ?? String(error),
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
          Number.isFinite(inferenceTimeoutMs) && inferenceTimeoutMs > 0
            ? inferenceTimeoutMs
            : 240000,
        ),
      );
      return await Promise.race([inferPromise, timeoutPromise]);
    } catch (error) {
      return {
        ok: false,
        error: error?.stack ?? String(error),
        elapsedMs: performance.now() - started,
      };
    }
  }, { expectedSplats, inferenceTimeoutMs, weightsPrecision });
  const nodeWatchdogMs =
    (Number.isFinite(inferenceTimeoutMs) && inferenceTimeoutMs > 0 ? inferenceTimeoutMs : 240000) +
    60000;
  const inference = await Promise.race([
    inferencePromise,
    new Promise((resolve) =>
      setTimeout(
        () =>
          resolve({
            ok: false,
            timeout: true,
            nodeWatchdog: true,
            elapsedMs: Date.now() - inferenceStartedAt,
          }),
        nodeWatchdogMs,
      ),
    ),
  ]);

  const tmpDir = process.env.BURN_SYNTH_WEB_TMP_DIR;
  if (tmpDir) {
    fs.mkdirSync(tmpDir, { recursive: true });
    const inferenceSummary = { ...inference };
    delete inferenceSummary.splatData;
    fs.writeFileSync(
      path.join(tmpDir, 'wasm_triposplat_result.json'),
      JSON.stringify(
        {
          inference: inferenceSummary,
          weightsPrecision,
          webgpu,
          modelRequests,
          failedModelResponses,
          pageErrors,
        },
        null,
        2,
      ),
    );
    fs.writeFileSync(
      path.join(tmpDir, 'wasm_triposplat_console.log'),
      `${consoleMessages.join('\n')}\n`,
    );
  }

  expect(inference.ok, `TripoSplat wasm inference failed: ${JSON.stringify(inference)}`).toBe(true);
  expect(inference.splatBytes).toBe(expectedSplats * 32);
  const wasmSplatBytes = Uint8Array.from(inference.splatData);
  const stats = parseSplatStats(wasmSplatBytes);
  expect(stats.count).toBe(expectedSplats);
  expect(stats.nonFinite).toBe(0);
  expect(stats.positiveAlpha).toBeGreaterThan(0);
  expect(stats.positiveScale).toBeGreaterThan(expectedSplats * 2);
  for (let axis = 0; axis < 3; axis += 1) {
    expect(Number.isFinite(stats.boundsMin[axis])).toBe(true);
    expect(Number.isFinite(stats.boundsMax[axis])).toBe(true);
    expect(stats.boundsMax[axis]).toBeGreaterThan(stats.boundsMin[axis]);
  }

  if (tmpDir) {
    fs.writeFileSync(path.join(tmpDir, 'wasm_triposplat_output.splat'), Buffer.from(wasmSplatBytes));
    fs.writeFileSync(
      path.join(tmpDir, 'wasm_triposplat_stats.json'),
      JSON.stringify({ stats, elapsedMs: inference.elapsedMs, weightsPrecision }, null, 2),
    );
    fs.writeFileSync(
      path.join(tmpDir, 'wasm_triposplat_console.log'),
      `${consoleMessages.join('\n')}\n`,
    );
  }

  const requiredManifestPatterns = [
    /\/assets\/models\/TripoSplat\/clip_vision\/dino_v3_vit_h(_f16)?\.bpk\.parts\.json$/,
    /\/assets\/models\/TripoSplat\/vae\/flux2_vae_encoder(_f16)?\.bpk\.parts\.json$/,
    /\/assets\/models\/TripoSplat\/diffusion_models\/triposplat_flow(_f16)?\.bpk\.parts\.json$/,
    /\/assets\/models\/TripoSplat\/vae\/triposplat_vae_decoder(_f16)?\.bpk\.parts\.json$/,
  ];
  for (const pattern of requiredManifestPatterns) {
    expect(
      modelRequests.some((url) => pattern.test(url)),
      `expected manifest matching ${pattern}, saw: ${modelRequests.join(' | ')}`,
    ).toBe(true);
  }
  const legacyShardManifestRequests = modelRequests.filter((url) =>
    url.endsWith('.bpk.shards.json'),
  );
  expect(
    legacyShardManifestRequests,
    `unexpected legacy shard manifest requests in TripoSplat wasm loader: ${legacyShardManifestRequests.join(' | ')}`,
  ).toEqual([]);
  const nonBenignFailedModelResponses = failedModelResponses.filter(
    (entry) =>
      !(
        entry.includes('404 ') &&
        (
          entry.includes('.bpk.parts.json') ||
          entry.includes('.bpk.part-') ||
          entry.includes('/assets/models/RMBG-1.4/model.bpk') ||
          entry.includes('/assets/models/RMBG-1.4/model_f16.bpk')
        )
      ),
  );
  expect(
    nonBenignFailedModelResponses,
    `failed model responses: ${nonBenignFailedModelResponses.join(' | ')}`,
  ).toEqual([]);
  const rmbgModelRequests = modelRequests.filter((url) =>
    url.includes('/assets/models/RMBG-1.4/'),
  );
  expect(
    rmbgModelRequests,
    `TripoSplat rmbg_model=none should not request RMBG artifacts: ${rmbgModelRequests.join(' | ')}`,
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

test('burn_synth wasm trellis inference path can run end-to-end', async ({ page }) => {
  test.setTimeout(1800000);
  test.skip(
    process.env.BURN_SYNTH_WEB_TRELLIS_SMOKE !== '1',
    'set BURN_SYNTH_WEB_TRELLIS_SMOKE=1 to enable Trellis wasm smoke',
  );

  const modelRequests = [];
  const failedModelResponses = [];
  const normalizeModelUrl = (url) => {
    if (url.includes('/www/assets/models/')) {
      return url.replace('/www/assets/models/', '/assets/models/');
    }
    if (url.includes('/www/assets/')) {
      return url.replace('/www/assets/', '/assets/models/');
    }
    return url;
  };

  page.on('request', (request) => {
    const url = normalizeModelUrl(request.url());
    if (url.includes('/assets/models/')) {
      modelRequests.push(url);
    }
  });
  page.on('response', (response) => {
    const url = normalizeModelUrl(response.url());
    if (url.includes('/assets/models/') && response.status() >= 400) {
      failedModelResponses.push(`${response.status()} ${url}`);
    }
  });

  await page.goto('http://127.0.0.1:4173/www/synth_api.html?model_source=local&sw=off', {
    waitUntil: 'domcontentloaded',
  });
  await page.click('#boot-start');
  await expect(page.locator('#boot-text')).toHaveText(/module ready/i, {
    timeout: 600000,
  });

  const inference = await page.evaluate(async () => {
    const started = performance.now();
    try {
      const imageResp = await fetch('/docs/input_chair.jpg');
      if (!imageResp.ok) {
        throw new Error(`failed to fetch docs/input_chair.jpg: ${imageResp.status}`);
      }
      const imageBytes = new Uint8Array(await imageResp.arrayBuffer());
      const options = new window.__burnSynthWasm.WasmInferOptions();
      options.set_backend('wgpu');
      options.set_synthesis_model('trellis');
      options.set_rmbg_model('rmbg14');
      options.set_quality('fast');
      options.set_seed(42n);

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
          900000,
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

  expect(inference.ok, `trellis wasm inference failed: ${JSON.stringify(inference)}`).toBe(true);
  expect(inference.glbBytes).toBeGreaterThan(0);
  expect(
    modelRequests.some((url) => url.includes('/assets/models/TRELLIS.2-4B/')),
    `expected TRELLIS model requests, saw: ${modelRequests.join(' | ')}`,
  ).toBe(true);
  expect(
    failedModelResponses,
    `failed model responses: ${failedModelResponses.join(' | ')}`,
  ).toEqual([]);
});
