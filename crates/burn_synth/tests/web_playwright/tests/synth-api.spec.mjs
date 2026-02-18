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

function relDiff(a, b) {
  const denom = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) / denom;
}

test('burn_synth wasm parts-based web inference produces a GLB', async ({ page }) => {
  test.setTimeout(1800000);

  const pageErrors = [];
  const consoleErrors = [];
  const modelRequests = [];
  const partRequests = new Set();
  const shardManifestRequests = new Set();
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
      if (url.endsWith('.bpk.parts.json') || url.includes('.bpk.part-')) {
        partRequests.add(url);
      }
      if (url.endsWith('.bpk.shards.json')) {
        shardManifestRequests.add(url);
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
      options.set_seed(42n);
      options.set_backend('wgpu');
      options.set_dino_backend('auto');
      const inferPromise = window.__burnSynthWasm
        .infer_glb_from_image_bytes_with_options(imageBytes, 'input_chair.jpg', options)
        .then((glb) => ({
          ok: true,
          glbBytes: glb.byteLength,
          glbData: Array.from(glb),
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
  const wasmGlbBytes = Uint8Array.from(inference.glbData);
  const tmpDir = process.env.BURN_SYNTH_WEB_TMP_DIR;
  if (tmpDir) {
    fs.mkdirSync(tmpDir, { recursive: true });
    fs.writeFileSync(path.join(tmpDir, 'wasm_output.glb'), Buffer.from(wasmGlbBytes));
  }
  const wasmStats = parseGlbStats(wasmGlbBytes);
  expect(wasmStats.vertexCount).toBeGreaterThan(0);
  expect(wasmStats.faceCount).toBeGreaterThan(0);

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
    ).toBeLessThan(0.15);
    for (let axis = 0; axis < 3; axis += 1) {
      expect(
        Math.abs(wasmStats.boundsMin[axis] - nativeStats.boundsMin[axis]),
        `boundsMin axis ${axis} drift`,
      ).toBeLessThan(0.08);
      expect(
        Math.abs(wasmStats.boundsMax[axis] - nativeStats.boundsMax[axis]),
        `boundsMax axis ${axis} drift`,
      ).toBeLessThan(0.08);
    }
  }
  expect(modelRequests.length).toBeGreaterThan(0);
  expect(partRequests.size).toBeGreaterThan(0);
  expect(
    Array.from(shardManifestRequests),
    `unexpected shard manifest requests in parts-first loader: ${Array.from(shardManifestRequests).join(' | ')}`,
  ).toEqual([]);
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
  expect(
    requestedTransformerF32PartsManifest,
    'expected fp32 transformer parts manifest request for wasm fp32 runtime',
  ).toBe(true);
  const requestedVaeF32PartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/MIDI-3D/vae/diffusion_pytorch_model.bpk.parts.json'),
  );
  expect(
    requestedVaeF32PartsManifest,
    'expected fp32 VAE parts manifest request for wasm fp32 runtime',
  ).toBe(true);
  const requestedRmbgPartsManifest = modelRequests.some((url) =>
    url.includes('/assets/models/RMBG-1.4/model_f16.bpk.parts.json') ||
    url.includes('/assets/models/RMBG-1.4/model.bpk.parts.json'),
  );
  expect(
    requestedRmbgPartsManifest,
    'expected RMBG parts manifest request for wasm runtime',
  ).toBe(true);
  const requestedTransformerF16Parts = modelRequests.some(
    (url) =>
      url.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model_f16.bpk.parts.json') ||
      url.includes('/assets/models/MIDI-3D/transformer/diffusion_pytorch_model_f16.bpk.part-'),
  );
  expect(
    requestedTransformerF16Parts,
    'unexpected fp16 transformer parts requests in wasm fp32 run',
  ).toBe(false);
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
