"use strict";

/**
 * Burn Synth model-weight Service Worker.
 *
 * Source-of-truth location:
 * - This file lives under `www/` so local web pages (`/www/index.html`,
 *   `/www/synth_api.html`) can register it directly during local development.
 *
 * Deployment location:
 * - GitHub Pages deploy copies this file to the published site root as
 *   `burn_synth_bpk_sw.js`.
 *
 * Why we copy to root at deploy time:
 * - Service worker scope is constrained by script URL path unless the host sets
 *   `Service-Worker-Allowed`.
 * - GitHub Pages does not provide per-file header control for that override.
 * - Root pages need a root-scoped worker, so the deployed copy must be at site root.
 */

const CACHE_PREFIX = "burn-synth-bpk";
const CACHE_VERSION = "v2";
const CACHE_NAME = `${CACHE_PREFIX}-${CACHE_VERSION}`;
const ABERRATION_MODEL_ORIGIN = "https://aberration.technology";

function isCacheableBpkPath(pathname) {
  return (
    pathname.endsWith(".bpk") ||
    pathname.endsWith(".bpk.parts.json") ||
    pathname.includes(".bpk.part-")
  );
}

function isCacheableModelRequest(requestUrl) {
  const url = new URL(requestUrl);
  const pathname = url.pathname;
  if (url.origin === self.location.origin) {
    // Same-origin cache coverage:
    // - local `/www/assets/...` during local dev/test
    // - deployed `/assets/...` bundles
    const isModelAssetPath =
      pathname.includes("/assets/models/") ||
      pathname.includes("/www/assets/") ||
      pathname.includes("/assets/");
    return isModelAssetPath && isCacheableBpkPath(pathname);
  }
  // Cross-origin cache coverage for production model host.
  const isAberrationModelPath =
    url.origin === ABERRATION_MODEL_ORIGIN && pathname.startsWith("/model/");
  return isAberrationModelPath && isCacheableBpkPath(pathname);
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((name) => name.startsWith(`${CACHE_PREFIX}-`) && name !== CACHE_NAME)
          .map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }
  if (!isCacheableModelRequest(request.url)) {
    return;
  }

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) {
        return cached;
      }

      const response = await fetch(request);
      if (response && (response.ok || response.type === "opaque")) {
        await cache.put(request, response.clone());
      }
      return response;
    })(),
  );
});
