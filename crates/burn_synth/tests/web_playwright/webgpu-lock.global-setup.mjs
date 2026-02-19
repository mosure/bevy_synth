import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const DEFAULT_TIMEOUT_SEC = 7200;
const LOCK_WAIT_MS = 2000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseTimeoutSeconds(raw) {
  const parsed = Number.parseInt(raw ?? '', 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TIMEOUT_SEC;
  }
  return parsed;
}

async function tryAcquireLockDir(lockDir) {
  try {
    await fs.mkdir(lockDir);
    return true;
  } catch (error) {
    if (error && error.code === 'EEXIST') {
      return false;
    }
    throw error;
  }
}

export default async function globalSetup() {
  if (process.env.BURN_SYNTH_WEBGPU_LOCK_HELD === '1') {
    return undefined;
  }

  const thisDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(thisDir, '../../../../');
  const lockDir =
    process.env.BURN_SYNTH_WEBGPU_LOCK_DIR ??
    path.join(repoRoot, 'target', '.webgpu-test.lockdir');
  const timeoutSec = parseTimeoutSeconds(
    process.env.BURN_SYNTH_WEBGPU_LOCK_TIMEOUT_SEC,
  );
  const deadline = Date.now() + timeoutSec * 1000;

  while (!(await tryAcquireLockDir(lockDir))) {
    if (Date.now() >= deadline) {
      throw new Error(
        `[web-e2e] timed out waiting for WebGPU lock after ${timeoutSec}s: ${lockDir}`,
      );
    }
    await sleep(LOCK_WAIT_MS);
  }

  const owner = {
    pid: process.pid,
    acquiredAt: new Date().toISOString(),
    runner: 'playwright',
  };
  await fs.writeFile(
    path.join(lockDir, 'owner.json'),
    JSON.stringify(owner),
    'utf8',
  );
  process.stdout.write(`[web-e2e] acquired playwright WebGPU lock: ${lockDir}\n`);

  return async () => {
    try {
      await fs.rm(lockDir, { recursive: true, force: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(
        `[web-e2e] failed to release WebGPU lock ${lockDir}: ${message}\n`,
      );
    }
  };
}
