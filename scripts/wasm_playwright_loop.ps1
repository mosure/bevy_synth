param(
    [string]$Package = "bevy_synth",
    [string]$Target = "wasm32-unknown-unknown",
    [string]$Profile = "wasm-release",
    [string]$OutDir = "www/out",
    [string]$PlaywrightDir = "tmp/playwright-smoke",
    [int]$PlaywrightTimeoutSec = 900,
    [string]$LogDir = "tmp/wasm-loop-logs"
)

$ErrorActionPreference = "Stop"

function Clear-RustFlagEnvironment {
    $varsToClear = @(
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
        "RUSTDOCFLAGS"
    )
    foreach ($name in $varsToClear) {
        if (Test-Path "Env:$name") {
            Remove-Item "Env:$name"
        }
    }
}

function Require-Path([string]$PathToCheck, [string]$ErrorMessage) {
    if (-not (Test-Path $PathToCheck)) {
        throw $ErrorMessage
    }
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$buildLog = Join-Path $LogDir "01-cargo-build.log"
$bindgenLog = Join-Path $LogDir "02-wasm-bindgen.log"
$playwrightOutLog = Join-Path $LogDir "03-playwright.stdout.log"
$playwrightErrLog = Join-Path $LogDir "03-playwright.stderr.log"

Write-Host "[wasm-loop] clearing rust flag environment variables"
Clear-RustFlagEnvironment

Write-Host "[wasm-loop] cargo build ($Package, $Target, $Profile)"
& cargo build -p $Package --target $Target --profile $Profile *> $buildLog
if ($LASTEXITCODE -ne 0) {
    throw "[wasm-loop] cargo build failed (see $buildLog)"
}

$artifactProfile = if ($Profile -eq "dev") { "debug" } else { $Profile }
$wasmPath = "target/$Target/$artifactProfile/$Package.wasm"
Require-Path $wasmPath "[wasm-loop] wasm artifact not found at $wasmPath"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
Write-Host "[wasm-loop] wasm-bindgen ($wasmPath -> $OutDir)"
& wasm-bindgen $wasmPath --out-dir $OutDir --target web --no-typescript *> $bindgenLog
if ($LASTEXITCODE -ne 0) {
    throw "[wasm-loop] wasm-bindgen failed (see $bindgenLog)"
}

Require-Path $PlaywrightDir "[wasm-loop] Playwright harness missing at $PlaywrightDir"
Remove-Item $playwrightOutLog -ErrorAction SilentlyContinue
Remove-Item $playwrightErrLog -ErrorAction SilentlyContinue

$playwrightCmd = "cd /d `"$((Resolve-Path $PlaywrightDir).Path)`" && npx playwright test --config playwright.config.mjs --workers=1 --reporter=line"
Write-Host "[wasm-loop] playwright smoke (timeout ${PlaywrightTimeoutSec}s)"
$p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $playwrightCmd -PassThru -WindowStyle Hidden -RedirectStandardOutput $playwrightOutLog -RedirectStandardError $playwrightErrLog
$completed = $p.WaitForExit($PlaywrightTimeoutSec * 1000)
if (-not $completed) {
    Stop-Process -Id $p.Id -Force
    throw "[wasm-loop] playwright timed out after ${PlaywrightTimeoutSec}s (stdout: $playwrightOutLog, stderr: $playwrightErrLog)"
}
if ($p.ExitCode -ne 0) {
    throw "[wasm-loop] playwright failed with exit code $($p.ExitCode) (stdout: $playwrightOutLog, stderr: $playwrightErrLog)"
}

Write-Host "[wasm-loop] success"
Write-Host "[wasm-loop] logs:"
Write-Host "  - $buildLog"
Write-Host "  - $bindgenLog"
Write-Host "  - $playwrightOutLog"
Write-Host "  - $playwrightErrLog"
