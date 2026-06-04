param(
    [string]$Target = "wasm32-unknown-unknown",
    [string]$Profile = "wasm-release",
    [string]$OutDir = "www/out"
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

function Build-WasmPackage {
    param(
        [string]$Package,
        [string[]]$Features = @()
    )
    $featureArgs = @()
    if ($Features.Count -gt 0) {
        $featureArgs = @("--features", ($Features -join ","))
    }
    Write-Host "Building $Package for $Target ($Profile)..."
    cargo build -p $Package --target $Target --profile $Profile @featureArgs

    $artifactProfile = if ($Profile -eq "dev") { "debug" } else { $Profile }
    $wasm = "target/$Target/$artifactProfile/$Package.wasm"
    if (-not (Test-Path $wasm)) {
        throw "WASM artifact not found at $wasm"
    }

    Write-Host "Running wasm-bindgen for $Package..."
    wasm-bindgen $wasm --out-dir $OutDir --target web --no-typescript
}

Clear-RustFlagEnvironment
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Build-WasmPackage -Package "bevy_synth"

try {
    Build-WasmPackage -Package "burn_synth" -Features @("wasm-api", "wasm-api-wgpu")
}
catch {
    Write-Warning "burn_synth wasm-api-wgpu build failed; retrying burn_synth with cpu-only wasm-api."
    Build-WasmPackage -Package "burn_synth" -Features @("wasm-api")
}

Write-Host "Web artifacts written to $OutDir"
