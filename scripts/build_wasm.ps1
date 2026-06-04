$ErrorActionPreference = "Stop"

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

cargo build --target wasm32-unknown-unknown --profile wasm-release
