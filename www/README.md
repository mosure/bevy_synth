# lattice web 🕊️🔥🎛️🎹

- `cargo build --release --target wasm32-unknown-unknown -p bevy_synth`
- `wasm-bindgen --out-dir ./www/out --target web target/wasm32-unknown-unknown/release/bevy_synth.wasm`
- bundle model assets into `www/assets/models`:
  - `pwsh ./scripts/bundle_web_assets.ps1`
- preview the bundle plan without copying files:
  - `pwsh ./scripts/bundle_web_assets.ps1 -DryRun`
- exclude RMBG-2.0 from the bundle (optional):
  - `pwsh ./scripts/bundle_web_assets.ps1 -ExcludeRmbg2`
- the bundler only copies deployment-needed model files:
  - only `*.bpk`
  - all JSON and `*.safetensors` are excluded
