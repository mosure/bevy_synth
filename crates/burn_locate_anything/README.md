# burn_locate_anything

Burn-side runtime and import scaffolding for NVIDIA LocateAnything visual grounding.

The public runtime surface is Burn-native. Upstream Python/Torch comparison tooling is kept as
repository validation support and is excluded from published crates.

`LocateAnythingRuntime` prepares images and text prompts, runs the MoonViT/projector and Qwen
decoder stack, and decodes LocateAnything box outputs. Full detection is opt-in with
`allow_experimental_native_detect=true` while broader checkpoint coverage is still being validated.
