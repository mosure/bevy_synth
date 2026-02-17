#[cfg(target_arch = "wasm32")]
// Generic loader precision helper (memory-oriented).
// TripoSG runtime parity uses burn_tripo::pipeline::runtime_parity policy helpers instead.
pub(crate) use burn_synth::model_loader::prefer_f16_burnpack;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use burn_synth::model_loader::{
    load_optional_text_candidates_from_root, load_optional_text_from_root,
    resolve_burnpack_asset_path_from_root, resolve_burnpack_asset_path_from_root_with_preference,
};

#[cfg(all(not(target_arch = "wasm32"), test, feature = "wgpu"))]
pub(crate) use burn_synth::model_loader::{
    burnpack_manifest_candidates, candidate_burnpack_names, parse_shard_manifest_bytes,
};
