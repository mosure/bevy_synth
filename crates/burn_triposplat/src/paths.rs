use std::path::{Path, PathBuf};

use crate::artifact::{TripoSplatArtifactSet, TripoSplatBurnpackPrecision};

pub fn resolve_triposplat_weights_root(candidate: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = candidate {
        return validate_or_report(path);
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("assets/models/TripoSplat"),
        manifest
            .parent()
            .and_then(|p| p.parent())
            .map(|workspace| workspace.join("assets/models/TripoSplat"))
            .unwrap_or_else(|| manifest.join("assets/models/TripoSplat")),
        PathBuf::from("assets/models/TripoSplat"),
        PathBuf::from("www/assets/models/TripoSplat"),
    ];
    for path in candidates {
        if has_any_triposplat_artifact(&path) {
            return validate_or_report(&path);
        }
    }
    Err("TripoSplat weights root not found; pass --triposplat-weights-root".to_string())
}

fn validate_or_report(path: &Path) -> Result<PathBuf, String> {
    let f16 = TripoSplatArtifactSet::new(path, TripoSplatBurnpackPrecision::F16);
    if f16.validate_burnpacks().is_ok() {
        return Ok(path.to_path_buf());
    }
    let f32 = TripoSplatArtifactSet::new(path, TripoSplatBurnpackPrecision::F32);
    if f32.validate_burnpacks().is_ok() {
        return Ok(path.to_path_buf());
    }
    f16.validate_burnpacks()?;
    Ok(path.to_path_buf())
}

fn has_any_triposplat_artifact(path: &Path) -> bool {
    path.join("clip_vision/dino_v3_vit_h_f16.bpk").exists()
        || path.join("clip_vision/dino_v3_vit_h.bpk").exists()
        || path.join("clip_vision/dino_v3_vit_h.safetensors").exists()
}
