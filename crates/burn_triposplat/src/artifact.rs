use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripoSplatBurnpackPrecision {
    F32,
    F16,
}

impl TripoSplatBurnpackPrecision {
    pub fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "",
            Self::F16 => "_f16",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripoSplatArtifact {
    pub component: &'static str,
    pub source_relative_path: &'static str,
    pub burnpack_stem: &'static str,
}

impl TripoSplatArtifact {
    pub fn source_path(self, root: &Path) -> PathBuf {
        root.join(self.source_relative_path)
    }

    pub fn burnpack_path(self, root: &Path, precision: TripoSplatBurnpackPrecision) -> PathBuf {
        root.join(self.component)
            .join(format!("{}{}.bpk", self.burnpack_stem, precision.suffix()))
    }

    pub fn parts_manifest_path(
        self,
        root: &Path,
        precision: TripoSplatBurnpackPrecision,
    ) -> PathBuf {
        let burnpack = self.burnpack_path(root, precision);
        let file_name = burnpack
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.bpk");
        burnpack.with_file_name(format!("{file_name}.parts.json"))
    }

    pub fn has_loadable_burnpack(
        self,
        root: &Path,
        precision: TripoSplatBurnpackPrecision,
    ) -> bool {
        self.burnpack_path(root, precision).exists()
            || self.parts_manifest_path(root, precision).exists()
    }

    pub fn is_triposplat_runtime_required(self) -> bool {
        self.burnpack_stem != "birefnet"
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripoSplatArtifactSet {
    pub root: PathBuf,
    pub precision: TripoSplatBurnpackPrecision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripoSplatCheckpointLayout {
    pub root: PathBuf,
}

pub const TRIPOSPLAT_ARTIFACTS: [TripoSplatArtifact; 5] = [
    TripoSplatArtifact {
        component: "clip_vision",
        source_relative_path: "clip_vision/dino_v3_vit_h.safetensors",
        burnpack_stem: "dino_v3_vit_h",
    },
    TripoSplatArtifact {
        component: "vae",
        source_relative_path: "vae/flux2-vae.safetensors",
        burnpack_stem: "flux2_vae_encoder",
    },
    TripoSplatArtifact {
        component: "diffusion_models",
        source_relative_path: "diffusion_models/triposplat_fp16.safetensors",
        burnpack_stem: "triposplat_flow",
    },
    TripoSplatArtifact {
        component: "vae",
        source_relative_path: "vae/triposplat_vae_decoder_fp16.safetensors",
        burnpack_stem: "triposplat_vae_decoder",
    },
    TripoSplatArtifact {
        component: "background_removal",
        source_relative_path: "background_removal/birefnet.safetensors",
        burnpack_stem: "birefnet",
    },
];

impl TripoSplatCheckpointLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn missing_sources(&self) -> Vec<PathBuf> {
        TRIPOSPLAT_ARTIFACTS
            .into_iter()
            .map(|artifact| artifact.source_path(&self.root))
            .filter(|path| !path.exists())
            .collect()
    }

    pub fn validate_sources(&self) -> Result<(), String> {
        let missing = self.missing_sources();
        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "missing TripoSplat source checkpoint(s): {}",
            missing
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

impl TripoSplatArtifactSet {
    pub fn new(root: impl Into<PathBuf>, precision: TripoSplatBurnpackPrecision) -> Self {
        Self {
            root: root.into(),
            precision,
        }
    }

    pub fn missing_burnpacks(&self) -> Vec<PathBuf> {
        TRIPOSPLAT_ARTIFACTS
            .into_iter()
            .map(|artifact| artifact.burnpack_path(&self.root, self.precision))
            .filter(|path| !path.exists())
            .collect()
    }

    pub fn missing_loadable_artifacts(&self) -> Vec<PathBuf> {
        TRIPOSPLAT_ARTIFACTS
            .into_iter()
            .filter(|artifact| artifact.is_triposplat_runtime_required())
            .filter(|artifact| !artifact.has_loadable_burnpack(&self.root, self.precision))
            .map(|artifact| artifact.burnpack_path(&self.root, self.precision))
            .collect()
    }

    pub fn missing_parts_manifests(&self) -> Vec<PathBuf> {
        TRIPOSPLAT_ARTIFACTS
            .into_iter()
            .filter(|artifact| artifact.is_triposplat_runtime_required())
            .map(|artifact| artifact.parts_manifest_path(&self.root, self.precision))
            .filter(|path| !path.exists())
            .collect()
    }

    pub fn validate_burnpacks(&self) -> Result<(), String> {
        let missing = self.missing_loadable_artifacts();
        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "missing TripoSplat {} burnpack artifact(s) or parts manifest(s): {}",
            self.precision.as_str(),
            missing
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_paths_match_upstream_layout() {
        let root = PathBuf::from("/models/TripoSplat");
        let dino = TRIPOSPLAT_ARTIFACTS[0];
        assert_eq!(
            dino.source_path(&root),
            root.join("clip_vision/dino_v3_vit_h.safetensors")
        );
        assert_eq!(
            dino.burnpack_path(&root, TripoSplatBurnpackPrecision::F16),
            root.join("clip_vision/dino_v3_vit_h_f16.bpk")
        );
    }

    #[test]
    fn parts_manifests_satisfy_loadable_artifact_contract() {
        let root = std::env::temp_dir().join(format!(
            "burn_triposplat_artifact_parts_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let required = TRIPOSPLAT_ARTIFACTS
            .into_iter()
            .filter(|artifact| artifact.is_triposplat_runtime_required())
            .collect::<Vec<_>>();
        for artifact in &required {
            let manifest = artifact.parts_manifest_path(&root, TripoSplatBurnpackPrecision::F16);
            std::fs::create_dir_all(manifest.parent().expect("manifest parent"))
                .expect("create artifact dir");
            std::fs::write(manifest, "{}").expect("write parts manifest");
        }

        let set = TripoSplatArtifactSet::new(&root, TripoSplatBurnpackPrecision::F16);
        assert_eq!(set.missing_burnpacks().len(), TRIPOSPLAT_ARTIFACTS.len());
        assert!(set.missing_loadable_artifacts().is_empty());
        assert!(set.missing_parts_manifests().is_empty());
        set.validate_burnpacks()
            .expect("parts manifests are loadable artifacts");

        let _ = std::fs::remove_dir_all(root);
    }
}
