use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum QuantizationMode {
    F32,
    F16,
    Both,
}

impl QuantizationMode {
    pub fn include_f32(self) -> bool {
        matches!(self, Self::F32 | Self::Both)
    }

    pub fn include_f16(self) -> bool {
        matches!(self, Self::F16 | Self::Both)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ArtifactPolicy {
    #[default]
    SingleFile,
    Parts {
        part_size_mib: u64,
    },
}

impl ArtifactPolicy {
    pub fn wants_parts(self) -> bool {
        matches!(self, Self::Parts { .. })
    }

    pub fn part_size_mib(self) -> Option<u64> {
        match self {
            Self::SingleFile => None,
            Self::Parts { part_size_mib } => Some(part_size_mib.max(1)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportArtifactRecord {
    pub label: String,
    pub source: String,
    pub output: String,
    pub precision: String,
    pub bytes_len: u64,
    pub sha256: String,
    pub parts_manifest: Option<String>,
    pub part_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImportReport {
    pub artifacts: Vec<ImportArtifactRecord>,
    pub skipped: Vec<String>,
    pub missing_sources: Vec<String>,
    pub notes: Vec<String>,
}
