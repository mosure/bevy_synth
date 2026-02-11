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
    Sharded {
        shard_size_mib: u64,
    },
    Both {
        shard_size_mib: u64,
    },
}

impl ArtifactPolicy {
    pub fn wants_shards(self) -> bool {
        !matches!(self, Self::SingleFile)
    }

    pub fn shard_size_mib(self) -> Option<u64> {
        match self {
            Self::SingleFile => None,
            Self::Sharded { shard_size_mib } | Self::Both { shard_size_mib } => {
                Some(shard_size_mib.max(1))
            }
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
    pub shard_manifest: Option<String>,
    pub shard_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImportReport {
    pub artifacts: Vec<ImportArtifactRecord>,
    pub skipped: Vec<String>,
    pub missing_sources: Vec<String>,
    pub notes: Vec<String>,
}
