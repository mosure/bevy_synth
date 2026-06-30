pub mod assets;
pub mod blob_burnpack;
pub mod cdn;
pub mod config;
pub mod decode;
pub mod import;
pub mod language;
pub mod native;
pub mod projector;
pub mod runtime;
pub mod tensor_io;
pub mod tokenizer;
pub mod vision;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use assets::{
    LocateAnythingAssetReport, LocateAnythingWeightFileStatus, inspect_model_assets,
    weight_file_for_tensor,
};
pub use cdn::{locate_anything_cdn_root_prefix, locate_anything_cdn_root_url};
pub use config::{
    LocateAnythingModelConfig, LocateAnythingTextConfig, LocateAnythingVisionBackboneConfig,
};
pub use decode::{
    DecodeMode, LocateAnythingTokenIds, ParallelBoxDecode, ParallelBoxDecodeConfig,
    ParallelPatternKind, decode_detections_from_text, decode_parallel_box_from_logits,
    decode_parallel_box_from_probs,
};
pub use native::{
    LocateAnythingNativeBatchInputs, LocateAnythingNativePrompt, prepare_native_batch_inputs,
};
pub use runtime::{
    BatchedDetectionRequest, LocateAnythingDetector, LocateAnythingRuntime,
    LocateAnythingRuntimeBackend, LocateAnythingRuntimeConfig,
};
pub use vision::{LOCATE_ANYTHING_CHECKPOINT_IN_TOKEN_LIMIT, LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Detection {
    pub label: String,
    pub bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub source_query: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DetectionQuery {
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_hint: Option<String>,
}

#[derive(Debug)]
pub enum LocateAnythingError {
    Config(String),
    Decode(String),
    Import(String),
    Io(String),
    Runtime(String),
    Unsupported(String),
}

impl fmt::Display for LocateAnythingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "configuration error: {err}"),
            Self::Decode(err) => write!(f, "decode error: {err}"),
            Self::Import(err) => write!(f, "import error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Runtime(err) => write!(f, "runtime error: {err}"),
            Self::Unsupported(err) => write!(f, "unsupported LocateAnything operation: {err}"),
        }
    }
}

impl std::error::Error for LocateAnythingError {}

impl From<std::io::Error> for LocateAnythingError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for LocateAnythingError {
    fn from(value: serde_json::Error) -> Self {
        Self::Config(value.to_string())
    }
}

pub type LocateAnythingResult<T> = Result<T, LocateAnythingError>;

pub(crate) fn normalize_bbox(mut bbox: [f32; 4]) -> [f32; 4] {
    for value in &mut bbox {
        if !value.is_finite() {
            *value = 0.0;
        }
        *value = value.clamp(0.0, 1.0);
    }
    if bbox[2] < bbox[0] {
        bbox.swap(0, 2);
    }
    if bbox[3] < bbox[1] {
        bbox.swap(1, 3);
    }
    bbox
}

pub(crate) fn normalize_point(mut point: [f32; 2]) -> [f32; 2] {
    for value in &mut point {
        if !value.is_finite() {
            *value = 0.0;
        }
        *value = value.clamp(0.0, 1.0);
    }
    point
}
