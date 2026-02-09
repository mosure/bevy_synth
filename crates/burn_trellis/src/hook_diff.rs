use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::Path;

use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

#[derive(Debug, Clone)]
pub struct HookTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct HookSnapshot {
    pub tensors: BTreeMap<String, HookTensor>,
    pub metadata: BTreeMap<String, String>,
}

impl HookSnapshot {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, HookDiffError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(HookDiffError::Io)?;
        let (_, header_metadata) =
            SafeTensors::read_metadata(&bytes).map_err(HookDiffError::SafeTensors)?;
        let safetensors = SafeTensors::deserialize(&bytes).map_err(HookDiffError::SafeTensors)?;

        let mut tensors = BTreeMap::new();
        for name in safetensors.names() {
            let view = safetensors
                .tensor(name)
                .map_err(HookDiffError::SafeTensors)?;
            let tensor = tensor_view_to_hook_tensor(&view)?;
            tensors.insert(name.to_string(), tensor);
        }

        let metadata = header_metadata
            .metadata()
            .as_ref()
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();

        Ok(Self { tensors, metadata })
    }
}

#[derive(Debug)]
pub enum HookDiffError {
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
    Decode(String),
}

impl Display for HookDiffError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::SafeTensors(err) => write!(f, "safetensors error: {err}"),
            Self::Decode(err) => write!(f, "decode error: {err}"),
        }
    }
}

impl std::error::Error for HookDiffError {}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricStats {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub rmse: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookDiffStatus {
    Match,
    MissingInActual,
    ShapeMismatch,
}

#[derive(Debug, Clone)]
pub struct HookDiffEntry {
    pub key: String,
    pub status: HookDiffStatus,
    pub reference_shape: Vec<usize>,
    pub actual_shape: Option<Vec<usize>>,
    pub stats: Option<MetricStats>,
}

#[derive(Debug, Clone, Default)]
pub struct HookDiffReport {
    pub entries: Vec<HookDiffEntry>,
    pub extra_in_actual: Vec<String>,
}

pub fn compare_hook_snapshots(
    reference: &HookSnapshot,
    actual: &HookSnapshot,
    prefix: Option<&str>,
) -> HookDiffReport {
    let mut keys = BTreeSet::new();
    for key in reference.tensors.keys() {
        if prefix.is_none_or(|p| key.starts_with(p)) {
            keys.insert(key.clone());
        }
    }

    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let Some(reference_tensor) = reference.tensors.get(&key) else {
            continue;
        };

        let actual_tensor = actual.tensors.get(&key);
        match actual_tensor {
            None => entries.push(HookDiffEntry {
                key,
                status: HookDiffStatus::MissingInActual,
                reference_shape: reference_tensor.shape.clone(),
                actual_shape: None,
                stats: None,
            }),
            Some(actual_tensor) if actual_tensor.shape != reference_tensor.shape => {
                entries.push(HookDiffEntry {
                    key,
                    status: HookDiffStatus::ShapeMismatch,
                    reference_shape: reference_tensor.shape.clone(),
                    actual_shape: Some(actual_tensor.shape.clone()),
                    stats: None,
                });
            }
            Some(actual_tensor) => entries.push(HookDiffEntry {
                key,
                status: HookDiffStatus::Match,
                reference_shape: reference_tensor.shape.clone(),
                actual_shape: Some(actual_tensor.shape.clone()),
                stats: Some(compute_stats(&actual_tensor.data, &reference_tensor.data)),
            }),
        }
    }

    let mut extra_in_actual: Vec<String> = actual
        .tensors
        .keys()
        .filter(|key| !reference.tensors.contains_key(*key))
        .filter(|key| prefix.is_none_or(|p| key.starts_with(p)))
        .cloned()
        .collect();
    extra_in_actual.sort();

    HookDiffReport {
        entries,
        extra_in_actual,
    }
}

pub fn compute_stats(actual: &[f32], reference: &[f32]) -> MetricStats {
    let len = actual.len().min(reference.len());
    if len == 0 {
        return MetricStats::default();
    }

    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;

    for i in 0..len {
        let diff = actual[i] - reference[i];
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        sum_sq += diff * diff;
    }

    let n = len as f32;
    MetricStats {
        mean_abs: sum_abs / n,
        max_abs,
        rmse: (sum_sq / n).sqrt(),
    }
}

fn tensor_view_to_hook_tensor(view: &TensorView<'_>) -> Result<HookTensor, HookDiffError> {
    let data = decode_tensor_data(view)?;
    Ok(HookTensor {
        shape: view.shape().to_vec(),
        data,
    })
}

fn decode_tensor_data(view: &TensorView<'_>) -> Result<Vec<f32>, HookDiffError> {
    let shape = view.shape();
    let numel = shape
        .iter()
        .fold(1usize, |acc, &value| acc.saturating_mul(value));
    let bytes = view.data();
    let dtype = view.dtype();
    let item_size = dtype_size(dtype);
    let expected = numel.saturating_mul(item_size);

    if bytes.len() != expected {
        return Err(HookDiffError::Decode(format!(
            "tensor byte length mismatch for dtype {:?}: expected {}, got {}",
            dtype,
            expected,
            bytes.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    for chunk in bytes.chunks_exact(item_size) {
        let value = match dtype {
            Dtype::BOOL => {
                if chunk[0] == 0 {
                    0.0
                } else {
                    1.0
                }
            }
            Dtype::U8 => chunk[0] as f32,
            Dtype::I8 => (chunk[0] as i8) as f32,
            Dtype::I16 => i16::from_le_bytes([chunk[0], chunk[1]]) as f32,
            Dtype::U16 => u16::from_le_bytes([chunk[0], chunk[1]]) as f32,
            Dtype::I32 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            Dtype::U32 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            Dtype::I64 => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            Dtype::U64 => u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            Dtype::F16 => {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            }
            Dtype::BF16 => {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16::from_bits(bits).to_f32()
            }
            Dtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            Dtype::F64 => {
                let value = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                value as f32
            }
            other => {
                return Err(HookDiffError::Decode(format!(
                    "unsupported safetensors dtype: {:?}",
                    other
                )));
            }
        };
        out.push(value);
    }
    Ok(out)
}

fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
        Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
        Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
        Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{HookDiffStatus, HookSnapshot, compare_hook_snapshots, compute_stats};

    #[test]
    fn compute_stats_reports_expected_values() {
        let stats = compute_stats(&[1.0, 2.0, 3.0], &[1.5, 1.0, 3.0]);
        assert!((stats.mean_abs - 0.5).abs() < 1e-6);
        assert!((stats.max_abs - 1.0).abs() < 1e-6);
        assert!((stats.rmse - (1.25f32 / 3.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn compare_reports_missing_and_shape_mismatch() {
        let mut reference = HookSnapshot::default();
        reference.tensors.insert(
            "a".to_string(),
            super::HookTensor {
                shape: vec![2],
                data: vec![0.0, 1.0],
            },
        );
        reference.tensors.insert(
            "b".to_string(),
            super::HookTensor {
                shape: vec![1, 2],
                data: vec![0.0, 1.0],
            },
        );

        let mut actual = HookSnapshot::default();
        actual.tensors.insert(
            "a".to_string(),
            super::HookTensor {
                shape: vec![2],
                data: vec![0.0, 2.0],
            },
        );
        actual.tensors.insert(
            "b".to_string(),
            super::HookTensor {
                shape: vec![2, 1],
                data: vec![0.0, 1.0],
            },
        );
        actual.tensors.insert(
            "extra".to_string(),
            super::HookTensor {
                shape: vec![1],
                data: vec![0.0],
            },
        );

        let report = compare_hook_snapshots(&reference, &actual, None);
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.extra_in_actual, vec!["extra".to_string()]);
        assert_eq!(report.entries[0].key, "a");
        assert_eq!(report.entries[0].status, HookDiffStatus::Match);
        assert!(report.entries[0].stats.is_some());
        assert_eq!(report.entries[1].key, "b");
        assert_eq!(report.entries[1].status, HookDiffStatus::ShapeMismatch);
        assert!(report.entries[1].stats.is_none());
    }
}
