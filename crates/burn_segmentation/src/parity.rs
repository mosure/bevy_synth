use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{BinaryMask, SegmentationError, SegmentationMask, SegmentationResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationParityConfig {
    pub min_mask_iou: f32,
    pub max_bbox_abs_error: f32,
    pub max_area_relative_error: f32,
}

impl Default for SegmentationParityConfig {
    fn default() -> Self {
        Self {
            min_mask_iou: 0.98,
            max_bbox_abs_error: 0.01,
            max_area_relative_error: 0.03,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationParitySummary {
    pub passed: bool,
    pub objects: Vec<SegmentationMaskParityReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationParityFixture {
    pub source: String,
    pub model: String,
    #[serde(default)]
    pub thresholds: SegmentationParityConfig,
    pub python_reference: Vec<SegmentationMask>,
    pub burn_observed: Vec<SegmentationMask>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationMaskParityReport {
    pub object_id: String,
    pub mask_iou: f32,
    pub bbox_max_abs_error: f32,
    pub area_relative_error: f32,
    pub passed: bool,
}

pub fn compare_mask_sets(
    reference: &[SegmentationMask],
    observed: &[SegmentationMask],
    config: &SegmentationParityConfig,
) -> SegmentationResult<SegmentationParitySummary> {
    let observed_by_id = observed
        .iter()
        .map(|mask| (mask.object_id.as_str(), mask))
        .collect::<BTreeMap<_, _>>();
    let mut reports = Vec::new();
    for reference_mask in reference {
        let observed_mask = observed_by_id
            .get(reference_mask.object_id.as_str())
            .copied()
            .ok_or_else(|| {
                SegmentationError::Image(format!(
                    "missing observed mask for object `{}`",
                    reference_mask.object_id
                ))
            })?;
        reports.push(compare_mask(reference_mask, observed_mask, config)?);
    }
    Ok(SegmentationParitySummary {
        passed: reports.iter().all(|report| report.passed),
        objects: reports,
    })
}

pub fn compare_parity_fixture(
    fixture: &SegmentationParityFixture,
) -> SegmentationResult<SegmentationParitySummary> {
    compare_mask_sets(
        &fixture.python_reference,
        &fixture.burn_observed,
        &fixture.thresholds,
    )
}

pub fn read_parity_fixture(
    path: impl AsRef<Path>,
) -> SegmentationResult<SegmentationParityFixture> {
    let path = path.as_ref();
    let bytes = fs::read(path)
        .map_err(|err| SegmentationError::Io(format!("read {}: {err}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| SegmentationError::Image(format!("parse {}: {err}", path.display())))
}

pub fn write_parity_summary(
    path: impl AsRef<Path>,
    summary: &SegmentationParitySummary,
) -> SegmentationResult<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    let bytes = serde_json::to_vec_pretty(summary)
        .map_err(|err| SegmentationError::Image(format!("serialize parity summary: {err}")))?;
    fs::write(path, bytes)
        .map_err(|err| SegmentationError::Io(format!("write {}: {err}", path.display())))
}

pub fn compare_mask(
    reference: &SegmentationMask,
    observed: &SegmentationMask,
    config: &SegmentationParityConfig,
) -> SegmentationResult<SegmentationMaskParityReport> {
    let reference_binary =
        BinaryMask::decode_rle(reference.width, reference.height, &reference.mask_rle)?;
    let observed_binary =
        BinaryMask::decode_rle(observed.width, observed.height, &observed.mask_rle)?;
    let mask_iou = reference_binary.iou(&observed_binary)?;
    let bbox_max_abs_error = reference
        .bbox
        .iter()
        .zip(observed.bbox.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let area_relative_error = if reference.area_px == 0 {
        f32::from(observed.area_px != 0)
    } else {
        ((observed.area_px as f32 - reference.area_px as f32).abs() / reference.area_px as f32)
            .max(0.0)
    };
    let passed = mask_iou >= config.min_mask_iou
        && bbox_max_abs_error <= config.max_bbox_abs_error
        && area_relative_error <= config.max_area_relative_error;
    Ok(SegmentationMaskParityReport {
        object_id: reference.object_id.clone(),
        mask_iou,
        bbox_max_abs_error,
        area_relative_error,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SegmentationModelKind, SegmentationPrompt};

    fn mask(object_id: &str, bbox: [f32; 4]) -> SegmentationMask {
        let binary = BinaryMask::from_normalized_bbox(20, 20, bbox).unwrap();
        SegmentationMask {
            object_id: object_id.to_string(),
            label: "chair".to_string(),
            bbox: binary.bbox_normalized().unwrap(),
            score: 1.0,
            width: 20,
            height: 20,
            area_px: binary.area_px(),
            mask_rle: binary.encode_rle(),
            mask_png_path: None,
            source_prompt: SegmentationPrompt {
                object_id: object_id.to_string(),
                label: "chair".to_string(),
                bbox,
                point: None,
                source_query: Some("chair".to_string()),
            },
            provider: "fixture".to_string(),
            model: SegmentationModelKind::BboxPrompt.label().to_string(),
        }
    }

    #[test]
    fn parity_comparison_passes_matching_masks() {
        let reference = vec![mask("chair_1", [0.1, 0.1, 0.4, 0.5])];
        let observed = vec![mask("chair_1", [0.1, 0.1, 0.4, 0.5])];
        let summary =
            compare_mask_sets(&reference, &observed, &SegmentationParityConfig::default()).unwrap();
        assert!(summary.passed);
        assert_eq!(summary.objects[0].mask_iou, 1.0);
    }

    #[test]
    fn parity_comparison_fails_shifted_masks() {
        let reference = vec![mask("chair_1", [0.1, 0.1, 0.4, 0.5])];
        let observed = vec![mask("chair_1", [0.4, 0.1, 0.7, 0.5])];
        let summary =
            compare_mask_sets(&reference, &observed, &SegmentationParityConfig::default()).unwrap();
        assert!(!summary.passed);
    }

    #[test]
    fn parity_fixture_compares_python_reference_to_burn_observed() {
        let fixture = SegmentationParityFixture {
            source: "python_sidecar".to_string(),
            model: "sam2".to_string(),
            thresholds: SegmentationParityConfig::default(),
            python_reference: vec![mask("chair_1", [0.1, 0.1, 0.4, 0.5])],
            burn_observed: vec![mask("chair_1", [0.1, 0.1, 0.4, 0.5])],
        };
        let summary = compare_parity_fixture(&fixture).unwrap();
        assert!(summary.passed);
    }
}
