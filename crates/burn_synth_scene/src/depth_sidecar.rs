use crate::pose_fit_prelude::*;

#[derive(Clone)]
pub(crate) struct DenseDepthCrop {
    pub(crate) depth_m: Vec<f32>,
    pub(crate) valid_mask: Vec<u8>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) valid_count: usize,
    pub(crate) source_path: PathBuf,
}

#[derive(Clone)]
pub(crate) struct LoadedSceneDepthMap {
    pub(crate) depth_m: Vec<f32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) raw_path: PathBuf,
    pub(crate) metadata_path: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseDepthComparison {
    pub(crate) mae_m: f32,
    pub(crate) normalized_loss: f32,
    pub(crate) sample_count: usize,
}

pub(crate) fn load_scene_depth_map_sidecar(
    evidence: &SceneGroundingEvidence,
) -> Result<Option<LoadedSceneDepthMap>, String> {
    let Some(depth_ref) = evidence.depth.as_ref() else {
        return Ok(None);
    };
    let Some(artifact_path) = depth_ref.artifact_path.as_ref() else {
        return Ok(None);
    };
    let artifact_path = PathBuf::from(artifact_path);
    if !artifact_path.exists() {
        return Err(format!(
            "depth evidence artifact does not exist: {}",
            artifact_path.display()
        ));
    }
    let metadata: Value = read_json_value(&artifact_path)?;
    let Some(sidecar) = metadata.get("depth_map_sidecar") else {
        return Ok(None);
    };
    let raw_path = sidecar
        .get("relative_raw_path")
        .and_then(Value::as_str)
        .or_else(|| sidecar.get("raw_path").and_then(Value::as_str))
        .map(|path| resolve_relative_to_artifact(&artifact_path, path))
        .ok_or_else(|| {
            format!(
                "depth evidence {} has depth_map_sidecar but no raw_path",
                artifact_path.display()
            )
        })?;
    let metadata_path = sidecar
        .get("metadata_path")
        .and_then(Value::as_str)
        .map(|path| resolve_relative_to_artifact(&artifact_path, path))
        .unwrap_or_else(|| artifact_path.with_file_name("depth_meters_f32le.json"));
    let width = sidecar
        .get("width")
        .and_then(Value::as_u64)
        .or_else(|| depth_ref.depth_map_size.map(|size| u64::from(size[0])))
        .ok_or_else(|| format!("depth sidecar {} is missing width", artifact_path.display()))?
        as usize;
    let height = sidecar
        .get("height")
        .and_then(Value::as_u64)
        .or_else(|| depth_ref.depth_map_size.map(|size| u64::from(size[1])))
        .ok_or_else(|| {
            format!(
                "depth sidecar {} is missing height",
                artifact_path.display()
            )
        })? as usize;
    let encoding = sidecar
        .get("encoding")
        .and_then(Value::as_str)
        .unwrap_or("f32le");
    if encoding != "f32le" {
        return Err(format!(
            "unsupported depth sidecar encoding {encoding:?} in {}",
            artifact_path.display()
        ));
    }
    let bytes = fs::read(&raw_path)
        .map_err(|err| format!("failed to read depth sidecar {}: {err}", raw_path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!(
            "depth sidecar {} byte length {} is not divisible by 4",
            raw_path.display(),
            bytes.len()
        ));
    }
    let depth_m = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| format!("depth sidecar shape overflows: {width}x{height}"))?;
    if depth_m.len() != expected {
        return Err(format!(
            "depth sidecar {} shape mismatch: expected {expected} values for {width}x{height}, got {}",
            raw_path.display(),
            depth_m.len()
        ));
    }
    Ok(Some(LoadedSceneDepthMap {
        depth_m,
        width,
        height,
        raw_path,
        metadata_path,
    }))
}

pub(crate) fn loaded_depth_map_report(depth_map: &LoadedSceneDepthMap) -> Value {
    let finite_positive_count = depth_map
        .depth_m
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .count();
    json!({
        "status": "loaded",
        "encoding": "f32le",
        "raw_path": depth_map.raw_path.display().to_string(),
        "metadata_path": depth_map.metadata_path.display().to_string(),
        "width": depth_map.width,
        "height": depth_map.height,
        "value_count": depth_map.depth_m.len(),
        "finite_positive_count": finite_positive_count,
    })
}

pub(crate) fn dense_depth_crop_from_sidecar(
    depth_map: &LoadedSceneDepthMap,
    crop_bbox: [f32; 4],
    resolution: usize,
    fallback_depth_m: Option<f32>,
) -> Option<DenseDepthCrop> {
    if resolution == 0 || depth_map.width == 0 || depth_map.height == 0 {
        return None;
    }
    let fallback = fallback_depth_m
        .filter(|value| value.is_finite() && *value > 0.0)
        .or_else(|| {
            let mut values = depth_map
                .depth_m
                .iter()
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .take(4096)
                .collect::<Vec<_>>();
            if values.is_empty() {
                None
            } else {
                values.sort_by(f32::total_cmp);
                Some(values[values.len() / 2])
            }
        })
        .unwrap_or(1.0);
    let mut depth_m = vec![fallback; resolution * resolution];
    let mut valid_mask = vec![0_u8; resolution * resolution];
    let mut valid_count = 0usize;
    for y in 0..resolution {
        for x in 0..resolution {
            let u = crop_bbox[0]
                + ((x as f32 + 0.5) / resolution as f32) * (crop_bbox[2] - crop_bbox[0]);
            let v = crop_bbox[1]
                + ((y as f32 + 0.5) / resolution as f32) * (crop_bbox[3] - crop_bbox[1]);
            let px = (u.clamp(0.0, 1.0) * (depth_map.width - 1) as f32).round() as usize;
            let py = (v.clamp(0.0, 1.0) * (depth_map.height - 1) as f32).round() as usize;
            let index = py * depth_map.width + px;
            let out_index = y * resolution + x;
            if let Some(value) = depth_map
                .depth_m
                .get(index)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
            {
                depth_m[out_index] = value;
                valid_mask[out_index] = 1;
                valid_count += 1;
            }
        }
    }
    (valid_count > 0).then(|| DenseDepthCrop {
        depth_m,
        valid_mask,
        width: resolution,
        height: resolution,
        valid_count,
        source_path: depth_map.raw_path.clone(),
    })
}

pub(crate) fn dense_depth_crop_report(crop: &DenseDepthCrop) -> Value {
    let mut values = crop
        .depth_m
        .iter()
        .copied()
        .zip(&crop.valid_mask)
        .filter_map(|(depth, valid)| (*valid != 0).then_some(depth))
        .collect::<Vec<_>>();
    values.sort_by(f32::total_cmp);
    json!({
        "source_path": crop.source_path.display().to_string(),
        "width": crop.width,
        "height": crop.height,
        "valid_count": crop.valid_count,
        "valid_coverage": crop.valid_count as f32 / crop.depth_m.len().max(1) as f32,
        "median_m": values.get(values.len().saturating_sub(1) / 2).copied(),
    })
}

pub(crate) fn resample_dense_depth_crop(
    crop: &DenseDepthCrop,
    width: usize,
    height: usize,
) -> Option<(Vec<f32>, Vec<f32>, usize)> {
    if width == 0 || height == 0 || crop.width == 0 || crop.height == 0 {
        return None;
    }
    let mut depth = vec![1.0; width * height];
    let mut valid = vec![0.0; width * height];
    let mut valid_count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let sx = ((x as f32 + 0.5) / width as f32 * crop.width as f32)
                .floor()
                .clamp(0.0, (crop.width - 1) as f32) as usize;
            let sy = ((y as f32 + 0.5) / height as f32 * crop.height as f32)
                .floor()
                .clamp(0.0, (crop.height - 1) as f32) as usize;
            let source_index = sy * crop.width + sx;
            let out_index = y * width + x;
            depth[out_index] = crop.depth_m[source_index];
            if crop.valid_mask[source_index] != 0 {
                valid[out_index] = 1.0;
                valid_count += 1;
            }
        }
    }
    Some((depth, valid, valid_count))
}

pub(crate) fn dense_depth_comparison(
    target: &DenseDepthCrop,
    target_mask: &[u8],
    surface_mask: &[u8],
    surface_depth: &[f32],
    max_depth_error_m: f32,
) -> Option<DenseDepthComparison> {
    let len = target
        .depth_m
        .len()
        .min(target_mask.len())
        .min(surface_mask.len())
        .min(surface_depth.len());
    let mut sum_abs = 0.0;
    let mut count = 0usize;
    for index in 0..len {
        if target.valid_mask.get(index).copied().unwrap_or(0) == 0
            || target_mask[index] == 0
            || surface_mask[index] == 0
        {
            continue;
        }
        let observed = surface_depth[index];
        let expected = target.depth_m[index];
        if observed.is_finite() && observed > 0.0 && expected.is_finite() && expected > 0.0 {
            sum_abs += (observed - expected).abs();
            count += 1;
        }
    }
    if count < 6 {
        return None;
    }
    let mae_m = sum_abs / count as f32;
    Some(DenseDepthComparison {
        mae_m,
        normalized_loss: (mae_m / max_depth_error_m.max(0.05)).clamp(0.0, 4.0),
        sample_count: count,
    })
}

fn read_json_value(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read JSON {}: {err}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| format!("failed to parse JSON {}: {err}", path.display()))
}

fn resolve_relative_to_artifact(artifact_path: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() || path.exists() {
        return path;
    }
    artifact_path
        .parent()
        .map(|parent| parent.join(&path))
        .unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_sidecar_loading_and_crop_matching_use_metric_depth_values() {
        let root = temp_depth_sidecar_dir("metric_depth");
        fs::create_dir_all(&root).expect("create temp dir");
        let raw_path = root.join("depth_meters_f32le.bin");
        let metadata_path = root.join("depth_evidence.json");
        let depths = (0..16)
            .map(|index| 1.0 + index as f32 * 0.1)
            .collect::<Vec<_>>();
        let mut raw = Vec::new();
        for depth in &depths {
            raw.extend_from_slice(&depth.to_le_bytes());
        }
        fs::write(&raw_path, raw).expect("write raw depth");
        write_json_file(
            &metadata_path,
            &json!({
                "depth_map_sidecar": {
                    "encoding": "f32le",
                    "relative_raw_path": "depth_meters_f32le.bin",
                    "width": 4,
                    "height": 4,
                }
            }),
        )
        .expect("write metadata");
        let evidence = SceneGroundingEvidence {
            source_image_path: "source.png".to_string(),
            depth: Some(crate::DepthEvidenceRef {
                provider: "depth-pro".to_string(),
                model: Some("depth-pro".to_string()),
                precision: Some("f16".to_string()),
                artifact_path: Some(metadata_path.display().to_string()),
                focal_length_px: Some(4.0),
                vertical_fov_degrees: Some(60.0),
                image_size: Some([4, 4]),
                depth_map_size: Some([4, 4]),
                floor_sample_count: None,
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: crate::EstimatedCamera {
                focal_length_px: Some(4.0),
                principal_point: Some([1.5, 1.5]),
                image_size: Some([4, 4]),
                vertical_fov_degrees: Some(60.0),
                confidence: Some(1.0),
            },
            floor: crate::EstimatedFloorPlane::default(),
            objects: Vec::new(),
        };

        let loaded = load_scene_depth_map_sidecar(&evidence)
            .expect("sidecar load")
            .expect("sidecar present");
        assert_eq!(loaded.width, 4);
        assert_eq!(loaded.height, 4);
        assert_eq!(loaded.depth_m, depths);
        let crop = dense_depth_crop_from_sidecar(&loaded, [0.0, 0.0, 1.0, 1.0], 4, None)
            .expect("depth crop");
        assert_eq!(crop.valid_count, 16);
        assert_eq!(crop.depth_m[5], depths[5]);
        let target_mask = vec![1_u8; 16];
        let surface_mask = vec![1_u8; 16];
        let matching =
            dense_depth_comparison(&crop, &target_mask, &surface_mask, &crop.depth_m, 0.35)
                .expect("matching comparison");
        assert!(matching.mae_m < 1.0e-6);
        let shifted_depth = crop
            .depth_m
            .iter()
            .map(|depth| depth + 0.35)
            .collect::<Vec<_>>();
        let shifted =
            dense_depth_comparison(&crop, &target_mask, &surface_mask, &shifted_depth, 0.35)
                .expect("shifted comparison");
        assert!(shifted.normalized_loss > matching.normalized_loss + 0.9);
        let _ = fs::remove_dir_all(root);
    }

    fn temp_depth_sidecar_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        env::temp_dir().join(format!("burn_synth_mcp_depth_sidecar_{label}_{nanos}"))
    }
}
