use crate::prelude::*;

#[derive(Clone)]
pub(crate) struct LoadedSceneDepthMap {
    pub(crate) depth_m: Vec<f32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) raw_path: PathBuf,
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
    }))
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
