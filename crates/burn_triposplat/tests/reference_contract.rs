#![cfg(feature = "import")]

use std::{fs, path::PathBuf};

use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;

#[test]
fn triposplat_reference_metadata_contract_reference() -> Result<(), Box<dyn std::error::Error>> {
    let Some(reference_path) = reference_json_path() else {
        eprintln!(
            "skipping: set TRIPOSPLAT_REFERENCE_JSON=/path/to/reference.json to validate upstream TripoSplat reference evidence"
        );
        return Ok(());
    };

    let json = fs::read_to_string(&reference_path)?;
    let reference: Value = serde_json::from_str(&json)?;
    assert_eq!(reference["seed"].as_i64(), Some(42));
    assert!(reference["steps"].as_i64().unwrap_or_default() > 0);
    assert_eq!(reference["guidance_scale"].as_f64(), Some(3.0));
    assert_eq!(reference["shift"].as_f64(), Some(3.0));
    assert_eq!(reference["erode_radius"].as_i64(), Some(1));

    let stages = reference["stages"]
        .as_object()
        .ok_or("reference.json missing stages object")?;
    for name in ["preprocess", "encode", "sample"] {
        assert!(stages.contains_key(name), "missing stage {name}");
    }
    assert_tensor_summary(&stages["encode"]["feature1"], "feature1", 3)?;
    assert_tensor_summary(&stages["encode"]["feature2"], "feature2", 3)?;
    assert_tensor_summary(&stages["sample"]["latent"], "latent", 3)?;

    let outputs = reference["outputs"]
        .as_array()
        .ok_or("reference.json missing outputs array")?;
    assert!(!outputs.is_empty(), "reference outputs must not be empty");
    for output in outputs {
        let count = output["gaussians"].as_u64().ok_or("missing gaussians")?;
        let bytes = output["splat_bytes"]
            .as_u64()
            .ok_or("missing splat_bytes")?;
        assert_eq!(
            bytes,
            count * 32,
            ".splat byte count must be 32 bytes per record"
        );
        let splat_path = output["splat_path"].as_str().ok_or("missing splat_path")?;
        assert!(
            PathBuf::from(splat_path).is_file(),
            "missing reference splat {splat_path}"
        );
    }

    if let Some(stage_path) = stage_tensors_path(&reference_path, &reference) {
        assert_stage_tensors(&stage_path)?;
    }

    Ok(())
}

fn reference_json_path() -> Option<PathBuf> {
    std::env::var_os("TRIPOSPLAT_REFERENCE_JSON").map(resolve_test_path)
}

fn stage_tensors_path(reference_path: &PathBuf, reference: &Value) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("TRIPOSPLAT_REFERENCE_STAGES") {
        return Some(resolve_test_path(path));
    }
    reference["stage_tensors"]["path"]
        .as_str()
        .map(PathBuf::from)
        .or_else(|| {
            Some(
                reference_path
                    .parent()?
                    .join("stage_tensors_f32.safetensors"),
            )
        })
        .filter(|path| path.is_file())
}

fn resolve_test_path(path: impl Into<PathBuf>) -> PathBuf {
    let path = path.into();
    if path.is_absolute() || path.exists() {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn assert_tensor_summary(
    value: &Value,
    name: &str,
    rank: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = value["shape"]
        .as_array()
        .ok_or_else(|| format!("{name} missing shape"))?;
    assert_eq!(shape.len(), rank, "{name} rank mismatch");
    assert!(
        shape.iter().all(|dim| dim.as_u64().unwrap_or_default() > 0),
        "{name} shape must be non-empty"
    );
    assert!(
        value["sha256_f32_le"].as_str().is_some(),
        "{name} missing sha256_f32_le"
    );
    for field in ["mean", "std", "min", "max"] {
        assert!(
            value[field].as_f64().is_some(),
            "{name} missing numeric field {field}"
        );
    }
    Ok(())
}

fn assert_stage_tensors(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    for (name, rank) in [
        ("image_rgb_0_1", 4usize),
        ("feature1", 3usize),
        ("feature2", 3usize),
        ("latent", 3usize),
    ] {
        let tensor = tensors.tensor(name)?;
        assert_eq!(tensor.dtype(), Dtype::F32, "{name} must be stored as f32");
        assert_eq!(tensor.shape().len(), rank, "{name} rank mismatch");
        assert!(
            tensor.shape().iter().all(|dim| *dim > 0),
            "{name} shape must be non-empty"
        );
    }
    Ok(())
}
