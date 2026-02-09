use std::path::PathBuf;

use burn_trellis::hook_diff::{HookSnapshot, compute_stats};
use burn_trellis::preprocess::{PreprocessConfig, preprocess_image_path};

const MEAN_ABS_TOLERANCE: f32 = 0.0;
const MAX_ABS_TOLERANCE: f32 = 0.0;
const RMSE_TOLERANCE: f32 = 0.0;

#[test]
fn trellis2_preprocess_matches_reference_hook() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let input_image = root.join("assets/hooks/trellis2_preprocess_input.png");
    let reference_hook = root.join("assets/hooks/trellis2_preprocess_reference.safetensors");
    if !input_image.exists() || !reference_hook.exists() {
        eprintln!(
            "Skipping preprocess hook correctness test: missing input or reference hook files."
        );
        return Ok(());
    }

    let output = preprocess_image_path(&input_image, PreprocessConfig::default())?;
    let output_shape = vec![output.height as usize, output.width as usize, 3usize];
    let output_data: Vec<f32> = output.rgb.iter().map(|&v| v as f32).collect();

    let reference = HookSnapshot::from_file(&reference_hook)?;
    let reference_tensor = reference
        .tensors
        .get("preprocess_image.output")
        .ok_or("missing preprocess_image.output tensor in reference hook")?;

    if reference_tensor.shape != output_shape {
        return Err(format!(
            "shape mismatch: reference {:?} vs burn {:?}",
            reference_tensor.shape, output_shape
        )
        .into());
    }
    if reference_tensor.data.len() != output_data.len() {
        return Err(format!(
            "data length mismatch: reference {} vs burn {}",
            reference_tensor.data.len(),
            output_data.len()
        )
        .into());
    }

    let stats = compute_stats(&output_data, &reference_tensor.data);
    eprintln!(
        "preprocess_image.output diff: mean_abs={:.6e}, max_abs={:.6e}, rmse={:.6e}",
        stats.mean_abs, stats.max_abs, stats.rmse
    );
    if stats.mean_abs > MEAN_ABS_TOLERANCE
        || stats.max_abs > MAX_ABS_TOLERANCE
        || stats.rmse > RMSE_TOLERANCE
    {
        return Err(format!(
            "preprocess hook mismatch: mean_abs={:.6e}, max_abs={:.6e}, rmse={:.6e}",
            stats.mean_abs, stats.max_abs, stats.rmse
        )
        .into());
    }
    Ok(())
}
