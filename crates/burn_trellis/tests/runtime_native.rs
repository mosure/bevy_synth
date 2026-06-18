use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "runtime-model")]
use burn_trellis::hook_diff::HookSnapshot;
#[cfg(feature = "runtime-model")]
use burn_trellis::mesh::load_obj_mesh;
use burn_trellis::pipeline::{Trellis2Pipeline, Trellis2PipelineConfig, TrellisRunOptions};
use image::{ImageBuffer, Rgba};

#[test]
fn native_pipeline_writes_mesh_and_preprocess_hook() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("burn_trellis_native_test_{unique}"));
    let weights_root = root.join("weights");
    let ckpts = weights_root.join("ckpts");
    std::fs::create_dir_all(&ckpts).expect("failed to create weights directory");

    std::fs::write(
        weights_root.join("pipeline.json"),
        r#"{
            "args": {
                "models": {
                    "shape": "ckpts/shape_model"
                }
            }
        }"#,
    )
    .expect("failed to write pipeline json");
    std::fs::write(ckpts.join("shape_model.json"), "{}").expect("failed to write model json");
    std::fs::write(ckpts.join("shape_model.bpk"), "fake").expect("failed to write model bpk");

    let input = root.join("input.png");
    let image = ImageBuffer::from_fn(8, 8, |x, y| {
        if (2..=5).contains(&x) && (1..=6).contains(&y) {
            Rgba([200u8, 120u8, 40u8, 255u8])
        } else {
            Rgba([0u8, 0u8, 0u8, 0u8])
        }
    });
    image.save(&input).expect("failed to save test image");

    let output = root.join("out.obj");
    let hook = root.join("hook.safetensors");

    let pipeline = Trellis2Pipeline::new(Trellis2PipelineConfig {
        weights_root,
        image_large_root: None,
    })
    .expect("pipeline should initialize");
    pipeline
        .validate_runtime()
        .expect("runtime assets should validate");

    let result = pipeline.infer_mesh_to_obj(
        &input,
        &output,
        &TrellisRunOptions {
            hook_output: Some(hook.clone()),
            ..TrellisRunOptions::default()
        },
    );

    #[cfg(not(feature = "runtime-model"))]
    {
        let err = result.expect_err("runtime-model-disabled pipeline should fail explicitly");
        assert!(
            err.to_string().contains("requires `runtime-model` feature"),
            "unexpected runtime-model-disabled error: {err}"
        );
    }

    #[cfg(feature = "runtime-model")]
    {
        result.expect("native pipeline should emit a mesh");

        assert!(output.exists(), "mesh output was not emitted");
        assert!(hook.exists(), "hook file was not emitted");

        let mesh = load_obj_mesh(&output).expect("failed to parse output OBJ");
        assert!(!mesh.vertices.is_empty(), "output mesh has no vertices");
        assert!(!mesh.faces.is_empty(), "output mesh has no faces");

        let snapshot = HookSnapshot::from_file(&hook).expect("failed to read hook");
        assert!(
            snapshot.tensors.contains_key("preprocess_image.output"),
            "preprocess hook tensor missing"
        );
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn runtime_validation_reports_missing_assets() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("burn_trellis_validate_test_{unique}"));
    let weights_root = root.join("weights");
    std::fs::create_dir_all(&weights_root).expect("failed to create test directory");

    std::fs::write(
        weights_root.join("pipeline.json"),
        r#"{
            "args": {
                "models": {
                    "shape": "ckpts/missing_shape"
                }
            }
        }"#,
    )
    .expect("failed to write pipeline json");

    let pipeline = Trellis2Pipeline::new(Trellis2PipelineConfig {
        weights_root: PathBuf::from(&weights_root),
        image_large_root: None,
    })
    .expect("pipeline should initialize");

    let err = pipeline
        .validate_runtime()
        .expect_err("validation should fail when model assets are missing");
    let message = err.to_string();
    assert!(message.contains("incomplete"));
    assert!(message.contains("missing_shape"));

    let _ = std::fs::remove_dir_all(root);
}
