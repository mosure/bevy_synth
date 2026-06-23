#[cfg(feature = "runtime-model")]
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "runtime-model")]
use super::{
    DecodeHookOverrides, DecodeShapeSubSample, FlowEulerSampleConfig, RuntimeDecodeModels,
    ShapeSLatSample, SparseFlowOpTimingSummary, TexSLatSample,
    cascade_resolution_accepts_token_budget, decode_latent_to_outputs, dense_cond_with_override,
    merge_voxel_attrs_for_decode, sparse_layout_from_batch_ids, sparse_layout_from_coords,
    validate_sparse_layout_rows,
};
use super::{
    SparseCoordCapSource, TrellisDecodeOutputMode, bake_pbr_from_voxels,
    runtime_max_sparse_coords_for_backend, summarize_material,
};
#[cfg(feature = "runtime-model")]
use crate::hook_diff::{HookSnapshot, compute_stats};
use crate::mesh::MeshPbrTextures;
#[cfg(feature = "runtime-model")]
use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
#[cfg(feature = "runtime-model")]
use crate::preprocess::PreprocessOutput;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::fdg_decoder::{FdgDecoderRuntime, decode_fdg_outputs};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::runtime_config::{
    RuntimeModelDebugConfig, set_runtime_model_debug_config,
};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_decoder::{
    SparseSubdivisionLogits, decoder_conv_telemetry, decoder_op_telemetry,
    reset_decoder_conv_telemetry, reset_decoder_op_telemetry,
};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_unet_vae_decoder::{
    SparseUnetVaeDecoderRuntime, decode_tex_outputs,
};
#[cfg(feature = "runtime-model")]
use crate::trellis_config::TrellisPipelineConfig;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::DefaultWgpuBackend;

#[cfg(feature = "runtime-model")]
fn env_flag(name: &str) -> bool {
    env_flag_default(name, false)
}

#[cfg(feature = "runtime-model")]
fn env_flag_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
}

#[test]
fn decode_output_modes_keep_mesh_postprocess_separate_from_pbr() {
    assert!(TrellisDecodeOutputMode::NativePbr.needs_native_pbr());
    assert!(TrellisDecodeOutputMode::NativePbr.needs_native_mesh_postprocess());

    assert!(!TrellisDecodeOutputMode::NativeMesh.needs_native_pbr());
    assert!(TrellisDecodeOutputMode::NativeMesh.needs_native_mesh_postprocess());

    assert!(!TrellisDecodeOutputMode::OvoxelHookExport.needs_native_pbr());
    assert!(!TrellisDecodeOutputMode::OvoxelHookExport.needs_native_mesh_postprocess());
}

#[cfg(feature = "runtime-model-wgpu")]
fn coords_to_default_wgpu_tensor(coords: &[[u32; 4]]) -> Tensor<DefaultWgpuBackend, 2, Int> {
    let device = <DefaultWgpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let mut flat = Vec::with_capacity(coords.len().saturating_mul(4));
    for (row_idx, coord) in coords.iter().enumerate() {
        for value in coord {
            let converted = i32::try_from(*value).unwrap_or_else(|_| {
                panic!(
                    "coords_to_default_wgpu_tensor overflow at row {} value {}",
                    row_idx, value
                )
            });
            flat.push(converted);
        }
    }
    Tensor::<DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(flat, [coords.len(), 4]),
        &device,
    )
}

#[cfg(feature = "runtime-model")]
fn print_decoder_op_telemetry(label: &str, top_n: usize) {
    let telemetry = decoder_op_telemetry();
    println!(
        "runtime_decoder_hook_alignment_report {label}_op_telemetry calls={} total_ms={:.2} readback_count={} readback_elements={}",
        telemetry.calls, telemetry.total_ms, telemetry.readback_count, telemetry.readback_elements
    );
    for (rank, op) in telemetry.ops.iter().take(top_n).enumerate() {
        println!(
            "runtime_decoder_hook_alignment_report {label}_op_telemetry.rank={} calls={} total_ms={:.2} max_ms={:.2} context={}",
            rank + 1,
            op.calls,
            op.total_ms,
            op.max_ms,
            op.context
        );
    }
}

#[cfg(feature = "runtime-model")]
fn print_decoder_conv_block_telemetry(
    label: &str,
    telemetry: &crate::runtime_model::sparse_decoder::DecoderConvTelemetry,
    top_n: usize,
) {
    for (rank, block) in telemetry.blocks.iter().take(top_n).enumerate() {
        println!(
            "runtime_decoder_hook_alignment_report {label}_conv_block.rank={} context={} conv_calls={} wgpu_calls={} wgpu_successes={} wgpu_failures={} dispatches={} chunked_calls={} max_chunk_rows={} input_bytes={} output_bytes={} neighbor_elements={}",
            rank + 1,
            block.context,
            block.conv_calls,
            block.wgpu_calls,
            block.wgpu_successes,
            block.wgpu_failures,
            block.dispatches,
            block.chunked_calls,
            block.max_chunk_rows,
            block.input_bytes,
            block.output_bytes,
            block.neighbor_elements
        );
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn rows_to_default_wgpu_tensor<const C: usize>(rows: &[[f32; C]]) -> Tensor<DefaultWgpuBackend, 2> {
    let device = <DefaultWgpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let mut flat = Vec::with_capacity(rows.len().saturating_mul(C));
    for row in rows {
        flat.extend_from_slice(row);
    }
    Tensor::<DefaultWgpuBackend, 2>::from_data(TensorData::new(flat, [rows.len(), C]), &device)
}

fn dummy_textures() -> MeshPbrTextures {
    let rgba = vec![
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    MeshPbrTextures {
        base_color: crate::mesh::MeshTexture {
            width: 2,
            height: 2,
            rgba8: rgba.clone(),
        },
        metallic_roughness: crate::mesh::MeshTexture {
            width: 2,
            height: 2,
            rgba8: vec![
                0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255,
            ],
        },
        normal: None,
        emissive: None,
        occlusion: None,
    }
}

#[cfg(feature = "runtime-model")]
fn dummy_preprocess() -> PreprocessOutput {
    PreprocessOutput {
        width: 1,
        height: 1,
        rgb: vec![128, 64, 32],
    }
}

#[cfg(feature = "runtime-model")]
fn dummy_shape_tex_samples() -> (ShapeSLatSample, TexSLatSample) {
    let shape = ShapeSLatSample {
        sampler_config: FlowEulerSampleConfig {
            steps: 1,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        },
        sigma_min: 1.0e-3,
        step_count: 1,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features: vec![[0.0; 32]],
        noise: vec![[0.0; 32]],
        step_0_pred_v: vec![[0.0; 32]],
        step_0_pred_v_pos: vec![[0.0; 32]],
        step_0_pred_v_neg: vec![[0.0; 32]],
        step_0_x_t: vec![[0.0; 32]],
        step_mid_x_t: vec![[0.0; 32]],
        step_last_x_t: vec![[0.0; 32]],
        coords: vec![[0, 0, 0, 0]],
        layout: vec![0..1],
        flow_ops: SparseFlowOpTimingSummary::default(),
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu: None,
        #[cfg(feature = "runtime-model-wgpu")]
        features_wgpu: None,
    };
    let tex = TexSLatSample {
        sampler_config: FlowEulerSampleConfig {
            steps: 1,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        },
        sigma_min: 1.0e-3,
        step_count: 1,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features: vec![[0.0; 32]],
        noise: vec![[0.0; 32]],
        step_0_pred_v: vec![[0.0; 32]],
        step_0_pred_v_pos: vec![[0.0; 32]],
        step_0_pred_v_neg: vec![[0.0; 32]],
        step_0_x_t: vec![[0.0; 32]],
        step_mid_x_t: vec![[0.0; 32]],
        step_last_x_t: vec![[0.0; 32]],
        shape_slat_cond: vec![[0.0; 32]],
        coords: vec![[0, 0, 0, 0]],
        layout: vec![0..1],
        flow_ops: SparseFlowOpTimingSummary::default(),
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu: None,
        #[cfg(feature = "runtime-model-wgpu")]
        features_wgpu: None,
    };
    (shape, tex)
}

#[test]
fn sparse_coord_cap_requires_explicit_override() {
    assert_eq!(runtime_max_sparse_coords_for_backend("wgpu", None), None);
    assert_eq!(
        runtime_max_sparse_coords_for_backend("wgpu", Some(4096)),
        Some((4096, SparseCoordCapSource::ExplicitRunConfig))
    );
}

#[test]
fn canonical_wgpu_no_host_readback_before_extraction() {
    // Roadmap gate alias: canonical runtime modules must not gain new `.into_data()`
    // readbacks outside the extraction baseline allowlist.
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root should be two levels above burn_trellis crate")
        .to_path_buf();
    let canonical_files = [
        "crates/burn_trellis/src/runtime_model/sparse_structure_decoder.rs",
        "crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs",
        "crates/burn_trellis/src/runtime_model/sparse_decoder_runtime_impl.rs",
        "crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs",
        "crates/burn_trellis/src/runtime_model/fdg_decoder.rs",
        "crates/burn_trellis/src/runtime_model/sparse_unet_vae_decoder.rs",
        "crates/burn_trellis/src/staged_pipeline_runtime_helpers.rs",
        "crates/burn_trellis/src/staged_pipeline_runtime_decode.rs",
        "crates/burn_trellis/src/staged_pipeline_sampling.rs",
    ];

    let baseline_path = repo_root.join("scripts/guards/canonical_runtime_into_data.baseline");
    let expected = fs::read_to_string(&baseline_path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", baseline_path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut occurrences = Vec::<(String, usize)>::new();
    for rel_path in canonical_files {
        let abs_path = repo_root.join(rel_path);
        let source = fs::read_to_string(&abs_path)
            .unwrap_or_else(|err| panic!("failed reading {}: {err}", abs_path.display()));
        let scan_source = source
            .find("\n#[cfg(test)]\nmod tests")
            .map_or(source.as_str(), |idx| &source[..idx]);
        for (offset, _) in scan_source.match_indices(".into_data(") {
            let line_no = scan_source[..offset]
                .bytes()
                .filter(|byte| *byte == b'\n')
                .count()
                + 1;
            occurrences.push((rel_path.to_string(), line_no));
        }
    }
    occurrences
        .sort_by(|(file_a, line_a), (file_b, line_b)| file_a.cmp(file_b).then(line_a.cmp(line_b)));

    let mut actual = Vec::with_capacity(occurrences.len());
    let mut current_file: Option<String> = None;
    let mut current_count = 0usize;
    for (file, _) in occurrences {
        if current_file.as_ref() != Some(&file) {
            current_file = Some(file.clone());
            current_count = 0;
        }
        current_count += 1;
        actual.push(format!("{file}#{current_count}"));
    }

    assert_eq!(
        actual, expected,
        "canonical runtime `.into_data()` baseline changed; run scripts/guard_canonical_runtime.sh and update baseline only if intentional"
    );
}

#[test]
fn sample_voxel_attr_returns_none_for_sparse_holes() {
    let mut voxel_map = std::collections::HashMap::new();
    voxel_map.insert(
        super::pack_coord(10, 10, 10),
        [0.2, 0.3, 0.4, 0.1, 0.9, 1.0],
    );

    let sampled = super::sample_voxel_attr([0.0, 0.0, 0.0], &voxel_map, [512, 512, 512])
        .expect("sampling with non-empty map should not error");
    assert!(
        sampled.is_none(),
        "sparse hole should remain uncovered instead of hard-failing"
    );
}

#[test]
fn decode_pbr_device_path_sparse_hole_failfast() {
    // Roadmap gate alias: sparse-hole semantics remain strict/no-rescue.
    sample_voxel_attr_returns_none_for_sparse_holes();
}

#[test]
fn sample_voxel_attr_returns_value_when_supported() {
    let mut voxel_map = std::collections::HashMap::new();
    voxel_map.insert(
        super::pack_coord(10, 10, 10),
        [0.2, 0.3, 0.4, 0.1, 0.9, 1.0],
    );

    let position = [
        (10.0 / 512.0) - 0.5,
        (10.0 / 512.0) - 0.5,
        (10.0 / 512.0) - 0.5,
    ];
    let sampled = super::sample_voxel_attr(position, &voxel_map, [512, 512, 512])
        .expect("sampling with non-empty map should not error");
    assert!(sampled.is_some(), "expected local voxel support to resolve");
}

#[test]
fn dense_voxel_lookup_sampling_matches_sparse_hash_sampling() {
    let voxel_coords = vec![[0, 8, 8, 8], [0, 9, 8, 8], [0, 8, 9, 8], [0, 9, 9, 8]];
    let voxel_attrs = vec![
        [0.2, 0.3, 0.4, 0.1, 0.9, 1.0],
        [0.6, 0.1, 0.2, 0.3, 0.7, 1.0],
        [0.1, 0.8, 0.2, 0.4, 0.6, 1.0],
        [0.9, 0.4, 0.1, 0.6, 0.2, 1.0],
    ];
    let spatial = [16, 16, 16];

    let lookup =
        super::build_voxel_attr_lookup(voxel_coords.as_slice(), voxel_attrs.as_slice(), spatial)
            .expect("lookup build should succeed");
    let (occupancy, attrs) = match lookup {
        super::VoxelAttrLookup::Dense {
            occupancy, attrs, ..
        } => (occupancy, attrs),
        super::VoxelAttrLookup::Sparse { .. } => {
            panic!("expected dense lookup for bounded spatial volume")
        }
    };

    let mut sparse_map = std::collections::HashMap::new();
    for (coord, attrs) in voxel_coords.iter().zip(voxel_attrs.iter()) {
        sparse_map.insert(super::pack_coord(coord[1], coord[2], coord[3]), *attrs);
    }

    let positions = [
        [(8.0 / 16.0) - 0.5, (8.0 / 16.0) - 0.5, (8.0 / 16.0) - 0.5],
        [(8.5 / 16.0) - 0.5, (8.5 / 16.0) - 0.5, (8.0 / 16.0) - 0.5],
        [0.0, 0.0, 0.0],
    ];
    for position in positions {
        let dense = super::sample_voxel_attr_dense(
            position,
            occupancy.as_slice(),
            attrs.as_slice(),
            spatial,
        )
        .expect("dense lookup should not fail");
        let sparse = super::sample_voxel_attr(position, &sparse_map, spatial)
            .expect("sparse lookup should not fail");
        assert_eq!(dense.is_some(), sparse.is_some());
        if let (Some(dense), Some(sparse)) = (dense, sparse) {
            for ch in 0..6 {
                let diff = (dense[ch] - sparse[ch]).abs();
                assert!(
                    diff <= 1.0e-5,
                    "dense/sparse mismatch at ch={ch}: dense={} sparse={} diff={diff}",
                    dense[ch],
                    sparse[ch]
                );
            }
        }
    }
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn pbr_bake_wgpu_dense_sampling_matches_cpu_sampling() {
    if std::env::var("BURN_WGPU_SMOKE").is_err() {
        eprintln!(
            "Skipping pbr_bake_wgpu_dense_sampling_matches_cpu_sampling: set BURN_WGPU_SMOKE=1"
        );
        return;
    }

    let vertices = vec![
        [-0.25, 0.0, -0.25],
        [0.25, 0.0, -0.25],
        [0.25, 0.0, 0.25],
        [-0.25, 0.0, 0.25],
    ];
    let faces = vec![[0, 1, 2], [0, 2, 3]];

    let mut vox_coords = Vec::new();
    let mut vox_attrs = Vec::new();
    for z in 0..32u32 {
        for x in 0..32u32 {
            vox_coords.push([0, x, 16, z]);
            let fx = x as f32 / 31.0;
            let fz = z as f32 / 31.0;
            vox_attrs.push([
                fx,
                (1.0 - fx) * 0.7,
                fz,
                0.15 + 0.6 * fz,
                0.2 + 0.5 * fx,
                1.0,
            ]);
        }
    }

    let (_uv_cpu, tex_cpu, _debug_cpu) = super::bake_pbr_from_voxels_with_options(
        vertices.as_slice(),
        faces.as_slice(),
        None,
        vox_coords.as_slice(),
        vox_attrs.as_slice(),
        32,
        None,
        false,
        false,
    )
    .expect("cpu pbr bake must succeed");
    let (_uv_wgpu, tex_wgpu, _debug_wgpu) = super::bake_pbr_from_voxels_with_options(
        vertices.as_slice(),
        faces.as_slice(),
        None,
        vox_coords.as_slice(),
        vox_attrs.as_slice(),
        32,
        None,
        false,
        true,
    )
    .expect("wgpu pbr bake must succeed");

    let tex_cpu = tex_cpu.expect("cpu textures");
    let tex_wgpu = tex_wgpu.expect("wgpu textures");
    let assert_texture_match = |label: &str, lhs: &[u8], rhs: &[u8]| {
        if lhs.len() != rhs.len() {
            panic!(
                "{label} texture byte length mismatch: cpu={} wgpu={}",
                lhs.len(),
                rhs.len()
            );
        }
        let mut max_abs_diff = 0u8;
        let mut first_diff = None;
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = a.abs_diff(*b);
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            if diff > 1 {
                first_diff = Some((idx, *a, *b, diff));
                break;
            }
        }
        if let Some((diff_idx, cpu, wgpu, diff)) = first_diff {
            let texel = diff_idx / 4;
            let channel = diff_idx % 4;
            panic!(
                "{label} texture mismatch at texel={} channel={}: cpu={} wgpu={} abs_diff={} max_abs_diff={}",
                texel, channel, cpu, wgpu, diff, max_abs_diff
            );
        }
        // WGSL FMA/operation ordering is not bitwise-identical to CPU scalar math, but
        // parity is still numerically correct when quantized bytes differ by <= 1 LSB.
        assert!(
            max_abs_diff <= 1,
            "{label} texture parity exceeded 1-LSB tolerance: max_abs_diff={}",
            max_abs_diff
        );
    };

    assert_texture_match(
        "base-color",
        tex_cpu.base_color.rgba8.as_slice(),
        tex_wgpu.base_color.rgba8.as_slice(),
    );
    assert_texture_match(
        "metallic-roughness",
        tex_cpu.metallic_roughness.rgba8.as_slice(),
        tex_wgpu.metallic_roughness.rgba8.as_slice(),
    );
}

#[cfg(feature = "runtime-model")]
#[test]
fn cascade_token_budget_accepts_equal_token_count_without_backoff() {
    assert!(cascade_resolution_accepts_token_budget(
        49_152, 49_152, 1536
    ));
    assert!(!cascade_resolution_accepts_token_budget(
        49_153, 49_152, 1536
    ));
    assert!(cascade_resolution_accepts_token_budget(
        80_000, 49_152, 1024
    ));
}

#[cfg(feature = "runtime-model")]
#[test]
fn cascade_quantize_token_cap_boundary_parity() {
    assert!(cascade_resolution_accepts_token_budget(1024, 1024, 1024));
    // Canonical cascade allows the floor resolution (1024) to proceed even if
    // token count remains above budget.
    assert!(cascade_resolution_accepts_token_budget(1025, 1024, 1024));
    assert!(!cascade_resolution_accepts_token_budget(1025, 1024, 1152));
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn cascade_quantize_wgpu_matches_host_sort_dedup_semantics() {
    if std::env::var("BURN_WGPU_SMOKE").is_err() {
        eprintln!(
            "Skipping cascade_quantize_wgpu_matches_host_sort_dedup_semantics: set BURN_WGPU_SMOKE=1"
        );
        return;
    }

    let hr_coords = vec![
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 63, 63, 63],
        [0, 64, 64, 64],
        [0, 127, 127, 127],
        [0, 128, 128, 128],
        [0, 255, 255, 255],
    ];
    let host = super::quantize_cascade_coords(hr_coords.as_slice(), 512, 64)
        .expect("host quantize should succeed");

    let device =
        <super::SparseFlowWgpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let mut flat = Vec::with_capacity(hr_coords.len().saturating_mul(4));
    for coord in hr_coords {
        for value in coord {
            flat.push(i32::try_from(value).expect("coord fits i32"));
        }
    }
    let rows = flat.len() / 4;
    let hr_coords_t = Tensor::<super::SparseFlowWgpuBackend, 1, Int>::from_data(
        TensorData::new(flat, [rows.saturating_mul(4)]),
        &device,
    )
    .reshape([rows, 4]);
    let quant_t = super::quantize_cascade_coords_wgpu(hr_coords_t, 512, 64)
        .expect("wgpu quantize should succeed");
    let [quant_rows, quant_cols] = quant_t.dims();
    assert_eq!(quant_cols, 4);
    let values = quant_t
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .expect("wgpu quantized coords into vec");
    assert_eq!(values.len(), quant_rows.saturating_mul(4));
    let mut got = Vec::with_capacity(quant_rows);
    for row_idx in 0..quant_rows {
        let base = row_idx.saturating_mul(4);
        got.push([
            u32::try_from(values[base]).expect("batch must be non-negative"),
            u32::try_from(values[base + 1]).expect("x must be non-negative"),
            u32::try_from(values[base + 2]).expect("y must be non-negative"),
            u32::try_from(values[base + 3]).expect("z must be non-negative"),
        ]);
    }
    assert_eq!(got, host);
}

#[cfg(feature = "runtime-model")]
#[test]
fn sparse_layout_from_coords_tracks_real_batched_ranges() {
    let coords = vec![[0, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [2, 1, 0, 0]];
    let layout = sparse_layout_from_coords(coords.as_slice()).expect("layout");
    assert_eq!(layout, vec![0..2, 2..2, 2..4]);
}

#[cfg(feature = "runtime-model")]
#[test]
fn sparse_layout_from_coords_rejects_non_grouped_batch_rows() {
    let coords = vec![[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]];
    let err = sparse_layout_from_coords(coords.as_slice())
        .expect_err("non-grouped batch coords must fail");
    assert!(err.contains("grouped by non-decreasing batch id"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn sparse_layout_from_batch_ids_tracks_real_batched_ranges() {
    let batch_ids = vec![0usize, 0usize, 2usize, 2usize];
    let layout = sparse_layout_from_batch_ids(batch_ids.as_slice(), "unit").expect("layout");
    assert_eq!(layout, vec![0..2, 2..2, 2..4]);
}

#[cfg(feature = "runtime-model")]
#[test]
fn sparse_layout_from_batch_ids_rejects_non_grouped_rows() {
    let batch_ids = vec![0usize, 1usize, 0usize];
    let err =
        sparse_layout_from_batch_ids(batch_ids.as_slice(), "unit").expect_err("layout must fail");
    assert!(err.contains("grouped by non-decreasing batch id"));
    assert!(err.contains("unit"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn validate_sparse_layout_rows_accepts_contiguous_layout() {
    let layout = vec![0..2, 2..2, 2..4];
    validate_sparse_layout_rows(layout.as_slice(), 4, "unit").expect("valid layout");
}

#[cfg(feature = "runtime-model")]
#[test]
fn validate_sparse_layout_rows_rejects_row_mismatch() {
    let layout = vec![0..2, 2..3];
    let err = validate_sparse_layout_rows(layout.as_slice(), 4, "unit").expect_err("must fail");
    assert!(err.contains("layout_rows=3 expected_rows=4"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn conditioning_path_requires_explicit_conditioning_tensors() {
    let preprocess = dummy_preprocess();
    let err = dense_cond_with_override(&preprocess, 32 * 32 + 5, 1024, None, "sparse_runtime")
        .expect_err("missing conditioning tensors must not synthesize fallback features");
    assert!(err.contains("missing TRELLIS image conditioning"));
    assert!(err.contains("no synthetic/degraded fallback"));
    assert!(err.contains("get_cond_512.out.cond"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn conditioning_path_uses_explicit_overrides_without_fallback() {
    let preprocess = dummy_preprocess();
    let expected = (0..12).map(|idx| idx as f32 * 0.1).collect::<Vec<_>>();
    let values = dense_cond_with_override(&preprocess, 4, 3, Some(expected.as_slice()), "unit")
        .expect("exact conditioning override should pass through unchanged");
    assert_eq!(values.as_ref(), expected.as_slice());
}

#[cfg(feature = "runtime-model")]
#[test]
fn staged_runtime_fails_fast_when_conditioning_unavailable() {
    let config = TrellisPipelineConfig::from_json_bytes(
        br#"{
                "name": "Trellis2ImageTo3DPipeline",
                "args": {
                    "models": {},
                    "default_pipeline_type": "1024_cascade"
                }
            }"#,
    )
    .expect("pipeline config should parse");
    let runtime = super::TrellisStageRuntime::from_args_with_assets(
        &config.args,
        None,
        None,
        None,
        false,
        None,
    );
    let err = runtime
        .run_profiled_with_overrides(
            &dummy_preprocess(),
            42,
            None,
            false,
            super::TrellisStageRunConfig::default(),
        )
        .expect_err("runtime must fail before staged sampling when conditioning is missing");
    assert!(err.contains("missing TRELLIS image conditioning"));
    assert!(err.contains("no synthetic/degraded fallback"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn runtime_conditioning_preflight_accepts_explicit_overrides() {
    let config = TrellisPipelineConfig::from_json_bytes(
        br#"{
                "name": "Trellis2ImageTo3DPipeline",
                "args": {
                    "models": {},
                    "default_pipeline_type": "1024_cascade"
                }
            }"#,
    )
    .expect("pipeline config should parse");
    let runtime = super::TrellisStageRuntime::from_args_with_assets(
        &config.args,
        None,
        None,
        None,
        false,
        None,
    );
    let overrides = super::TrellisNoiseOverrides {
        cond_512: Some(vec![0.25]),
        neg_cond_512: Some(vec![0.0]),
        cond_1024: Some(vec![0.5]),
        neg_cond_1024: Some(vec![0.0]),
        ..Default::default()
    };
    let conditioning = runtime
        .resolve_runtime_conditioning(&dummy_preprocess(), Some(&overrides))
        .expect("explicit overrides should satisfy conditioning preflight");
    assert_eq!(conditioning.cond_512.len(), (32 * 32 + 5) * 1024);
    assert_eq!(conditioning.neg_cond_512.len(), (32 * 32 + 5) * 1024);
    assert_eq!(
        conditioning.cond_1024.as_ref().map(|values| values.len()),
        Some((64 * 64 + 5) * 1024)
    );
    assert_eq!(
        conditioning
            .neg_cond_1024
            .as_ref()
            .map(|values| values.len()),
        Some((64 * 64 + 5) * 1024)
    );
}

#[cfg(feature = "runtime-model")]
#[test]
fn merge_voxel_attrs_strict_rejects_coord_mismatch() {
    let err = merge_voxel_attrs_for_decode(
        &[[0, 1, 1, 1], [0, 2, 2, 2]],
        &[[0, 1, 1, 1]],
        &[[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
        true,
    )
    .expect_err("strict mode should reject tex/shape coord mismatch");
    assert!(err.contains("coords differ"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn merge_voxel_attrs_non_strict_rejects_coord_mismatch() {
    let err = merge_voxel_attrs_for_decode(
        &[[0, 1, 1, 1], [0, 2, 2, 2]],
        &[[0, 1, 1, 1]],
        &[[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]],
        false,
    )
    .expect_err("non-strict mode should also reject tex/shape coord mismatch");
    assert!(err.contains("coords differ"));
}

#[test]
fn pbr_bake_produces_textures_and_uvs() {
    let vertices = vec![[-0.02, 0.0, -0.02], [0.02, 0.0, -0.02], [0.0, 0.0, 0.02]];
    let faces = vec![[0, 1, 2]];
    let vox_coords = vec![[0, 16, 16, 16], [0, 20, 16, 16], [0, 16, 20, 16]];
    let vox_attrs = vec![
        [0.8, 0.2, 0.1, 0.1, 0.8, 1.0],
        [0.1, 0.8, 0.2, 0.3, 0.6, 1.0],
        [0.2, 0.1, 0.8, 0.5, 0.4, 1.0],
    ];

    let (uvs, textures, debug) =
        bake_pbr_from_voxels(&vertices, &faces, &vox_coords, &vox_attrs, 32)
            .expect("pbr bake should succeed");
    assert_eq!(uvs.len(), vertices.len());
    let textures = textures.expect("pbr textures should exist");
    assert!(textures.base_color.width >= 64);
    assert_eq!(
        textures.base_color.rgba8.len(),
        (textures.base_color.width * textures.base_color.height * 4) as usize
    );
    assert!(debug.raster_mask.iter().any(|value| *value != 0));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn pbr_bake_reused_projection_bvh_matches_owned_projection_bvh() {
    let vertices = vec![[-0.02, 0.0, -0.02], [0.02, 0.0, -0.02], [0.0, 0.0, 0.02]];
    let faces = vec![[0, 1, 2]];
    let projection_vertices = vertices.clone();
    let projection_faces = faces.clone();
    let projection_source = super::PbrProjectionSource {
        vertices: projection_vertices.as_slice(),
        faces: projection_faces.as_slice(),
    };
    let vox_coords = vec![[0, 16, 16, 16], [0, 20, 16, 16], [0, 16, 20, 16]];
    let vox_attrs = vec![
        [0.8, 0.2, 0.1, 0.1, 0.8, 1.0],
        [0.1, 0.8, 0.2, 0.3, 0.6, 1.0],
        [0.2, 0.1, 0.8, 0.5, 0.4, 1.0],
    ];

    let (owned_mesh, owned_textures, _) = super::bake_pbr_from_voxels_with_options(
        vertices.as_slice(),
        faces.as_slice(),
        Some(projection_source),
        vox_coords.as_slice(),
        vox_attrs.as_slice(),
        32,
        None,
        false,
        false,
    )
    .expect("owned projection bvh bake should succeed");
    let projection_bvh = super::build_projection_bvh_for_pbr(projection_source)
        .expect("projection bvh should build");
    let (reused_mesh, reused_textures, _) =
        super::bake_pbr_from_voxels_with_options_and_projection_bvh(
            vertices.as_slice(),
            faces.as_slice(),
            Some(projection_source),
            Some(&projection_bvh),
            vox_coords.as_slice(),
            vox_attrs.as_slice(),
            32,
            None,
            false,
            false,
        )
        .expect("reused projection bvh bake should succeed");

    assert_eq!(reused_mesh.vertices, owned_mesh.vertices);
    assert_eq!(reused_mesh.faces, owned_mesh.faces);
    assert_eq!(reused_mesh.uvs, owned_mesh.uvs);
    assert_eq!(reused_textures, owned_textures);
}

#[test]
fn pbr_debug_samples_are_first_hit_bounded() {
    let vertices = vec![[-0.03, 0.0, -0.03], [0.03, 0.0, -0.03], [0.0, 0.0, 0.03]];
    // Duplicate the face to force overdraw. Debug hooks should still record one
    // accepted sample per covered texel, not every raster candidate.
    let faces = vec![[0, 1, 2], [0, 1, 2]];
    let vox_coords = vec![[0, 16, 16, 16], [0, 18, 16, 16], [0, 16, 18, 16]];
    let vox_attrs = vec![
        [0.8, 0.2, 0.1, 0.1, 0.8, 1.0],
        [0.1, 0.8, 0.2, 0.3, 0.6, 1.0],
        [0.2, 0.1, 0.8, 0.5, 0.4, 1.0],
    ];

    let (_, _, debug) = bake_pbr_from_voxels(&vertices, &faces, &vox_coords, &vox_attrs, 32)
        .expect("pbr bake should succeed");
    let covered = debug
        .raster_mask
        .iter()
        .filter(|value| **value != 0)
        .count();
    assert!(covered > 0);
    assert_eq!(debug.sample_positions.len(), covered);
    assert_eq!(debug.sample_attrs.len(), covered);
    assert!(debug.sample_positions.len() <= debug.texture_width * debug.texture_height);
}

#[test]
fn pbr_quantization_tracks_float_buffers() {
    let vertices = vec![
        [-0.03, 0.0, -0.03],
        [0.03, 0.0, -0.03],
        [0.03, 0.0, 0.03],
        [-0.03, 0.0, 0.03],
    ];
    let faces = vec![[0, 1, 2], [0, 2, 3]];
    let vox_coords = vec![
        [0, 16, 16, 16],
        [0, 17, 16, 16],
        [0, 17, 17, 16],
        [0, 16, 17, 16],
    ];
    let vox_attrs = vec![
        [0.2, 0.3, 0.4, 0.2, 0.7, 1.0],
        [0.5, 0.6, 0.7, 0.4, 0.5, 1.0],
        [0.8, 0.6, 0.3, 0.6, 0.4, 1.0],
        [0.4, 0.2, 0.1, 0.1, 0.8, 1.0],
    ];

    let (_, _, debug) = bake_pbr_from_voxels(&vertices, &faces, &vox_coords, &vox_attrs, 32)
        .expect("pbr bake should succeed");
    assert!(!debug.base_color_float.is_empty());
    assert_eq!(
        debug.base_color_float.len(),
        debug.texture_width * debug.texture_height
    );
    assert_eq!(
        debug.metallic_float.len(),
        debug.texture_width * debug.texture_height
    );

    for (idx, rgba) in debug.base_color_float.iter().enumerate() {
        let off = idx * 4;
        let expected = [
            (rgba[0].clamp(0.0, 1.0) * 255.0).round() as i32,
            (rgba[1].clamp(0.0, 1.0) * 255.0).round() as i32,
            (rgba[2].clamp(0.0, 1.0) * 255.0).round() as i32,
            (debug.alpha_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32,
        ];
        for (channel, expected_value) in expected.iter().enumerate() {
            let actual = debug.base_color_rgba_u8[off + channel] as i32;
            assert!(
                (actual - *expected_value).abs() <= 1,
                "base channel mismatch idx={idx} ch={channel}: actual={actual}, expected={}",
                expected_value
            );
        }

        let expected_metallic = (debug.metallic_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32;
        let expected_roughness =
            (debug.roughness_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32;
        let mr = &debug.metallic_roughness_u8[off..off + 4];
        assert!((mr[1] as i32 - expected_roughness).abs() <= 1);
        assert!((mr[2] as i32 - expected_metallic).abs() <= 1);
    }
}

#[test]
fn pbr_inpaint_fills_uncovered_texels_without_hiding_raster_mask() {
    let texture_size = 3;
    let texels = texture_size * texture_size;
    let center = 4;
    let mut mask = vec![0u8; texels];
    mask[center] = 255;
    let mut base_color_float = vec![[0.0f32; 4]; texels];
    let mut metallic_float = vec![0.0f32; texels];
    let mut roughness_float = vec![1.0f32; texels];
    let mut alpha_float = vec![0.0f32; texels];
    base_color_float[center] = [0.25, 0.5, 0.75, 1.0];
    metallic_float[center] = 0.4;
    roughness_float[center] = 0.6;
    alpha_float[center] = 1.0;

    super::inpaint_texture_channels(
        texture_size,
        mask.as_mut_slice(),
        base_color_float.as_mut_slice(),
        metallic_float.as_mut_slice(),
        roughness_float.as_mut_slice(),
        alpha_float.as_mut_slice(),
    )
    .expect("inpaint should fill from the single covered texel");

    for idx in 0..texels {
        for channel in 0..4 {
            assert!(
                (base_color_float[idx][channel] - base_color_float[center][channel]).abs() < 1.0e-6
            );
        }
        assert!((metallic_float[idx] - metallic_float[center]).abs() < 1.0e-6);
        assert!((roughness_float[idx] - roughness_float[center]).abs() < 1.0e-6);
        assert!((alpha_float[idx] - alpha_float[center]).abs() < 1.0e-6);
    }
    assert_eq!(mask.iter().filter(|value| **value != 0).count(), 1);
    assert_eq!(mask[center], 255);
}

#[test]
fn pbr_glb_output_uvs_flip_v_from_raster_uvs() {
    let output = super::glb_output_uvs_from_raster_uvs(&[[0.25, 0.2], [1.2, -0.2]]);
    assert_eq!(output, vec![[0.25, 0.8], [1.0, 1.0]]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_pbr_simple_dc_remesh_extracts_nonempty_surface() {
    let vertices = vec![
        [-0.25, -0.25, -0.25],
        [0.25, -0.25, -0.25],
        [0.25, 0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25],
        [0.25, -0.25, 0.25],
        [0.25, 0.25, 0.25],
        [-0.25, 0.25, 0.25],
    ];
    let faces = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];

    let bvh = super::build_projection_bvh_for_pbr(super::PbrProjectionSource {
        vertices: vertices.as_slice(),
        faces: faces.as_slice(),
    })
    .expect("projection bvh should build");
    let (remeshed_vertices, remeshed_faces) =
        super::remesh_narrow_band_simple_dc_with_projection_bvh(&bvh, 32, 1.0)
            .expect("simple dc remesh should extract a surface");

    assert!(!remeshed_vertices.is_empty());
    assert!(!remeshed_faces.is_empty());
    assert!(remeshed_faces.iter().all(|face| {
        face.iter()
            .all(|index| (*index as usize) < remeshed_vertices.len())
    }));
    assert!(remeshed_vertices.iter().all(|vertex| {
        vertex
            .iter()
            .all(|component| component.is_finite() && component.abs() <= 0.6)
    }));
}

#[test]
fn pbr_uv_domain_is_portable_and_seam_split() {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let faces = vec![[0, 1, 2], [0, 1, 3]];
    let domain = super::build_uv_raster_domain(vertices.as_slice(), faces.as_slice(), 64);

    assert_eq!(domain.output_faces.len(), faces.len());
    assert_eq!(domain.raster_faces.len(), faces.len());
    assert_eq!(domain.output_vertices.len(), faces.len() * 3);
    assert_eq!(domain.raster_vertices.len(), faces.len() * 3);
    assert_eq!(domain.output_uvs.len(), faces.len() * 3);
    assert_eq!(domain.raster_uvs.len(), faces.len() * 3);
    assert!(domain.output_vertices.len() > vertices.len());
    for uv in domain.raster_uvs.iter().chain(domain.output_uvs.iter()) {
        assert!((0.0..=1.0).contains(&uv[0]), "u out of range: {uv:?}");
        assert!((0.0..=1.0).contains(&uv[1]), "v out of range: {uv:?}");
    }
}

#[test]
fn pbr_bake_benchmark_report() {
    if std::env::var("TRELLIS2_PBR_BENCH").is_err() {
        eprintln!("skipping: set TRELLIS2_PBR_BENCH=1 to run pbr_bake_benchmark_report");
        return;
    }

    let grid = env_usize("TRELLIS2_PBR_BENCH_GRID").unwrap_or(96).max(8);
    let warmup = env_usize("TRELLIS2_PBR_BENCH_WARMUP").unwrap_or(1).max(1);
    let iters = env_usize("TRELLIS2_PBR_BENCH_ITERS").unwrap_or(3).max(1);
    let fallback_res = env_usize("TRELLIS2_PBR_BENCH_FALLBACK_RES")
        .unwrap_or(64)
        .clamp(16, 512) as u32;
    let prefer_wgpu_sampling = env_usize("TRELLIS2_PBR_BENCH_WGPU").unwrap_or(0) != 0;

    let mut vertices = Vec::with_capacity((grid + 1) * (grid + 1));
    for z in 0..=grid {
        for x in 0..=grid {
            let xf = (x as f32 / grid as f32 - 0.5) * 0.2;
            let zf = (z as f32 / grid as f32 - 0.5) * 0.2;
            vertices.push([xf, 0.0, zf]);
        }
    }
    let mut faces = Vec::with_capacity(grid * grid * 2);
    for z in 0..grid {
        for x in 0..grid {
            let row = z * (grid + 1);
            let i0 = (row + x) as u32;
            let i1 = (row + x + 1) as u32;
            let i2 = (row + x + grid + 1) as u32;
            let i3 = (row + x + grid + 2) as u32;
            faces.push([i0, i1, i3]);
            faces.push([i0, i3, i2]);
        }
    }

    let mut vox_coords = Vec::with_capacity(64 * 64);
    let mut vox_attrs = Vec::with_capacity(64 * 64);
    for z in 0..64u32 {
        for x in 0..64u32 {
            vox_coords.push([0, x, 32, z]);
            let fx = x as f32 / 63.0;
            let fz = z as f32 / 63.0;
            vox_attrs.push([
                fx,
                (1.0 - fx) * 0.8,
                fz,
                0.1 + 0.7 * fz,
                0.2 + 0.7 * fx,
                1.0,
            ]);
        }
    }

    let mut run_ms = Vec::with_capacity(iters);
    let mut covered_texels = 0usize;
    for step in 0..(warmup + iters) {
        let start = std::time::Instant::now();
        let (_uvs, textures, debug) = super::bake_pbr_from_voxels_with_options(
            vertices.as_slice(),
            faces.as_slice(),
            None,
            vox_coords.as_slice(),
            vox_attrs.as_slice(),
            fallback_res,
            None,
            false,
            prefer_wgpu_sampling,
        )
        .expect("pbr bench bake must succeed");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if step >= warmup {
            run_ms.push(elapsed_ms);
            if let Some(tex) = textures.as_ref() {
                covered_texels = tex
                    .base_color
                    .rgba8
                    .chunks_exact(4)
                    .filter(|rgba| rgba[3] != 0)
                    .count();
            }
            let _ = debug;
        }
    }

    run_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ms = run_ms.iter().sum::<f64>() / run_ms.len() as f64;
    let min_ms = *run_ms.first().unwrap_or(&0.0);
    let p50_ms = run_ms[(run_ms.len() - 1) / 2];
    let p90_idx = (((run_ms.len() - 1) as f64) * 0.9).round() as usize;
    let p90_ms = run_ms[p90_idx.min(run_ms.len() - 1)];

    println!(
        concat!(
            "PBR_BENCH_RESULT,",
            "grid={},",
            "vertices={},",
            "triangles={},",
            "voxels={},",
            "fallback_res={},",
            "iters={},",
            "mean_ms={:.3},",
            "min_ms={:.3},",
            "p50_ms={:.3},",
            "p90_ms={:.3},",
            "covered_texels={}"
        ),
        grid,
        vertices.len(),
        faces.len(),
        vox_coords.len(),
        fallback_res,
        iters,
        mean_ms,
        min_ms,
        p50_ms,
        p90_ms,
        covered_texels
    );
}

#[test]
fn material_summary_prefers_texture_data_when_available() {
    let textures = dummy_textures();
    let material = summarize_material(&[[0.0; 6]], Some(&textures)).expect("material");
    assert!(material.base_color[0] > 0.1);
    assert!(material.alpha > 0.8);
}

#[cfg(feature = "runtime-model")]
#[test]
fn decode_missing_runtime_decoders_errors_when_not_strict() {
    let (shape, tex) = dummy_shape_tex_samples();
    let err = decode_latent_to_outputs(
        &shape,
        &tex,
        "512",
        None,
        None,
        None,
        false,
        false,
        DecodeHookOverrides::default(),
        Default::default(),
        RuntimeDecodeModels::default(),
    )
    .expect_err("decode should fail when runtime decoders are missing");
    assert!(err.contains("shape runtime decoder is required"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn decode_missing_runtime_decoders_errors_when_strict() {
    let (shape, tex) = dummy_shape_tex_samples();
    let err = decode_latent_to_outputs(
        &shape,
        &tex,
        "512",
        None,
        None,
        None,
        true,
        false,
        DecodeHookOverrides::default(),
        Default::default(),
        RuntimeDecodeModels::default(),
    )
    .expect_err("strict decode should fail when runtime decoders are missing");
    assert!(err.contains("shape runtime decoder is required"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn decode_rejects_decode_hook_overrides() {
    let (shape, tex) = dummy_shape_tex_samples();
    let override_levels = vec![DecodeShapeSubSample {
        coords: vec![[0, 0, 0, 0]],
        feats: vec![[0.0; 8]],
        spatial_shape: [1, 1, 1],
    }];
    let err = decode_latent_to_outputs(
        &shape,
        &tex,
        "512",
        None,
        None,
        None,
        true,
        false,
        DecodeHookOverrides {
            decode_shape_subs: Some(override_levels.as_slice()),
            ..DecodeHookOverrides::default()
        },
        Default::default(),
        RuntimeDecodeModels::default(),
    )
    .expect_err("decode should reject hook overrides on canonical runtime path");
    assert!(err.contains("decode hook override tensors are disabled"));
}

#[cfg(feature = "runtime-model")]
#[test]
fn runtime_decoder_hook_alignment_report() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference_path = std::env::var("TRELLIS2_DECODER_REFERENCE_HOOK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            root.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors")
        });
    if !reference_path.exists() {
        trellis_stage_log!(
            "Skipping runtime_decoder_hook_alignment_report: missing reference hook '{}'",
            reference_path.display()
        );
        return;
    }
    let reference = HookSnapshot::from_file(&reference_path).expect("reference hook should load");

    let has_decode_inputs = reference
        .tensors
        .contains_key("decode_shape_slat.input.coords")
        && reference
            .tensors
            .contains_key("decode_shape_slat.input.feats");
    let strict_subdiv_checks = env_flag("TRELLIS2_PARITY_STRICT")
        || env_flag("TRELLIS2_E2E_STRICT")
        || env_flag("TRELLIS2_DECODER_SUBDIV_REQUIRE_DECODE_INPUTS");
    assert!(
        has_decode_inputs,
        "runtime_decoder_hook_alignment_report: reference hook '{}' must include decode_shape_slat.input.* keys",
        reference_path.display()
    );
    let shape_coords = tensor_to_coords4(
        reference
            .tensors
            .get("decode_shape_slat.input.coords")
            .expect("missing decode_shape_slat.input.coords"),
    )
    .expect("decode input coords should decode");
    let shape_feats = tensor_to_rows::<32>(
        reference
            .tensors
            .get("decode_shape_slat.input.feats")
            .expect("missing decode_shape_slat.input.feats"),
    )
    .expect("decode input feats should decode");
    let tex_coords_key = if reference
        .tensors
        .contains_key("sample_tex_slat.slat.coords")
    {
        "sample_tex_slat.slat.coords"
    } else {
        "decode_tex_slat.input.coords"
    };
    let tex_feats_key = if reference.tensors.contains_key("sample_tex_slat.slat.feats") {
        "sample_tex_slat.slat.feats"
    } else {
        "decode_tex_slat.input.feats"
    };
    let tex_coords = tensor_to_coords4(
        reference
            .tensors
            .get(tex_coords_key)
            .unwrap_or_else(|| panic!("missing {tex_coords_key}")),
    )
    .expect("tex coords should decode");
    let tex_feats = tensor_to_rows::<32>(
        reference
            .tensors
            .get(tex_feats_key)
            .unwrap_or_else(|| panic!("missing {tex_feats_key}")),
    )
    .expect("tex feats should decode");
    let reference_voxel_coords = tensor_to_coords4(
        reference
            .tensors
            .get("decode_tex_slat.voxels.coords")
            .expect("missing decode_tex_slat.voxels.coords"),
    )
    .expect("reference voxel coords should decode");
    let reference_voxel_feats = tensor_to_rows::<6>(
        reference
            .tensors
            .get("decode_tex_slat.voxels.feats")
            .expect("missing decode_tex_slat.voxels.feats"),
    )
    .expect("reference voxel feats should decode");
    let reference_subdivisions = load_reference_subdivisions(&reference)
        .expect("reference shape subdivisions should decode");

    let mut rows = shape_coords
        .len()
        .min(shape_feats.len())
        .min(tex_coords.len())
        .min(tex_feats.len());
    assert!(rows > 0, "reference hooks must contain slat rows");
    if let Ok(value) = std::env::var("TRELLIS2_DECODER_TEST_MAX_ROWS")
        && let Ok(cap) = value.trim().parse::<usize>()
        && cap > 0
        && rows > cap
    {
        assert!(
            !strict_subdiv_checks,
            "runtime_decoder_hook_alignment_report: TRELLIS2_DECODER_TEST_MAX_ROWS={} is not allowed in strict subdivision mode because sparse conv neighborhoods depend on full coordinate context",
            cap
        );
        rows = cap;
    }

    let weights_root = resolve_trellis2_weights_root(None);
    if !weights_root.exists() {
        trellis_stage_log!(
            "Skipping runtime_decoder_hook_alignment_report: missing weights root '{}'",
            weights_root.display()
        );
        return;
    }
    let image_large_root = resolve_trellis2_image_large_root(None);
    let image_large_root_opt = if image_large_root.exists() {
        Some(image_large_root)
    } else {
        None
    };

    let pipeline_bytes =
        std::fs::read(weights_root.join("pipeline.json")).expect("pipeline.json should load");
    let pipeline = TrellisPipelineConfig::from_json_bytes(pipeline_bytes.as_slice())
        .expect("pipeline config should parse");
    let shape_stem = pipeline
        .args
        .models
        .get("shape_slat_decoder")
        .expect("shape_slat_decoder model stem missing");
    let tex_stem = pipeline
        .args
        .models
        .get("tex_slat_decoder")
        .expect("tex_slat_decoder model stem missing");
    set_runtime_model_debug_config(RuntimeModelDebugConfig {
        sparse_decoder_conv_f16: env_flag_default("TRELLIS2_DECODER_CONV_F16", true),
        ..RuntimeModelDebugConfig::default()
    });

    let shape_decoder = FdgDecoderRuntime::load_from_stem(
        weights_root.as_path(),
        image_large_root_opt.as_deref(),
        shape_stem.as_str(),
        false,
    )
    .expect("shape decoder should load");
    let tex_decoder = SparseUnetVaeDecoderRuntime::load_from_stem(
        weights_root.as_path(),
        image_large_root_opt.as_deref(),
        tex_stem.as_str(),
        false,
    )
    .expect("tex decoder should load");

    reset_decoder_conv_telemetry();
    reset_decoder_op_telemetry();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_decoded = {
        // Canonical decode parity must exercise the tensor-native entrypoint.
        // The host decode API is intentionally blocked on runtime WGPU path.
        let shape_coords_t = coords_to_default_wgpu_tensor(&shape_coords[..rows]);
        let shape_feats_t = rows_to_default_wgpu_tensor::<32>(&shape_feats[..rows]);
        let decoded = shape_decoder
            .decode_sparse_result_with_tensors(shape_coords_t, shape_feats_t)
            .expect("shape decoder should run");
        decode_fdg_outputs(&decoded, shape_decoder.voxel_margin())
            .expect("shape decoder outputs should decode")
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let shape_decoded = shape_decoder
        .decode_sparse(&shape_coords[..rows], &shape_feats[..rows])
        .expect("shape decoder should run");
    let shape_subdivisions_host = shape_decoded
        .subdivisions
        .iter()
        .map(|sub| {
            let coords =
                sub.coords_host("runtime_decoder_hook_alignment_report shape subdivision coords")?;
            let logits =
                sub.logits_host("runtime_decoder_hook_alignment_report shape subdivision logits")?;
            SparseSubdivisionLogits::from_host(sub.spatial_shape, coords, logits)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("shape decoder subdivisions should materialize");
    let shape_conv_telemetry = decoder_conv_telemetry();
    print_decoder_op_telemetry("shape_decoder", 16);
    print_decoder_conv_block_telemetry("shape_decoder", &shape_conv_telemetry, 20);
    println!(
        "runtime_decoder_hook_alignment_report shape_decoder_telemetry conv_calls={} wgpu_calls={} wgpu_successes={} wgpu_failures={} dispatches={} chunked_calls={} max_chunk_rows={} input_bytes={} output_bytes={} neighbor_elements={}",
        shape_conv_telemetry.conv_calls,
        shape_conv_telemetry.wgpu_calls,
        shape_conv_telemetry.wgpu_successes,
        shape_conv_telemetry.wgpu_failures,
        shape_conv_telemetry.dispatches,
        shape_conv_telemetry.chunked_calls,
        shape_conv_telemetry.max_chunk_rows,
        shape_conv_telemetry.input_bytes,
        shape_conv_telemetry.output_bytes,
        shape_conv_telemetry.neighbor_elements
    );
    if env_flag("TRELLIS2_DECODER_REQUIRE_WGPU_SUCCESS") {
        assert!(
            shape_conv_telemetry.wgpu_successes > 0,
            "runtime_decoder_hook_alignment_report: expected shape decoder wgpu successes > 0"
        );
        assert!(
            shape_conv_telemetry.wgpu_failures == 0,
            "runtime_decoder_hook_alignment_report: shape decoder had wgpu failures={}",
            shape_conv_telemetry.wgpu_failures
        );
        assert!(
            shape_conv_telemetry.dispatches > 0,
            "runtime_decoder_hook_alignment_report: expected shape decoder wgpu dispatches > 0"
        );
    }
    #[cfg(feature = "runtime-model-wgpu")]
    if shape_conv_telemetry.wgpu_calls > 0 {
        assert!(
            shape_decoded
                .subdivisions
                .iter()
                .all(|sub| sub.device_tensors().is_some()),
            "runtime_decoder_hook_alignment_report: shape decoder produced host-only subdivisions despite wgpu path"
        );
    }
    let default_subdiv_threshold = if strict_subdiv_checks {
        Some(1.0e-2f32)
    } else {
        None
    };
    let global_subdiv_max_mean_abs =
        env_f32("TRELLIS2_DECODER_SUBDIV_MAX_MEAN_ABS").or(default_subdiv_threshold);
    let global_subdiv_max_rmse =
        env_f32("TRELLIS2_DECODER_SUBDIV_MAX_RMSE").or(default_subdiv_threshold);
    let global_subdiv_max_abs =
        env_f32("TRELLIS2_DECODER_SUBDIV_MAX_ABS").or(default_subdiv_threshold);
    let mut compared_subdiv_levels = 0usize;
    for (level, reference_sub) in reference_subdivisions.iter().enumerate() {
        let Some(actual_sub) = shape_subdivisions_host.get(level) else {
            if strict_subdiv_checks {
                panic!(
                    "runtime_decoder_hook_alignment_report: missing actual subdivision level {} (actual_levels={} reference_levels={})",
                    level,
                    shape_subdivisions_host.len(),
                    reference_subdivisions.len()
                );
            }
            continue;
        };
        compared_subdiv_levels += 1;
        let (sub_stats, sub_overlap, actual_sub_rows, reference_sub_rows) =
            compare_subdivision_overlap(actual_sub, reference_sub);
        let (actual_min, actual_max, actual_mean) = tensor_stats(actual_sub.logits.as_slice());
        let (reference_min, reference_max, reference_mean) =
            tensor_stats(reference_sub.logits.as_slice());
        println!(
            "runtime_decoder_hook_alignment_report shape_subdiv.level={} overlap={} actual_rows={} reference_rows={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e} actual[min,max,mean]=[{:.6e},{:.6e},{:.6e}] reference[min,max,mean]=[{:.6e},{:.6e},{:.6e}]",
            level,
            sub_overlap,
            actual_sub_rows,
            reference_sub_rows,
            sub_stats.mean_abs,
            sub_stats.max_abs,
            sub_stats.rmse,
            actual_min,
            actual_max,
            actual_mean,
            reference_min,
            reference_max,
            reference_mean
        );
        if let Some(top_k) = env_usize("TRELLIS2_DECODER_SUBDIV_TOPK")
            && top_k > 0
        {
            for (rank, entry) in top_subdivision_diffs(actual_sub, reference_sub, top_k)
                .into_iter()
                .enumerate()
            {
                println!(
                    "runtime_decoder_hook_alignment_report shape_subdiv.level={} top_diff.rank={} coord=[{},{},{},{}] child={} abs_diff={:.6e} actual={:.6e} reference={:.6e}",
                    level,
                    rank + 1,
                    entry.coord[0],
                    entry.coord[1],
                    entry.coord[2],
                    entry.coord[3],
                    entry.child,
                    entry.abs_diff,
                    entry.actual,
                    entry.reference
                );
            }
        }
        if let Some(top_k_coords) = env_usize("TRELLIS2_DECODER_SUBDIV_COORD_TOPK")
            && top_k_coords > 0
        {
            let coord_diff = top_subdivision_coord_diffs(actual_sub, reference_sub, top_k_coords);
            if !coord_diff.missing_in_actual.is_empty()
                || !coord_diff.extra_in_actual.is_empty()
                || coord_diff.duplicate_actual_rows > 0
                || coord_diff.duplicate_reference_rows > 0
            {
                println!(
                    "runtime_decoder_hook_alignment_report shape_subdiv.level={} coord_diff missing_in_actual={} extra_in_actual={} duplicate_actual_rows={} duplicate_reference_rows={}",
                    level,
                    coord_diff.missing_in_actual.len(),
                    coord_diff.extra_in_actual.len(),
                    coord_diff.duplicate_actual_rows,
                    coord_diff.duplicate_reference_rows
                );
                for (rank, coord) in coord_diff.missing_in_actual.iter().enumerate() {
                    println!(
                        "runtime_decoder_hook_alignment_report shape_subdiv.level={} coord_missing.rank={} coord=[{},{},{},{}]",
                        level,
                        rank + 1,
                        coord[0],
                        coord[1],
                        coord[2],
                        coord[3]
                    );
                }
                for (rank, coord) in coord_diff.extra_in_actual.iter().enumerate() {
                    println!(
                        "runtime_decoder_hook_alignment_report shape_subdiv.level={} coord_extra.rank={} coord=[{},{},{},{}]",
                        level,
                        rank + 1,
                        coord[0],
                        coord[1],
                        coord[2],
                        coord[3]
                    );
                }
            }
        }
        let inspect_coord = std::env::var(format!(
            "TRELLIS2_DECODER_SUBDIV_LEVEL{}_INSPECT_COORD",
            level
        ))
        .ok()
        .or_else(|| std::env::var("TRELLIS2_DECODER_SUBDIV_INSPECT_COORD").ok())
        .and_then(|value| parse_coord4_env(value.as_str()));
        if let Some(coord) = inspect_coord {
            let actual_row = subdivision_row_for_coord(actual_sub, coord);
            let reference_row = subdivision_row_for_coord(reference_sub, coord);
            println!(
                "runtime_decoder_hook_alignment_report shape_subdiv.level={} inspect_coord=[{},{},{},{}] actual_present={} reference_present={}",
                level,
                coord[0],
                coord[1],
                coord[2],
                coord[3],
                actual_row.is_some(),
                reference_row.is_some()
            );
            if let Some(row) = actual_row {
                println!(
                    "runtime_decoder_hook_alignment_report shape_subdiv.level={} inspect_coord_actual logits=[{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}]",
                    level, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                );
            }
            if let Some(row) = reference_row {
                println!(
                    "runtime_decoder_hook_alignment_report shape_subdiv.level={} inspect_coord_reference logits=[{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}]",
                    level, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                );
            }
        }
        if strict_subdiv_checks {
            assert!(
                sub_overlap > 0,
                "runtime_decoder_hook_alignment_report: subdivision level {} has zero coord overlap (actual_rows={} reference_rows={})",
                level,
                actual_sub_rows,
                reference_sub_rows
            );
        }

        let level_max_mean_abs = env_f32(&format!(
            "TRELLIS2_DECODER_SUBDIV_LEVEL{}_MAX_MEAN_ABS",
            level
        ))
        .or(global_subdiv_max_mean_abs);
        if let Some(limit) = level_max_mean_abs {
            assert!(
                sub_stats.mean_abs <= limit,
                "runtime_decoder_hook_alignment_report: subdivision level {} mean_abs {:.6e} exceeded limit {:.6e}",
                level,
                sub_stats.mean_abs,
                limit
            );
        }
        let level_max_rmse = env_f32(&format!("TRELLIS2_DECODER_SUBDIV_LEVEL{}_MAX_RMSE", level))
            .or(global_subdiv_max_rmse);
        if let Some(limit) = level_max_rmse {
            assert!(
                sub_stats.rmse <= limit,
                "runtime_decoder_hook_alignment_report: subdivision level {} rmse {:.6e} exceeded limit {:.6e}",
                level,
                sub_stats.rmse,
                limit
            );
        }
        let level_max_abs = env_f32(&format!("TRELLIS2_DECODER_SUBDIV_LEVEL{}_MAX_ABS", level))
            .or(global_subdiv_max_abs);
        if let Some(limit) = level_max_abs {
            assert!(
                sub_stats.max_abs <= limit,
                "runtime_decoder_hook_alignment_report: subdivision level {} max_abs {:.6e} exceeded limit {:.6e}",
                level,
                sub_stats.max_abs,
                limit
            );
        }
    }
    if strict_subdiv_checks {
        assert!(
            compared_subdiv_levels > 0,
            "runtime_decoder_hook_alignment_report: strict subdivision checks compared zero levels"
        );
        assert!(
            shape_subdivisions_host.len() == reference_subdivisions.len(),
            "runtime_decoder_hook_alignment_report: strict subdivision checks require equal level count (actual={} reference={})",
            shape_subdivisions_host.len(),
            reference_subdivisions.len()
        );
    }
    reset_decoder_conv_telemetry();
    reset_decoder_op_telemetry();
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_decoded = {
        let tex_coords_t = coords_to_default_wgpu_tensor(&tex_coords[..rows]);
        let tex_feats_t = rows_to_default_wgpu_tensor::<32>(&tex_feats[..rows]);
        let decoded = tex_decoder
            .decode_with_guidance_result_with_tensors(
                tex_coords_t,
                tex_feats_t,
                shape_decoded.subdivisions.as_slice(),
            )
            .expect("tex decoder should run");
        decode_tex_outputs(&decoded).expect("tex decoder outputs should decode")
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let tex_decoded = tex_decoder
        .decode_with_guidance(
            &tex_coords[..rows],
            &tex_feats[..rows],
            shape_decoded.subdivisions.as_slice(),
        )
        .expect("tex decoder should run");
    let tex_conv_telemetry = decoder_conv_telemetry();
    print_decoder_op_telemetry("tex_decoder", 16);
    print_decoder_conv_block_telemetry("tex_decoder", &tex_conv_telemetry, 20);
    println!(
        "runtime_decoder_hook_alignment_report tex_decoder_telemetry conv_calls={} wgpu_calls={} wgpu_successes={} wgpu_failures={} dispatches={} chunked_calls={} max_chunk_rows={} input_bytes={} output_bytes={} neighbor_elements={}",
        tex_conv_telemetry.conv_calls,
        tex_conv_telemetry.wgpu_calls,
        tex_conv_telemetry.wgpu_successes,
        tex_conv_telemetry.wgpu_failures,
        tex_conv_telemetry.dispatches,
        tex_conv_telemetry.chunked_calls,
        tex_conv_telemetry.max_chunk_rows,
        tex_conv_telemetry.input_bytes,
        tex_conv_telemetry.output_bytes,
        tex_conv_telemetry.neighbor_elements
    );
    if env_flag("TRELLIS2_DECODER_REQUIRE_WGPU_SUCCESS") {
        assert!(
            tex_conv_telemetry.wgpu_successes > 0,
            "runtime_decoder_hook_alignment_report: expected tex decoder wgpu successes > 0"
        );
        assert!(
            tex_conv_telemetry.wgpu_failures == 0,
            "runtime_decoder_hook_alignment_report: tex decoder had wgpu failures={}",
            tex_conv_telemetry.wgpu_failures
        );
        assert!(
            tex_conv_telemetry.dispatches > 0,
            "runtime_decoder_hook_alignment_report: expected tex decoder wgpu dispatches > 0"
        );
    }
    if tex_conv_telemetry.wgpu_calls > 0 {
        assert!(
            tex_conv_telemetry.wgpu_calls == tex_conv_telemetry.conv_calls,
            "runtime_decoder_hook_alignment_report: tex decoder device-path invariant violated (wgpu_calls={} conv_calls={})",
            tex_conv_telemetry.wgpu_calls,
            tex_conv_telemetry.conv_calls
        );
        assert!(
            tex_conv_telemetry.wgpu_failures == 0,
            "runtime_decoder_hook_alignment_report: tex decoder had wgpu failures={}",
            tex_conv_telemetry.wgpu_failures
        );
    }
    if env_flag("TRELLIS2_DECODER_DEBUG_REFERENCE_GUIDE")
        && shape_decoded.subdivisions.len() <= reference_subdivisions.len()
        && let Ok(tex_decoded_reference_guides) = tex_decoder.decode_with_guidance(
            &tex_coords[..rows],
            &tex_feats[..rows],
            &reference_subdivisions[..shape_decoded.subdivisions.len()],
        )
    {
        let (
            ref_guide_stats,
            ref_guide_overlap,
            ref_guide_actual_total,
            ref_guide_reference_total,
            _,
        ) = compare_tex_voxel_overlap(
            tex_decoded_reference_guides.coords.as_slice(),
            tex_decoded_reference_guides.attrs.as_slice(),
            reference_voxel_coords.as_slice(),
            reference_voxel_feats.as_slice(),
        );
        println!(
            "runtime_decoder_hook_alignment_report reference_guide overlap={} actual_voxels={} reference_voxels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            ref_guide_overlap,
            ref_guide_actual_total,
            ref_guide_reference_total,
            ref_guide_stats.mean_abs,
            ref_guide_stats.max_abs,
            ref_guide_stats.rmse
        );
    }

    assert!(
        !shape_decoded.coords.is_empty(),
        "decoded shape coords should not be empty"
    );
    assert!(
        !tex_decoded.coords.is_empty(),
        "decoded tex coords should not be empty"
    );

    let (stats, overlap, actual_total, reference_total, per_channel) = compare_tex_voxel_overlap(
        tex_decoded.coords.as_slice(),
        tex_decoded.attrs.as_slice(),
        reference_voxel_coords.as_slice(),
        reference_voxel_feats.as_slice(),
    );
    println!(
        "runtime_decoder_hook_alignment_report overlap={} actual_voxels={} reference_voxels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        overlap, actual_total, reference_total, stats.mean_abs, stats.max_abs, stats.rmse
    );
    for (channel, channel_stats) in per_channel.iter().enumerate() {
        println!(
            "runtime_decoder_hook_alignment_report channel={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            channel, channel_stats.mean_abs, channel_stats.max_abs, channel_stats.rmse
        );
    }
    assert!(
        overlap > 0,
        "expected overlapping decode voxels with reference hooks"
    );
    assert!(
        stats.mean_abs.is_finite() && stats.max_abs.is_finite() && stats.rmse.is_finite(),
        "decoder diff stats must be finite"
    );
    if let Some(min_overlap) = env_usize("TRELLIS2_DECODER_MIN_OVERLAP") {
        assert!(
            overlap >= min_overlap,
            "decoder overlap {} below TRELLIS2_DECODER_MIN_OVERLAP={}",
            overlap,
            min_overlap
        );
    }
    if let Some(max_mean_abs) = env_f32("TRELLIS2_DECODER_MAX_MEAN_ABS") {
        assert!(
            stats.mean_abs <= max_mean_abs,
            "decoder mean_abs {:.6e} exceeded TRELLIS2_DECODER_MAX_MEAN_ABS={:.6e}",
            stats.mean_abs,
            max_mean_abs
        );
    }
    if let Some(max_rmse) = env_f32("TRELLIS2_DECODER_MAX_RMSE") {
        assert!(
            stats.rmse <= max_rmse,
            "decoder rmse {:.6e} exceeded TRELLIS2_DECODER_MAX_RMSE={:.6e}",
            stats.rmse,
            max_rmse
        );
    }
    if let Some(max_abs) = env_f32("TRELLIS2_DECODER_MAX_ABS") {
        assert!(
            stats.max_abs <= max_abs,
            "decoder max_abs {:.6e} exceeded TRELLIS2_DECODER_MAX_ABS={:.6e}",
            stats.max_abs,
            max_abs
        );
    }
}

#[cfg(feature = "runtime-model")]
#[test]
fn runtime_decoder_stage0_subdivision_alignment_report() {
    let reference_path = match std::env::var("TRELLIS2_DECODER_REFERENCE_HOOK") {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            trellis_stage_log!(
                "Skipping runtime_decoder_stage0_subdivision_alignment_report: set TRELLIS2_DECODER_REFERENCE_HOOK to a stage0 alignment hook."
            );
            return;
        }
    };
    if !reference_path.exists() {
        trellis_stage_log!(
            "Skipping runtime_decoder_stage0_subdivision_alignment_report: missing reference hook '{}'",
            reference_path.display()
        );
        return;
    }
    let reference = HookSnapshot::from_file(&reference_path).expect("reference hook should load");

    let strict_subdiv_checks = env_flag("TRELLIS2_PARITY_STRICT")
        || env_flag("TRELLIS2_E2E_STRICT")
        || env_flag("TRELLIS2_DECODER_SUBDIV_REQUIRE_DECODE_INPUTS");
    let has_decode_inputs = reference
        .tensors
        .contains_key("decode_shape_slat.input.coords")
        && reference
            .tensors
            .contains_key("decode_shape_slat.input.feats");
    assert!(
        has_decode_inputs,
        "runtime_decoder_stage0_subdivision_alignment_report: reference hook '{}' must include decode_shape_slat.input.* keys",
        reference_path.display()
    );
    let shape_coords = tensor_to_coords4(
        reference
            .tensors
            .get("decode_shape_slat.input.coords")
            .expect("missing decode_shape_slat.input.coords"),
    )
    .expect("decode input coords should decode");
    let shape_feats = tensor_to_rows::<32>(
        reference
            .tensors
            .get("decode_shape_slat.input.feats")
            .expect("missing decode_shape_slat.input.feats"),
    )
    .expect("decode input feats should decode");
    let mut rows = shape_coords.len().min(shape_feats.len());
    assert!(rows > 0, "reference hook must contain stage0 rows");
    if let Some(cap) = env_usize("TRELLIS2_DECODER_TEST_MAX_ROWS")
        && cap > 0
        && rows > cap
    {
        assert!(
            !strict_subdiv_checks,
            "runtime_decoder_stage0_subdivision_alignment_report: TRELLIS2_DECODER_TEST_MAX_ROWS={} is not allowed in strict subdivision mode because sparse conv neighborhoods depend on full coordinate context",
            cap
        );
        rows = cap;
    }
    let reference_subdivisions = load_reference_subdivisions(&reference)
        .expect("reference shape subdivisions should decode");
    let Some(reference_stage0) = reference_subdivisions.first() else {
        trellis_stage_log!(
            "Skipping runtime_decoder_stage0_subdivision_alignment_report: no decode_shape_slat.subs.0 in '{}'",
            reference_path.display()
        );
        return;
    };

    let weights_root = resolve_trellis2_weights_root(None);
    if !weights_root.exists() {
        trellis_stage_log!(
            "Skipping runtime_decoder_stage0_subdivision_alignment_report: missing weights root '{}'",
            weights_root.display()
        );
        return;
    }
    let image_large_root = resolve_trellis2_image_large_root(None);
    let image_large_root_opt = if image_large_root.exists() {
        Some(image_large_root)
    } else {
        None
    };

    let pipeline_bytes =
        std::fs::read(weights_root.join("pipeline.json")).expect("pipeline.json should load");
    let pipeline = TrellisPipelineConfig::from_json_bytes(pipeline_bytes.as_slice())
        .expect("pipeline config should parse");
    let shape_stem = pipeline
        .args
        .models
        .get("shape_slat_decoder")
        .expect("shape_slat_decoder model stem missing");
    let shape_decoder = FdgDecoderRuntime::load_from_stem(
        weights_root.as_path(),
        image_large_root_opt.as_deref(),
        shape_stem.as_str(),
        false,
    )
    .expect("shape decoder should load");

    let stage0 = shape_decoder
        .stage0_subdivision_logits(&shape_coords[..rows], &shape_feats[..rows])
        .expect("shape stage0 subdivision should run");
    // Materialize once for alignment reporting; runtime path stays tensor-native.
    let stage0_host = SparseSubdivisionLogits::from_host(
        stage0.spatial_shape,
        stage0
            .coords_host("runtime_decoder_stage0_subdivision_alignment_report coords")
            .expect("stage0 coords should materialize"),
        stage0
            .logits_host("runtime_decoder_stage0_subdivision_alignment_report logits")
            .expect("stage0 logits should materialize"),
    )
    .expect("stage0 host materialization should be valid");
    let (stats, overlap, actual_rows, reference_rows) =
        compare_subdivision_overlap(&stage0_host, reference_stage0);
    println!(
        "runtime_decoder_stage0_subdivision_alignment_report input_source={} overlap={} actual_rows={} reference_rows={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        "decode_shape_slat.input",
        overlap,
        actual_rows,
        reference_rows,
        stats.mean_abs,
        stats.max_abs,
        stats.rmse
    );
    if let Some(top_k) = env_usize("TRELLIS2_DECODER_SUBDIV_STAGE0_TOPK")
        && top_k > 0
    {
        for (rank, entry) in top_subdivision_diffs(&stage0_host, reference_stage0, top_k)
            .into_iter()
            .enumerate()
        {
            println!(
                "runtime_decoder_stage0_subdivision_alignment_report top_diff.rank={} coord=[{},{},{},{}] child={} abs_diff={:.6e} actual={:.6e} reference={:.6e}",
                rank + 1,
                entry.coord[0],
                entry.coord[1],
                entry.coord[2],
                entry.coord[3],
                entry.child,
                entry.abs_diff,
                entry.actual,
                entry.reference
            );
        }
    }
    assert!(
        overlap > 0,
        "expected overlapping stage0 subdivision coords"
    );
    assert!(
        stats.mean_abs.is_finite() && stats.max_abs.is_finite() && stats.rmse.is_finite(),
        "stage0 subdivision diff stats must be finite"
    );
    if let Some(limit) =
        env_f32("TRELLIS2_DECODER_SUBDIV_STAGE0_MAX_MEAN_ABS").or(if strict_subdiv_checks {
            Some(1.0e-2f32)
        } else {
            None
        })
    {
        assert!(
            stats.mean_abs <= limit,
            "stage0 subdivision mean_abs {:.6e} exceeded limit {:.6e}",
            stats.mean_abs,
            limit
        );
    }
    if let Some(limit) =
        env_f32("TRELLIS2_DECODER_SUBDIV_STAGE0_MAX_RMSE").or(if strict_subdiv_checks {
            Some(1.0e-2f32)
        } else {
            None
        })
    {
        assert!(
            stats.rmse <= limit,
            "stage0 subdivision rmse {:.6e} exceeded limit {:.6e}",
            stats.rmse,
            limit
        );
    }
    if let Some(limit) =
        env_f32("TRELLIS2_DECODER_SUBDIV_STAGE0_MAX_ABS").or(if strict_subdiv_checks {
            Some(1.0e-2f32)
        } else {
            None
        })
    {
        assert!(
            stats.max_abs <= limit,
            "stage0 subdivision max_abs {:.6e} exceeded limit {:.6e}",
            stats.max_abs,
            limit
        );
    }
}

#[cfg(feature = "runtime-model")]
fn tensor_to_coords4(tensor: &crate::hook_diff::HookTensor) -> Result<Vec<[u32; 4]>, String> {
    if tensor.shape.len() != 2 || tensor.shape[1] != 4 {
        return Err(format!(
            "expected coords tensor shape [N,4], got {:?}",
            tensor.shape
        ));
    }
    let rows = tensor.shape[0];
    if tensor.data.len() != rows * 4 {
        return Err(format!(
            "coords tensor element count mismatch: expected {}, got {}",
            rows * 4,
            tensor.data.len()
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let base = row_idx * 4;
        out.push([
            tensor.data[base].round().max(0.0) as u32,
            tensor.data[base + 1].round().max(0.0) as u32,
            tensor.data[base + 2].round().max(0.0) as u32,
            tensor.data[base + 3].round().max(0.0) as u32,
        ]);
    }
    Ok(out)
}

#[cfg(feature = "runtime-model")]
fn tensor_to_rows<const C: usize>(
    tensor: &crate::hook_diff::HookTensor,
) -> Result<Vec<[f32; C]>, String> {
    if tensor.shape.len() != 2 || tensor.shape[1] != C {
        return Err(format!(
            "expected row tensor shape [N,{C}], got {:?}",
            tensor.shape
        ));
    }
    let rows = tensor.shape[0];
    if tensor.data.len() != rows * C {
        return Err(format!(
            "row tensor element count mismatch: expected {}, got {}",
            rows * C,
            tensor.data.len()
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let base = row_idx * C;
        let mut row = [0.0f32; C];
        row.copy_from_slice(&tensor.data[base..base + C]);
        out.push(row);
    }
    Ok(out)
}

#[cfg(feature = "runtime-model")]
fn tensor_to_spatial_shape3(tensor: &crate::hook_diff::HookTensor) -> Result<[u32; 3], String> {
    if tensor.shape.len() != 1 || tensor.shape[0] != 3 {
        return Err(format!(
            "expected spatial shape tensor [3], got {:?}",
            tensor.shape
        ));
    }
    if tensor.data.len() != 3 {
        return Err(format!(
            "spatial shape tensor element count mismatch: expected 3, got {}",
            tensor.data.len()
        ));
    }
    Ok([
        tensor.data[0].round().max(0.0) as u32,
        tensor.data[1].round().max(0.0) as u32,
        tensor.data[2].round().max(0.0) as u32,
    ])
}

#[cfg(feature = "runtime-model")]
fn load_reference_subdivisions(
    hook: &HookSnapshot,
) -> Result<Vec<SparseSubdivisionLogits>, String> {
    let mut levels = Vec::new();
    for level in 0usize..16 {
        let coords_key = format!("decode_shape_slat.subs.{level}.coords");
        let feats_key = format!("decode_shape_slat.subs.{level}.feats");
        let spatial_key = format!("decode_shape_slat.subs.{level}.spatial_shape");
        let (Some(coords_tensor), Some(feats_tensor), Some(spatial_tensor)) = (
            hook.tensors.get(coords_key.as_str()),
            hook.tensors.get(feats_key.as_str()),
            hook.tensors.get(spatial_key.as_str()),
        ) else {
            break;
        };
        let coords = tensor_to_coords4(coords_tensor)?;
        let feats = tensor_to_rows::<8>(feats_tensor)?;
        let spatial_shape = tensor_to_spatial_shape3(spatial_tensor)?;
        if coords.len() != feats.len() {
            return Err(format!(
                "reference subdivision level {} coords/feats mismatch: {} vs {}",
                level,
                coords.len(),
                feats.len()
            ));
        }
        let mut logits = Vec::with_capacity(feats.len() * 8);
        for row in feats {
            logits.extend_from_slice(row.as_slice());
        }
        levels.push(SparseSubdivisionLogits::from_host(
            spatial_shape,
            coords,
            logits,
        )?);
    }
    Ok(levels)
}

#[cfg(feature = "runtime-model")]
fn compare_subdivision_overlap(
    actual: &SparseSubdivisionLogits,
    reference: &SparseSubdivisionLogits,
) -> (crate::hook_diff::MetricStats, usize, usize, usize) {
    let actual_coords = actual
        .coords_host("compare_subdivision_overlap actual coords")
        .expect("actual subdivision coords should materialize");
    let actual_logits = actual
        .logits_host("compare_subdivision_overlap actual logits")
        .expect("actual subdivision logits should materialize");
    let reference_coords = reference
        .coords_host("compare_subdivision_overlap reference coords")
        .expect("reference subdivision coords should materialize");
    let reference_logits = reference
        .logits_host("compare_subdivision_overlap reference logits")
        .expect("reference subdivision logits should materialize");

    let mut actual_map: HashMap<[u32; 4], Vec<f32>> =
        HashMap::with_capacity(actual_coords.len().saturating_mul(2));
    for (idx, coord) in actual_coords.iter().copied().enumerate() {
        let row = &actual_logits[idx * 8..(idx + 1) * 8];
        actual_map.insert(coord, row.to_vec());
    }
    let mut reference_map: HashMap<[u32; 4], Vec<f32>> =
        HashMap::with_capacity(reference_coords.len().saturating_mul(2));
    for (idx, coord) in reference_coords.iter().copied().enumerate() {
        let row = &reference_logits[idx * 8..(idx + 1) * 8];
        reference_map.insert(coord, row.to_vec());
    }
    let mut actual_flat = Vec::new();
    let mut reference_flat = Vec::new();
    for (coord, reference_row) in &reference_map {
        if let Some(actual_row) = actual_map.get(coord) {
            actual_flat.extend_from_slice(actual_row.as_slice());
            reference_flat.extend_from_slice(reference_row.as_slice());
        }
    }
    let overlap = actual_flat.len() / 8;
    let stats = compute_stats(actual_flat.as_slice(), reference_flat.as_slice());
    (stats, overlap, actual_map.len(), reference_map.len())
}

#[cfg(feature = "runtime-model")]
#[derive(Clone, Copy, Debug)]
struct SubdivisionDiffEntry {
    coord: [u32; 4],
    child: usize,
    abs_diff: f32,
    actual: f32,
    reference: f32,
}

#[cfg(feature = "runtime-model")]
fn top_subdivision_diffs(
    actual: &SparseSubdivisionLogits,
    reference: &SparseSubdivisionLogits,
    k: usize,
) -> Vec<SubdivisionDiffEntry> {
    if k == 0 {
        return Vec::new();
    }
    let mut actual_map: HashMap<[u32; 4], [f32; 8]> =
        HashMap::with_capacity(actual.coords.len().saturating_mul(2));
    for (idx, coord) in actual.coords.iter().copied().enumerate() {
        let mut row = [0.0f32; 8];
        row.copy_from_slice(&actual.logits[idx * 8..(idx + 1) * 8]);
        actual_map.insert(coord, row);
    }

    let mut out = Vec::new();
    for (idx, coord) in reference.coords.iter().copied().enumerate() {
        let Some(actual_row) = actual_map.get(&coord) else {
            continue;
        };
        let reference_row = &reference.logits[idx * 8..(idx + 1) * 8];
        for child in 0..8 {
            let actual_value = actual_row[child];
            let reference_value = reference_row[child];
            out.push(SubdivisionDiffEntry {
                coord,
                child,
                abs_diff: (actual_value - reference_value).abs(),
                actual: actual_value,
                reference: reference_value,
            });
        }
    }
    out.sort_by(|a, b| b.abs_diff.total_cmp(&a.abs_diff));
    out.truncate(k);
    out
}

#[cfg(feature = "runtime-model")]
#[derive(Clone, Debug)]
struct SubdivisionCoordDiff {
    missing_in_actual: Vec<[u32; 4]>,
    extra_in_actual: Vec<[u32; 4]>,
    duplicate_actual_rows: usize,
    duplicate_reference_rows: usize,
}

#[cfg(feature = "runtime-model")]
fn top_subdivision_coord_diffs(
    actual: &SparseSubdivisionLogits,
    reference: &SparseSubdivisionLogits,
    limit: usize,
) -> SubdivisionCoordDiff {
    let mut actual_counts: HashMap<[u32; 4], usize> =
        HashMap::with_capacity(actual.coords.len().saturating_mul(2));
    for coord in actual.coords.iter().copied() {
        *actual_counts.entry(coord).or_insert(0) += 1;
    }
    let mut reference_counts: HashMap<[u32; 4], usize> =
        HashMap::with_capacity(reference.coords.len().saturating_mul(2));
    for coord in reference.coords.iter().copied() {
        *reference_counts.entry(coord).or_insert(0) += 1;
    }

    let duplicate_actual_rows = actual_counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();
    let duplicate_reference_rows = reference_counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();

    let mut missing_in_actual = reference_counts
        .keys()
        .copied()
        .filter(|coord| !actual_counts.contains_key(coord))
        .collect::<Vec<_>>();
    let mut extra_in_actual = actual_counts
        .keys()
        .copied()
        .filter(|coord| !reference_counts.contains_key(coord))
        .collect::<Vec<_>>();

    missing_in_actual.sort_unstable();
    extra_in_actual.sort_unstable();
    if limit > 0 {
        missing_in_actual.truncate(limit);
        extra_in_actual.truncate(limit);
    }

    SubdivisionCoordDiff {
        missing_in_actual,
        extra_in_actual,
        duplicate_actual_rows,
        duplicate_reference_rows,
    }
}

#[cfg(feature = "runtime-model")]
fn parse_coord4_env(value: &str) -> Option<[u32; 4]> {
    let mut parts = value.trim().split(',');
    let b = parts.next()?.trim().parse::<u32>().ok()?;
    let x = parts.next()?.trim().parse::<u32>().ok()?;
    let y = parts.next()?.trim().parse::<u32>().ok()?;
    let z = parts.next()?.trim().parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some([b, x, y, z])
}

#[cfg(feature = "runtime-model")]
fn subdivision_row_for_coord(sub: &SparseSubdivisionLogits, coord: [u32; 4]) -> Option<[f32; 8]> {
    for (idx, current) in sub.coords.iter().copied().enumerate() {
        if current == coord {
            let mut row = [0.0f32; 8];
            row.copy_from_slice(&sub.logits[idx * 8..(idx + 1) * 8]);
            return Some(row);
        }
    }
    None
}

#[cfg(feature = "runtime-model")]
fn tensor_stats(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min_value = values[0];
    let mut max_value = values[0];
    let mut sum = 0.0f32;
    for value in values {
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
        sum += *value;
    }
    (min_value, max_value, sum / values.len() as f32)
}

#[cfg(feature = "runtime-model")]
fn compare_tex_voxel_overlap(
    actual_coords: &[[u32; 4]],
    actual_attrs: &[[f32; 6]],
    reference_coords: &[[u32; 4]],
    reference_attrs: &[[f32; 6]],
) -> (
    crate::hook_diff::MetricStats,
    usize,
    usize,
    usize,
    [crate::hook_diff::MetricStats; 6],
) {
    let mut actual = HashMap::with_capacity(actual_coords.len().saturating_mul(2));
    for (coord, attr) in actual_coords
        .iter()
        .copied()
        .zip(actual_attrs.iter().copied())
    {
        actual.insert(coord, attr);
    }
    let mut reference = HashMap::with_capacity(reference_coords.len().saturating_mul(2));
    for (coord, attr) in reference_coords
        .iter()
        .copied()
        .zip(reference_attrs.iter().copied())
    {
        reference.insert(coord, attr);
    }

    let mut actual_flat = Vec::new();
    let mut reference_flat = Vec::new();
    let mut actual_channels = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut reference_channels = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for (coord, reference_attr) in &reference {
        if let Some(actual_attr) = actual.get(coord) {
            actual_flat.extend(actual_attr);
            reference_flat.extend(reference_attr);
            for channel in 0..6 {
                actual_channels[channel].push(actual_attr[channel]);
                reference_channels[channel].push(reference_attr[channel]);
            }
        }
    }
    let overlap = actual_flat.len() / 6;
    let stats = compute_stats(actual_flat.as_slice(), reference_flat.as_slice());
    let per_channel = [
        compute_stats(
            actual_channels[0].as_slice(),
            reference_channels[0].as_slice(),
        ),
        compute_stats(
            actual_channels[1].as_slice(),
            reference_channels[1].as_slice(),
        ),
        compute_stats(
            actual_channels[2].as_slice(),
            reference_channels[2].as_slice(),
        ),
        compute_stats(
            actual_channels[3].as_slice(),
            reference_channels[3].as_slice(),
        ),
        compute_stats(
            actual_channels[4].as_slice(),
            reference_channels[4].as_slice(),
        ),
        compute_stats(
            actual_channels[5].as_slice(),
            reference_channels[5].as_slice(),
        ),
    ];
    (stats, overlap, actual.len(), reference.len(), per_channel)
}
