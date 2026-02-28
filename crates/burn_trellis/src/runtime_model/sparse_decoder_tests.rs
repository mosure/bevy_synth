use std::fs;
use std::sync::{Mutex, MutexGuard};
#[cfg(not(feature = "runtime-model-wgpu"))]
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "runtime-model-wgpu")]
use burn::prelude::Backend;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "runtime-model-wgpu")]
use super::decoder_wgpu_neighbor_from_coords;
#[cfg(not(feature = "runtime-model-wgpu"))]
use super::sparse_subm_conv_forward_legacy;
use super::{
    decoder_conv_impl, linear_forward, logits_to_mask, resolve_model_weight_candidates,
    sparse_subm_conv_forward, DecoderConvCache, DecoderConvImpl, DecoderRuntimeConfig, LinearLayer,
    SparseConvLayer,
};
#[cfg(feature = "runtime-model-wgpu")]
use super::{
    decoder_wgpu_clear_cache_after_decode, decoder_wgpu_device_math_allow_fp16,
    decoder_wgpu_device_math_enabled, decoder_wgpu_device_math_max_state_bytes,
    decoder_wgpu_max_neighbor_bytes, decoder_wgpu_max_output_bytes, decoder_wgpu_max_tensor_bytes,
    decoder_wgpu_reduce_chunk_rows, decoder_wgpu_tensor_cache_max, decoder_wgpu_use_tensor_cache,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock_guard() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_guide_subdivision_tensor_handoff_parity() {
    if std::env::var("BURN_WGPU_SMOKE").is_err() {
        eprintln!(
            "Skipping decoder_guide_subdivision_tensor_handoff_parity: set BURN_WGPU_SMOKE=1"
        );
        return;
    }

    let device = <super::DefaultWgpuBackend as Backend>::Device::default();
    let coords_t = Tensor::<super::DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(vec![0i32, 0, 0, 0, 0, 1, 0, 0], [2, 4]),
        &device,
    );
    let logits_t = Tensor::<super::DefaultWgpuBackend, 2>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.2, 0.4, 0.8, 0.1, -0.2, 0.7, 0.0, 0.3, 0.0,
                0.9,
            ],
            [2, 8],
        ),
        &device,
    );
    let active_indices_t = Tensor::<super::DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(vec![0i32, 0, 0, 3, 1, 7], [3, 2]),
        &device,
    );
    let child_coords_t = Tensor::<super::DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(vec![0i32, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 1], [3, 4]),
        &device,
    );
    let child_linear_idx_t = Tensor::<super::DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(vec![0i32, 3, 15], [3]),
        &device,
    );

    let guide = super::SparseSubdivisionLogits::from_device_tensors_with_active_and_children(
        [2, 1, 1],
        coords_t.clone(),
        logits_t.clone(),
        Some(active_indices_t.clone()),
        Some((child_coords_t.clone(), child_linear_idx_t.clone())),
    )
    .expect("guide tensor pack should succeed");
    let guide_logits_t =
        super::guide_subdivision_logits_tensor_for_parent_wgpu(coords_t, &guide, &device, 0)
            .expect("guide logits tensor handoff should pass through unchanged");
    assert_eq!(guide_logits_t.dims(), [2, 8]);
    let handed_active = super::guide_subdivision_active_indices_tensor_for_parent_wgpu(2, &guide, 0)
        .expect("guide active-index tensor handoff should pass through unchanged");
    assert_eq!(handed_active.dims(), [3, 2]);
    let handed_active_values = handed_active
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .expect("guide active-index tensor should materialize for parity check");
    assert_eq!(handed_active_values, vec![0i32, 0, 0, 3, 1, 7]);
    let (handed_child_coords, handed_child_linear) =
        super::guide_subdivision_child_tensors_for_parent_wgpu(2, &guide, 0)
            .expect("guide child tensors should hand off unchanged");
    assert_eq!(handed_child_coords.dims(), [3, 4]);
    assert_eq!(handed_child_linear.dims(), [3]);
    let handed_child_coords = handed_child_coords
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .expect("guide child coord tensor should materialize for parity check");
    let handed_child_linear = handed_child_linear
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .expect("guide child linear tensor should materialize for parity check");
    assert_eq!(handed_child_coords, vec![0i32, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 1]);
    assert_eq!(handed_child_linear, vec![0i32, 3, 15]);

    let coords_t_no_child = Tensor::<super::DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(vec![0i32, 0, 0, 0, 0, 1, 0, 0], [2, 4]),
        &device,
    );
    let guide_without_child =
        super::SparseSubdivisionLogits::from_device_tensors_with_active_and_children(
        [2, 1, 1],
        coords_t_no_child,
        logits_t.clone(),
        Some(active_indices_t),
        None,
    )
    .expect("guide tensor pack without child tensors should succeed");
    let err = super::guide_subdivision_child_tensors_for_parent_wgpu(2, &guide_without_child, 0)
        .expect_err("guide child tensors are required on canonical guide handoff");
    assert!(err.contains("requires tensor-native guide child tensors"));

    let coords_t_no_active = Tensor::<super::DefaultWgpuBackend, 2, Int>::from_data(
        TensorData::new(vec![0i32, 0, 0, 0, 0, 1, 0, 0], [2, 4]),
        &device,
    );
    let guide_without_active =
        super::SparseSubdivisionLogits::from_device_tensors_with_active_and_children(
        [2, 1, 1],
        coords_t_no_active,
        logits_t,
        None,
        None,
    )
    .expect("guide tensor pack without active indices should succeed");
    let err = super::guide_subdivision_active_indices_tensor_for_parent_wgpu(
        2,
        &guide_without_active,
        0,
    )
    .expect_err("guide active-index tensor must be required on canonical guide handoff");
    assert!(err.contains("requires tensor-native guide active indices"));
}

fn make_unit_conv_3x1x1(weight: [f32; 3]) -> SparseConvLayer {
    SparseConvLayer {
        in_channels: 1,
        out_channels: 1,
        kernel_d: 3,
        kernel_h: 1,
        kernel_w: 1,
        in_channels_per_group: 1,
        out_channels_per_group: 1,
        groups: 1,
        weight: weight.to_vec(),
        bias: vec![0.0],
        flex_packed_weight: None,
    }
}

#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.state >> 40) as u32) | 1;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn sparse_conv_uses_neighbor_voxels() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
        std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
    }
    let coords = vec![[0, 0, 0, 0], [0, 1, 0, 0]];
    let input = vec![1.0f32, 2.0f32];
    // kernel offsets: [-1, 0, +1]
    let layer = make_unit_conv_3x1x1([10.0, 1.0, 100.0]);

    let output = sparse_subm_conv_forward(
        coords.as_slice(),
        input.as_slice(),
        &layer,
        "test conv",
        &mut DecoderConvCache::default(),
        #[cfg(feature = "runtime-model-wgpu")]
        None,
    );
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let err = output.expect_err("wgpu conv path should fail fast without context");
        assert!(
            err.contains("context unavailable"),
            "unexpected error: {err}"
        );
        unsafe {
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let output = output.expect("sparse conv should succeed");
        assert_eq!(output.len(), 2);
        // x=0: center(1*1) + right-neighbor(2*100)
        assert!((output[0] - 201.0).abs() < 1.0e-5);
        // x=1: left-neighbor(1*10) + center(2*1)
        assert!((output[1] - 12.0).abs() < 1.0e-5);
        unsafe {
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
}

#[test]
fn sparse_conv_flex_matches_legacy_path() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
        std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
        std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "flex_gmm");
    }
    let mut rng = Lcg::new(123);
    let layer = SparseConvLayer {
        in_channels: 4,
        out_channels: 6,
        kernel_d: 3,
        kernel_h: 1,
        kernel_w: 1,
        in_channels_per_group: 2,
        out_channels_per_group: 3,
        groups: 2,
        weight: (0..(6 * 3 * 2)).map(|_| rng.next_f32()).collect(),
        bias: (0..6).map(|_| rng.next_f32()).collect(),
        flex_packed_weight: None,
    };
    let coords: Vec<[u32; 4]> = (0..32u32).map(|x| [0, x, 0, 0]).collect();
    let input: Vec<f32> = (0..coords.len() * layer.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let legacy =
        sparse_subm_conv_forward_legacy(coords.as_slice(), input.as_slice(), &layer, "legacy")
            .expect("legacy conv");
    let fused = sparse_subm_conv_forward(
        coords.as_slice(),
        input.as_slice(),
        &layer,
        "fused",
        &mut DecoderConvCache::default(),
        #[cfg(feature = "runtime-model-wgpu")]
        None,
    );
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let err = fused.expect_err("wgpu conv path should fail fast without context");
        assert!(
            err.contains("context unavailable"),
            "unexpected error: {err}"
        );
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let fused = fused.expect("fused conv");
        assert_eq!(legacy.len(), fused.len());
        for (idx, (lhs, rhs)) in legacy.iter().zip(fused.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= 1.0e-5,
                "mismatch idx={idx}: legacy={lhs} fused={rhs} diff={diff}"
            );
        }
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
}

#[test]
fn decoder_neighbor_cache_reuses_across_coord_allocations() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
        std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
    }
    let layer = make_unit_conv_3x1x1([0.1, 0.2, 0.3]);
    let config = super::flex_config_for_layer(&layer);
    let coords_a: Vec<[u32; 4]> = (0..16u32).map(|x| [0, x, 0, 0]).collect();
    let coords_b = coords_a.clone();
    let mut cache = DecoderConvCache::default();

    let key_a = {
        let (key, rows) = cache
            .neighbor_rows_with_key(&config, coords_a.as_slice())
            .expect("cache build");
        assert_eq!(rows.len(), coords_a.len() * 3);
        key
    };
    let len_after_a = cache.neighbor_rows.len();
    let key_b = {
        let (key, rows) = cache
            .neighbor_rows_with_key(&config, coords_b.as_slice())
            .expect("cache hit");
        assert_eq!(rows.len(), coords_b.len() * 3);
        key
    };

    assert_eq!(key_a, key_b);
    assert_eq!(cache.neighbor_rows.len(), len_after_a);
}

#[test]
fn decoder_neighbor_cache_reuse_reduces_repeated_conv_time() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "flex_gmm");
        std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
        std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
    }
    let mut rng = Lcg::new(991);
    let layer = SparseConvLayer {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        weight: (0..(128 * 3 * 3 * 3 * 64))
            .map(|_| rng.next_f32())
            .collect(),
        bias: (0..128).map(|_| rng.next_f32()).collect(),
        flex_packed_weight: None,
    };
    let coords: Vec<[u32; 4]> = (0..4096u32).map(|x| [0, x, 0, 0]).collect();
    let input: Vec<f32> = (0..coords.len() * layer.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let err = sparse_subm_conv_forward(
            coords.as_slice(),
            input.as_slice(),
            &layer,
            "cold",
            &mut DecoderConvCache::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            None,
        )
        .expect_err("wgpu conv path should fail fast without context");
        assert!(
            err.contains("context unavailable"),
            "unexpected error: {err}"
        );
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let iterations = 12usize;
        let cold_start = Instant::now();
        for _ in 0..iterations {
            let _ = sparse_subm_conv_forward(
                coords.as_slice(),
                input.as_slice(),
                &layer,
                "cold",
                &mut DecoderConvCache::default(),
                #[cfg(feature = "runtime-model-wgpu")]
                None,
            )
            .expect("cold conv");
        }
        let cold = cold_start.elapsed();

        let mut warm_cache = DecoderConvCache::default();
        let warm_start = Instant::now();
        for _ in 0..iterations {
            let _ = sparse_subm_conv_forward(
                coords.as_slice(),
                input.as_slice(),
                &layer,
                "warm",
                &mut warm_cache,
                #[cfg(feature = "runtime-model-wgpu")]
                None,
            )
            .expect("warm conv");
        }
        let warm = warm_start.elapsed();
        eprintln!(
            "decoder cache perf: cold={:?} warm={:?} ratio={:.3}",
            cold,
            warm,
            warm.as_secs_f64() / cold.as_secs_f64().max(1.0e-12)
        );
        assert!(
            warm <= cold,
            "expected persistent neighbor cache to be no slower than rebuilding; cold={cold:?} warm={warm:?}"
        );

        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }
}

#[test]
fn decoder_default_child_cap_is_uncapped_without_strict_mode() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        1,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
}

#[test]
fn parity_strict_defaults_to_uncapped_children() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::set_var("TRELLIS2_PARITY_STRICT", "1");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        1,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
    }
}

#[test]
fn explicit_zero_child_cap_env_means_uncapped() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::set_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT", "0");
    }
    let logits = vec![1.0f32; 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        1,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
}

#[test]
fn large_rows_default_child_cap_is_uncapped() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 4_096 * 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        4_096,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
}

#[test]
fn large_rows_parity_strict_disables_default_child_cap() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::set_var("TRELLIS2_PARITY_STRICT", "1");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 4_096 * 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        4_096,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
    }
}

#[test]
fn large_rows_decode_input_strict_disables_default_child_cap() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::set_var("TRELLIS2_DECODER_SUBDIV_REQUIRE_DECODE_INPUTS", "1");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 4_096 * 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        4_096,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_SUBDIV_REQUIRE_DECODE_INPUTS");
    }
}

#[test]
fn explicit_child_cap_env_is_ignored_in_canonical_mode() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
        std::env::set_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT", "3");
    }
    let logits = vec![1.0f32; 4_096 * 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        4_096,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
}

#[test]
fn large_rows_uncapped_env_disables_default_child_cap() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::set_var("TRELLIS2_DECODER_UNCAPPED", "1");
        std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
    }
    let logits = vec![1.0f32; 4_096 * 8];
    let mask = logits_to_mask(
        logits.as_slice(),
        4_096,
        true,
        &DecoderRuntimeConfig::default(),
    )
    .expect("mask");
    let selected = mask[0].iter().filter(|flag| **flag).count();
    assert_eq!(selected, 8);
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
    }
}

#[test]
fn decoder_conv_auto_defaults_to_flex() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
        std::env::remove_var("TRELLIS2_PARITY_STRICT");
        std::env::remove_var("TRELLIS2_E2E_STRICT");
        std::env::remove_var("TRELLIS2_DECODER_DISABLE_WGPU");
    }
    #[cfg(feature = "runtime-model-wgpu")]
    assert_eq!(decoder_conv_impl(), DecoderConvImpl::Wgpu);
    #[cfg(not(feature = "runtime-model-wgpu"))]
    assert_eq!(decoder_conv_impl(), DecoderConvImpl::FlexGmm);
}

#[test]
fn decoder_conv_auto_does_not_force_legacy_in_strict_mode() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
        std::env::set_var("TRELLIS2_E2E_STRICT", "1");
    }
    #[cfg(feature = "runtime-model-wgpu")]
    assert_eq!(decoder_conv_impl(), DecoderConvImpl::Wgpu);
    #[cfg(not(feature = "runtime-model-wgpu"))]
    assert_eq!(decoder_conv_impl(), DecoderConvImpl::FlexGmm);
    unsafe {
        std::env::remove_var("TRELLIS2_E2E_STRICT");
    }
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_wgpu_neighbor_source_defaults_to_coords_kernel() {
    let _guard = env_lock_guard();
    assert!(decoder_wgpu_neighbor_from_coords());
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_wgpu_cache_controls_have_expected_defaults() {
    let _guard = env_lock_guard();
    assert!(!decoder_wgpu_clear_cache_after_decode());
    assert_eq!(decoder_wgpu_tensor_cache_max(), 128);
    assert!(decoder_wgpu_use_tensor_cache());
    // Runtime behavior is canonical and should not drift based on environment toggles.
    unsafe {
        std::env::set_var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE", "1");
        std::env::set_var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX", "8");
    }
    assert!(!decoder_wgpu_clear_cache_after_decode());
    assert_eq!(decoder_wgpu_tensor_cache_max(), 128);
    assert!(decoder_wgpu_use_tensor_cache());
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE");
        std::env::remove_var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX");
    }
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_wgpu_chunk_guards_have_expected_defaults() {
    let _guard = env_lock_guard();
    assert_eq!(decoder_wgpu_max_output_bytes(), 512 * 1024 * 1024);
    assert_eq!(decoder_wgpu_max_neighbor_bytes(), 256 * 1024 * 1024);
    assert_eq!(decoder_wgpu_max_tensor_bytes(), i32::MAX as usize);
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_wgpu_reduce_chunk_rows_halves_with_alignment() {
    let _guard = env_lock_guard();
    assert_eq!(decoder_wgpu_reduce_chunk_rows(1), 1);
    assert_eq!(decoder_wgpu_reduce_chunk_rows(2), 1);
    assert_eq!(decoder_wgpu_reduce_chunk_rows(130), 64);
    assert_eq!(decoder_wgpu_reduce_chunk_rows(2048), 1024);
}

#[cfg(feature = "runtime-model-wgpu")]
#[test]
fn decoder_wgpu_device_math_control_defaults_enabled() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH");
        std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16");
        std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "wgpu");
        std::env::remove_var("TRELLIS2_DECODER_DISABLE_WGPU");
    }
    assert!(decoder_wgpu_device_math_enabled());
    assert!(decoder_wgpu_device_math_allow_fp16());
    assert_eq!(
        decoder_wgpu_device_math_max_state_bytes(),
        512 * 1024 * 1024
    );
    unsafe {
        std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH", "0");
    }
    assert!(decoder_wgpu_device_math_enabled());
    unsafe {
        std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH", "1");
        std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16", "0");
    }
    assert!(decoder_wgpu_device_math_allow_fp16());
    unsafe {
        std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16", "1");
        std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "legacy");
    }
    assert!(decoder_wgpu_device_math_enabled());
    unsafe {
        std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH");
        std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16");
        std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
    }
}

#[test]
fn linear_forward_matches_naive_matmul() {
    let layer = LinearLayer {
        in_channels: 3,
        out_channels: 2,
        // [out, in]
        weight: vec![
            1.0, 2.0, 3.0, // out0
            -1.0, 0.5, 4.0, // out1
        ],
        bias: vec![0.25, -0.5],
    };
    let input = vec![
        2.0, -1.0, 0.5, // row0
        -3.0, 4.0, 1.0, // row1
    ];
    let output = linear_forward(input.as_slice(), 2, &layer, "test linear")
        .expect("linear forward should succeed");
    assert_eq!(output.len(), 4);

    let mut expected = Vec::new();
    for row in 0..2 {
        let x = &input[row * 3..(row + 1) * 3];
        // out0
        expected.push(layer.bias[0] + x[0] * 1.0 + x[1] * 2.0 + x[2] * 3.0);
        // out1
        expected.push(layer.bias[1] - x[0] + x[1] * 0.5 + x[2] * 4.0);
    }
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1.0e-5, "got={got} want={want}");
    }
}

#[test]
fn model_weight_candidates_prefer_bpk_variants() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_BPK_PRECISION");
        std::env::remove_var("BURN_SYNTH_BPK_PRECISION");
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("burn_trellis_decoder_candidates_{unique}"));
    let ckpts = root.join("ckpts");
    fs::create_dir_all(&ckpts).expect("create ckpts");
    fs::write(ckpts.join("shape.safetensors"), b"safe").expect("write safetensors");
    fs::write(ckpts.join("shape.bpk"), b"bpk").expect("write bpk");
    fs::write(ckpts.join("shape_f16.bpk"), b"bpk_f16").expect("write f16 bpk");

    let candidates = resolve_model_weight_candidates("ckpts/shape", root.as_path(), None);
    assert!(!candidates.is_empty(), "expected weight candidates");
    assert_eq!(candidates[0], ckpts.join("shape_f16.bpk"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn model_weight_candidates_ignore_env_precision_overrides() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::set_var("TRELLIS2_BPK_PRECISION", "f32");
        std::env::remove_var("BURN_SYNTH_BPK_PRECISION");
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("burn_trellis_decoder_candidates_{unique}"));
    let ckpts = root.join("ckpts");
    fs::create_dir_all(&ckpts).expect("create ckpts");
    fs::write(ckpts.join("shape.safetensors"), b"safe").expect("write safetensors");
    fs::write(ckpts.join("shape.bpk"), b"bpk").expect("write bpk");
    fs::write(ckpts.join("shape_f16.bpk"), b"bpk_f16").expect("write f16 bpk");

    let candidates = resolve_model_weight_candidates("ckpts/shape", root.as_path(), None);
    assert!(!candidates.is_empty(), "expected weight candidates");
    assert_eq!(candidates[0], ckpts.join("shape_f16.bpk"));

    unsafe {
        std::env::remove_var("TRELLIS2_BPK_PRECISION");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn model_weight_candidates_include_parts_manifest_without_base_file() {
    let _guard = env_lock_guard();
    unsafe {
        std::env::remove_var("TRELLIS2_BPK_PRECISION");
        std::env::remove_var("BURN_SYNTH_BPK_PRECISION");
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("burn_trellis_decoder_parts_{unique}"));
    let ckpts = root.join("ckpts");
    fs::create_dir_all(&ckpts).expect("create ckpts");
    fs::write(ckpts.join("shape.safetensors"), b"safe").expect("write safetensors");
    fs::write(
        ckpts.join("shape_f16.bpk.parts.json"),
        br#"{
  "version": 1,
  "source_file": "shape_f16.bpk",
  "source_modified_unix_ms": 0,
  "total_bytes": 4,
  "max_part_bytes": 4,
  "parts": [{"path": "shape_f16.bpk.part-00000.bpk", "bytes": 4, "sha256": "", "tensors": 1}]
}"#,
    )
    .expect("write parts manifest");

    let candidates = resolve_model_weight_candidates("ckpts/shape", root.as_path(), None);
    assert!(!candidates.is_empty(), "expected weight candidates");
    assert_eq!(candidates[0], ckpts.join("shape_f16.bpk"));

    let _ = fs::remove_dir_all(root);
}
