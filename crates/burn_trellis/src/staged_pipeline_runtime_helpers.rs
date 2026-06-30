fn runtime_parity_strict() -> bool {
    false
}

#[cfg(feature = "runtime-model")]
fn runtime_lazy_model_load_enabled() -> bool {
    true
}

#[cfg(feature = "runtime-model")]
fn load_flow_runtime_from_spec(
    spec: Option<&FlowRuntimeLoadSpec>,
) -> Option<SparseStructureFlowRuntime> {
    let spec = spec?;
    let load_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: {} runtime load begin (model='{}', prefer_wgpu={})",
        spec.stage_label,
        spec.model_stem,
        spec.prefer_wgpu
    );
    match SparseStructureFlowRuntime::load_from_stem(
        spec.weights_root.as_path(),
        spec.image_large_root.as_deref(),
        spec.model_stem.as_str(),
        spec.prefer_wgpu,
        spec.slat_dense_resolution,
    ) {
        Ok(runtime) => {
            match spec.stage_label {
                "sparse flow" => {
                    trellis_stage_log!(
                        "burn_trellis: sparse flow runtime backend = {} (load_ms={:.2})",
                        runtime.backend_name(),
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                "shape slat" => {
                    let key = spec.flow_key.as_deref().unwrap_or("shape_slat_flow_model");
                    trellis_stage_log!(
                        "burn_trellis: shape slat runtime backend = {} (flow={}, dense_res={}, load_ms={:.2})",
                        runtime.backend_name(),
                        key,
                        runtime.config().resolution,
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                "tex slat" => {
                    let key = spec.flow_key.as_deref().unwrap_or("tex_slat_flow_model");
                    trellis_stage_log!(
                        "burn_trellis: tex slat runtime backend = {} (flow={}, dense_res={}, load_ms={:.2})",
                        runtime.backend_name(),
                        key,
                        runtime.config().resolution,
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                _ => {}
            }
            Some(runtime)
        }
        Err(err) => {
            match spec.stage_label {
                "sparse flow" => {
                    trellis_stage_log!(
                        "burn_trellis: sparse flow runtime model unavailable after {:.2} ms ({err}); sparse stage requires runtime model and will fail fast.",
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                "shape slat" => {
                    let key = spec.flow_key.as_deref().unwrap_or("shape_slat_flow_model");
                    trellis_stage_log!(
                        "burn_trellis: shape slat runtime model unavailable for key '{}' after {:.2} ms ({err}); shape stage requires runtime model and will fail fast.",
                        key,
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                "tex slat" => {
                    let key = spec.flow_key.as_deref().unwrap_or("tex_slat_flow_model");
                    trellis_stage_log!(
                        "burn_trellis: tex slat runtime model unavailable for key '{}' after {:.2} ms ({err}); tex stage requires runtime model and will fail fast.",
                        key,
                        load_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                _ => {}
            }
            None
        }
    }
}

#[cfg(feature = "runtime-model")]
fn load_sparse_structure_decoder_from_spec(
    spec: Option<&SparseStructureDecoderLoadSpec>,
) -> Option<SparseStructureDecoderRuntime> {
    let spec = spec?;
    let load_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: sparse structure decoder runtime load begin (model='{}', prefer_wgpu={})",
        spec.model_stem,
        spec.prefer_wgpu
    );
    match SparseStructureDecoderRuntime::load_from_stem(
        spec.weights_root.as_path(),
        spec.image_large_root.as_deref(),
        spec.model_stem.as_str(),
        spec.prefer_wgpu,
    ) {
        Ok(runtime) => {
            trellis_stage_log!(
                "burn_trellis: sparse structure decoder runtime backend = {} (load_ms={:.2})",
                runtime.backend_name(),
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(runtime)
        }
        Err(err) => {
            trellis_stage_log!(
                "burn_trellis: sparse structure decoder runtime unavailable after {:.2} ms ({err}); sparse stage will fail fast when runtime sparse flow is active.",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            None
        }
    }
}

#[cfg(feature = "runtime-model")]
fn load_shape_decoder_from_spec(
    spec: Option<&DecoderRuntimeLoadSpec>,
) -> Option<FdgDecoderRuntime> {
    let spec = spec?;
    if !matches!(spec.kind, DecoderRuntimeKind::Shape) {
        return None;
    }
    let load_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: shape decoder runtime load begin (model='{}', prefer_wgpu={})",
        spec.model_stem,
        spec.prefer_wgpu
    );
    match FdgDecoderRuntime::load_from_stem(
        spec.weights_root.as_path(),
        spec.image_large_root.as_deref(),
        spec.model_stem.as_str(),
        spec.prefer_wgpu,
    ) {
        Ok(runtime) => {
            trellis_stage_log!(
                "burn_trellis: shape decoder runtime load complete (load_ms={:.2})",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(runtime)
        }
        Err(err) => {
            trellis_stage_log!(
                "burn_trellis: shape decoder runtime unavailable after {:.2} ms ({err}); decode stage will fail until runtime decoder assets are available.",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            None
        }
    }
}

#[cfg(feature = "runtime-model")]
fn load_tex_decoder_from_spec(
    spec: Option<&DecoderRuntimeLoadSpec>,
) -> Option<SparseUnetVaeDecoderRuntime> {
    let spec = spec?;
    if !matches!(spec.kind, DecoderRuntimeKind::Tex) {
        return None;
    }
    let load_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: tex decoder runtime load begin (model='{}', prefer_wgpu={})",
        spec.model_stem,
        spec.prefer_wgpu
    );
    match SparseUnetVaeDecoderRuntime::load_from_stem(
        spec.weights_root.as_path(),
        spec.image_large_root.as_deref(),
        spec.model_stem.as_str(),
        spec.prefer_wgpu,
    ) {
        Ok(runtime) => {
            trellis_stage_log!(
                "burn_trellis: tex decoder runtime load complete (load_ms={:.2})",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(runtime)
        }
        Err(err) => {
            trellis_stage_log!(
                "burn_trellis: tex decoder runtime unavailable after {:.2} ms ({err}); decode stage will fail until runtime decoder assets are available.",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            None
        }
    }
}

#[cfg(feature = "runtime-model")]
fn load_image_conditioning_from_spec(
    spec: Option<&ImageConditioningLoadSpec>,
) -> Option<TrellisImageConditioningRuntime> {
    let spec = spec?;
    let load_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: image conditioning runtime load begin (model='{}', prefer_wgpu={})",
        spec.model_name,
        spec.prefer_wgpu
    );
    match TrellisImageConditioningRuntime::load_from_model_name(
        spec.weights_root.as_path(),
        spec.image_large_root.as_deref(),
        spec.model_name.as_str(),
        spec.prefer_wgpu,
    ) {
        Ok(runtime) => {
            trellis_stage_log!(
                "burn_trellis: image conditioning runtime backend = {} (load_ms={:.2})",
                runtime.backend_name(),
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            Some(runtime)
        }
        Err(err) => {
            trellis_stage_log!(
                "burn_trellis: image conditioning runtime unavailable after {:.2} ms ({err}); inference will now fail fast before staged sampling.",
                load_start.elapsed().as_secs_f64() * 1000.0
            );
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(
                &format!(
                    "burn_trellis: image conditioning runtime load failed after {:.2} ms: {err}",
                    load_start.elapsed().as_secs_f64() * 1000.0
                )
                .into(),
            );
            None
        }
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn reset_neighbor_build_stats() {
    reset_neighbor_rows_build_stats();
    reset_sparse_wgpu_kernel_stats();
}

#[cfg(not(feature = "runtime-model-wgpu"))]
fn reset_neighbor_build_stats() {}

#[cfg(feature = "runtime-model-wgpu")]
fn log_neighbor_build_stats(stage: &str) {
    let stats = neighbor_rows_build_stats();
    let conv_stats = sparse_wgpu_kernel_stats();
    let hash_probe_avg = if stats.device_hash_rows == 0 {
        0.0
    } else {
        stats.device_hash_probe_total as f64 / stats.device_hash_rows as f64
    };
    trellis_stage_log!(
        "burn_trellis: neighbor-map telemetry [{stage}] cache_hits={} cache_misses={} host_builds={} device_builds={} device_scan_builds={} device_hash_builds={} device_scan_ms={:.2} device_hash_ms={:.2} hash_rows={} hash_probe_total={} hash_probe_avg={:.2} hash_probe_max={} hash_fail_rows={}",
        stats.cache_hits,
        stats.cache_misses,
        stats.host_builds,
        stats.device_builds,
        stats.device_scan_builds,
        stats.device_hash_builds,
        stats.device_scan_build_ns as f64 / 1_000_000.0,
        stats.device_hash_build_ns as f64 / 1_000_000.0,
        stats.device_hash_rows,
        stats.device_hash_probe_total,
        hash_probe_avg,
        stats.device_hash_probe_max,
        stats.device_hash_insert_fail_rows
    );
    trellis_stage_log!(
        "burn_trellis: sparse-wgpu conv telemetry [{stage}] calls={} splitk_calls={} fused_variant_calls={} single_group_specialized_calls={} dispatches={} rows={} output_elements={} elapsed_ms={:.2}",
        conv_stats.calls,
        conv_stats.splitk_calls,
        conv_stats.fused_variant_calls,
        conv_stats.single_group_specialized_calls,
        conv_stats.total_dispatches,
        conv_stats.total_rows,
        conv_stats.total_output_elements,
        conv_stats.total_elapsed_ns as f64 / 1_000_000.0
    );
}

#[cfg(not(feature = "runtime-model-wgpu"))]
fn log_neighbor_build_stats(_stage: &str) {}

#[cfg(feature = "runtime-model")]
fn log_decoder_conv_telemetry(stage: &str, telemetry: &DecoderConvTelemetry) {
    trellis_stage_log!(
        "burn_trellis: decoder conv telemetry [{stage}] conv_calls={} wgpu_calls={} wgpu_successes={} wgpu_failures={} dispatches={} chunked_calls={} max_chunk_rows={} input_bytes={} output_bytes={} neighbor_elements={}",
        telemetry.conv_calls,
        telemetry.wgpu_calls,
        telemetry.wgpu_successes,
        telemetry.wgpu_failures,
        telemetry.dispatches,
        telemetry.chunked_calls,
        telemetry.max_chunk_rows,
        telemetry.input_bytes,
        telemetry.output_bytes,
        telemetry.neighbor_elements
    );
    for block in telemetry.blocks.iter() {
        trellis_stage_log!(
            "burn_trellis: decoder conv telemetry [{stage}] block='{}' conv_calls={} wgpu_calls={} wgpu_successes={} wgpu_failures={} dispatches={} chunked_calls={} max_chunk_rows={} input_bytes={} output_bytes={} neighbor_elements={}",
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

#[cfg(feature = "runtime-model")]
fn log_decoder_op_telemetry(stage: &str, telemetry: &DecoderOpTelemetry) {
    trellis_stage_log!(
        "burn_trellis: decoder op telemetry [{stage}] calls={} total_ms={:.3} readback_count={} readback_elements={}",
        telemetry.calls,
        telemetry.total_ms,
        telemetry.readback_count,
        telemetry.readback_elements
    );
    for op in telemetry.ops.iter() {
        let mean_ms = if op.calls > 0 {
            op.total_ms / op.calls as f64
        } else {
            0.0
        };
        trellis_stage_log!(
            "burn_trellis: decoder op telemetry [{stage}] op='{}' calls={} total_ms={:.3} mean_ms={:.3} max_ms={:.3}",
            op.context,
            op.calls,
            op.total_ms,
            mean_ms,
            op.max_ms
        );
    }
}

fn runtime_max_sparse_coords_for_backend(
    _backend_name: &str,
    explicit_limit: Option<usize>,
) -> Option<(usize, SparseCoordCapSource)> {
    if let Some(limit) = explicit_limit {
        return Some((limit, SparseCoordCapSource::ExplicitRunConfig));
    }
    None
}

fn resolve_sampler_settings(
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
) -> (FlowEulerGuidanceIntervalSampler, FlowEulerSampleConfig, f32) {
    if let Some(override_config) = sampler_override {
        let sampler = FlowEulerGuidanceIntervalSampler::new(override_config.sigma_min);
        return (sampler, override_config.config, override_config.sigma_min);
    }
    let sigma_min = sampler_config.args.sigma_min;
    let (sampler, config) =
        FlowEulerGuidanceIntervalSampler::from_params(sigma_min, &sampler_config.params);
    (sampler, config, sigma_min)
}

fn dense_noise_with_override(
    rng: &mut Lcg,
    expected_len: usize,
    override_values: Option<&[f32]>,
    stage: &str,
) -> Vec<f32> {
    if let Some(values) = override_values {
        if values.len() == expected_len {
            return values.to_vec();
        }
        trellis_stage_log!(
            "burn_trellis: ignoring {stage} noise override due to len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        );
    }
    (0..expected_len).map(|_| rng.next_normal_f32()).collect()
}

#[cfg(feature = "runtime-model")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn build_dense_runtime_noise(
    rng: &mut Lcg,
    channels: usize,
    voxel_count: usize,
    dense_override: Option<&[f32]>,
    sparse_row_override: Option<&SparseRowNoiseOverride>,
    active_coords: &[[u32; 4]],
    sparse_resolution: usize,
    dense_resolution: usize,
    stage: &str,
) -> Vec<f32> {
    let mut noise = dense_noise_with_override(
        rng,
        channels.saturating_mul(voxel_count),
        dense_override,
        stage,
    );
    if let Some(override_rows) = sparse_row_override {
        merge_sparse_row_noise_override(
            noise.as_mut_slice(),
            override_rows,
            active_coords,
            channels,
            sparse_resolution,
            dense_resolution,
            stage,
        );
    }
    noise
}

#[cfg(feature = "runtime-model")]
fn resize_override_values(values: &[f32], expected_len: usize) -> Option<Vec<f32>> {
    if expected_len == 0 {
        return Some(Vec::new());
    }
    if values.is_empty() {
        return None;
    }
    if values.len() == expected_len {
        return Some(values.to_vec());
    }
    if values.len() == 1 {
        return Some(vec![values[0]; expected_len]);
    }

    let src_last = values.len() - 1;
    let dst_last = expected_len - 1;
    let mut out = Vec::with_capacity(expected_len);
    for dst_idx in 0..expected_len {
        let src_pos = dst_idx as f64 * src_last as f64 / dst_last.max(1) as f64;
        let src_floor = src_pos.floor() as usize;
        let src_ceil = src_pos.ceil() as usize;
        if src_floor == src_ceil {
            out.push(values[src_floor]);
            continue;
        }
        let t = (src_pos - src_floor as f64) as f32;
        let a = values[src_floor];
        let b = values[src_ceil];
        out.push(a * (1.0 - t) + b * t);
    }
    Some(out)
}

#[cfg(feature = "runtime-model")]
fn validated_cond_override(
    expected_len: usize,
    override_values: Option<&[f32]>,
    stage: &str,
    value_kind: &str,
) -> Result<Option<Vec<f32>>, String> {
    let Some(values) = override_values else {
        return Ok(None);
    };
    if values.len() == expected_len {
        return Ok(Some(values.to_vec()));
    }
    if runtime_parity_strict() {
        return Err(format!(
            "strict mode rejects {stage} {value_kind} override len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        ));
    }
    if let Some(resized) = resize_override_values(values, expected_len) {
        trellis_stage_log!(
            "burn_trellis: resized {stage} {value_kind} override from {} to {} values",
            values.len(),
            expected_len
        );
        return Ok(Some(resized));
    }
    trellis_stage_log!(
        "burn_trellis: ignoring {stage} {value_kind} override due to len mismatch (expected {}, got {})",
        expected_len,
        values.len()
    );
    Ok(None)
}

#[cfg(feature = "runtime-model")]
fn cond_override_for_tokens(
    overrides: Option<&TrellisNoiseOverrides>,
    cond_tokens: usize,
) -> (Option<&[f32]>, Option<&[f32]>) {
    const TOKENS_512: usize = 32 * 32 + 5;
    const TOKENS_1024: usize = 64 * 64 + 5;
    let Some(overrides) = overrides else {
        return (None, None);
    };
    match cond_tokens {
        TOKENS_512 => (
            overrides.cond_512.as_deref(),
            overrides.neg_cond_512.as_deref(),
        ),
        TOKENS_1024 => (
            overrides.cond_1024.as_deref(),
            overrides.neg_cond_1024.as_deref(),
        ),
        _ => (None, None),
    }
}

#[cfg(feature = "runtime-model")]
fn cond_hook_key_hint(cond_tokens: usize) -> &'static str {
    const TOKENS_512: usize = 32 * 32 + 5;
    const TOKENS_1024: usize = 64 * 64 + 5;
    match cond_tokens {
        TOKENS_512 => "get_cond_512.out.cond/get_cond_512.out.neg_cond",
        TOKENS_1024 => "get_cond_1024.out.cond/get_cond_1024.out.neg_cond",
        _ => "get_cond_*.out.cond/get_cond_*.out.neg_cond",
    }
}

#[cfg(feature = "runtime-model")]
fn missing_runtime_conditioning_error(
    stage: &str,
    cond_tokens: usize,
    cond_channels: usize,
) -> String {
    let expected = cond_tokens.saturating_mul(cond_channels);
    let hook_key_hint = cond_hook_key_hint(cond_tokens);
    format!(
        "missing TRELLIS image conditioning for stage '{stage}' (expected {expected} values from {cond_tokens} tokens x {cond_channels} channels). required conditioning keys: '{hook_key_hint}'. no synthetic/degraded fallback is allowed before conditioning is available."
    )
}

#[cfg(feature = "runtime-model")]
fn dense_cond_with_override<'a>(
    preprocess: &PreprocessOutput,
    cond_tokens: usize,
    cond_channels: usize,
    override_values: Option<&'a [f32]>,
    stage: &str,
) -> Result<Cow<'a, [f32]>, String> {
    let expected = cond_tokens.saturating_mul(cond_channels);
    if let Some(values) = override_values {
        if values.len() == expected {
            return Ok(Cow::Borrowed(values));
        }
        if runtime_parity_strict() {
            return Err(format!(
                "strict mode rejects {stage} cond override len mismatch (expected {}, got {})",
                expected,
                values.len()
            ));
        }
        if let Some(resized) = resize_override_values(values, expected) {
            trellis_stage_log!(
                "burn_trellis: resized {stage} cond override from {} to {} values",
                values.len(),
                expected
            );
            return Ok(Cow::Owned(resized));
        }
        trellis_stage_log!(
            "burn_trellis: ignoring {stage} cond override due to len mismatch (expected {}, got {})",
            expected,
            values.len()
        );
    }
    let _ = preprocess;
    Err(missing_runtime_conditioning_error(
        stage,
        cond_tokens,
        cond_channels,
    ))
}

#[cfg(feature = "runtime-model")]
fn dense_neg_cond_with_override<'a>(
    expected_len: usize,
    override_values: Option<&'a [f32]>,
    stage: &str,
) -> Result<Cow<'a, [f32]>, String> {
    if let Some(values) = override_values {
        if values.len() == expected_len {
            return Ok(Cow::Borrowed(values));
        }
        if runtime_parity_strict() {
            return Err(format!(
                "strict mode rejects {stage} neg-cond override len mismatch (expected {}, got {})",
                expected_len,
                values.len()
            ));
        }
        if let Some(resized) = resize_override_values(values, expected_len) {
            trellis_stage_log!(
                "burn_trellis: resized {stage} neg-cond override from {} to {} values",
                values.len(),
                expected_len
            );
            return Ok(Cow::Owned(resized));
        }
        trellis_stage_log!(
            "burn_trellis: ignoring {stage} neg-cond override due to len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        );
    }
    Ok(Cow::Owned(vec![0.0; expected_len]))
}

fn sparse_row_noise_map(override_rows: &SparseRowNoiseOverride) -> HashMap<u64, [f32; 32]> {
    let count = override_rows.coords.len().min(override_rows.feats.len());
    let mut out = HashMap::with_capacity(count * 2);
    for idx in 0..count {
        let coord = override_rows.coords[idx];
        out.insert(
            pack_coord(coord[1], coord[2], coord[3]),
            override_rows.feats[idx],
        );
    }
    out
}

#[cfg(feature = "runtime-model")]
#[allow(dead_code)]
fn merge_sparse_row_noise_override(
    dense_noise: &mut [f32],
    override_rows: &SparseRowNoiseOverride,
    active_coords: &[[u32; 4]],
    channels: usize,
    sparse_resolution: usize,
    dense_resolution: usize,
    stage: &str,
) {
    if channels == 0 || dense_noise.is_empty() {
        return;
    }
    let voxel_count = dense_noise.len() / channels.max(1);
    if voxel_count == 0 || dense_noise.len() != channels * voxel_count {
        return;
    }

    let active_keys: HashSet<u64> = active_coords
        .iter()
        .map(|coord| pack_coord(coord[1], coord[2], coord[3]))
        .collect();
    let count = override_rows.coords.len().min(override_rows.feats.len());
    let mut merged = 0usize;
    for idx in 0..count {
        let coord = override_rows.coords[idx];
        let key = pack_coord(coord[1], coord[2], coord[3]);
        if !active_keys.contains(&key) {
            continue;
        }
        let dense_idx = map_coord_to_dense_flat(coord, sparse_resolution, dense_resolution);
        if dense_idx >= voxel_count {
            continue;
        }
        let row = override_rows.feats[idx];
        for ch in 0..channels.min(32) {
            dense_noise[ch * voxel_count + dense_idx] = row[ch];
        }
        merged += 1;
    }
    if runtime_stage_debug_enabled() {
        trellis_stage_log!(
            "burn_trellis: merged {merged} sparse-row noise overrides for stage {stage}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
struct Lcg {
    state: u64,
    cached_normal: Option<f32>,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 + 0.5) * (1.0 / 4_294_967_296.0)
    }

    fn next_open01(&mut self) -> f32 {
        self.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON)
    }

    fn next_normal_f32(&mut self) -> f32 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }
        let u1 = self.next_open01();
        let u2 = self.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }
}
