#[cfg(feature = "runtime-model-wgpu")]
use crate::runtime_model::types::extraction::tensor_i32_to_vec;
#[cfg(feature = "runtime-model-wgpu")]
use crate::sampler::FlowEulerSampleTrace;

#[allow(clippy::too_many_arguments)]
fn sample_sparse_structure(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    coords_override: Option<&[[u32; 4]]>,
    cond_override: Option<&[f32]>,
    neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    capture_sampler_trace: bool,
    _parity_strict: bool,
    materialize_host_coords: bool,
    max_sparse_coords_override: Option<usize>,
    #[cfg(feature = "runtime-model")] sparse_flow: Option<&SparseStructureFlowRuntime>,
    #[cfg(feature = "runtime-model")] sparse_structure_decoder: Option<
        &SparseStructureDecoderRuntime,
    >,
) -> Result<SparseStructureSample, String> {
    #[cfg(feature = "runtime-model")]
    {
        let sparse_flow = sparse_flow.ok_or_else(|| {
            "burn_trellis: sparse flow runtime is required for sparse_structure stage".to_string()
        })?;
        let sparse_structure_decoder = sparse_structure_decoder.ok_or_else(|| {
            "burn_trellis: sparse structure decoder runtime is required for sparse_structure stage"
                .to_string()
        })?;
        return sample_sparse_structure_with_model(
            preprocess,
            resolution,
            rng,
            noise_override,
            coords_override,
            cond_override,
            neg_cond_override,
            sampler_config,
            sampler_override,
            capture_sampler_trace,
            materialize_host_coords,
            max_sparse_coords_override,
            sparse_flow,
            sparse_structure_decoder,
        );
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (
            preprocess,
            resolution,
            rng,
            noise_override,
            coords_override,
            cond_override,
            neg_cond_override,
            sampler_config,
            sampler_override,
            capture_sampler_trace,
            materialize_host_coords,
            max_sparse_coords_override,
        );
        Err("burn_trellis: sparse_structure stage requires `runtime-model` feature".to_string())
    }
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_sparse_structure_with_model(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    coords_override: Option<&[[u32; 4]]>,
    cond_override: Option<&[f32]>,
    neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    capture_sampler_trace: bool,
    materialize_host_coords: bool,
    max_sparse_coords_override: Option<usize>,
    sparse_flow: &SparseStructureFlowRuntime,
    sparse_structure_decoder: &SparseStructureDecoderRuntime,
) -> Result<SparseStructureSample, String> {
    #[cfg(feature = "runtime-model-wgpu")]
    let materialize_host_coords = if sparse_flow.backend_name() == "wgpu"
        && sparse_structure_decoder.backend_name() == "wgpu"
    {
        false
    } else {
        materialize_host_coords
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let _ = materialize_host_coords;
    let cond_prepare_start = Instant::now();
    let config = sparse_flow.config();
    let flow_resolution = config.resolution;
    let channels = config.in_channels;
    let flow_voxels = flow_resolution * flow_resolution * flow_resolution;
    let noise = dense_noise_with_override(
        rng,
        channels * flow_voxels,
        noise_override,
        "sparse_runtime",
    );

    let cond_tokens = 32 * 32 + 5;
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "sparse_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: sparse flow cond override rejected ({err})"
            ));
        }
    };
    let neg_cond =
        match dense_neg_cond_with_override(cond.len(), neg_cond_override, "sparse_runtime") {
            Ok(cond) => cond,
            Err(err) => {
                return Err(format!(
                    "burn_trellis: sparse flow neg-cond override rejected ({err})"
                ));
            }
        };
    if runtime_stage_debug_enabled() {
        let mut diff_sum = 0.0f64;
        let mut diff_max = 0.0f32;
        for (pos, neg) in cond.iter().zip(neg_cond.iter()) {
            let diff = (pos - neg).abs();
            diff_sum += diff as f64;
            diff_max = diff_max.max(diff);
        }
        let diff_mean = if cond.is_empty() {
            0.0
        } else {
            (diff_sum / cond.len() as f64) as f32
        };
        trellis_stage_log!(
            "burn_trellis: sparse cond override delta mean_abs={:.6} max_abs={:.6} len={}",
            diff_mean,
            diff_max,
            cond.len()
        );
    }
    let cond_tensor = match sparse_flow.prepare_condition(cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: sparse flow cond preparation failed ({err})"
            ));
        }
    };
    let neg_cond_tensor = match sparse_flow.prepare_condition(neg_cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: sparse flow negative cond preparation failed ({err})"
            ));
        }
    };
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    let cond_prepare_ms = cond_prepare_start.elapsed().as_secs_f64() * 1000.0;
    let sample_start = Instant::now();
    reset_sparse_flow_op_telemetry();
    #[cfg(feature = "runtime-model-wgpu")]
    let use_tensor_sparse_handoff = sparse_flow.backend_name() == "wgpu"
        && sparse_structure_decoder.backend_name() == "wgpu"
        && !capture_sampler_trace
        && coords_override.is_none();
    #[cfg(feature = "runtime-model-wgpu")]
    let (trace, latent, latent_tensor_wgpu) = if use_tensor_sparse_handoff {
        let latent_tensor = sparse_flow
            .sample_final_tensor_wgpu(
                noise.as_slice(),
                sample_cfg,
                sigma_min,
                &cond_tensor,
                &neg_cond_tensor,
                None,
            )
            .map_err(|err| format!("burn_trellis: sparse flow model prediction failed ({err})"))?;
        (
            FlowEulerSampleTrace {
                steps: sample_cfg.steps,
                samples: Vec::new(),
                step_0_pred_v: Vec::new(),
                step_0_pred_v_pos: Vec::new(),
                step_0_pred_v_neg: Vec::new(),
                step_0_x_t: Vec::new(),
                step_mid_x_t: Vec::new(),
                step_last_x_t: Vec::new(),
            },
            Vec::new(),
            Some(latent_tensor),
        )
    } else {
        let trace = sparse_flow
            .sample_with_trace(
                noise.as_slice(),
                sample_cfg,
                sigma_min,
                &cond_tensor,
                &neg_cond_tensor,
                None,
                capture_sampler_trace,
            )
            .map_err(|err| format!("burn_trellis: sparse flow model prediction failed ({err})"))?;
        let latent = trace.samples.clone();
        (trace, latent, None)
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let (trace, latent) = {
        let trace = sparse_flow
            .sample_with_trace(
                noise.as_slice(),
                sample_cfg,
                sigma_min,
                &cond_tensor,
                &neg_cond_tensor,
                None,
                capture_sampler_trace,
            )
            .map_err(|err| format!("burn_trellis: sparse flow model prediction failed ({err})"))?;
        let latent = trace.samples.clone();
        (trace, latent)
    };
    let sample_ms = sample_start.elapsed().as_secs_f64() * 1000.0;
    let flow_ops = sparse_flow_op_telemetry();
    let flow_ops_summary = current_sparse_flow_op_timing_summary();
    trellis_stage_log!(
        "burn_trellis: sparse flow op telemetry [sparse_runtime] self_attn_calls={} self_attn_ms={:.2} cross_attn_calls={} cross_attn_ms={:.2} mlp_calls={} mlp_ms={:.2}",
        flow_ops.self_attn_calls,
        flow_ops.self_attn_ns as f64 / 1_000_000.0,
        flow_ops.cross_attn_calls,
        flow_ops.cross_attn_ns as f64 / 1_000_000.0,
        flow_ops.mlp_calls,
        flow_ops.mlp_ns as f64 / 1_000_000.0
    );
    let post_start = Instant::now();
    if runtime_stage_debug_enabled() && !latent.is_empty() {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_v = 0.0f64;
        for value in latent.iter().copied() {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            sum_v += value as f64;
        }
        let mean_v = (sum_v / latent.len() as f64) as f32;
        trellis_stage_log!(
            "burn_trellis: sparse latent stats rows={} min={:.6} max={:.6} mean={:.6}",
            latent.len(),
            min_v,
            max_v,
            mean_v
        );
    }
    #[cfg(feature = "runtime-model-wgpu")]
    if runtime_stage_debug_enabled() && latent.is_empty()
        && let Some(latent_t) = latent_tensor_wgpu.as_ref()
    {
        let [batch, channels, depth, height, width] = latent_t.dims();
        trellis_stage_log!(
            "burn_trellis: sparse latent tensor stats backend=wgpu dims=[{batch},{channels},{depth},{height},{width}]"
        );
    }
    let max_sparse_cap = runtime_max_sparse_coords_for_backend(
        sparse_flow.backend_name(),
        max_sparse_coords_override,
    );
    let max_sparse_coords = max_sparse_cap.map(|(limit, _)| limit);
    #[cfg(feature = "runtime-model-wgpu")]
    let (coords, coords_wgpu, layout) = if let Some(override_coords) = coords_override {
        if runtime_stage_debug_enabled() {
            trellis_stage_log!(
                "burn_trellis: sparse runtime using hook coord override rows={}",
                override_coords.len()
            );
        }
        let layout = sparse_layout_from_coords(override_coords)?;
        if sparse_flow.backend_name() == "wgpu" {
            let coords_t = coords_u32_to_wgpu_tensor(override_coords)?;
            let coords_host = if materialize_host_coords {
                override_coords.to_vec()
            } else {
                Vec::new()
            };
            (coords_host, Some(coords_t), layout)
        } else {
            (override_coords.to_vec(), None, layout)
        }
    } else {
        let sampled = if let Some(latent_t) = latent_tensor_wgpu.as_ref() {
            sparse_structure_decoder
                .decode_to_sparse_coords_wgpu_latent_tensor(
                    latent_t.clone(),
                    resolution,
                    max_sparse_coords,
                )
                .map_err(|err| {
                    format!(
                        "burn_trellis: sparse structure decoder failed after tensor-native flow sampling ({err})"
                    )
                })?
        } else {
            sparse_structure_decoder
                .decode_to_sparse_coords(
                    latent.as_slice(),
                    flow_resolution,
                    resolution,
                    max_sparse_coords,
                )
                .map_err(|err| {
                    format!("burn_trellis: sparse structure decoder failed after flow sampling ({err})")
                })?
        };
        if sampled.rows() == 0 {
            return Err(
                "burn_trellis: sparse structure decoder produced zero active coordinates"
                    .to_string(),
            );
        }
        let sampled_rows = sampled.rows();
        let sampled_wgpu = sampled.coords_tensor();
        let sampled_host = if materialize_host_coords {
            sampled
                .coords_host("sparse structure decode coord materialization for staged runtime")?
        } else {
            Vec::new()
        };
        if let Some((limit, source)) = max_sparse_cap {
            trellis_stage_log!(
                "burn_trellis: sparse coords after threshold/cap = {} (limit={}, source={})",
                sampled_rows,
                limit,
                source.as_str()
            );
        }
        let sampled_layout = if sampled_rows == 0 {
            Vec::new()
        } else if !sampled_host.is_empty() {
            sparse_layout_from_coords(sampled_host.as_slice())?
        } else if sampled_wgpu.is_some() {
            // Sparse-structure decoder currently emits single-batch coords (`batch=0`).
            // Keep canonical WGPU path device-resident by deriving layout from row count
            // instead of materializing coord batches on host.
            vec![0..sampled_rows]
        } else {
            return Err(
                "burn_trellis: sparse structure stage requires either host coords or device coord tensor to derive layout"
                    .to_string(),
            );
        };
        (sampled_host, sampled_wgpu, sampled_layout)
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let (coords, layout) = if let Some(override_coords) = coords_override {
        if runtime_stage_debug_enabled() {
            trellis_stage_log!(
                "burn_trellis: sparse runtime using hook coord override rows={}",
                override_coords.len()
            );
        }
        let coords = override_coords.to_vec();
        let layout = sparse_layout_from_coords(coords.as_slice())?;
        (coords, layout)
    } else {
        let sampled = sparse_structure_decoder
            .decode_to_sparse_coords(
                latent.as_slice(),
                flow_resolution,
                resolution,
                max_sparse_coords,
            )
            .map_err(|err| {
                format!("burn_trellis: sparse structure decoder failed after flow sampling ({err})")
            })?;
        if sampled.rows() == 0 {
            return Err(
                "burn_trellis: sparse structure decoder produced zero active coordinates"
                    .to_string(),
            );
        }
        let sampled_host = sampled
            .coords_host("sparse structure decode coord materialization for staged runtime")?;
        if let Some((limit, source)) = max_sparse_cap {
            trellis_stage_log!(
                "burn_trellis: sparse coords after threshold/cap = {} (limit={}, source={})",
                sampled_host.len(),
                limit,
                source.as_str()
            );
        }
        let layout = sparse_layout_from_coords(sampled_host.as_slice())?;
        (sampled_host, layout)
    };
    #[cfg(feature = "runtime-model-wgpu")]
    if coords.is_empty()
        && coords_wgpu
            .as_ref()
            .map_or(true, |coords_t| coords_t.dims()[0] == 0)
    {
        return Err(
            "burn_trellis: sparse structure stage produced no coordinates after overrides"
                .to_string(),
        );
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    if coords.is_empty() {
        return Err(
            "burn_trellis: sparse structure stage produced no coordinates after overrides"
                .to_string(),
        );
    }
    let postprocess_ms = post_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: sparse runtime profile cond_prep={cond_prepare_ms:.2} ms sample={sample_ms:.2} ms post={postprocess_ms:.2} ms total={:.2} ms",
        cond_prepare_ms + sample_ms + postprocess_ms
    );
    Ok(SparseStructureSample {
        source: match (
            sparse_flow.backend_name(),
            sparse_structure_decoder.backend_name(),
        ) {
            ("wgpu", "wgpu") => SparseStructureStageSource::RuntimeModelWgpu,
            _ => SparseStructureStageSource::RuntimeModelCpu,
        },
        sampler_config: sample_cfg,
        sigma_min,
        step_count: trace.steps,
        resolution,
        flow_resolution,
        flow_channels: channels,
        noise,
        step_0_pred_v: trace.step_0_pred_v,
        step_0_pred_v_pos: trace.step_0_pred_v_pos,
        step_0_pred_v_neg: trace.step_0_pred_v_neg,
        step_0_x_t: trace.step_0_x_t,
        step_mid_x_t: trace.step_mid_x_t,
        step_last_x_t: trace.step_last_x_t,
        latent,
        coords,
        layout,
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu,
        runtime_profile: Some(SparseStructureRuntimeProfile {
            cond_prepare_ms,
            sample_ms,
            postprocess_ms,
            flow_ops: flow_ops_summary,
        }),
    })
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn build_sparse_runtime_noise(
    rng: &mut Lcg,
    channels: usize,
    coords: &[[u32; 4]],
    dense_override: Option<&[f32]>,
    sparse_row_override: Option<&SparseRowNoiseOverride>,
    sparse_resolution: usize,
    dense_resolution: usize,
    stage: &str,
) -> Vec<f32> {
    let rows = coords.len();
    if rows == 0 || channels == 0 {
        return Vec::new();
    }
    let mut noise = (0..rows.saturating_mul(channels))
        .map(|_| rng.next_normal_f32())
        .collect::<Vec<_>>();

    if let Some(values) = dense_override {
        let voxel_count = dense_resolution
            .saturating_mul(dense_resolution)
            .saturating_mul(dense_resolution);
        let expected = channels.saturating_mul(voxel_count);
        if voxel_count == 0 || values.len() != expected {
            trellis_stage_log!(
                "burn_trellis: ignoring {stage} dense noise override due to len mismatch (expected {}, got {})",
                expected,
                values.len()
            );
        } else {
            for (row_idx, coord) in coords.iter().enumerate() {
                let dense_idx =
                    map_coord_to_dense_flat(*coord, sparse_resolution.max(1), dense_resolution);
                let row_base = row_idx.saturating_mul(channels);
                for ch in 0..channels {
                    noise[row_base + ch] = values[ch * voxel_count + dense_idx];
                }
            }
        }
    }

    if let Some(override_rows) = sparse_row_override {
        let override_map = sparse_row_noise_map(override_rows);
        let mut merged = 0usize;
        for (row_idx, coord) in coords.iter().enumerate() {
            let key = pack_coord(coord[1], coord[2], coord[3]);
            let Some(row) = override_map.get(&key) else {
                continue;
            };
            let row_base = row_idx.saturating_mul(channels);
            for ch in 0..channels.min(32) {
                noise[row_base + ch] = row[ch];
            }
            merged += 1;
        }
        if runtime_stage_debug_enabled() {
            trellis_stage_log!(
                "burn_trellis: merged {merged} sparse-row noise overrides for stage {stage}"
            );
        }
    }
    noise
}

#[cfg(feature = "runtime-model")]
fn build_sparse_runtime_noise_rows_only(
    rng: &mut Lcg,
    channels: usize,
    rows: usize,
    sparse_resolution: usize,
    dense_resolution: usize,
    dense_override: Option<&[f32]>,
    sparse_row_override: Option<&SparseRowNoiseOverride>,
    stage: &str,
) -> Result<Vec<f32>, String> {
    if rows == 0 || channels == 0 {
        return Ok(Vec::new());
    }
    let mut noise = (0..rows.saturating_mul(channels))
        .map(|_| rng.next_normal_f32())
        .collect::<Vec<_>>();
    if let Some(override_rows) = sparse_row_override {
        if override_rows.feats.len() != override_rows.coords.len() {
            return Err(format!(
                "burn_trellis: {stage} sparse-row noise override has mismatched coords/feats rows (coords={}, feats={})",
                override_rows.coords.len(),
                override_rows.feats.len()
            ));
        }
        if override_rows.feats.len() != rows {
            return Err(format!(
                "burn_trellis: {stage} sparse-row noise override rows ({}) must match runtime sparse rows ({rows}) on canonical device-token path",
                override_rows.feats.len()
            ));
        }
        if let Some(values) = dense_override {
            let dense_res = dense_resolution.max(1);
            let voxel_count = dense_res
                .saturating_mul(dense_res)
                .saturating_mul(dense_res);
            let expected = channels.saturating_mul(voxel_count);
            if values.len() != expected {
                return Err(format!(
                    "burn_trellis: {stage} dense noise override len mismatch on canonical device-token path (expected {expected}, got {})",
                    values.len()
                ));
            }
            for (row_idx, coord) in override_rows.coords.iter().enumerate() {
                let dense_idx = map_coord_to_dense_flat(
                    *coord,
                    sparse_resolution.max(1),
                    dense_res,
                );
                let row_base = row_idx.saturating_mul(channels);
                for ch in 0..channels {
                    noise[row_base + ch] = values[ch * voxel_count + dense_idx];
                }
            }
        }
        // Canonical device-token path forbids host coord completion. For parity
        // harnesses we still permit sparse-row override injection by row index.
        for (row_idx, row) in override_rows.feats.iter().enumerate() {
            let row_base = row_idx.saturating_mul(channels);
            for ch in 0..channels.min(32) {
                noise[row_base + ch] = row[ch];
            }
        }
    } else if dense_override.is_some() {
        return Err(format!(
            "burn_trellis: {stage} dense noise override on canonical device-token path requires sparse-row override coords"
        ));
    }
    Ok(noise)
}

#[cfg(feature = "runtime-model")]
fn sparse_row_noise_override_rows(override_rows: &SparseRowNoiseOverride) -> usize {
    override_rows.coords.len().min(override_rows.feats.len())
}

#[cfg(feature = "runtime-model")]
fn require_sparse_row_noise_override_rows<'a>(
    override_rows: Option<&'a SparseRowNoiseOverride>,
    rows: usize,
    stage: &str,
) -> Result<Option<&'a SparseRowNoiseOverride>, String> {
    let Some(override_rows) = override_rows else {
        return Ok(None);
    };
    let override_len = sparse_row_noise_override_rows(override_rows);
    if override_rows.coords.len() != override_rows.feats.len() {
        return Err(format!(
            "burn_trellis: {stage} sparse-row noise override has mismatched coords/feats rows (coords={}, feats={})",
            override_rows.coords.len(),
            override_rows.feats.len()
        ));
    }
    if override_len != rows {
        return Err(format!(
            "burn_trellis: {stage} sparse-row noise override rows ({override_len}) must match runtime sparse rows ({rows})"
        ));
    }
    Ok(Some(override_rows))
}

#[cfg(feature = "runtime-model")]
fn optional_sparse_row_noise_override_for_rows<'a>(
    override_rows: Option<&'a SparseRowNoiseOverride>,
    rows: usize,
    stage: &str,
) -> Option<&'a SparseRowNoiseOverride> {
    let override_rows = override_rows?;
    let override_len = sparse_row_noise_override_rows(override_rows);
    if override_len == rows && override_rows.coords.len() == override_rows.feats.len() {
        return Some(override_rows);
    }
    if runtime_stage_debug_enabled() {
        trellis_stage_log!(
            "burn_trellis: skipping generic {stage} sparse-row noise override rows={} for runtime sparse rows={rows}",
            override_len
        );
    }
    None
}

fn sparse_layout_from_batch_ids(
    batch_ids: &[usize],
    context: &str,
) -> Result<Vec<std::ops::Range<usize>>, String> {
    if batch_ids.is_empty() {
        return Ok(Vec::new());
    }
    let max_batch = batch_ids.iter().copied().max().unwrap_or(0);
    let mut layout = Vec::with_capacity(max_batch.saturating_add(1));
    let mut cursor = 0usize;
    for batch_idx in 0..=max_batch {
        let start = cursor;
        while cursor < batch_ids.len() && batch_ids[cursor] == batch_idx {
            cursor += 1;
        }
        layout.push(start..cursor);
    }
    if cursor != batch_ids.len() {
        let offending_batch = batch_ids[cursor];
        return Err(format!(
            "{context}: sparse coords must be grouped by non-decreasing batch id from 0..N; first offending row={} batch={}",
            cursor,
            offending_batch
        ));
    }
    Ok(layout)
}

fn sparse_layout_from_coords(coords: &[[u32; 4]]) -> Result<Vec<std::ops::Range<usize>>, String> {
    let mut batch_ids = Vec::with_capacity(coords.len());
    for coord in coords {
        batch_ids.push(coord[0] as usize);
    }
    sparse_layout_from_batch_ids(batch_ids.as_slice(), "sparse_layout_from_coords")
}

fn validate_sparse_layout_rows(
    layout: &[std::ops::Range<usize>],
    rows: usize,
    context: &str,
) -> Result<(), String> {
    let mut expected_start = 0usize;
    for (batch_idx, range) in layout.iter().enumerate() {
        if range.start > range.end {
            return Err(format!(
                "{context}: sparse layout start>end for batch {}: {}..{}",
                batch_idx, range.start, range.end
            ));
        }
        if range.start != expected_start {
            return Err(format!(
                "{context}: sparse layout must be contiguous from row 0; batch {} starts at {} but expected {}",
                batch_idx, range.start, expected_start
            ));
        }
        expected_start = range.end;
    }
    if expected_start != rows {
        return Err(format!(
            "{context}: sparse layout row mismatch: layout_rows={} expected_rows={}",
            expected_start, rows
        ));
    }
    Ok(())
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_shape_slat_with_model(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    sparse_layout_input: &[std::ops::Range<usize>],
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    capture_sampler_trace: bool,
    #[cfg(feature = "runtime-model-wgpu")] coords_wgpu: Option<
        Tensor<SparseFlowWgpuBackend, 2, Int>,
    >,
    shape_flow: &SparseStructureFlowRuntime,
) -> Result<ShapeSLatSample, String> {
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    #[cfg(feature = "runtime-model-wgpu")]
    let use_device_coords = coords_wgpu.is_some();
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let use_device_coords = false;
    #[cfg(feature = "runtime-model-wgpu")]
    if shape_flow.backend_name() == "wgpu" && !use_device_coords {
        return Err(
            "burn_trellis: shape slat canonical wgpu path requires device coord tensor; host coord completion is disabled"
                .to_string(),
        );
    }
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_rows_wgpu = coords_wgpu
        .as_ref()
        .map(|coords_t| coords_t.dims()[0])
        .unwrap_or(0);
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_rows_wgpu = 0usize;
    let sparse_row_count = if use_device_coords {
        coords_rows_wgpu
    } else {
        coords.len()
    };
    if sparse_row_count == 0 {
        return Ok(ShapeSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: capture_sampler_trace.then_some(Vec::new()),
            features: Vec::new(),
            noise: Vec::new(),
            step_0_pred_v: Vec::new(),
            step_0_pred_v_pos: Vec::new(),
            step_0_pred_v_neg: Vec::new(),
            step_0_x_t: Vec::new(),
            step_mid_x_t: Vec::new(),
            step_last_x_t: Vec::new(),
            coords: Vec::new(),
            layout: Vec::new(),
            flow_ops: SparseFlowOpTimingSummary::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu: None,
            #[cfg(feature = "runtime-model-wgpu")]
            features_wgpu: None,
        });
    }
    let config = shape_flow.config();
    if config.out_channels == 0 {
        return Err("burn_trellis: shape flow runtime has out_channels=0".to_string());
    }
    let feature_channels = 32usize.min(config.out_channels);
    let sparse_layout = if use_device_coords {
        sparse_layout_input.to_vec()
    } else {
        sparse_layout_from_coords(coords)?
    };
    validate_sparse_layout_rows(
        sparse_layout.as_slice(),
        sparse_row_count,
        "shape_slat_runtime layout validation",
    )?;
    let dense_resolution = config.resolution.max(1);
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_for_noise_storage;
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_for_noise = if use_device_coords && coords.is_empty() && noise_dense_override.is_some()
    {
        let coords_t = coords_wgpu.as_ref().ok_or_else(|| {
            "burn_trellis: shape slat dense noise override requires device coords".to_string()
        })?;
        coords_for_noise_storage = coords_wgpu_tensor_to_host(
            coords_t.clone(),
            "burn_trellis: shape slat coord materialization for dense noise",
        )?;
        coords_for_noise_storage.as_slice()
    } else {
        coords
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_for_noise = coords;
    let noise = if use_device_coords {
        if !coords_for_noise.is_empty() && (noise_dense_override.is_some() || noise_override.is_some()) {
            build_sparse_runtime_noise(
                rng,
                config.out_channels,
                coords_for_noise,
                noise_dense_override,
                noise_override,
                sparse_resolution,
                dense_resolution,
                "shape_slat_runtime",
            )
        } else {
            build_sparse_runtime_noise_rows_only(
                rng,
                config.out_channels,
                sparse_row_count,
                sparse_resolution,
                dense_resolution,
                noise_dense_override,
                noise_override,
                "shape_slat_runtime",
            )?
        }
    } else {
        build_sparse_runtime_noise(
            rng,
            config.out_channels,
            coords,
            noise_dense_override,
            noise_override,
            sparse_resolution,
            dense_resolution,
            "shape_slat_runtime",
        )
    };

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "shape_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: shape slat cond override rejected ({err})"
            ));
        }
    };
    let neg_cond =
        match dense_neg_cond_with_override(cond.len(), neg_cond_override, "shape_slat_runtime") {
            Ok(cond) => cond,
            Err(err) => {
                return Err(format!(
                    "burn_trellis: shape slat neg-cond override rejected ({err})"
                ));
            }
        };
    let cond_tensor = match shape_flow.prepare_condition(cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: shape slat cond preparation failed ({err})"
            ));
        }
    };
    let neg_cond_tensor = match shape_flow.prepare_condition(neg_cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: shape slat negative cond preparation failed ({err})"
            ));
        }
    };
    #[cfg(feature = "runtime-model-wgpu")]
    let materialize_host_rows = capture_sampler_trace || shape_flow.backend_name() != "wgpu";
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let materialize_host_rows = true;
    reset_sparse_flow_op_telemetry();
    #[cfg(feature = "runtime-model-wgpu")]
    let trace = if shape_flow.backend_name() == "wgpu" {
        let coords_t = coords_wgpu
            .as_ref()
            .ok_or_else(|| {
                "burn_trellis: shape slat canonical wgpu path missing device coords".to_string()
            })?;
        let device = coords_t.device();
        let noise_t = Tensor::<SparseFlowWgpuBackend, 1>::from_floats(noise.as_slice(), &device)
            .reshape([sparse_row_count, config.out_channels]);
        shape_flow.sample_sparse_rows_with_trace_wgpu_inputs(
            coords_t.clone(),
            noise_t,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            None,
            sparse_layout.clone(),
            sparse_resolution.max(1),
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    } else {
        let sparse_noise = shape_flow
            .sparse_tensor_from_host_layout(
                coords.to_vec(),
                noise.clone(),
                sparse_layout.clone(),
                config.out_channels,
                sparse_resolution.max(1),
            )
            .map_err(|err| format!("burn_trellis: shape slat sparse tensor assembly failed ({err})"))?;
        shape_flow.sample_sparse_rows_with_trace(
            &sparse_noise,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            None,
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let trace = {
        let sparse_noise = shape_flow
            .sparse_tensor_from_host_layout(
                coords.to_vec(),
                noise.clone(),
                sparse_layout.clone(),
                config.out_channels,
                sparse_resolution.max(1),
            )
            .map_err(|err| format!("burn_trellis: shape slat sparse tensor assembly failed ({err})"))?;
        shape_flow.sample_sparse_rows_with_trace(
            &sparse_noise,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            None,
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    };
    let trace = match trace {
        Ok(trace) => trace,
        Err(err) => {
            return Err(format!(
                "burn_trellis: shape slat runtime prediction failed ({err})"
            ));
        }
    };
    let flow_ops = sparse_flow_op_telemetry();
    let flow_ops_summary = current_sparse_flow_op_timing_summary();
    trellis_stage_log!(
        "burn_trellis: sparse flow op telemetry [shape_slat] self_attn_calls={} self_attn_ms={:.2} cross_attn_calls={} cross_attn_ms={:.2} mlp_calls={} mlp_ms={:.2}",
        flow_ops.self_attn_calls,
        flow_ops.self_attn_ns as f64 / 1_000_000.0,
        flow_ops.cross_attn_calls,
        flow_ops.cross_attn_ns as f64 / 1_000_000.0,
        flow_ops.mlp_calls,
        flow_ops.mlp_ns as f64 / 1_000_000.0
    );

    let mut features = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut noise_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_pos_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_neg_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_mid_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_last_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    if materialize_host_rows {
        let gathered_channels = feature_channels.min(trace.row_channels);
        for row_idx in 0..sparse_row_count {
            let gathered_base = row_idx.saturating_mul(trace.row_channels);
            let noise_base = row_idx.saturating_mul(config.out_channels);
            let mut row = [0.0f32; 32];
            let mut noise_row = [0.0f32; 32];
            let mut step_0_pred_v_row = [0.0f32; 32];
            let mut step_0_pred_v_pos_row = [0.0f32; 32];
            let mut step_0_pred_v_neg_row = [0.0f32; 32];
            let mut step_0_row = [0.0f32; 32];
            let mut step_mid_row = [0.0f32; 32];
            let mut step_last_row = [0.0f32; 32];
            for ch in 0..gathered_channels {
                let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
                let std = normalization
                    .std
                    .get(ch)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1.0e-6);
                let sampled = trace.samples[gathered_base + ch];
                row[ch] = sampled * std + mean;
                noise_row[ch] = noise[noise_base + ch];
                step_0_pred_v_row[ch] = trace.step_0_pred_v[gathered_base + ch];
                step_0_pred_v_pos_row[ch] = trace.step_0_pred_v_pos[gathered_base + ch];
                step_0_pred_v_neg_row[ch] = trace.step_0_pred_v_neg[gathered_base + ch];
                step_0_row[ch] = trace.step_0_x_t[gathered_base + ch];
                step_mid_row[ch] = trace.step_mid_x_t[gathered_base + ch];
                step_last_row[ch] = trace.step_last_x_t[gathered_base + ch];
            }
            features.push(row);
            noise_rows.push(noise_row);
            step_0_pred_v_rows.push(step_0_pred_v_row);
            step_0_pred_v_pos_rows.push(step_0_pred_v_pos_row);
            step_0_pred_v_neg_rows.push(step_0_pred_v_neg_row);
            step_0_rows.push(step_0_row);
            step_mid_rows.push(step_mid_row);
            step_last_rows.push(step_last_row);
        }
    }
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_out = if use_device_coords && !materialize_host_rows {
        Vec::new()
    } else if !coords.is_empty() {
        coords.to_vec()
    } else {
        coords_for_noise.to_vec()
    };
    #[cfg(feature = "runtime-model-wgpu")]
    let features_wgpu = if shape_flow.backend_name() == "wgpu" {
        let samples_t = trace.samples_wgpu.clone().ok_or_else(|| {
            "burn_trellis: shape slat canonical wgpu path missing device trace rows; host tensorization fallback is disabled"
                .to_string()
        })?;
        Some(denormalize_and_pad_trace_rows_wgpu(
            samples_t,
            normalization,
            "shape slat trace denorm",
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_out = coords.to_vec();
    Ok(ShapeSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features,
        noise: noise_rows,
        step_0_pred_v: step_0_pred_v_rows,
        step_0_pred_v_pos: step_0_pred_v_pos_rows,
        step_0_pred_v_neg: step_0_pred_v_neg_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        coords: coords_out,
        layout: sparse_layout,
        flow_ops: flow_ops_summary,
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu: if shape_flow.backend_name() == "wgpu" {
            coords_wgpu
        } else {
            None
        },
        #[cfg(feature = "runtime-model-wgpu")]
        features_wgpu,
    })
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_tex_slat_with_model(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    capture_sampler_trace: bool,
    #[cfg(feature = "runtime-model-wgpu")] coords_wgpu: Option<
        Tensor<SparseFlowWgpuBackend, 2, Int>,
    >,
    tex_flow: &SparseStructureFlowRuntime,
) -> Result<TexSLatSample, String> {
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    #[cfg(feature = "runtime-model-wgpu")]
    let use_device_coords = coords_wgpu.is_some();
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let use_device_coords = false;
    #[cfg(feature = "runtime-model-wgpu")]
    if tex_flow.backend_name() == "wgpu" && !use_device_coords {
        return Err(
            "burn_trellis: tex slat canonical wgpu path requires device coord tensor; host coord completion is disabled"
                .to_string(),
        );
    }
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_rows_wgpu = coords_wgpu
        .as_ref()
        .map(|coords_t| coords_t.dims()[0])
        .unwrap_or(0);
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_rows_wgpu = 0usize;
    let sparse_row_count = if use_device_coords {
        coords_rows_wgpu
    } else {
        shape_slat.coords.len()
    };
    if sparse_row_count == 0 {
        return Ok(TexSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: capture_sampler_trace.then_some(Vec::new()),
            features: Vec::new(),
            noise: Vec::new(),
            step_0_pred_v: Vec::new(),
            step_0_pred_v_pos: Vec::new(),
            step_0_pred_v_neg: Vec::new(),
            step_0_x_t: Vec::new(),
            step_mid_x_t: Vec::new(),
            step_last_x_t: Vec::new(),
            shape_slat_cond: Vec::new(),
            coords: Vec::new(),
            layout: Vec::new(),
            flow_ops: SparseFlowOpTimingSummary::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu: None,
            #[cfg(feature = "runtime-model-wgpu")]
            features_wgpu: None,
        });
    }
    let config = tex_flow.config();
    if config.out_channels == 0 {
        return Err("burn_trellis: tex flow runtime has out_channels=0".to_string());
    }
    let feature_channels = 32usize.min(config.out_channels);
    let sparse_layout = if use_device_coords {
        shape_slat.layout.clone()
    } else {
        sparse_layout_from_coords(shape_slat.coords.as_slice())?
    };
    validate_sparse_layout_rows(
        sparse_layout.as_slice(),
        sparse_row_count,
        "tex_slat_runtime layout validation",
    )?;
    let dense_resolution = config.resolution.max(1);
    let concat_channels = config.in_channels.saturating_sub(config.out_channels);
    if concat_channels == 0 {
        return Err("burn_trellis: tex flow runtime has no concat channels".to_string());
    }
    let shape_rows_host = shape_slat.features.get(..sparse_row_count);
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_rows_wgpu = shape_slat.features_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    if tex_flow.backend_name() == "wgpu" && shape_rows_wgpu.is_none() {
        return Err(
            "burn_trellis: tex slat canonical wgpu path requires device shape rows for concat; host concat fallback is disabled"
                .to_string(),
        );
    }
    #[cfg(feature = "runtime-model-wgpu")]
    let concat_rows_host = if tex_flow.backend_name() == "wgpu" {
        None
    } else {
        let rows = shape_rows_host.ok_or_else(|| {
            format!(
                "burn_trellis: tex slat shape feature row mismatch: shape_rows={} expected_rows={}",
                shape_slat.features.len(),
                sparse_row_count
            )
        })?;
        Some(build_shape_concat_rows_host(
            rows,
            concat_channels,
            shape_normalization,
        ))
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let concat_rows_host = {
        let rows = shape_rows_host.ok_or_else(|| {
            format!(
                "burn_trellis: tex slat shape feature row mismatch: shape_rows={} expected_rows={}",
                shape_slat.features.len(),
                sparse_row_count
            )
        })?;
        build_shape_concat_rows_host(rows, concat_channels, shape_normalization)
    };
    let shape_cond_rows_host = shape_rows_host.map(|rows| build_shape_cond_rows_host(rows, shape_normalization));
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_for_noise_storage;
    #[cfg(feature = "runtime-model-wgpu")]
    let coords_for_noise = if use_device_coords
        && shape_slat.coords.is_empty()
        && noise_dense_override.is_some()
    {
        let coords_t = coords_wgpu.as_ref().ok_or_else(|| {
            "burn_trellis: tex slat dense noise override requires device coords".to_string()
        })?;
        coords_for_noise_storage = coords_wgpu_tensor_to_host(
            coords_t.clone(),
            "burn_trellis: tex slat coord materialization for dense noise",
        )?;
        coords_for_noise_storage.as_slice()
    } else {
        shape_slat.coords.as_slice()
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_for_noise = shape_slat.coords.as_slice();
    let noise = if use_device_coords {
        if !coords_for_noise.is_empty()
            && (noise_dense_override.is_some() || noise_override.is_some())
        {
            build_sparse_runtime_noise(
                rng,
                config.out_channels,
                coords_for_noise,
                noise_dense_override,
                noise_override,
                sparse_resolution,
                dense_resolution,
                "tex_slat_runtime",
            )
        } else {
            build_sparse_runtime_noise_rows_only(
                rng,
                config.out_channels,
                sparse_row_count,
                sparse_resolution,
                dense_resolution,
                noise_dense_override,
                noise_override,
                "tex_slat_runtime",
            )?
        }
    } else {
        build_sparse_runtime_noise(
            rng,
            config.out_channels,
            shape_slat.coords.as_slice(),
            noise_dense_override,
            noise_override,
            sparse_resolution,
            dense_resolution,
            "tex_slat_runtime",
        )
    };

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "tex_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: tex slat cond override rejected ({err})"
            ));
        }
    };
    let neg_cond =
        match dense_neg_cond_with_override(cond.len(), neg_cond_override, "tex_slat_runtime") {
            Ok(cond) => cond,
            Err(err) => {
                return Err(format!(
                    "burn_trellis: tex slat neg-cond override rejected ({err})"
                ));
            }
        };
    let cond_tensor = match tex_flow.prepare_condition(cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: tex slat cond preparation failed ({err})"
            ));
        }
    };
    let neg_cond_tensor = match tex_flow.prepare_condition(neg_cond.as_ref(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            return Err(format!(
                "burn_trellis: tex slat negative cond preparation failed ({err})"
            ));
        }
    };
    #[cfg(feature = "runtime-model-wgpu")]
    let materialize_host_rows = capture_sampler_trace || tex_flow.backend_name() != "wgpu";
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let materialize_host_rows = true;
    reset_sparse_flow_op_telemetry();
    #[cfg(feature = "runtime-model-wgpu")]
    let trace = if tex_flow.backend_name() == "wgpu" {
        let coords_t = coords_wgpu
            .as_ref()
            .ok_or_else(|| {
                "burn_trellis: tex slat canonical wgpu path missing device coords".to_string()
            })?;
        let device = coords_t.device();
        let noise_t = Tensor::<SparseFlowWgpuBackend, 1>::from_floats(noise.as_slice(), &device)
            .reshape([sparse_row_count, config.out_channels]);
        let shape_rows_t = shape_rows_wgpu.ok_or_else(|| {
            "burn_trellis: tex slat canonical wgpu path missing device shape rows".to_string()
        })?;
        let concat_t = build_shape_concat_tensor_wgpu(
            shape_rows_t,
            sparse_row_count,
            concat_channels,
            shape_normalization,
            "tex slat concat shape tensor build",
        )?;
        tex_flow.sample_sparse_rows_with_trace_wgpu_inputs(
            coords_t.clone(),
            noise_t,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            Some(concat_t),
            sparse_layout.clone(),
            sparse_resolution.max(1),
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    } else {
        let concat_rows = concat_rows_host.as_ref().ok_or_else(|| {
            "burn_trellis: tex slat host concat rows are unavailable for cpu/runtime fallback path"
                .to_string()
        })?;
        let sparse_noise = tex_flow
            .sparse_tensor_from_host_layout(
                shape_slat.coords.clone(),
                noise.clone(),
                sparse_layout.clone(),
                config.out_channels,
                sparse_resolution.max(1),
            )
            .map_err(|err| format!("burn_trellis: tex slat sparse tensor assembly failed ({err})"))?;
        let concat_owned = tex_flow
            .varlen_tensor_from_host_layout(
                concat_rows.clone(),
                sparse_layout.clone(),
                concat_channels,
            )
            .map_err(|err| format!("burn_trellis: tex slat concat tensor assembly failed ({err})"))?;
        tex_flow.sample_sparse_rows_with_trace(
            &sparse_noise,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            Some(&concat_owned),
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let trace = {
        let concat_rows = concat_rows_host.as_slice();
        let sparse_noise = tex_flow
            .sparse_tensor_from_host_layout(
                shape_slat.coords.clone(),
                noise.clone(),
                sparse_layout.clone(),
                config.out_channels,
                sparse_resolution.max(1),
            )
            .map_err(|err| format!("burn_trellis: tex slat sparse tensor assembly failed ({err})"))?;
        let concat_owned = tex_flow
            .varlen_tensor_from_host_layout(
                concat_rows.to_vec(),
                sparse_layout.clone(),
                concat_channels,
            )
            .map_err(|err| format!("burn_trellis: tex slat concat tensor assembly failed ({err})"))?;
        tex_flow.sample_sparse_rows_with_trace(
            &sparse_noise,
            sample_cfg,
            sigma_min,
            &cond_tensor,
            &neg_cond_tensor,
            Some(&concat_owned),
            feature_channels,
            capture_sampler_trace,
            materialize_host_rows,
        )
    };
    let trace = match trace {
        Ok(trace) => trace,
        Err(err) => {
            return Err(format!(
                "burn_trellis: tex slat runtime prediction failed ({err})"
            ));
        }
    };
    let flow_ops = sparse_flow_op_telemetry();
    let flow_ops_summary = current_sparse_flow_op_timing_summary();
    trellis_stage_log!(
        "burn_trellis: sparse flow op telemetry [tex_slat] self_attn_calls={} self_attn_ms={:.2} cross_attn_calls={} cross_attn_ms={:.2} mlp_calls={} mlp_ms={:.2}",
        flow_ops.self_attn_calls,
        flow_ops.self_attn_ns as f64 / 1_000_000.0,
        flow_ops.cross_attn_calls,
        flow_ops.cross_attn_ns as f64 / 1_000_000.0,
        flow_ops.mlp_calls,
        flow_ops.mlp_ns as f64 / 1_000_000.0
    );

    let mut features = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut noise_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_pos_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_pred_v_neg_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_0_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_mid_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut step_last_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    let mut shape_cond_rows = Vec::with_capacity(if materialize_host_rows {
        sparse_row_count
    } else {
        0
    });
    if materialize_host_rows {
        let gathered_channels = feature_channels.min(trace.row_channels);
        for idx in 0..sparse_row_count {
            let gathered_base = idx.saturating_mul(trace.row_channels);
            let noise_base = idx.saturating_mul(config.out_channels);
            let mut row = [0.0f32; 32];
            let mut noise_row = [0.0f32; 32];
            let mut step_0_pred_v_row = [0.0f32; 32];
            let mut step_0_pred_v_pos_row = [0.0f32; 32];
            let mut step_0_pred_v_neg_row = [0.0f32; 32];
            let mut step_0_row = [0.0f32; 32];
            let mut step_mid_row = [0.0f32; 32];
            let mut step_last_row = [0.0f32; 32];
            let shape_cond = shape_cond_rows_host
                .as_ref()
                .and_then(|rows| rows.get(idx))
                .copied()
                .unwrap_or([0.0f32; 32]);
            for ch in 0..gathered_channels {
                let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
                let std = normalization
                    .std
                    .get(ch)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1.0e-6);
                let sampled = trace.samples[gathered_base + ch];
                row[ch] = sampled * std + mean;
                noise_row[ch] = noise[noise_base + ch];
                step_0_pred_v_row[ch] = trace.step_0_pred_v[gathered_base + ch];
                step_0_pred_v_pos_row[ch] = trace.step_0_pred_v_pos[gathered_base + ch];
                step_0_pred_v_neg_row[ch] = trace.step_0_pred_v_neg[gathered_base + ch];
                step_0_row[ch] = trace.step_0_x_t[gathered_base + ch];
                step_mid_row[ch] = trace.step_mid_x_t[gathered_base + ch];
                step_last_row[ch] = trace.step_last_x_t[gathered_base + ch];
            }
            features.push(row);
            noise_rows.push(noise_row);
            step_0_pred_v_rows.push(step_0_pred_v_row);
            step_0_pred_v_pos_rows.push(step_0_pred_v_pos_row);
            step_0_pred_v_neg_rows.push(step_0_pred_v_neg_row);
            step_0_rows.push(step_0_row);
            step_mid_rows.push(step_mid_row);
            step_last_rows.push(step_last_row);
            shape_cond_rows.push(shape_cond);
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    let coords_out = if use_device_coords && !materialize_host_rows {
        Vec::new()
    } else if !shape_slat.coords.is_empty() {
        shape_slat.coords.clone()
    } else {
        coords_for_noise.to_vec()
    };
    #[cfg(feature = "runtime-model-wgpu")]
    let features_wgpu = if tex_flow.backend_name() == "wgpu" {
        let samples_t = trace.samples_wgpu.clone().ok_or_else(|| {
            "burn_trellis: tex slat canonical wgpu path missing device trace rows; host tensorization fallback is disabled"
                .to_string()
        })?;
        Some(denormalize_and_pad_trace_rows_wgpu(
            samples_t,
            normalization,
            "tex slat trace denorm",
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let coords_out = shape_slat.coords.clone();

    Ok(TexSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features,
        noise: noise_rows,
        step_0_pred_v: step_0_pred_v_rows,
        step_0_pred_v_pos: step_0_pred_v_pos_rows,
        step_0_pred_v_neg: step_0_pred_v_neg_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        shape_slat_cond: shape_cond_rows,
        coords: coords_out,
        layout: sparse_layout,
        flow_ops: flow_ops_summary,
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu: if tex_flow.backend_name() == "wgpu" {
            coords_wgpu
        } else {
            None
        },
        #[cfg(feature = "runtime-model-wgpu")]
        features_wgpu,
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_shape_slat(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    sparse_layout: &[std::ops::Range<usize>],
    slat_override: Option<&SparseRowNoiseOverride>,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    capture_sampler_trace: bool,
    parity_strict: bool,
    #[cfg(feature = "runtime-model-wgpu")] coords_wgpu: Option<
        Tensor<SparseFlowWgpuBackend, 2, Int>,
    >,
    #[cfg(feature = "runtime-model")] shape_flow: Option<&SparseStructureFlowRuntime>,
) -> Result<ShapeSLatSample, String> {
    let (_sampler, sample_cfg, sigma_min) =
        resolve_sampler_settings(sampler_config, sampler_override);
    const DENSE_CHANNELS: usize = 32;

    if let Some(override_rows) = slat_override {
        trellis_stage_log!(
            "burn_trellis: using shape_slat hook slat override rows={} (strict={})",
            override_rows.coords.len(),
            parity_strict
        );
        let (dense_resolution, dense_noise) = if let Some(values) = noise_dense_override {
            if !values.len().is_multiple_of(DENSE_CHANNELS) {
                if parity_strict {
                    return Err(format!(
                        "strict mode requires shape_slat dense noise len divisible by {DENSE_CHANNELS}; got {}",
                        values.len()
                    ));
                }
                (0usize, None)
            } else {
                let voxel_count = values.len() / DENSE_CHANNELS;
                let dense_resolution = (voxel_count as f64).cbrt().round() as usize;
                if dense_resolution == 0
                    || dense_resolution
                        .saturating_mul(dense_resolution)
                        .saturating_mul(dense_resolution)
                        != voxel_count
                {
                    if parity_strict {
                        return Err(format!(
                            "strict mode requires cubic shape_slat dense noise (channels={DENSE_CHANNELS}, len={})",
                            values.len()
                        ));
                    }
                    (0usize, None)
                } else {
                    (
                        dense_resolution,
                        capture_sampler_trace.then_some(values.to_vec()),
                    )
                }
            }
        } else {
            (0usize, None)
        };
        let override_noise_map = noise_override.map(sparse_row_noise_map);
        let noise_rows = override_rows
            .coords
            .iter()
            .map(|coord| {
                override_noise_map
                    .as_ref()
                    .and_then(|map| map.get(&pack_coord(coord[1], coord[2], coord[3])))
                    .copied()
                    .unwrap_or([0.0; 32])
            })
            .collect::<Vec<_>>();
        let features = override_rows.feats.clone();
        let override_layout = sparse_layout_from_coords(override_rows.coords.as_slice())?;
        return Ok(ShapeSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution,
            dense_channels: if dense_noise.is_some() {
                DENSE_CHANNELS
            } else {
                0
            },
            dense_noise,
            features: features.clone(),
            noise: noise_rows,
            step_0_pred_v: features.clone(),
            step_0_pred_v_pos: features.clone(),
            step_0_pred_v_neg: features.clone(),
            step_0_x_t: features.clone(),
            step_mid_x_t: features.clone(),
            step_last_x_t: features,
            coords: override_rows.coords.clone(),
            layout: override_layout,
            flow_ops: SparseFlowOpTimingSummary::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu: None,
            #[cfg(feature = "runtime-model-wgpu")]
            features_wgpu: None,
        });
    }

    #[cfg(feature = "runtime-model")]
    {
        let shape_flow = shape_flow.ok_or_else(|| {
            "burn_trellis: shape_slat runtime-model path requires shape flow runtime".to_string()
        })?;
        return sample_shape_slat_with_model(
            preprocess,
            coords,
            sparse_layout,
            rng,
            noise_override,
            noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu,
            shape_flow,
        );
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (
            preprocess,
            coords,
            sparse_layout,
            rng,
            noise_override,
            noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            parity_strict,
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu,
        );
        Err("burn_trellis: shape_slat stage requires `runtime-model` feature".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_tex_slat(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    slat_override: Option<&SparseRowNoiseOverride>,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    capture_sampler_trace: bool,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] tex_flow: Option<&SparseStructureFlowRuntime>,
) -> Result<TexSLatSample, String> {
    let (_sampler, sample_cfg, sigma_min) =
        resolve_sampler_settings(sampler_config, sampler_override);
    const DENSE_CHANNELS: usize = 32;

    if let Some(override_rows) = slat_override {
        trellis_stage_log!(
            "burn_trellis: using tex_slat hook slat override rows={} (strict={})",
            override_rows.coords.len(),
            parity_strict
        );
        let (dense_resolution, dense_noise) = if let Some(values) = noise_dense_override {
            if !values.len().is_multiple_of(DENSE_CHANNELS) {
                if parity_strict {
                    return Err(format!(
                        "strict mode requires tex_slat dense noise len divisible by {DENSE_CHANNELS}; got {}",
                        values.len()
                    ));
                }
                (0usize, None)
            } else {
                let voxel_count = values.len() / DENSE_CHANNELS;
                let dense_resolution = (voxel_count as f64).cbrt().round() as usize;
                if dense_resolution == 0
                    || dense_resolution
                        .saturating_mul(dense_resolution)
                        .saturating_mul(dense_resolution)
                        != voxel_count
                {
                    if parity_strict {
                        return Err(format!(
                            "strict mode requires cubic tex_slat dense noise (channels={DENSE_CHANNELS}, len={})",
                            values.len()
                        ));
                    }
                    (0usize, None)
                } else {
                    (
                        dense_resolution,
                        capture_sampler_trace.then_some(values.to_vec()),
                    )
                }
            }
        } else {
            (0usize, None)
        };
        let override_noise_map = noise_override.map(sparse_row_noise_map);
        let noise_rows = override_rows
            .coords
            .iter()
            .map(|coord| {
                override_noise_map
                    .as_ref()
                    .and_then(|map| map.get(&pack_coord(coord[1], coord[2], coord[3])))
                    .copied()
                    .unwrap_or([0.0; 32])
            })
            .collect::<Vec<_>>();

        let mut shape_cond_map = HashMap::with_capacity(shape_slat.coords.len());
        for (idx, coord) in shape_slat.coords.iter().enumerate() {
            let shape_hint = shape_slat.features[idx];
            let mut shape_cond = [0.0f32; 32];
            for ch in 0..32 {
                let mean = shape_normalization.mean.get(ch).copied().unwrap_or(0.0);
                let std = shape_normalization
                    .std
                    .get(ch)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1.0e-6);
                shape_cond[ch] = (shape_hint[ch] - mean) / std;
            }
            shape_cond_map.insert(pack_coord(coord[1], coord[2], coord[3]), shape_cond);
        }
        let shape_slat_cond = override_rows
            .coords
            .iter()
            .map(|coord| {
                shape_cond_map
                    .get(&pack_coord(coord[1], coord[2], coord[3]))
                    .copied()
                    .unwrap_or([0.0; 32])
            })
            .collect::<Vec<_>>();
        let features = override_rows.feats.clone();
        let override_layout = sparse_layout_from_coords(override_rows.coords.as_slice())?;
        return Ok(TexSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution,
            dense_channels: if dense_noise.is_some() {
                DENSE_CHANNELS
            } else {
                0
            },
            dense_noise,
            features: features.clone(),
            noise: noise_rows,
            step_0_pred_v: features.clone(),
            step_0_pred_v_pos: features.clone(),
            step_0_pred_v_neg: features.clone(),
            step_0_x_t: features.clone(),
            step_mid_x_t: features.clone(),
            step_last_x_t: features,
            shape_slat_cond,
            coords: override_rows.coords.clone(),
            layout: override_layout,
            flow_ops: SparseFlowOpTimingSummary::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu: None,
            #[cfg(feature = "runtime-model-wgpu")]
            features_wgpu: None,
        });
    }

    #[cfg(feature = "runtime-model")]
    {
        let tex_flow = tex_flow.ok_or_else(|| {
            "burn_trellis: tex_slat runtime-model path requires tex flow runtime".to_string()
        })?;
        return sample_tex_slat_with_model(
            preprocess,
            shape_slat,
            rng,
            noise_override,
            noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            shape_normalization,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            #[cfg(feature = "runtime-model-wgpu")]
            shape_slat.coords_wgpu.clone(),
            tex_flow,
        );
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (
            preprocess,
            shape_slat,
            rng,
            noise_override,
            noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            shape_normalization,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            parity_strict,
        );
        Err("burn_trellis: tex_slat stage requires `runtime-model` feature".to_string())
    }
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_shape_slat_cascade_runtime(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    sparse_layout: &[std::ops::Range<usize>],
    #[cfg(feature = "runtime-model-wgpu")] coords_wgpu: Option<
        Tensor<SparseFlowWgpuBackend, 2, Int>,
    >,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    normalization: &TrellisNormalization,
    lr_sparse_resolution: usize,
    lr_resolution: usize,
    target_resolution: usize,
    max_num_tokens: usize,
    capture_sampler_trace: bool,
    parity_strict: bool,
    shape_flow_512: Option<&SparseStructureFlowRuntime>,
    shape_flow_1024: Option<&SparseStructureFlowRuntime>,
    shape_decoder: Option<&FdgDecoderRuntime>,
) -> Result<(ShapeSLatSample, Option<ShapeSLatSample>, usize, usize), String> {
    let shape_flow_512 = shape_flow_512.ok_or_else(|| {
        "burn_trellis: cascade pipeline requires shape_slat_flow_model_512 runtime".to_string()
    })?;
    let shape_flow_1024 = shape_flow_1024.ok_or_else(|| {
        "burn_trellis: cascade pipeline requires shape_slat_flow_model_1024 runtime".to_string()
    })?;
    let shape_decoder = shape_decoder.ok_or_else(|| {
        "burn_trellis: cascade pipeline requires shape decoder runtime for coordinate upsample"
            .to_string()
    })?;
    if lr_resolution == 0 {
        return Err("burn_trellis: cascade lr_resolution must be > 0".to_string());
    }
    if target_resolution == 0 || !target_resolution.is_multiple_of(16) {
        return Err(format!(
            "burn_trellis: cascade target resolution must be a positive multiple of 16, got {}",
            target_resolution
        ));
    }
    if max_num_tokens == 0 {
        return Err("burn_trellis: cascade max_num_tokens must be > 0".to_string());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    let lr_sparse_rows = coords_wgpu
        .as_ref()
        .map(|coords_t| coords_t.dims()[0])
        .unwrap_or(coords.len());
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let lr_sparse_rows = coords.len();
    let explicit_lr_noise_override = require_sparse_row_noise_override_rows(
        cond_overrides.and_then(|overrides| overrides.shape_noise_lr.as_ref()),
        lr_sparse_rows,
        "shape_slat_lr_runtime",
    )?;
    let lr_noise_override = explicit_lr_noise_override.or_else(|| {
        optional_sparse_row_noise_override_for_rows(
            noise_override,
            lr_sparse_rows,
            "shape_slat_lr_runtime",
        )
    });

    let shape_lr = sample_shape_slat(
        preprocess,
        coords,
        sparse_layout,
        None,
        rng,
        lr_noise_override,
        noise_dense_override,
        cond_overrides,
        sampler_config,
        sampler_override,
        normalization,
        lr_sparse_resolution,
        capture_sampler_trace,
        parity_strict,
        #[cfg(feature = "runtime-model-wgpu")]
        coords_wgpu,
        Some(shape_flow_512),
    )?;

    #[cfg(feature = "runtime-model-wgpu")]
    let hr_coords_sparse = if shape_flow_1024.backend_name() == "wgpu" {
        let shape_coords_t = shape_lr.coords_wgpu.as_ref().ok_or_else(|| {
            "burn_trellis: cascade canonical wgpu path requires device coords from low-resolution shape stage"
                .to_string()
        })?;
        let shape_rows_t = shape_lr.features_wgpu.as_ref().ok_or_else(|| {
            "burn_trellis: cascade canonical wgpu path requires device shape row tensor from low-resolution shape stage"
                .to_string()
        })?;
        shape_decoder.upsample_coords_result_with_tensors(shape_coords_t.clone(), shape_rows_t.clone(), 4)?
    } else if let Some(shape_coords_t) = shape_lr.coords_wgpu.as_ref() {
        let shape_rows_t = shape_lr.features_wgpu.as_ref().ok_or_else(|| {
            "burn_trellis: cascade tensor-coord upsample path requires device shape rows; host completion fallback is disabled"
                .to_string()
        })?;
        shape_decoder.upsample_coords_result_with_tensors(
            shape_coords_t.clone(),
            shape_rows_t.clone(),
            4,
        )?
    } else {
        shape_decoder.upsample_coords_result(shape_lr.coords.as_slice(), shape_lr.features.as_slice(), 4)?
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let hr_coords_sparse =
        shape_decoder.upsample_coords_result(shape_lr.coords.as_slice(), shape_lr.features.as_slice(), 4)?;
    if hr_coords_sparse.rows() == 0 {
        return Err("burn_trellis: cascade decoder upsample produced zero coordinates".to_string());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    if shape_flow_1024.backend_name() == "wgpu" {
        let hr_coords_t = hr_coords_sparse.coords_tensor().ok_or_else(|| {
            "burn_trellis: cascade decoder upsample returned host-only coords on canonical wgpu path"
                .to_string()
        })?;
        let mut effective_resolution = target_resolution;
        let hr_coords_quantized_t = loop {
            let sparse_resolution = (effective_resolution / 16).max(1);
            let quantized_t =
                quantize_cascade_coords_wgpu(hr_coords_t.clone(), lr_resolution, sparse_resolution)?;
            let [num_tokens, coord_cols] = quantized_t.dims();
            if coord_cols != 4 {
                return Err(format!(
                    "burn_trellis: cascade quantized coord tensor must have 4 columns, got {}",
                    coord_cols
                ));
            }
            if num_tokens == 0 {
                return Err(format!(
                    "burn_trellis: cascade quantization produced zero coordinates at resolution {}",
                    effective_resolution
                ));
            }
            if cascade_resolution_accepts_token_budget(
                num_tokens,
                max_num_tokens,
                effective_resolution,
            ) {
                if effective_resolution != target_resolution {
                    trellis_stage_log!(
                        "burn_trellis: cascade reducing decode resolution from {} to {} due to max_num_tokens={} (tokens={})",
                        target_resolution,
                        effective_resolution,
                        max_num_tokens,
                        num_tokens
                    );
                }
                break quantized_t;
            }
            effective_resolution = effective_resolution.saturating_sub(128).max(1024);
        };
        let effective_sparse_resolution = (effective_resolution / 16).max(1);
        let hr_noise_override = require_sparse_row_noise_override_rows(
            cond_overrides.and_then(|overrides| overrides.shape_noise_hr.as_ref()),
            hr_coords_quantized_t.dims()[0],
            "shape_slat_hr_runtime",
        )?
        .or_else(|| {
            optional_sparse_row_noise_override_for_rows(
                noise_override,
                hr_coords_quantized_t.dims()[0],
                "shape_slat_hr_runtime",
            )
        });
        let hr_coords_quantized_host =
            if noise_dense_override.is_some() || hr_noise_override.is_some() {
                coords_wgpu_tensor_to_host(
                    hr_coords_quantized_t.clone(),
                    "burn_trellis: cascade quantized coord materialization for dense noise",
                )?
            } else {
                Vec::new()
            };
        let hr_layout = if !hr_coords_quantized_host.is_empty() {
            sparse_layout_from_coords(hr_coords_quantized_host.as_slice())?
        } else {
            // Canonical runtime path is single-image/single-batch; avoid host
            // coord extraction for layout when no coordinate-indexed override
            // needs it, keeping cascade handoff fully device-resident.
            vec![0..hr_coords_quantized_t.dims()[0]]
        };
        let shape_hr = sample_shape_slat(
            preprocess,
            hr_coords_quantized_host.as_slice(),
            hr_layout.as_slice(),
            None,
            rng,
            hr_noise_override,
            noise_dense_override,
            cond_overrides,
            sampler_config,
            sampler_override,
            normalization,
            effective_sparse_resolution,
            capture_sampler_trace,
            parity_strict,
            Some(hr_coords_quantized_t),
            Some(shape_flow_1024),
        )?;
        return Ok((
            shape_hr,
            Some(shape_lr),
            effective_sparse_resolution,
            effective_resolution,
        ));
    }

    let hr_coords = hr_coords_sparse
        .coords_host("burn_trellis: cascade decoder upsample coord materialization")?;

    let mut effective_resolution = target_resolution;
    let hr_coords_quantized = loop {
        let sparse_resolution = (effective_resolution / 16).max(1);
        let quantized =
            quantize_cascade_coords(hr_coords.as_slice(), lr_resolution, sparse_resolution)?;
        if quantized.is_empty() {
            return Err(format!(
                "burn_trellis: cascade quantization produced zero coordinates at resolution {}",
                effective_resolution
            ));
        }
        let num_tokens = quantized.len();
        if cascade_resolution_accepts_token_budget(num_tokens, max_num_tokens, effective_resolution)
        {
            if effective_resolution != target_resolution {
                trellis_stage_log!(
                    "burn_trellis: cascade reducing decode resolution from {} to {} due to max_num_tokens={} (tokens={})",
                    target_resolution,
                    effective_resolution,
                    max_num_tokens,
                    num_tokens
                );
            }
            break quantized;
        }
        effective_resolution = effective_resolution.saturating_sub(128).max(1024);
    };
    let effective_sparse_resolution = (effective_resolution / 16).max(1);
    let hr_layout = sparse_layout_from_coords(hr_coords_quantized.as_slice())?;
    let hr_noise_override = require_sparse_row_noise_override_rows(
        cond_overrides.and_then(|overrides| overrides.shape_noise_hr.as_ref()),
        hr_coords_quantized.len(),
        "shape_slat_hr_runtime",
    )?
    .or_else(|| {
        optional_sparse_row_noise_override_for_rows(
            noise_override,
            hr_coords_quantized.len(),
            "shape_slat_hr_runtime",
        )
    });
    let shape_hr = sample_shape_slat(
        preprocess,
        hr_coords_quantized.as_slice(),
        hr_layout.as_slice(),
        None,
        rng,
        hr_noise_override,
        noise_dense_override,
        cond_overrides,
        sampler_config,
        sampler_override,
        normalization,
        effective_sparse_resolution,
        capture_sampler_trace,
        parity_strict,
        #[cfg(feature = "runtime-model-wgpu")]
        None,
        Some(shape_flow_1024),
    )?;
    Ok((
        shape_hr,
        Some(shape_lr),
        effective_sparse_resolution,
        effective_resolution,
    ))
}

#[cfg(feature = "runtime-model")]
fn cascade_resolution_accepts_token_budget(
    num_tokens: usize,
    max_num_tokens: usize,
    effective_resolution: usize,
) -> bool {
    num_tokens <= max_num_tokens || effective_resolution == 1024
}

#[cfg(feature = "runtime-model")]
fn quantize_cascade_coords(
    hr_coords: &[[u32; 4]],
    lr_resolution: usize,
    target_sparse_resolution: usize,
) -> Result<Vec<[u32; 4]>, String> {
    if lr_resolution == 0 || target_sparse_resolution == 0 {
        return Err(format!(
            "burn_trellis: cascade quantization requires lr_resolution>0 and target_sparse_resolution>0 (got lr={}, target={})",
            lr_resolution, target_sparse_resolution
        ));
    }

    let mut quantized = Vec::with_capacity(hr_coords.len());
    let src = lr_resolution as f32;
    let dst = target_sparse_resolution as f32;
    for coord in hr_coords.iter().copied() {
        let quant_axis = |value: u32| -> u32 {
            // TRELLIS.2: ((hr_coord + 0.5) / lr_resolution * (hr_resolution // 16)).int()
            let scaled = (((value as f32) + 0.5) / src * dst) as i64;
            scaled.clamp(0, target_sparse_resolution.saturating_sub(1) as i64) as u32
        };
        quantized.push([
            coord[0],
            quant_axis(coord[1]),
            quant_axis(coord[2]),
            quant_axis(coord[3]),
        ]);
    }
    quantized.sort_unstable();
    quantized.dedup();
    Ok(quantized)
}

#[cfg(feature = "runtime-model")]
fn build_shape_cond_rows_host(
    shape_rows: &[[f32; 32]],
    normalization: &TrellisNormalization,
) -> Vec<[f32; 32]> {
    let mut out = Vec::with_capacity(shape_rows.len());
    for shape_feat in shape_rows.iter().copied() {
        let mut shape_cond = [0.0f32; 32];
        for ch in 0..32 {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            shape_cond[ch] = (shape_feat[ch] - mean) / std;
        }
        out.push(shape_cond);
    }
    out
}

#[cfg(feature = "runtime-model")]
fn build_shape_concat_rows_host(
    shape_rows: &[[f32; 32]],
    concat_channels: usize,
    normalization: &TrellisNormalization,
) -> Vec<f32> {
    let mut concat_rows = vec![0.0f32; shape_rows.len().saturating_mul(concat_channels)];
    for (row_idx, shape_feat) in shape_rows.iter().copied().enumerate() {
        let row_base = row_idx.saturating_mul(concat_channels);
        for ch in 0..concat_channels.min(32) {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            concat_rows[row_base + ch] = (shape_feat[ch] - mean) / std;
        }
    }
    concat_rows
}

#[cfg(feature = "runtime-model-wgpu")]
fn denormalize_and_pad_trace_rows_wgpu(
    rows_t: Tensor<SparseFlowWgpuBackend, 2>,
    normalization: &TrellisNormalization,
    context: &str,
) -> Result<Tensor<SparseFlowWgpuBackend, 2>, String> {
    let [rows, channels] = rows_t.dims();
    if channels == 0 {
        return Ok(Tensor::<SparseFlowWgpuBackend, 2>::zeros([rows, 32], &rows_t.device()));
    }
    if channels > 32 {
        return Err(format!(
            "{context}: sparse flow row tensor has {} channels; expected <= 32",
            channels
        ));
    }
    let device = rows_t.device();
    let mut mean = Vec::with_capacity(channels);
    let mut std = Vec::with_capacity(channels);
    for ch in 0..channels {
        mean.push(normalization.mean.get(ch).copied().unwrap_or(0.0));
        std.push(
            normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6),
        );
    }
    let mean_t = Tensor::<SparseFlowWgpuBackend, 1>::from_floats(mean.as_slice(), &device)
        .reshape([1, channels]);
    let std_t =
        Tensor::<SparseFlowWgpuBackend, 1>::from_floats(std.as_slice(), &device).reshape([1, channels]);
    let denorm = rows_t.mul(std_t).add(mean_t);
    if channels == 32 {
        return Ok(denorm);
    }
    let pad = Tensor::<SparseFlowWgpuBackend, 2>::zeros([rows, 32 - channels], &device);
    Ok(Tensor::cat(vec![denorm, pad], 1))
}

#[cfg(feature = "runtime-model-wgpu")]
fn build_shape_concat_tensor_wgpu(
    shape_rows_t: Tensor<SparseFlowWgpuBackend, 2>,
    rows: usize,
    concat_channels: usize,
    normalization: &TrellisNormalization,
    context: &str,
) -> Result<Tensor<SparseFlowWgpuBackend, 2>, String> {
    let [shape_rows, shape_cols] = shape_rows_t.dims();
    if shape_cols != 32 {
        return Err(format!(
            "{context}: shape feature tensor must have 32 columns, got {}",
            shape_cols
        ));
    }
    if shape_rows < rows {
        return Err(format!(
            "{context}: shape feature tensor row mismatch: shape_rows={} expected_rows>={}",
            shape_rows, rows
        ));
    }
    let shape_rows_t = if shape_rows == rows {
        shape_rows_t
    } else {
        shape_rows_t.slice([0..rows, 0..32])
    };
    let used_channels = concat_channels.min(32);
    if used_channels == 0 {
        return Ok(Tensor::<SparseFlowWgpuBackend, 2>::zeros([rows, concat_channels], &shape_rows_t.device()));
    }
    let device = shape_rows_t.device();
    let shape_used_t = if used_channels == 32 {
        shape_rows_t
    } else {
        shape_rows_t.slice([0..rows, 0..used_channels])
    };
    let mut mean = Vec::with_capacity(used_channels);
    let mut std = Vec::with_capacity(used_channels);
    for ch in 0..used_channels {
        mean.push(normalization.mean.get(ch).copied().unwrap_or(0.0));
        std.push(
            normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6),
        );
    }
    let mean_t = Tensor::<SparseFlowWgpuBackend, 1>::from_floats(mean.as_slice(), &device)
        .reshape([1, used_channels]);
    let std_t = Tensor::<SparseFlowWgpuBackend, 1>::from_floats(std.as_slice(), &device)
        .reshape([1, used_channels]);
    let concat_used_t = shape_used_t.sub(mean_t).div(std_t);
    if used_channels == concat_channels {
        return Ok(concat_used_t);
    }
    let pad = Tensor::<SparseFlowWgpuBackend, 2>::zeros(
        [rows, concat_channels.saturating_sub(used_channels)],
        &device,
    );
    Ok(Tensor::cat(vec![concat_used_t, pad], 1))
}

#[cfg(feature = "runtime-model-wgpu")]
fn coords_u32_to_wgpu_tensor(
    coords: &[[u32; 4]],
) -> Result<Tensor<SparseFlowWgpuBackend, 2, Int>, String> {
    let device: <SparseFlowWgpuBackend as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let mut flat = Vec::with_capacity(coords.len().saturating_mul(4));
    for (row_idx, coord) in coords.iter().enumerate() {
        for value in coord {
            let converted = i32::try_from(*value).map_err(|_| {
                format!(
                    "burn_trellis: sparse override coord conversion overflow at row {} value {}",
                    row_idx, value
                )
            })?;
            flat.push(converted);
        }
    }
    Ok(Tensor::<SparseFlowWgpuBackend, 1, Int>::from_data(
        TensorData::new(flat, [coords.len().saturating_mul(4)]),
        &device,
    )
    .reshape([coords.len(), 4]))
}

#[cfg(feature = "runtime-model-wgpu")]
fn coords_wgpu_tensor_to_host(
    coords_t: Tensor<SparseFlowWgpuBackend, 2, Int>,
    context: &str,
) -> Result<Vec<[u32; 4]>, String> {
    let [rows, cols] = coords_t.dims();
    if cols != 4 {
        return Err(format!("{context}: coord tensor must have 4 columns, got {cols}"));
    }
    let values = tensor_i32_to_vec(coords_t, context)?;
    if values.len() != rows.saturating_mul(4) {
        return Err(format!(
            "{context}: coord tensor length mismatch: got={} expected={}",
            values.len(),
            rows.saturating_mul(4)
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let base = row_idx.saturating_mul(4);
        let to_u32 = |value: i32| -> Result<u32, String> {
            u32::try_from(value)
                .map_err(|_| format!("{context}: negative coordinate value {value} at row {row_idx}"))
        };
        out.push([
            to_u32(values[base])?,
            to_u32(values[base + 1])?,
            to_u32(values[base + 2])?,
            to_u32(values[base + 3])?,
        ]);
    }
    Ok(out)
}

#[cfg(feature = "runtime-model-wgpu")]
fn quantize_cascade_coords_wgpu(
    hr_coords_t: Tensor<SparseFlowWgpuBackend, 2, Int>,
    lr_resolution: usize,
    target_sparse_resolution: usize,
) -> Result<Tensor<SparseFlowWgpuBackend, 2, Int>, String> {
    if lr_resolution == 0 || target_sparse_resolution == 0 {
        return Err(format!(
            "burn_trellis: cascade quantization requires lr_resolution>0 and target_sparse_resolution>0 (got lr={}, target={})",
            lr_resolution, target_sparse_resolution
        ));
    }
    let [rows, coord_cols] = hr_coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "burn_trellis: cascade upsample coord tensor must have 4 columns, got {}",
            coord_cols
        ));
    }
    let device = hr_coords_t.device();
    if rows == 0 {
        return Ok(Tensor::<SparseFlowWgpuBackend, 2, Int>::zeros([0, 4], &device));
    }
    let scale = i32::try_from(target_sparse_resolution).map_err(|_| {
        format!(
            "burn_trellis: cascade target sparse resolution {} exceeds i32 range",
            target_sparse_resolution
        )
    })?;
    if scale <= 0 {
        return Err(format!(
            "burn_trellis: cascade target sparse resolution must be > 0, got {}",
            target_sparse_resolution
        ));
    }
    let max_axis = i32::try_from(target_sparse_resolution.saturating_sub(1)).map_err(|_| {
        format!(
            "burn_trellis: cascade target sparse resolution {} exceeds i32 range",
            target_sparse_resolution
        )
    })?;
    let batch_col = hr_coords_t.clone().slice([0..rows, 0..1]);
    let xyz_q = hr_coords_t
        .slice([0..rows, 1..4])
        .float()
        .add_scalar(0.5)
        .div_scalar(lr_resolution as f32)
        .mul_scalar(target_sparse_resolution as f32)
        .int()
        .clamp(0i32, max_axis);
    let quantized_t = Tensor::cat(vec![batch_col, xyz_q], 1);

    // Canonicalize ordering and dedup in-tensor, matching host sort+dedup semantics.
    let batch = quantized_t.clone().slice([0..rows, 0..1]).squeeze_dim(1);
    let x = quantized_t.clone().slice([0..rows, 1..2]).squeeze_dim(1);
    let y = quantized_t.clone().slice([0..rows, 2..3]).squeeze_dim(1);
    let z = quantized_t.clone().slice([0..rows, 3..4]).squeeze_dim(1);
    let key_t = batch
        .mul_scalar(scale)
        .add(x)
        .mul_scalar(scale)
        .add(y)
        .mul_scalar(scale)
        .add(z);
    let (sorted_keys, sorted_idx) = key_t.sort_with_indices(0);
    let sorted_coords = quantized_t.select(0, sorted_idx);
    if rows == 1 {
        return Ok(sorted_coords);
    }
    let first_prev_key = sorted_keys.clone().slice([0..1]).sub_scalar(1);
    let shifted_prev_keys = sorted_keys.clone().slice([0..rows - 1]);
    let prev_keys = Tensor::cat(vec![first_prev_key, shifted_prev_keys], 0);
    let keep_mask = sorted_keys.not_equal(prev_keys);
    let keep_idx_rows = keep_mask.argwhere();
    let [keep_rows, keep_cols] = keep_idx_rows.dims();
    if keep_cols != 1 {
        return Err(format!(
            "burn_trellis: cascade keep-index tensor must have 1 column, got {}",
            keep_cols
        ));
    }
    if keep_rows == 0 {
        return Ok(Tensor::<SparseFlowWgpuBackend, 2, Int>::zeros([0, 4], &device));
    }
    let idx_col = Tensor::<SparseFlowWgpuBackend, 1, Int>::from_ints([0], &device);
    let keep_idx = keep_idx_rows.select(1, idx_col).squeeze_dim(1);
    Ok(sorted_coords.select(0, keep_idx))
}
