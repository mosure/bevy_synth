fn linear_forward(
    input: &[f32],
    rows: usize,
    layer: &LinearLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    let op_start = Instant::now();
    let result = (|| {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let expected = rows
            .checked_mul(layer.in_channels)
            .ok_or_else(|| format!("{context}: input size overflow"))?;
        if input.len() != expected {
            return Err(format!(
                "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
                input.len(),
                expected,
                rows,
                layer.in_channels
            ));
        }
        if layer.bias.len() != layer.out_channels {
            return Err(format!(
                "{context}: bias len {} does not match out_channels {}",
                layer.bias.len(),
                layer.out_channels
            ));
        }
        let mut output = vec![0.0f32; rows * layer.out_channels];
        for row_idx in 0..rows {
            let base = row_idx * layer.out_channels;
            output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
        }
        unsafe {
            matrixmultiply::sgemm(
                rows,
                layer.in_channels,
                layer.out_channels,
                1.0,
                input.as_ptr(),
                layer.in_channels as isize,
                1,
                layer.weight.as_ptr(),
                1,
                layer.in_channels as isize,
                1.0,
                output.as_mut_ptr(),
                layer.out_channels as isize,
                1,
            );
        }
        Ok(output)
    })();
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn linear_forward_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    layer: &LinearLayer,
    context: &str,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let op_start = Instant::now();
    let result = (|| {
        let [rows, in_channels] = input.dims();
        if in_channels != layer.in_channels {
            return Err(format!(
                "{context}: invalid input channels {}, expected {}",
                in_channels, layer.in_channels
            ));
        }
        if layer.bias.len() != layer.out_channels {
            return Err(format!(
                "{context}: bias len {} does not match out_channels {}",
                layer.bias.len(),
                layer.out_channels
            ));
        }
        if rows == 0 {
            return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
                [0, layer.out_channels],
                &context_gpu.device,
            ));
        }
        // WGPU matmul for very skinny outputs (<=8 channels) and large row counts
        // can exhibit pathological compile/dispatch latency. Keep this path on a
        // dedicated single-dispatch kernel to avoid the multi-pass reduction
        // overhead of generic tensor-op formulations in decode hotspots.
        if layer.out_channels <= 8 && rows >= 32_768 {
            if decoder_conv_debug_enabled() {
                eprintln!(
                    "burn_trellis: using skinny linear path '{}' rows={} in_channels={} out_channels={}",
                    context, rows, in_channels, layer.out_channels
                );
            }
            let weight_t = context_gpu.linear_weight_tensor(layer);
            let bias_t = context_gpu.linear_bias_tensor(layer);
            let output = linear_skinny_forward_wgpu(input.clone(), weight_t, bias_t)
                .map_err(|err| format!("{context}: skinny linear kernel path failed: {err}"))?;
            decoder_wgpu_linear_parity_check(input, output.clone(), layer, context)?;
            return Ok(output);
        }

        let bytes_per_row = layer
            .out_channels
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{context}: output row byte size overflow"))?;
        let output_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or_else(|| format!("{context}: output byte size overflow"))?;
        let max_tensor_bytes = decoder_wgpu_max_tensor_bytes();
        if output_bytes > max_tensor_bytes {
            return Err(format!(
                "{context}: output exceeds wgpu tensor limit (bytes={} max_tensor_bytes={})",
                output_bytes, max_tensor_bytes
            ));
        }
        let max_output_bytes = decoder_wgpu_max_output_bytes();
        // Large-row matmul dispatches can become pathological on WGPU in this
        // decoder path; proactively chunk very large row counts.
        let linear_chunk_rows_cap = 16_384usize;

        let weight_t = context_gpu.linear_weight_tensor(layer).swap_dims(0, 1);
        let bias_t = context_gpu
            .linear_bias_tensor(layer)
            .reshape([1, layer.out_channels]);

        if output_bytes <= max_output_bytes && rows <= linear_chunk_rows_cap {
            let output = input.clone().matmul(weight_t).add(bias_t);
            decoder_wgpu_linear_parity_check(input, output.clone(), layer, context)?;
            return Ok(output);
        }

        // Historical guardrail kept wide decoder MLPs at 4k-row chunks, which
        // over-fragmented high-row decode matmuls into many small dispatches.
        // Keep a bounded cap for very wide outputs, but allow materially larger
        // chunks to reduce launch/sync overhead on canonical WGPU decode.
        let force_chunk_rows_cap = if layer.out_channels >= 1024 && rows >= 32_768 {
            8_192
        } else {
            linear_chunk_rows_cap
        };
        let chunk_rows = decoder_wgpu_chunk_rows(rows, bytes_per_row, max_output_bytes)
            .min(linear_chunk_rows_cap)
            .min(force_chunk_rows_cap)
            .max(1);
        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: chunking wgpu linear '{}' rows={} chunk_rows={} out_channels={} output_bytes={} max_output_bytes={}",
                context, rows, chunk_rows, layer.out_channels, output_bytes, max_output_bytes
            );
        }
        // Avoid `Tensor::cat(chunks, 0)` here: cat needs a full destination tensor while
        // keeping all chunk outputs alive, which can double peak memory and OOM on large
        // decode MLP shapes. Preallocate once and slice-assign each chunk directly.
        let mut output = Tensor::<DefaultWgpuBackend, 2>::zeros(
            [rows, layer.out_channels],
            &context_gpu.device,
        );
        let total_chunks = rows.div_ceil(chunk_rows);
        let mut chunk_idx = 0usize;
        let mut start = 0usize;
        while start < rows {
            let end = (start + chunk_rows).min(rows);
            chunk_idx += 1;
            let chunk_start = Instant::now();
            let input_chunk = input.clone().slice([start..end, 0..in_channels]);
            let output_chunk = input_chunk.matmul(weight_t.clone()).add(bias_t.clone());
            if decoder_conv_debug_enabled() {
                let chunk_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "burn_trellis: wgpu linear '{}' chunk {}/{} rows=[{}..{}) elapsed_ms={:.2}",
                    context, chunk_idx, total_chunks, start, end, chunk_ms
                );
            }
            output = output.slice_assign([start..end, 0..layer.out_channels], output_chunk);
            start = end;
        }
        decoder_wgpu_linear_parity_check(input, output.clone(), layer, context)?;
        Ok(output)
    })();
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn layer_norm_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    context: &str,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let op_start = Instant::now();
    let result = (|| {
        let [rows, channels] = input.dims();
        if rows == 0 || channels == 0 {
            return Ok(input);
        }
        if let Some(weight) = weight
            && weight.len() != channels
        {
            return Err(format!(
                "layer_norm_wgpu: invalid weight len {}, expected {}",
                weight.len(),
                channels
            ));
        }
        if let Some(bias) = bias
            && bias.len() != channels
        {
            return Err(format!(
                "layer_norm_wgpu: invalid bias len {}, expected {}",
                bias.len(),
                channels
            ));
        }
        let weight_t = if let Some(weight) = weight {
            context_gpu.vector_tensor(weight)
        } else {
            Tensor::<DefaultWgpuBackend, 1>::ones([channels], &context_gpu.device)
        };
        let bias_t = if let Some(bias) = bias {
            context_gpu.vector_tensor(bias)
        } else {
            Tensor::<DefaultWgpuBackend, 1>::zeros([channels], &context_gpu.device)
        };
        layer_norm_affine_forward_wgpu(input, weight_t, bias_t, eps)
            .map_err(|err| format!("{context}: layer norm affine kernel path failed: {err}"))
    })();
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn layer_norm_silu_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    context: &str,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let op_start = Instant::now();
    let result = (|| {
        let [rows, channels] = input.dims();
        if rows == 0 || channels == 0 {
            return Ok(input);
        }
        if let Some(weight) = weight
            && weight.len() != channels
        {
            return Err(format!(
                "layer_norm_silu_wgpu: invalid weight len {}, expected {}",
                weight.len(),
                channels
            ));
        }
        if let Some(bias) = bias
            && bias.len() != channels
        {
            return Err(format!(
                "layer_norm_silu_wgpu: invalid bias len {}, expected {}",
                bias.len(),
                channels
            ));
        }
        let weight_t = if let Some(weight) = weight {
            context_gpu.vector_tensor(weight)
        } else {
            Tensor::<DefaultWgpuBackend, 1>::ones([channels], &context_gpu.device)
        };
        let bias_t = if let Some(bias) = bias {
            context_gpu.vector_tensor(bias)
        } else {
            Tensor::<DefaultWgpuBackend, 1>::zeros([channels], &context_gpu.device)
        };
        layer_norm_affine_silu_forward_wgpu(input, weight_t, bias_t, eps)
            .map_err(|err| format!("{context}: layer norm silu kernel path failed: {err}"))
    })();
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn layer_norm_silu_wgpu_with_fp16_fences(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    compute_fp16: bool,
    context: &str,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    if !compute_fp16 {
        return layer_norm_silu_wgpu(context_gpu, input, weight, bias, eps, context);
    }
    let mut output = layer_norm_wgpu(
        context_gpu,
        input,
        weight,
        bias,
        eps,
        format!("{context} layer_norm").as_str(),
    )?;
    output = quantize_f16_tensor_wgpu(output);
    output = silu_wgpu(output, format!("{context} silu").as_str());
    Ok(quantize_f16_tensor_wgpu(output))
}

#[cfg(feature = "runtime-model-wgpu")]
fn silu_wgpu(input: Tensor<DefaultWgpuBackend, 2>, context: &str) -> Tensor<DefaultWgpuBackend, 2> {
    let op_start = Instant::now();
    let output = input.clone().mul(sigmoid(input));
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    output
}

#[cfg(feature = "runtime-model-wgpu")]
fn tensor_bytes_f32(rows: usize, channels: usize, context: &str) -> Result<usize, String> {
    rows.checked_mul(channels)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!("{context}: tensor byte size overflow (rows={rows}, channels={channels})")
        })
}

#[cfg(feature = "runtime-model-wgpu")]
fn convnext_block_mlp_forward_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input_t: Tensor<DefaultWgpuBackend, 2>,
    residual_t: Tensor<DefaultWgpuBackend, 2>,
    block: &ConvNeXtBlock,
    stage_idx: usize,
    block_idx: usize,
    compute_fp16: bool,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let [rows, channels] = input_t.dims();
    if rows == 0 {
        return Ok(input_t);
    }
    if channels != block.mlp_0.in_channels {
        return Err(format!(
            "decoder stage {stage_idx} block {block_idx} mlp_0 input mismatch: channels={} expected={}",
            channels, block.mlp_0.in_channels
        ));
    }
    if block.mlp_0.out_channels != block.mlp_2.in_channels {
        return Err(format!(
            "decoder stage {stage_idx} block {block_idx} mlp hidden mismatch: mlp_0.out_channels={} mlp_2.in_channels={}",
            block.mlp_0.out_channels, block.mlp_2.in_channels
        ));
    }
    let [residual_rows, residual_channels] = residual_t.dims();
    if residual_rows != rows || residual_channels != block.mlp_2.out_channels {
        return Err(format!(
            "decoder stage {stage_idx} block {block_idx} residual mismatch: residual=[{},{}] expected=[{},{}]",
            residual_rows, residual_channels, rows, block.mlp_2.out_channels
        ));
    }

    let max_output_bytes = decoder_wgpu_max_output_bytes();
    let hidden_bytes = tensor_bytes_f32(
        rows,
        block.mlp_0.out_channels,
        format!("decoder stage {stage_idx} block {block_idx} mlp_0").as_str(),
    )?;

    let hidden_bytes_per_row = block
        .mlp_0
        .out_channels
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!("decoder stage {stage_idx} block {block_idx} hidden row bytes overflow")
        })?;
    let output_bytes_per_row = block
        .mlp_2
        .out_channels
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!("decoder stage {stage_idx} block {block_idx} output row bytes overflow")
        })?;
    // Keep wasm conservative, but avoid over-fragmenting native WGPU decoder
    // MLPs. The 512-base decoder stages top out at [32768, 512] hidden chunks
    // here, which stays well below the normal tensor/output guards while
    // cutting hundreds of small matmul/slice dispatches from decode.
    #[cfg(target_arch = "wasm32")]
    let mlp_chunk_rows_cap = 8_192usize;
    #[cfg(not(target_arch = "wasm32"))]
    let mlp_chunk_rows_cap = 32_768usize;
    let chunk_rows = decoder_wgpu_chunk_rows(rows, hidden_bytes_per_row, max_output_bytes)
        .min(decoder_wgpu_chunk_rows(
            rows,
            output_bytes_per_row,
            max_output_bytes,
        ))
        .min(mlp_chunk_rows_cap)
        .max(1);

    if decoder_conv_debug_enabled() {
        eprintln!(
            "burn_trellis: chunking wgpu convnext mlp stage={} block={} rows={} chunk_rows={} hidden_channels={} output_channels={} hidden_bytes={}",
            stage_idx,
            block_idx,
            rows,
            chunk_rows,
            block.mlp_0.out_channels,
            block.mlp_2.out_channels,
            hidden_bytes
        );
    }

    // Keep residual tensor as the destination buffer and slice-assign each MLP chunk.
    // This avoids materializing a second full [rows, channels] tensor for large blocks.
    let mut output_t = residual_t;
    let mut start = 0usize;
    while start < rows {
        let end = (start + chunk_rows).min(rows);
        let input_chunk = input_t.clone().slice([start..end, 0..channels]);
        let mut hidden_chunk = linear_forward_wgpu(
            context_gpu,
            input_chunk,
            &block.mlp_0,
            format!("stage {stage_idx} block {block_idx} mlp_0(wgpu_math chunk[{start}:{end}])")
                .as_str(),
        )?;
        if compute_fp16 {
            hidden_chunk = quantize_f16_tensor_wgpu(hidden_chunk);
        }
        let mut hidden_chunk = silu_wgpu(
            hidden_chunk,
            format!("stage {stage_idx} block {block_idx} silu(wgpu_math chunk[{start}:{end}])")
                .as_str(),
        );
        if compute_fp16 {
            hidden_chunk = quantize_f16_tensor_wgpu(hidden_chunk);
        }
        let mut output_chunk = linear_forward_wgpu(
            context_gpu,
            hidden_chunk,
            &block.mlp_2,
            format!("stage {stage_idx} block {block_idx} mlp_2(wgpu_math chunk[{start}:{end}])")
                .as_str(),
        )?;
        if compute_fp16 {
            output_chunk = quantize_f16_tensor_wgpu(output_chunk);
        }
        let residual_chunk = output_t
            .clone()
            .slice([start..end, 0..block.mlp_2.out_channels]);
        let mut combined_chunk = output_chunk.add(residual_chunk);
        if compute_fp16 {
            combined_chunk = quantize_f16_tensor_wgpu(combined_chunk);
        }
        output_t = output_t.slice_assign([start..end, 0..block.mlp_2.out_channels], combined_chunk);
        start = end;
    }

    Ok(output_t)
}

#[cfg(feature = "runtime-model-wgpu")]
fn tensor_to_vec_f32(
    tensor: Tensor<DefaultWgpuBackend, 2>,
    context: &str,
) -> Result<Vec<f32>, String> {
    let op_start = Instant::now();
    let result = crate::runtime_model::types::extraction::tensor_f32_to_vec(tensor, context);
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    if let Ok(values) = result.as_ref() {
        telemetry_record_readback(values.len());
    }
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn tensor_to_vec_i32(
    tensor: Tensor<DefaultWgpuBackend, 2, Int>,
    context: &str,
) -> Result<Vec<i32>, String> {
    let op_start = Instant::now();
    let result = crate::runtime_model::types::extraction::tensor_i32_to_vec(tensor, context);
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    if let Ok(values) = result.as_ref() {
        telemetry_record_readback(values.len());
    }
    result
}

#[cfg(feature = "runtime-model-wgpu")]
fn tensor_to_coords_u32(
    tensor: Tensor<DefaultWgpuBackend, 2, Int>,
    context: &str,
) -> Result<Vec<[u32; 4]>, String> {
    let [rows, cols] = tensor.dims();
    if cols != 4 {
        return Err(format!(
            "{context}: coord tensor must have 4 columns, got {cols}"
        ));
    }
    let flat = tensor_to_vec_i32(tensor, context)?;
    if flat.len() != rows.saturating_mul(4) {
        return Err(format!(
            "{context}: coord tensor length mismatch: len={} expected={}",
            flat.len(),
            rows.saturating_mul(4)
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let base = row_idx.saturating_mul(4);
        let to_u32 = |value: i32| -> Result<u32, String> {
            u32::try_from(value).map_err(|_| {
                format!("{context}: negative coordinate value {value} at row {row_idx}")
            })
        };
        out.push([
            to_u32(flat[base])?,
            to_u32(flat[base + 1])?,
            to_u32(flat[base + 2])?,
            to_u32(flat[base + 3])?,
        ]);
    }
    Ok(out)
}

#[cfg(feature = "runtime-model-wgpu")]
fn coords_tensor_from_u32_slice(
    coords: &[[u32; 4]],
    device: &WgpuDevice,
) -> Result<Tensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let mut flat = Vec::with_capacity(rows.saturating_mul(4));
    for (row_idx, coord) in coords.iter().enumerate() {
        for value in coord {
            let converted = i32::try_from(*value).map_err(|_| {
                format!(
                    "coord conversion overflow at row {} value {} for wgpu tensor path",
                    row_idx, value
                )
            })?;
            flat.push(converted);
        }
    }
    Ok(Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(flat, [rows.saturating_mul(4)]),
        device,
    )
    .reshape([rows, 4]))
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_conv_parity_context_matches(context: &str) -> bool {
    let Ok(needle) = std::env::var("TRELLIS2_DECODER_WGPU_CONV_PARITY_CONTEXT") else {
        return false;
    };
    let needle = needle.trim();
    !needle.is_empty() && (needle == "*" || context.contains(needle))
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_conv_parity_strict() -> bool {
    std::env::var("TRELLIS2_DECODER_WGPU_CONV_PARITY_STRICT")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_compare_flat_stats(actual: &[f32], expected: &[f32]) -> (f32, f32, f32) {
    let count = actual.len().min(expected.len());
    if count == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (lhs, rhs) in actual.iter().zip(expected.iter()).take(count) {
        let diff = (*lhs - *rhs).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
    }
    (
        (sum_abs / count as f64) as f32,
        max_abs,
        (sum_sq / count as f64).sqrt() as f32,
    )
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_neighbor_parity_check(
    config: &FlexConvConfig,
    coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
    context: &str,
) -> Result<(), String> {
    if !decoder_wgpu_conv_parity_context_matches(context) {
        return Ok(());
    }
    let coords = tensor_to_coords_u32(coords_t, format!("{context} parity coords").as_str())?;
    let actual_neighbors =
        tensor_to_vec_i32(neighbor_t, format!("{context} parity neighbors").as_str())?;
    let expected_neighbors = build_neighbor_rows(config, coords.as_slice())
        .map_err(|err| format!("{context}: cpu neighbor reference failed: {err}"))?;
    let mismatches = actual_neighbors
        .iter()
        .zip(expected_neighbors.iter())
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!(
        "burn_trellis: decoder_wgpu_neighbor_parity context=\"{}\" rows={} kernel_rows={} mismatches={} total={}",
        context,
        coords.len(),
        kernel_rows(config)?,
        mismatches,
        actual_neighbors.len().min(expected_neighbors.len())
    );
    if decoder_wgpu_conv_parity_strict() {
        assert_eq!(
            actual_neighbors, expected_neighbors,
            "decoder WGPU neighbor rows diverged for context '{}'",
            context
        );
    }
    Ok(())
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_sparse_conv_parity_check(
    config: &FlexConvConfig,
    input_t: Tensor<DefaultWgpuBackend, 2>,
    neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
    weight_t: Tensor<DefaultWgpuBackend, 5>,
    bias_t: Tensor<DefaultWgpuBackend, 1>,
    output_t: Tensor<DefaultWgpuBackend, 2>,
    context: &str,
) -> Result<(), String> {
    if !decoder_wgpu_conv_parity_context_matches(context) {
        return Ok(());
    }
    let input = tensor_to_vec_f32(input_t, format!("{context} parity input").as_str())?;
    let neighbor_rows =
        tensor_to_vec_i32(neighbor_t, format!("{context} parity neighbor rows").as_str())?;
    let weight = weight_t
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: failed weight tensor extraction: {err:?}"))?;
    let bias = bias_t
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: failed bias tensor extraction: {err:?}"))?;
    let actual = tensor_to_vec_f32(output_t, format!("{context} parity output").as_str())?;
    let expected = sparse_subm_conv_forward_flex_precomputed(
        config,
        SparseSubmConvWeights {
            weight: weight.as_slice(),
            bias: bias.as_slice(),
        },
        input.as_slice(),
        neighbor_rows.as_slice(),
        None,
    )
    .map_err(|err| format!("{context}: cpu sparse conv reference failed: {err}"))?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{context}: WGPU/reference sparse conv output length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let (mean_abs, max_abs, rmse) =
        decoder_wgpu_compare_flat_stats(actual.as_slice(), expected.as_slice());
    println!(
        "burn_trellis: decoder_wgpu_conv_parity context=\"{}\" rows={} in_channels={} out_channels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        context,
        actual.len() / config.out_channels.max(1),
        config.in_channels,
        config.out_channels,
        mean_abs,
        max_abs,
        rmse
    );
    if decoder_wgpu_conv_parity_strict() {
        assert!(
            mean_abs <= 1.0e-4 && max_abs <= 1.0e-3,
            "decoder WGPU sparse conv diverged for context '{}': mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            context,
            mean_abs,
            max_abs,
            rmse
        );
    }
    Ok(())
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_linear_parity_context_matches(context: &str) -> bool {
    let Ok(needle) = std::env::var("TRELLIS2_DECODER_WGPU_LINEAR_PARITY_CONTEXT") else {
        return false;
    };
    let needle = needle.trim();
    !needle.is_empty() && (needle == "*" || context.contains(needle))
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_linear_parity_check(
    input_t: Tensor<DefaultWgpuBackend, 2>,
    output_t: Tensor<DefaultWgpuBackend, 2>,
    layer: &LinearLayer,
    context: &str,
) -> Result<(), String> {
    if !decoder_wgpu_linear_parity_context_matches(context) {
        return Ok(());
    }
    let [rows, _channels] = input_t.dims();
    let input = tensor_to_vec_f32(input_t, format!("{context} linear parity input").as_str())?;
    let actual = tensor_to_vec_f32(output_t, format!("{context} linear parity output").as_str())?;
    let expected = linear_forward(
        input.as_slice(),
        rows,
        layer,
        format!("{context} linear parity reference").as_str(),
    )?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{context}: WGPU/reference linear output length mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let (mean_abs, max_abs, rmse) =
        decoder_wgpu_compare_flat_stats(actual.as_slice(), expected.as_slice());
    println!(
        "burn_trellis: decoder_wgpu_linear_parity context=\"{}\" rows={} in_channels={} out_channels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        context,
        rows,
        layer.in_channels,
        layer.out_channels,
        mean_abs,
        max_abs,
        rmse
    );
    if decoder_wgpu_conv_parity_strict() {
        assert!(
            mean_abs <= 1.0e-4 && max_abs <= 1.0e-3,
            "decoder WGPU linear diverged for context '{}': mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            context,
            mean_abs,
            max_abs,
            rmse
        );
    }
    Ok(())
}

#[cfg(feature = "runtime-model-wgpu")]
fn subdivision_active_indices_wgpu(
    logits_t: Tensor<DefaultWgpuBackend, 2>,
    enforce_non_empty: bool,
    runtime_config: &DecoderRuntimeConfig,
) -> Result<Tensor<DefaultWgpuBackend, 2, Int>, String> {
    if enforce_non_empty {
        return Err(
            "decoder subdivision enforce_non_empty is unsupported on canonical device path"
                .to_string(),
        );
    }
    let [rows, cols] = logits_t.dims();
    if cols != 8 {
        return Err(format!(
            "decoder subdivision logits tensor must have 8 columns, got {cols}"
        ));
    }
    if let Some(limit) = decoder_max_children_per_parent(rows) {
        return Err(format!(
            "decoder subdivision child cap ({limit}) is unsupported on canonical device path"
        ));
    }
    let device = logits_t.device();
    let child_thresholds = decoder_subdivision_child_thresholds(runtime_config);
    let threshold_t =
        Tensor::<DefaultWgpuBackend, 1>::from_floats(child_thresholds.as_slice(), &device)
            .reshape([1, 8]);
    let mask_t = logits_t.sub(threshold_t).greater_elem(0.0);
    Ok(mask_t.argwhere())
}

#[cfg(feature = "runtime-model-wgpu")]
fn expand_subdivision_coords_and_linear_indices_wgpu(
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    active_indices_t: Tensor<DefaultWgpuBackend, 2, Int>,
) -> Result<
    (
        Tensor<DefaultWgpuBackend, 2, Int>,
        Tensor<DefaultWgpuBackend, 1, Int>,
    ),
    String,
> {
    let [_rows, coord_cols] = parent_coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "decoder parent coord tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let [active_rows, active_cols] = active_indices_t.dims();
    if active_cols != 2 {
        return Err(format!(
            "decoder active index tensor must have 2 columns, got {active_cols}"
        ));
    }
    let device = parent_coords_t.device();
    if active_rows == 0 {
        return Ok((
            Tensor::<DefaultWgpuBackend, 2, Int>::zeros([0, 4], &device),
            Tensor::<DefaultWgpuBackend, 1, Int>::zeros([0], &device),
        ));
    }
    let idx_parent_col = Tensor::<DefaultWgpuBackend, 1, Int>::from_ints([0], &device);
    let idx_child_col = Tensor::<DefaultWgpuBackend, 1, Int>::from_ints([1], &device);
    let parent_idx = active_indices_t
        .clone()
        .select(1, idx_parent_col)
        .squeeze_dim(1);
    let child_idx = active_indices_t.select(1, idx_child_col).squeeze_dim(1);
    let linear_idx = parent_idx.clone().mul_scalar(8).add(child_idx.clone());

    let parent_coords_selected = parent_coords_t.select(0, parent_idx);
    let offsets = Tensor::<DefaultWgpuBackend, 1, Int>::from_ints(
        [
            0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
        ],
        &device,
    )
    .reshape([8, 3]);
    let child_offsets = offsets.select(0, child_idx);
    let idx_batch_col = Tensor::<DefaultWgpuBackend, 1, Int>::from_ints([0], &device);
    let batch_col = parent_coords_selected
        .clone()
        .select(1, idx_batch_col)
        .reshape([active_rows, 1]);
    let xyz_child = parent_coords_selected
        .slice([0..active_rows, 1..4])
        .mul_scalar(2)
        .add(child_offsets);
    let child_coords_t = Tensor::cat(vec![batch_col, xyz_child], 1);
    Ok((child_coords_t, linear_idx))
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NeighborTensorConfigKey {
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    axis_order: [usize; 3],
    axis_sign: [i32; 3],
}

#[cfg(feature = "runtime-model-wgpu")]
impl NeighborTensorConfigKey {
    fn from_config(config: &FlexConvConfig) -> Self {
        // Neighbor topology depends on kernel geometry and axis mapping only.
        // Channel-group dimensions affect GEMM shape, not neighbor lookup.
        Self {
            kernel_d: config.kernel_d,
            kernel_h: config.kernel_h,
            kernel_w: config.kernel_w,
            axis_order: config.axis_order,
            axis_sign: config.axis_sign,
        }
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn quantize_f16_tensor_wgpu(
    tensor: Tensor<DefaultWgpuBackend, 2>,
) -> Tensor<DefaultWgpuBackend, 2> {
    tensor
        .cast(burn::tensor::FloatDType::F16)
        .cast(burn::tensor::FloatDType::F32)
}

#[cfg(feature = "runtime-model-wgpu")]
fn convnext_blocks_forward_wgpu_tensor(
    context_gpu: &mut DecoderWgpuConvContext,
    coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    mut state_t: Tensor<DefaultWgpuBackend, 2>,
    stage_idx: usize,
    stage_channels: usize,
    blocks: &[ConvNeXtBlock],
    compute_fp16: bool,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "decoder wgpu convnext coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let [state_rows, state_channels] = state_t.dims();
    if state_rows != rows || state_channels != stage_channels {
        return Err(format!(
            "decoder wgpu convnext tensor dims mismatch: got=[{},{}] expected=[{},{}]",
            state_rows, state_channels, rows, stage_channels
        ));
    }
    let mut neighbor_tensors = HashMap::<NeighborTensorConfigKey, Tensor<DefaultWgpuBackend, 2, Int>>::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let residual = state_t.clone();
        let config = flex_config_for_layer(&block.conv);
        let neighbor_key = NeighborTensorConfigKey::from_config(&config);
        let neighbor_t = if let Some(cached) = neighbor_tensors.get(&neighbor_key) {
            cached.clone()
        } else {
            let built = neighbor_rows_tensor_from_coords_tensor(&config, coords_t.clone())?;
            neighbor_tensors.insert(neighbor_key, built.clone());
            built
        };
        let kernel_rows = kernel_rows(&config)?;
        state_t = context_gpu.forward_with_neighbor_tensor_tensor(
            &config,
            &block.conv,
            state_t,
            format!("stage {stage_idx} block {block_idx} conv(wgpu_math)").as_str(),
            rows,
            kernel_rows,
            neighbor_t,
        )?;
        if compute_fp16 {
            state_t = quantize_f16_tensor_wgpu(state_t);
        }
        state_t = layer_norm_wgpu(
            context_gpu,
            state_t,
            Some(block.norm_weight.as_slice()),
            Some(block.norm_bias.as_slice()),
            LAYER_NORM32_EPS,
            format!("stage {stage_idx} block {block_idx} layer_norm(wgpu_math)").as_str(),
        )?;
        if compute_fp16 {
            state_t = quantize_f16_tensor_wgpu(state_t);
        }
        state_t = convnext_block_mlp_forward_wgpu(
            context_gpu,
            state_t,
            residual,
            block,
            stage_idx,
            block_idx,
            compute_fp16,
        )?;
    }
    Ok(state_t)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderConvImpl {
    Legacy,
    #[cfg(not(feature = "runtime-model-wgpu"))]
    FlexGmm,
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu,
}

fn decoder_conv_impl() -> DecoderConvImpl {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        DecoderConvImpl::Wgpu
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        DecoderConvImpl::FlexGmm
    }
}

fn decoder_conv_debug_enabled() -> bool {
    false
}

pub(crate) fn reset_decoder_conv_telemetry() {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        if let Ok(mut state) = decoder_conv_telemetry_state().lock() {
            *state = DecoderConvTelemetryState::default();
        }
    }
}

pub(crate) fn reset_decoder_op_telemetry() {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        if let Ok(mut state) = decoder_op_telemetry_state().lock() {
            *state = DecoderOpTelemetryState::default();
        }
    }
}

pub(crate) fn decoder_conv_telemetry() -> DecoderConvTelemetry {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let Ok(state) = decoder_conv_telemetry_state().lock() else {
            return DecoderConvTelemetry::default();
        };
        let mut blocks = state.blocks.values().cloned().collect::<Vec<_>>();
        blocks.sort_by(|lhs, rhs| {
            rhs.dispatches
                .cmp(&lhs.dispatches)
                .then_with(|| rhs.wgpu_calls.cmp(&lhs.wgpu_calls))
                .then_with(|| rhs.conv_calls.cmp(&lhs.conv_calls))
                .then_with(|| lhs.context.cmp(&rhs.context))
        });
        DecoderConvTelemetry {
            conv_calls: state.total.conv_calls,
            wgpu_calls: state.total.wgpu_calls,
            wgpu_successes: state.total.wgpu_successes,
            wgpu_failures: state.total.wgpu_failures,
            dispatches: state.total.dispatches,
            chunked_calls: state.total.chunked_calls,
            max_chunk_rows: state.total.max_chunk_rows,
            input_bytes: state.total.input_bytes,
            output_bytes: state.total.output_bytes,
            neighbor_elements: state.total.neighbor_elements,
            blocks,
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        DecoderConvTelemetry::default()
    }
}

pub(crate) fn decoder_op_telemetry() -> DecoderOpTelemetry {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let Ok(state) = decoder_op_telemetry_state().lock() else {
            return DecoderOpTelemetry::default();
        };
        let mut ops = state.ops.values().cloned().collect::<Vec<_>>();
        ops.sort_by(|lhs, rhs| {
            rhs.total_ms
                .partial_cmp(&lhs.total_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| rhs.calls.cmp(&lhs.calls))
                .then_with(|| lhs.context.cmp(&rhs.context))
        });
        DecoderOpTelemetry {
            calls: state.calls,
            total_ms: state.total_ms,
            readback_count: state.readback_count,
            readback_elements: state.readback_elements,
            ops,
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        DecoderOpTelemetry::default()
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_conv_telemetry_state() -> &'static Mutex<DecoderConvTelemetryState> {
    DECODER_CONV_TELEMETRY.get_or_init(|| Mutex::new(DecoderConvTelemetryState::default()))
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_op_telemetry_state() -> &'static Mutex<DecoderOpTelemetryState> {
    DECODER_OP_TELEMETRY.get_or_init(|| Mutex::new(DecoderOpTelemetryState::default()))
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_update<F>(context: &str, mut update: F)
where
    F: FnMut(&mut DecoderConvBlockTelemetry),
{
    let Ok(mut state) = decoder_conv_telemetry_state().lock() else {
        return;
    };
    update(&mut state.total);
    let block =
        state
            .blocks
            .entry(context.to_string())
            .or_insert_with(|| DecoderConvBlockTelemetry {
                context: context.to_string(),
                ..DecoderConvBlockTelemetry::default()
            });
    update(block);
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_op_duration(context: &str, elapsed_ms: f64) {
    let Ok(mut state) = decoder_op_telemetry_state().lock() else {
        return;
    };
    state.calls = state.calls.saturating_add(1);
    state.total_ms += elapsed_ms;
    let entry = state
        .ops
        .entry(context.to_string())
        .or_insert_with(|| DecoderOpTimingTelemetry {
            context: context.to_string(),
            ..DecoderOpTimingTelemetry::default()
        });
    entry.calls = entry.calls.saturating_add(1);
    entry.total_ms += elapsed_ms;
    entry.max_ms = entry.max_ms.max(elapsed_ms);
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_readback(elements: usize) {
    let Ok(mut state) = decoder_op_telemetry_state().lock() else {
        return;
    };
    state.readback_count = state.readback_count.saturating_add(1);
    state.readback_elements = state.readback_elements.saturating_add(elements as u64);
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_conv_call(context: &str) {
    telemetry_update(context, |stats| {
        stats.conv_calls += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_call(context: &str) {
    telemetry_update(context, |stats| {
        stats.wgpu_calls += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_failure(context: &str) {
    telemetry_update(context, |stats| {
        stats.wgpu_failures += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_success(
    context: &str,
    dispatches: u64,
    chunked: bool,
    max_chunk_rows: usize,
    input_bytes: usize,
    output_bytes: usize,
    neighbor_elements: usize,
) {
    telemetry_update(context, |stats| {
        stats.wgpu_successes += 1;
        stats.dispatches += dispatches;
        if chunked {
            stats.chunked_calls += 1;
        }
        stats.max_chunk_rows = stats.max_chunk_rows.max(max_chunk_rows);
        stats.input_bytes = stats.input_bytes.saturating_add(input_bytes as u64);
        stats.output_bytes = stats.output_bytes.saturating_add(output_bytes as u64);
        stats.neighbor_elements = stats
            .neighbor_elements
            .saturating_add(neighbor_elements as u64);
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_neighbor_from_coords() -> bool {
    true
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_clear_cache_after_decode() -> bool {
    // Keep decoder tensor caches resident to avoid invalidating live tensor views before
    // decode stage-boundary readback completes in strict canonical WGPU flow.
    false
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_tensor_cache_max() -> usize {
    DECODER_WGPU_TENSOR_CACHE_MAX
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_use_tensor_cache() -> bool {
    if decoder_wgpu_clear_cache_after_decode() {
        return false;
    }
    decoder_wgpu_tensor_cache_max() > 0
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_device_math_enabled() -> bool {
    if decoder_conv_impl() != DecoderConvImpl::Wgpu {
        return false;
    }
    true
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_device_math_allow_fp16() -> bool {
    true
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_device_math_max_state_bytes() -> usize {
    // Canonical WGPU decoder path is bounded by tensor addressability; per-dispatch
    // limits are enforced by sparse-conv chunking (`decoder_wgpu_max_output_bytes`).
    decoder_wgpu_max_tensor_bytes()
}

fn flex_config_for_layer(layer: &SparseConvLayer) -> FlexConvConfig {
    FlexConvConfig {
        in_channels: layer.in_channels,
        out_channels: layer.out_channels,
        kernel_d: layer.kernel_d,
        kernel_h: layer.kernel_h,
        kernel_w: layer.kernel_w,
        in_channels_per_group: layer.in_channels_per_group,
        out_channels_per_group: layer.out_channels_per_group,
        groups: layer.groups,
        axis_order: conv_kernel_axis_order(),
        axis_sign: conv_kernel_axis_signs(),
    }
}

fn sparse_subm_conv_forward(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
    conv_cache: &mut DecoderConvCache,
    #[cfg(feature = "runtime-model-wgpu")] wgpu_context: Option<&mut DecoderWgpuConvContext>,
) -> Result<Vec<f32>, String> {
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_conv_call(context);

    let op_start = Instant::now();
    let result = (|| {
        let config = flex_config_for_layer(layer);
        let weights = SparseSubmConvWeights {
            weight: layer.weight.as_slice(),
            bias: layer.bias.as_slice(),
        };
        let conv_impl = decoder_conv_impl();

        #[cfg(feature = "runtime-model-wgpu")]
        if conv_impl == DecoderConvImpl::Wgpu {
            let context_gpu = wgpu_context.ok_or_else(|| {
                format!(
                    "burn_trellis: wgpu sparse conv context unavailable in '{context}'; refusing fallback behavior"
                )
            })?;
            telemetry_record_wgpu_call(context);
            let wgpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if decoder_wgpu_neighbor_from_coords() {
                    context_gpu.forward_with_coords(&config, layer, input, context, coords)
                } else {
                    let (neighbor_key, neighbor_rows) =
                        conv_cache.neighbor_rows_with_key(&config, coords)?;
                    context_gpu.forward_with_neighbor_rows(
                        &config,
                        layer,
                        input,
                        context,
                        neighbor_key,
                        neighbor_rows,
                    )
                }
            }));
            match wgpu_result {
                Ok(Ok(output)) => return Ok(output),
                Ok(Err(err)) => {
                    telemetry_record_wgpu_failure(context);
                    if err.contains("BufferTooBig") {
                        context_gpu.wgpu_failed = true;
                        if decoder_conv_debug_enabled() {
                            eprintln!(
                                "burn_trellis: wgpu conv disabling after buffer-too-big in '{context}': {err}"
                            );
                        }
                    } else if decoder_conv_debug_enabled() {
                        eprintln!("burn_trellis: wgpu conv failed in '{context}': {err}");
                    }
                    return Err(format!(
                        "burn_trellis: wgpu sparse conv failed in '{context}': {err}"
                    ));
                }
                Err(payload) => {
                    telemetry_record_wgpu_failure(context);
                    context_gpu.wgpu_failed = true;
                    let panic_message = panic_payload_to_string(payload);
                    if decoder_conv_debug_enabled() {
                        eprintln!(
                            "burn_trellis: wgpu conv panicked in '{context}': {panic_message}"
                        );
                    }
                    return Err(format!(
                        "burn_trellis: wgpu sparse conv panicked in '{context}': {panic_message}"
                    ));
                }
            }
        }

        if conv_impl != DecoderConvImpl::Legacy {
            let (_neighbor_key, neighbor_rows) =
                conv_cache.neighbor_rows_with_key(&config, coords)?;
            return sparse_subm_conv_forward_flex_precomputed(
                &config,
                weights,
                input,
                neighbor_rows,
                layer.flex_packed_weight.as_deref(),
            )
            .map_err(|err| format!("burn_trellis: flex sparse conv failed in '{context}': {err}"));
        }

        sparse_subm_conv_forward_legacy(coords, input, layer, context)
    })();
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

fn sparse_subm_conv_forward_legacy(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows
        .checked_mul(layer.in_channels)
        .ok_or_else(|| format!("{context}: input size overflow"))?;
    if input.len() != expected {
        return Err(format!(
            "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
            input.len(),
            expected,
            rows,
            layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }

    let mut output = vec![0.0f32; rows * layer.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * layer.out_channels;
        output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
    }

    let mut coord_to_row = HashMap::with_capacity(rows.saturating_mul(2));
    for (row_idx, coord) in coords.iter().copied().enumerate() {
        coord_to_row.insert(coord, row_idx);
    }

    let center_d = (layer.kernel_d / 2) as i32;
    let center_h = (layer.kernel_h / 2) as i32;
    let center_w = (layer.kernel_w / 2) as i32;
    let axis_order = conv_kernel_axis_order();
    let axis_sign = conv_kernel_axis_signs();
    for (out_row_idx, out_coord) in coords.iter().copied().enumerate().take(rows) {
        let batch = out_coord[0];
        let ox = out_coord[1] as i32;
        let oy = out_coord[2] as i32;
        let oz = out_coord[3] as i32;
        let out_base = out_row_idx * layer.out_channels;

        for kd_idx in 0..layer.kernel_d {
            for kh_idx in 0..layer.kernel_h {
                for kw_idx in 0..layer.kernel_w {
                    let deltas = [
                        axis_sign[0] * (kd_idx as i32 - center_d),
                        axis_sign[1] * (kh_idx as i32 - center_h),
                        axis_sign[2] * (kw_idx as i32 - center_w),
                    ];
                    let mut spatial = [ox, oy, oz];
                    spatial[axis_order[0]] += deltas[0];
                    spatial[axis_order[1]] += deltas[1];
                    spatial[axis_order[2]] += deltas[2];
                    if spatial[0] < 0 || spatial[1] < 0 || spatial[2] < 0 {
                        continue;
                    }
                    let neighbor = [
                        batch,
                        spatial[0] as u32,
                        spatial[1] as u32,
                        spatial[2] as u32,
                    ];
                    let Some(in_row_idx) = coord_to_row.get(&neighbor).copied() else {
                        continue;
                    };
                    let in_row = &input
                        [in_row_idx * layer.in_channels..(in_row_idx + 1) * layer.in_channels];
                    for group_idx in 0..layer.groups {
                        let in_group_base = group_idx * layer.in_channels_per_group;
                        let out_group_base = group_idx * layer.out_channels_per_group;
                        for out_local in 0..layer.out_channels_per_group {
                            let out_idx = out_group_base + out_local;
                            let weight_base =
                                (((out_idx * layer.kernel_d + kd_idx) * layer.kernel_h + kh_idx)
                                    * layer.kernel_w
                                    + kw_idx)
                                    * layer.in_channels_per_group;
                            let mut accum = 0.0f32;
                            for in_local in 0..layer.in_channels_per_group {
                                accum += in_row[in_group_base + in_local]
                                    * layer.weight[weight_base + in_local];
                            }
                            output[out_base + out_idx] += accum;
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

fn conv_kernel_axis_order() -> [usize; 3] {
    [0, 1, 2]
}

fn conv_kernel_axis_signs() -> [i32; 3] {
    [1, 1, 1]
}

fn layer_norm_inplace(
    data: &mut [f32],
    rows: usize,
    channels: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    context: &str,
) -> Result<(), String> {
    let op_start = Instant::now();
    let result = (|| {
        if rows == 0 || channels == 0 {
            return Ok(());
        }
        if data.len() != rows * channels {
            return Err(format!(
                "layer_norm_inplace: invalid data len {}, expected {}",
                data.len(),
                rows * channels
            ));
        }
        if let Some(weight) = weight
            && weight.len() != channels
        {
            return Err(format!(
                "layer_norm_inplace: invalid weight len {}, expected {}",
                weight.len(),
                channels
            ));
        }
        if let Some(bias) = bias
            && bias.len() != channels
        {
            return Err(format!(
                "layer_norm_inplace: invalid bias len {}, expected {}",
                bias.len(),
                channels
            ));
        }

        for row_idx in 0..rows {
            let base = row_idx * channels;
            let row = &mut data[base..base + channels];
            let mean = row.iter().copied().sum::<f32>() / channels as f32;
            let var = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / channels as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for ch in 0..channels {
                let mut value = (row[ch] - mean) * inv_std;
                if let Some(weight) = weight {
                    value *= weight[ch];
                }
                if let Some(bias) = bias {
                    value += bias[ch];
                }
                row[ch] = value;
            }
        }
        Ok(())
    })();
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
    result
}

fn silu_inplace(data: &mut [f32], context: &str) {
    let op_start = Instant::now();
    for value in data {
        *value = *value / (1.0 + (-*value).exp());
    }
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
}

fn quantize_f16_inplace(data: &mut [f32]) {
    for value in data {
        *value = f16::from_f32(*value).to_f32();
    }
}

fn row_center_logits(data: &mut [f32], rows: usize, context: &str) {
    let op_start = Instant::now();
    if rows == 0 {
        return;
    }
    if data.len() != rows * 8 {
        return;
    }
    for row_idx in 0..rows {
        let row = &mut data[row_idx * 8..(row_idx + 1) * 8];
        let mean = row.iter().copied().sum::<f32>() / 8.0;
        for value in row {
            *value -= mean;
        }
    }
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
}

fn add_inplace(lhs: &mut [f32], rhs: &[f32], context: &str) {
    let op_start = Instant::now();
    if lhs.len() != rhs.len() {
        return;
    }
    for (left, right) in lhs.iter_mut().zip(rhs.iter()) {
        *left += *right;
    }
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_op_duration(context, op_start.elapsed().as_secs_f64() * 1000.0);
}

fn logits_to_mask(
    logits: &[f32],
    rows: usize,
    enforce_non_empty: bool,
    runtime_config: &DecoderRuntimeConfig,
) -> Result<Vec<[bool; 8]>, String> {
    if logits.len() != rows * 8 {
        return Err(format!(
            "subdivision logits len {} does not match rows*8={}",
            logits.len(),
            rows * 8
        ));
    }
    let mut out = Vec::with_capacity(rows);
    let max_children = decoder_max_children_per_parent(rows);
    let child_thresholds = decoder_subdivision_child_thresholds(runtime_config);
    for row_idx in 0..rows {
        let mut mask = [false; 8];
        let row = &logits[row_idx * 8..(row_idx + 1) * 8];
        for child in 0..8 {
            mask[child] = row[child] > child_thresholds[child];
        }
        if let Some(max_children) = max_children {
            let selected = mask.iter().filter(|flag| **flag).count();
            if selected > max_children {
                let mut order = (0..8usize).collect::<Vec<_>>();
                order.sort_by(|a, b| {
                    row[*b]
                        .partial_cmp(&row[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut limited = [false; 8];
                for idx in order.into_iter().take(max_children) {
                    limited[idx] = true;
                }
                mask = limited;
            }
        }
        if enforce_non_empty && !mask.iter().any(|flag| *flag) {
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (idx, value) in row.iter().enumerate().skip(1) {
                if *value > best_val {
                    best_val = *value;
                    best_idx = idx;
                }
            }
            mask[best_idx] = true;
        }
        out.push(mask);
    }
    Ok(out)
}

fn decoder_subdivision_child_thresholds(runtime_config: &DecoderRuntimeConfig) -> [f32; 8] {
    let mut thresholds = [runtime_config.subdivision_threshold; 8];
    for (idx, threshold) in thresholds.iter_mut().enumerate() {
        let child = runtime_config.subdivision_child_thresholds[idx];
        if child.is_finite() {
            *threshold = child;
        }
    }
    thresholds
}

fn decoder_max_children_per_parent(_rows: usize) -> Option<usize> {
    None
}

#[cfg(feature = "runtime-model-wgpu")]
fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_output_bytes() -> usize {
    // Keep decode conv dispatches device-resident with a larger default guard to
    // reduce avoidable chunking in large upsample blocks.
    512 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_sparse_conv_max_output_bytes() -> usize {
    // Sparse-conv kernels can require additional transient buffers beyond output.
    // Keep per-dispatch output conservative: larger native dispatches were
    // measured slower on the 512-base decoder and changed final occupancy by
    // one voxel on the frozen chair reference.
    96 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_im2col_max_bytes() -> usize {
    // The matmul route materializes a logical im2col tensor. Keep an explicit cap
    // so it is only used for measured decoder-hot chunks that fit the current
    // native WGPU memory envelope.
    3 * 1024 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_use_im2col_matmul(
    config: &FlexConvConfig,
    rows: usize,
    kernel_rows: usize,
) -> bool {
    if config.groups != 1
        || config.in_channels_per_group != config.in_channels
        || config.out_channels_per_group != config.out_channels
    {
        return false;
    }
    if kernel_rows != 27 {
        return false;
    }
    let inner_work = kernel_rows.saturating_mul(config.in_channels);
    let output_work = rows.saturating_mul(config.out_channels);
    if inner_work < 1_024 || output_work < 1_000_000 {
        return false;
    }
    let Some(im2col_elements) = rows
        .checked_mul(kernel_rows)
        .and_then(|value| value.checked_mul(config.in_channels))
    else {
        return false;
    };
    let im2col_bytes = im2col_elements.saturating_mul(core::mem::size_of::<f32>());
    im2col_bytes <= decoder_wgpu_im2col_max_bytes()
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_weight_bytes() -> usize {
    // Large monolithic decoder weight uploads can OOM on some adapters before the
    // chunked row-dispatch logic even runs. Keep a bounded per-dispatch weight
    // upload size and split output channels instead of falling back to host.
    128 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_upsample_conv1_max_output_bytes() -> usize {
    // Prefer direct conv1 materialization while it fits the normal WGPU output guard:
    // the chunked gather path is reserved for truly oversized stages.
    decoder_wgpu_max_tensor_bytes()
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_tensor_bytes() -> usize {
    i32::MAX as usize
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_input_bytes() -> usize {
    // Input tensors use the same addressability constraints as outputs on the
    // canonical WGPU path; a lower fixed cap causes false-positive aborts in
    // high-row decode stages that are otherwise chunk-safe.
    decoder_wgpu_max_tensor_bytes()
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_neighbor_bytes() -> usize {
    256 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_chunk_rows(rows: usize, bytes_per_row: usize, max_output_bytes: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    if bytes_per_row == 0 {
        return rows;
    }
    let by_bytes = (max_output_bytes / bytes_per_row).max(1).min(rows);
    let aligned = by_bytes - (by_bytes % 64);
    if aligned > 0 { aligned } else { by_bytes }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_reduce_chunk_rows(chunk_rows: usize) -> usize {
    if chunk_rows <= 1 {
        return 1;
    }
    let halved = (chunk_rows / 2).max(1);
    let aligned = halved - (halved % 64);
    if aligned > 0 { aligned } else { halved }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_is_buffer_too_big(err: &str) -> bool {
    err.contains("BufferTooBig")
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_hotspot_min_output_bytes() -> usize {
    384 * 1024 * 1024
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_hotspot_fused_enabled() -> bool {
    true
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_forward_config_for_call(
    config: &FlexConvConfig,
    rows: usize,
    output_bytes: usize,
    max_output_bytes: usize,
) -> SparseWgpuForwardConfig {
    if !decoder_wgpu_hotspot_fused_enabled() {
        return SparseWgpuForwardConfig::default();
    }

    let hotspot = output_bytes >= decoder_wgpu_hotspot_min_output_bytes()
        || output_bytes > max_output_bytes
        || rows >= 131_072;
    if hotspot && config.out_channels_per_group >= 4 && config.out_channels >= 4 {
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::FusedOc4,
            split_k: Some(1),
        }
    } else {
        SparseWgpuForwardConfig::default()
    }
}

fn channel2spatial(
    coords: &[[u32; 4]],
    feats: &[f32],
    in_channels: usize,
    subdivision_mask: &[[bool; 8]],
) -> Result<(Vec<[u32; 4]>, Vec<f32>), String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if feats.len() != rows * in_channels {
        return Err(format!(
            "channel2spatial: invalid feats len {}, expected {}",
            feats.len(),
            rows * in_channels
        ));
    }
    if !in_channels.is_multiple_of(8) {
        return Err(format!(
            "channel2spatial: in_channels={} is not divisible by 8",
            in_channels
        ));
    }
    if subdivision_mask.len() != rows {
        return Err(format!(
            "channel2spatial: subdivision rows {} do not match coords rows {}",
            subdivision_mask.len(),
            rows
        ));
    }

    let out_channels = in_channels / 8;
    let mut out_coords = Vec::new();
    let mut out_feats = Vec::new();
    for row_idx in 0..rows {
        let coord = coords[row_idx];
        let row_feats = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for (child, selected) in subdivision_mask[row_idx].iter().enumerate().take(8usize) {
            if !*selected {
                continue;
            }
            let cx = (child & 1) as u32;
            let cy = ((child >> 1) & 1) as u32;
            let cz = ((child >> 2) & 1) as u32;
            out_coords.push([
                coord[0],
                coord[1].saturating_mul(2).saturating_add(cx),
                coord[2].saturating_mul(2).saturating_add(cy),
                coord[3].saturating_mul(2).saturating_add(cz),
            ]);
            let child_base = child * out_channels;
            out_feats.extend_from_slice(&row_feats[child_base..child_base + out_channels]);
        }
    }
    Ok((out_coords, out_feats))
}

fn repeat_interleave_channels(
    feats: &[f32],
    rows: usize,
    in_channels: usize,
    repeat_factor: usize,
) -> Vec<f32> {
    if rows == 0 || in_channels == 0 || repeat_factor == 0 {
        return Vec::new();
    }
    let out_channels = in_channels * repeat_factor;
    let mut out = Vec::with_capacity(rows * out_channels);
    for row_idx in 0..rows {
        let row = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for value in row {
            for _ in 0..repeat_factor {
                out.push(*value);
            }
        }
    }
    out
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_upsample_parity_stage() -> Option<usize> {
    std::env::var("TRELLIS2_DECODER_WGPU_UPSAMPLE_PARITY_STAGE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_conv_flex_reference_for_layer(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    let config = flex_config_for_layer(layer);
    let neighbor_rows = build_neighbor_rows(&config, coords)
        .map_err(|err| format!("{context}: failed building cpu neighbor rows: {err}"))?;
    sparse_subm_conv_forward_flex_precomputed(
        &config,
        SparseSubmConvWeights {
            weight: layer.weight.as_slice(),
            bias: layer.bias.as_slice(),
        },
        input,
        neighbor_rows.as_slice(),
        layer.flex_packed_weight.as_deref(),
    )
    .map_err(|err| format!("{context}: cpu flex sparse conv failed: {err}"))
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_upsample_parity_check(
    stage_idx: usize,
    up: &C2SUpsampleBlock,
    parent_feats_t: Tensor<DefaultWgpuBackend, 2>,
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    active_indices_t: Tensor<DefaultWgpuBackend, 2, Int>,
    actual_state_t: Tensor<DefaultWgpuBackend, 2>,
) -> Result<(), String> {
    if decoder_wgpu_upsample_parity_stage() != Some(stage_idx) {
        return Ok(());
    }
    let context = format!("stage {stage_idx} upsample parity");
    let parent_coords =
        tensor_to_coords_u32(parent_coords_t, format!("{context} parent coords").as_str())?;
    let parent_feats =
        tensor_to_vec_f32(parent_feats_t, format!("{context} parent feats").as_str())?;
    let active_indices =
        tensor_to_vec_i32(active_indices_t, format!("{context} active indices").as_str())?;
    if active_indices.len() % 2 != 0 {
        return Err(format!(
            "{context}: active index tensor must have pairs, got len={}",
            active_indices.len()
        ));
    }
    let parent_rows = parent_coords.len();
    if parent_feats.len() != parent_rows.saturating_mul(up.in_channels) {
        return Err(format!(
            "{context}: parent feature len mismatch got={} expected={}",
            parent_feats.len(),
            parent_rows.saturating_mul(up.in_channels)
        ));
    }
    let mut subdivision_mask = vec![[false; 8]; parent_rows];
    for pair in active_indices.chunks_exact(2) {
        let parent_idx = usize::try_from(pair[0])
            .map_err(|_| format!("{context}: negative active parent index {}", pair[0]))?;
        let child_idx = usize::try_from(pair[1])
            .map_err(|_| format!("{context}: negative active child index {}", pair[1]))?;
        if parent_idx >= parent_rows || child_idx >= 8 {
            return Err(format!(
                "{context}: active index out of range parent={} child={} parent_rows={}",
                parent_idx, child_idx, parent_rows
            ));
        }
        subdivision_mask[parent_idx][child_idx] = true;
    }

    let mut h_norm = parent_feats.clone();
    layer_norm_inplace(
        h_norm.as_mut_slice(),
        parent_rows,
        up.in_channels,
        Some(up.norm1_weight.as_slice()),
        Some(up.norm1_bias.as_slice()),
        LAYER_NORM32_EPS,
        format!("{context} norm1").as_str(),
    )?;
    quantize_f16_inplace(h_norm.as_mut_slice());
    silu_inplace(h_norm.as_mut_slice(), format!("{context} silu1").as_str());
    quantize_f16_inplace(h_norm.as_mut_slice());

    let mut h_conv1 = sparse_conv_flex_reference_for_layer(
        parent_coords.as_slice(),
        h_norm.as_slice(),
        &up.conv1,
        format!("{context} conv1").as_str(),
    )?;
    quantize_f16_inplace(h_conv1.as_mut_slice());
    let (child_coords, mut h_up) = channel2spatial(
        parent_coords.as_slice(),
        h_conv1.as_slice(),
        up.out_channels
            .checked_mul(8)
            .ok_or_else(|| format!("{context}: up.out_channels * 8 overflow"))?,
        subdivision_mask.as_slice(),
    )?;
    let (child_coords_skip, x_up) = channel2spatial(
        parent_coords.as_slice(),
        parent_feats.as_slice(),
        up.in_channels,
        subdivision_mask.as_slice(),
    )?;
    if child_coords != child_coords_skip {
        return Err(format!("{context}: conv and skip child coords diverged"));
    }

    let skip_in_channels = up.in_channels / 8;
    if skip_in_channels == 0 || up.out_channels % skip_in_channels != 0 {
        return Err(format!(
            "{context}: invalid skip channel ratio in={} out={}",
            up.in_channels, up.out_channels
        ));
    }
    let repeat_factor = up.out_channels / skip_in_channels;
    let skip = repeat_interleave_channels(
        x_up.as_slice(),
        child_coords.len(),
        skip_in_channels,
        repeat_factor,
    );
    let child_rows = child_coords.len();
    layer_norm_inplace(
        h_up.as_mut_slice(),
        child_rows,
        up.out_channels,
        None,
        None,
        LAYER_NORM32_EPS,
        format!("{context} layer_norm").as_str(),
    )?;
    quantize_f16_inplace(h_up.as_mut_slice());
    silu_inplace(h_up.as_mut_slice(), format!("{context} silu2").as_str());
    quantize_f16_inplace(h_up.as_mut_slice());
    let mut expected = sparse_conv_flex_reference_for_layer(
        child_coords.as_slice(),
        h_up.as_slice(),
        &up.conv2,
        format!("{context} conv2").as_str(),
    )?;
    quantize_f16_inplace(expected.as_mut_slice());
    add_inplace(
        expected.as_mut_slice(),
        skip.as_slice(),
        format!("{context} skip_add").as_str(),
    );
    quantize_f16_inplace(expected.as_mut_slice());

    let actual =
        tensor_to_vec_f32(actual_state_t, format!("{context} actual state").as_str())?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{context}: actual/expected upsample feature len mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let (mean_abs, max_abs, rmse) =
        decoder_wgpu_compare_flat_stats(actual.as_slice(), expected.as_slice());
    println!(
        "burn_trellis: decoder_wgpu_upsample_parity stage={} parent_rows={} child_rows={} channels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        stage_idx,
        parent_rows,
        child_rows,
        up.out_channels,
        mean_abs,
        max_abs,
        rmse
    );
    if decoder_wgpu_conv_parity_strict() {
        assert!(
            mean_abs <= 1.0e-4 && max_abs <= 1.0e-3,
            "decoder WGPU upsample diverged at stage {}: mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            stage_idx,
            mean_abs,
            max_abs,
            rmse
        );
    }
    Ok(())
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_upsample_conv1_select_parity_check(
    stage_idx: usize,
    up: &C2SUpsampleBlock,
    parent_feats_norm_t: Tensor<DefaultWgpuBackend, 2>,
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    linear_idx_t: Tensor<DefaultWgpuBackend, 1, Int>,
    selected_t: Tensor<DefaultWgpuBackend, 2>,
) -> Result<(), String> {
    if decoder_wgpu_upsample_parity_stage() != Some(stage_idx) {
        return Ok(());
    }
    let context = format!("stage {stage_idx} upsample conv1-select parity");
    let parent_coords =
        tensor_to_coords_u32(parent_coords_t, format!("{context} parent coords").as_str())?;
    let parent_feats =
        tensor_to_vec_f32(parent_feats_norm_t, format!("{context} parent feats norm").as_str())?;
    let linear_idx = linear_idx_t
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .map_err(|err| format!("{context}: failed linear index extraction: {err:?}"))?;
    let h_conv1 = sparse_conv_flex_reference_for_layer(
        parent_coords.as_slice(),
        parent_feats.as_slice(),
        &up.conv1,
        format!("{context} conv1").as_str(),
    )?;
    let full_rows = parent_coords
        .len()
        .checked_mul(8)
        .ok_or_else(|| format!("{context}: parent_rows * 8 overflow"))?;
    if h_conv1.len() != full_rows.saturating_mul(up.out_channels) {
        return Err(format!(
            "{context}: conv1 output len mismatch got={} expected={}",
            h_conv1.len(),
            full_rows.saturating_mul(up.out_channels)
        ));
    }
    let mut expected = Vec::with_capacity(linear_idx.len().saturating_mul(up.out_channels));
    for value in &linear_idx {
        let idx = usize::try_from(*value)
            .map_err(|_| format!("{context}: negative linear index {value}"))?;
        if idx >= full_rows {
            return Err(format!(
                "{context}: linear index {} out of range full_rows={}",
                idx, full_rows
            ));
        }
        let start = idx.saturating_mul(up.out_channels);
        expected.extend_from_slice(&h_conv1[start..start + up.out_channels]);
    }
    let actual = tensor_to_vec_f32(selected_t, format!("{context} selected").as_str())?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{context}: selected feature len mismatch actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let (mean_abs, max_abs, rmse) =
        decoder_wgpu_compare_flat_stats(actual.as_slice(), expected.as_slice());
    println!(
        "burn_trellis: decoder_wgpu_upsample_conv1_select_parity stage={} rows={} channels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
        stage_idx,
        linear_idx.len(),
        up.out_channels,
        mean_abs,
        max_abs,
        rmse
    );
    Ok(())
}

fn map_guide_subdivision_logits(
    coords: &[[u32; 4]],
    guide: &SparseSubdivisionLogits,
) -> Result<Vec<f32>, String> {
    let guide_coords = guide.coords_host("guide subdivision coord materialization")?;
    let guide_logits = guide.logits_host("guide subdivision logits materialization")?;
    if guide_logits.len() != guide_coords.len() * 8 {
        return Err(format!(
            "guide subdivision logits invalid length: logits={} coords={}",
            guide_logits.len(),
            guide_coords.len()
        ));
    }
    let mut map = HashMap::with_capacity(guide_coords.len() * 2);
    for (idx, coord) in guide_coords.iter().enumerate() {
        let row = &guide_logits[idx * 8..(idx + 1) * 8];
        map.insert(*coord, row.to_vec());
    }

    let mut out = Vec::with_capacity(coords.len() * 8);
    for coord in coords {
        if let Some(row) = map.get(coord) {
            out.extend_from_slice(row);
        } else {
            return Err(format!(
                "guide subdivision logits missing coord {:?}",
                coord
            ));
        }
    }
    Ok(out)
}

#[cfg(feature = "runtime-model-wgpu")]
fn guide_subdivision_logits_tensor_for_parent_wgpu(
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    guide: &SparseSubdivisionLogits,
    device: &WgpuDevice,
    stage_idx: usize,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let [parent_rows, parent_cols] = parent_coords_t.dims();
    if parent_cols != 4 {
        return Err(format!(
            "decoder stage {} parent coords tensor must have 4 columns, got {}",
            stage_idx, parent_cols
        ));
    }
    if let Some((guide_coords_t, guide_logits_t)) = guide.device_tensors() {
        let [guide_rows, guide_cols] = guide_coords_t.dims();
        if guide_cols != 4 {
            return Err(format!(
                "decoder stage {} guide coords tensor must have 4 columns, got {}",
                stage_idx, guide_cols
            ));
        }
        let [logit_rows, logit_cols] = guide_logits_t.dims();
        if logit_cols != 8 {
            return Err(format!(
                "decoder stage {} guide logits tensor must have 8 columns, got {}",
                stage_idx, logit_cols
            ));
        }
        if guide_rows != parent_rows || logit_rows != parent_rows {
            return Err(format!(
                "decoder stage {} guide tensor rows mismatch: parent_rows={} guide_coord_rows={} guide_logit_rows={}",
                stage_idx, parent_rows, guide_rows, logit_rows
            ));
        }
        return Ok(guide_logits_t);
    }
    let _ = (parent_coords_t, device);
    Err(format!(
        "decoder stage {stage_idx} requires tensor-native guide subdivisions on wgpu path; host guide readback fallback is disabled"
    ))
}

#[cfg(feature = "runtime-model-wgpu")]
fn guide_subdivision_active_indices_tensor_for_parent_wgpu(
    parent_rows: usize,
    guide: &SparseSubdivisionLogits,
    stage_idx: usize,
) -> Result<Tensor<DefaultWgpuBackend, 2, Int>, String> {
    // Reuse active child indices produced by the guide (shape decoder) so tex decode
    // follows the same subdivision topology and avoids redundant argwhere passes.
    let Some(active_indices_t) = guide.active_indices_tensor() else {
        return Err(format!(
            "decoder stage {stage_idx} requires tensor-native guide active indices on wgpu path; guide argwhere recompute fallback is disabled"
        ));
    };
    let [active_rows, active_cols] = active_indices_t.dims();
    if active_cols != 2 {
        return Err(format!(
            "decoder stage {} guide active-index tensor must have 2 columns, got {}",
            stage_idx, active_cols
        ));
    }
    if active_rows > parent_rows.saturating_mul(8) {
        return Err(format!(
            "decoder stage {} guide active-index tensor has too many rows: active_rows={} max_rows={}",
            stage_idx,
            active_rows,
            parent_rows.saturating_mul(8)
        ));
    }
    Ok(active_indices_t)
}

#[cfg(feature = "runtime-model-wgpu")]
fn guide_subdivision_child_tensors_for_parent_wgpu(
    parent_rows: usize,
    guide: &SparseSubdivisionLogits,
    stage_idx: usize,
) -> Result<
    (
        Tensor<DefaultWgpuBackend, 2, Int>,
        Tensor<DefaultWgpuBackend, 1, Int>,
    ),
    String,
> {
    // Reuse child coords + parent*8 linear mapping from the guide (shape decoder)
    // so tex decode skips redundant subdivision-expansion kernels.
    let Some((child_coords_t, child_linear_idx_t)) = guide.child_tensors() else {
        return Err(format!(
            "decoder stage {stage_idx} requires tensor-native guide child tensors on wgpu path; guide expansion fallback is disabled"
        ));
    };
    let [child_rows, child_cols] = child_coords_t.dims();
    if child_cols != 4 {
        return Err(format!(
            "decoder stage {} guide child-coord tensor must have 4 columns, got {}",
            stage_idx, child_cols
        ));
    }
    let [linear_rows] = child_linear_idx_t.dims();
    if linear_rows != child_rows {
        return Err(format!(
            "decoder stage {} guide child tensor row mismatch: child_rows={} linear_rows={}",
            stage_idx, child_rows, linear_rows
        ));
    }
    if child_rows > parent_rows.saturating_mul(8) {
        return Err(format!(
            "decoder stage {} guide child tensor has too many rows: child_rows={} max_rows={}",
            stage_idx,
            child_rows,
            parent_rows.saturating_mul(8)
        ));
    }
    Ok((child_coords_t, child_linear_idx_t))
}

fn spatial_shape_from_coords(coords: &[[u32; 4]]) -> [u32; 3] {
    if coords.is_empty() {
        return [1, 1, 1];
    }
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut max_z = 0u32;
    for coord in coords {
        max_x = max_x.max(coord[1]);
        max_y = max_y.max(coord[2]);
        max_z = max_z.max(coord[3]);
    }
    [
        max_x.saturating_add(1),
        max_y.saturating_add(1),
        max_z.saturating_add(1),
    ]
}
