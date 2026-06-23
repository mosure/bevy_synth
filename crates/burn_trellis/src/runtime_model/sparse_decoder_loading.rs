fn load_linear(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;
    let weight_fp16 = round_vec_to_f16(weight.as_slice());
    let bias_fp16 = round_vec_to_f16(bias.as_slice());

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
        weight_fp16,
        bias_fp16,
    })
}

fn load_linear_dynamic(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;
    let weight_fp16 = round_vec_to_f16(weight.as_slice());
    let bias_fp16 = round_vec_to_f16(bias.as_slice());

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
        weight_fp16,
        bias_fp16,
    })
}

fn load_sparse_conv(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<SparseConvLayer, String> {
    let (w_shape, weight) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 5 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=5, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let kd = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let in_channels_per_group = w_shape[4];
    if kd == 0 || kh == 0 || kw == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid kernel dims ({kd},{kh},{kw})"
        ));
    }
    if in_channels_per_group == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid in_channels_per_group=0"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }
    let in_channels = if expected_in > 0 {
        expected_in
    } else {
        in_channels_per_group
    };
    if in_channels < in_channels_per_group || !in_channels.is_multiple_of(in_channels_per_group) {
        return Err(format!(
            "tensor '{weight_key}' expected_in={in_channels} is incompatible with in_per_group={in_channels_per_group}"
        ));
    }
    let groups = in_channels / in_channels_per_group;
    if groups == 0 || !out_channels.is_multiple_of(groups) {
        return Err(format!(
            "tensor '{weight_key}' has incompatible grouped channels (groups={groups}, out_channels={out_channels})"
        ));
    }
    let out_channels_per_group = out_channels / groups;

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let expected_weight_len = out_channels
        .checked_mul(kd)
        .and_then(|value| value.checked_mul(kh))
        .and_then(|value| value.checked_mul(kw))
        .and_then(|value| value.checked_mul(in_channels_per_group))
        .ok_or_else(|| format!("tensor '{weight_key}' weight shape product overflow"))?;
    if weight.len() != expected_weight_len {
        return Err(format!(
            "tensor '{weight_key}' element count mismatch: expected {expected_weight_len}, got {}",
            weight.len()
        ));
    }

    #[cfg(target_arch = "wasm32")]
    let flex_packed_weight = None;
    #[cfg(not(target_arch = "wasm32"))]
    let flex_packed_weight = {
        let flex_pack_config = FlexConvConfig {
            in_channels,
            out_channels,
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            in_channels_per_group,
            out_channels_per_group,
            groups,
            axis_order: [0, 1, 2],
            axis_sign: [1, 1, 1],
        };
        Some(pack_flex_weight(&flex_pack_config, weight.as_slice())?)
    };

    Ok(SparseConvLayer {
        in_channels,
        out_channels,
        kernel_d: kd,
        kernel_h: kh,
        kernel_w: kw,
        in_channels_per_group,
        out_channels_per_group,
        groups,
        weight_fp16: round_vec_to_f16(weight.as_slice()),
        bias_fp16: round_vec_to_f16(bias.as_slice()),
        weight,
        bias,
        flex_packed_weight,
    })
}

fn round_vec_to_f16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .map(|value| f16::from_f32(value).to_f32())
        .collect()
}

fn load_vector(
    safetensors: &SafeTensors<'_>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let (shape, data) = load_tensor_f32(safetensors, key)?;
    if shape.len() != 1 {
        return Err(format!(
            "tensor '{key}' expected rank=1, got rank={}",
            shape.len()
        ));
    }
    if expected_len > 0 && shape[0] != expected_len {
        return Err(format!(
            "tensor '{key}' expected len={expected_len}, got len={}",
            shape[0]
        ));
    }
    Ok(data)
}

fn load_tensor_f32(
    safetensors: &SafeTensors<'_>,
    key: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let view = safetensors
        .tensor(key)
        .map_err(|err| format!("missing tensor '{key}' in safetensors: {err}"))?;
    let shape = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F32 => bytes_to_f32(view.data())?,
        Dtype::F16 => bytes_to_f16(view.data())?,
        Dtype::BF16 => bytes_to_bf16(view.data())?,
        other => {
            return Err(format!(
                "tensor '{key}' has unsupported dtype {other:?}; expected f32/f16/bf16"
            ));
        }
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| format!("tensor '{key}' shape product overflow: {:?}", shape))?;
    if data.len() != expected {
        return Err(format!(
            "tensor '{key}' element count mismatch: expected {expected}, got {}",
            data.len()
        ));
    }
    Ok((shape, data))
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid f32 tensor payload byte length {}; must be divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bytes_to_f16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid f16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn bytes_to_bf16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid bf16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn flatten_rows_32(rows: &[[f32; 32]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * 32);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}


fn load_weight_backing(path: &Path) -> Result<WeightsBacking, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        let bytes = load_blob_bytes_from_burnpack_or_parts(path, load_burnpack_blob_bytes)?;
        return Ok(WeightsBacking::Bytes(bytes));
    }

    if crate::virtual_fs::has_virtual_file(path) {
        let bytes = crate::virtual_fs::read(path).map_err(|err| {
            format!(
                "failed to read virtual sparse decoder weights '{}': {err}",
                path.display()
            )
        })?;
        return Ok(WeightsBacking::Bytes(bytes));
    }

    let file = File::open(path).map_err(|err| {
        format!(
            "failed to open sparse decoder weights '{}': {err}",
            path.display()
        )
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        format!(
            "failed to mmap sparse decoder weights '{}': {err}",
            path.display()
        )
    })?;
    Ok(WeightsBacking::Mmap(mmap))
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    load_blob_bytes_from_blob_burnpack(path)
}

fn resolve_model_weight_candidates(
    model_stem: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> Vec<PathBuf> {
    let source =
        resolve_model_source_path(model_stem, "safetensors", weights_root, image_large_root);
    let burnpack = source.with_extension("bpk");
    let burnpack_f16 = with_file_stem_suffix(&burnpack, F16_SUFFIX);
    let source_f16 = with_file_stem_suffix(&source, F16_SUFFIX);
    let prefer_f16 = prefer_f16_burnpack();
    let candidates = if prefer_f16 {
        vec![burnpack_f16, burnpack, source_f16, source]
    } else {
        vec![burnpack, burnpack_f16, source, source_f16]
    };
    candidates
        .into_iter()
        .filter(|path| candidate_exists_or_has_parts(path))
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    true
}

fn resolve_model_source_path(
    stem: &str,
    ext: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        let image_large_root = image_large_root.unwrap_or(weights_root);
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}
