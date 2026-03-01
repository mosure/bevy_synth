#[derive(Default)]
struct DecodeHookOverrides<'a> {
    decode_shape_subs: Option<&'a [DecodeShapeSubSample]>,
    decode_tex_voxels: Option<&'a DecodeTexVoxelSample>,
    decode_mesh_vertices: Option<&'a [[f32; 3]]>,
    decode_mesh_faces: Option<&'a [[u32; 3]]>,
}

#[cfg(feature = "runtime-model")]
#[derive(Clone, Copy, Default)]
struct RuntimeDecodeModels<'a> {
    shape_decoder: Option<&'a FdgDecoderRuntime>,
    tex_decoder: Option<&'a SparseUnetVaeDecoderRuntime>,
}

fn decode_latent_to_outputs(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    final_resolution_override: Option<usize>,
    target_faces: Option<usize>,
    parity_strict: bool,
    capture_debug_artifacts: bool,
    decode_overrides: DecodeHookOverrides<'_>,
    #[cfg(feature = "runtime-model")] runtime_decoders: RuntimeDecodeModels<'_>,
) -> Result<DecodedLatentOutput, String> {
    let has_decode_override = decode_overrides.decode_shape_subs.is_some()
        || decode_overrides.decode_tex_voxels.is_some()
        || decode_overrides.decode_mesh_vertices.is_some()
        || decode_overrides.decode_mesh_faces.is_some();
    if has_decode_override {
        return Err(
            "burn_trellis: decode hook override tensors are disabled on canonical runtime decode path"
                .to_string(),
        );
    }

    #[cfg(feature = "runtime-model")]
    {
        let Some(shape_decoder) = runtime_decoders.shape_decoder else {
            return Err(
                "burn_trellis: shape runtime decoder is required (missing `shape_slat_decoder` runtime)"
                    .to_string(),
            );
        };
        let Some(tex_decoder) = runtime_decoders.tex_decoder else {
            return Err(
                "burn_trellis: tex runtime decoder is required (missing `tex_slat_decoder` runtime)"
                    .to_string(),
            );
        };
        let decoded = decode_latent_with_runtime_decoders(
            shape,
            tex,
            RuntimeDecodeRequest {
                pipeline_type,
                final_resolution: final_resolution_override
                    .unwrap_or_else(|| final_resolution_for_pipeline(pipeline_type)),
                target_faces,
                parity_strict,
                capture_debug_artifacts,
                shape_decoder,
                tex_decoder,
                shape_guide_subdivisions: None,
            },
        )
        .map_err(|err| format!("burn_trellis: runtime decode pipeline failed: {err}"))?;
        let _ = decode_overrides;
        Ok(decoded)
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (
            shape,
            tex,
            pipeline_type,
            final_resolution_override,
            target_faces,
            parity_strict,
            capture_debug_artifacts,
            decode_overrides,
        );
        Err("burn_trellis: TRELLIS decode requires `runtime-model` feature".to_string())
    }
}

#[cfg(feature = "runtime-model")]
fn merge_voxel_attrs_for_decode(
    shape_coords: &[[u32; 4]],
    tex_coords: &[[u32; 4]],
    tex_attrs: &[[f32; 6]],
    _parity_strict: bool,
) -> Result<Vec<[f32; 6]>, String> {
    if tex_coords.len() != tex_attrs.len() {
        return Err(format!(
            "decode tex voxel output mismatch: coords={} attrs={}",
            tex_coords.len(),
            tex_attrs.len()
        ));
    }
    if shape_coords.is_empty() {
        return Ok(Vec::new());
    }

    if shape_coords.len() == tex_coords.len() && shape_coords == tex_coords {
        return Ok(tex_attrs.to_vec());
    }
    Err(format!(
        "decode tex voxel coords differ from shape coords (shape_rows={} tex_rows={})",
        shape_coords.len(),
        tex_coords.len()
    ))
}

#[cfg(feature = "runtime-model")]
type MeshVertices = Vec<[f32; 3]>;
#[cfg(feature = "runtime-model")]
type MeshFaces = Vec<[u32; 3]>;
#[cfg(feature = "runtime-model")]
type MeshSanitizeResult = (MeshVertices, MeshFaces, usize, usize, usize);

#[cfg(feature = "runtime-model")]
fn sanitize_mesh_geometry(vertices: MeshVertices, faces: MeshFaces) -> MeshSanitizeResult {
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut sanitized_vertices = Vec::with_capacity(vertices.len());
    for (idx, vertex) in vertices.into_iter().enumerate() {
        if vertex.iter().all(|component| component.is_finite()) {
            remap[idx] = sanitized_vertices.len() as u32;
            sanitized_vertices.push(vertex);
        }
    }
    let dropped_vertices = remap.iter().filter(|&&mapped| mapped == u32::MAX).count();

    let mut sanitized_faces = Vec::with_capacity(faces.len());
    let mut dropped_invalid_faces = 0usize;
    let mut dropped_degenerate_faces = 0usize;
    for [a, b, c] in faces {
        let map_index = |value: u32| -> Option<u32> {
            let idx = value as usize;
            if idx >= remap.len() {
                return None;
            }
            let mapped = remap[idx];
            (mapped != u32::MAX).then_some(mapped)
        };
        let Some(a_mapped) = map_index(a) else {
            dropped_invalid_faces += 1;
            continue;
        };
        let Some(b_mapped) = map_index(b) else {
            dropped_invalid_faces += 1;
            continue;
        };
        let Some(c_mapped) = map_index(c) else {
            dropped_invalid_faces += 1;
            continue;
        };
        if a_mapped == b_mapped || b_mapped == c_mapped || a_mapped == c_mapped {
            dropped_degenerate_faces += 1;
            continue;
        }
        sanitized_faces.push([a_mapped, b_mapped, c_mapped]);
    }

    (
        sanitized_vertices,
        sanitized_faces,
        dropped_vertices,
        dropped_invalid_faces,
        dropped_degenerate_faces,
    )
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn decimate_mesh_for_face_budget(
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    target_faces: usize,
) -> Result<(), String> {
    if target_faces == 0 || faces.len() <= target_faces || faces.is_empty() || vertices.is_empty() {
        return Ok(());
    }

    let mut indices = Vec::with_capacity(faces.len() * 3);
    for face in faces.iter() {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for runtime decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(vertices.as_slice());
    let adapter = meshopt::VertexDataAdapter::new(
        vertices_bytes,
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut result_error = 0.0f32;
    let mut simplified = Vec::<u32>::new();
    for error_limit in [0.02f32, 0.05, 0.1, 0.25, 0.5, 1.0] {
        let mut stage_error = 0.0f32;
        let candidate = meshopt::simplify(
            &indices,
            &adapter,
            target_index_count,
            error_limit,
            meshopt::SimplifyOptions::None,
            Some(&mut stage_error),
        );
        if candidate.len() < 3 {
            continue;
        }
        result_error = stage_error;
        simplified = candidate;
        if simplified.len() <= target_index_count {
            break;
        }
    }
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(
            &indices,
            &adapter,
            target_index_count,
            result_error.max(0.25),
            None,
        );
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) = meshopt::generate_vertex_remap(vertices.as_slice(), Some(&simplified));
    let remapped_vertices = meshopt::remap_vertex_buffer(vertices.as_slice(), vertex_count, &remap);
    let remapped_indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if remapped_indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }
    let remapped_faces = remapped_indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();

    *vertices = remapped_vertices;
    *faces = remapped_faces;
    Ok(())
}

#[cfg(feature = "runtime-model")]
struct RuntimeDecodeRequest<'a> {
    pipeline_type: &'a str,
    final_resolution: usize,
    target_faces: Option<usize>,
    parity_strict: bool,
    capture_debug_artifacts: bool,
    shape_decoder: &'a FdgDecoderRuntime,
    tex_decoder: &'a SparseUnetVaeDecoderRuntime,
    shape_guide_subdivisions: Option<&'a [SparseSubdivisionLogits]>,
}

#[cfg(feature = "runtime-model")]
fn decode_latent_with_runtime_decoders(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    request: RuntimeDecodeRequest<'_>,
) -> Result<DecodedLatentOutput, String> {
    let RuntimeDecodeRequest {
        pipeline_type,
        final_resolution,
        target_faces,
        parity_strict,
        capture_debug_artifacts,
        shape_decoder,
        tex_decoder,
        shape_guide_subdivisions,
    } = request;
    let stage_debug = runtime_stage_debug_enabled();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_coords_wgpu = shape.coords_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_coords_wgpu = tex.coords_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_features_wgpu = shape.features_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_features_wgpu = tex.features_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    // Runtime decode mode must follow tensor residency, not compile-time cfg:
    // strict canonical WGPU stays fail-fast, while explicit host decode runs
    // still work when crate is built with runtime-model-wgpu enabled.
    let using_device_decode_inputs = runtime_decode_uses_device_inputs(
        shape_coords_wgpu.is_some(),
        tex_coords_wgpu.is_some(),
        shape_features_wgpu.is_some(),
        tex_features_wgpu.is_some(),
    );
    #[cfg(feature = "runtime-model-wgpu")]
    let decode_stage_fenced = !using_device_decode_inputs || runtime_stage_fence_enabled();
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let decode_stage_fenced = true;
    let shape_coord_rows = if !shape.coords.is_empty() {
        shape.coords.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            shape_coords_wgpu
                .as_ref()
                .map(|coords_t| coords_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let tex_coord_rows = if !tex.coords.is_empty() {
        tex.coords.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            tex_coords_wgpu
                .as_ref()
                .map(|coords_t| coords_t.dims()[0])
                .unwrap_or(shape_coord_rows)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            shape_coord_rows
        }
    };
    let shape_feature_rows = if !shape.features.is_empty() {
        shape.features.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            shape_features_wgpu
                .as_ref()
                .map(|rows_t| rows_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let tex_feature_rows = if !tex.features.is_empty() {
        tex.features.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            tex_features_wgpu
                .as_ref()
                .map(|rows_t| rows_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let count = shape_coord_rows
        .min(tex_coord_rows)
        .min(shape_feature_rows)
        .min(tex_feature_rows);
    if count == 0 {
        return Err("runtime decode received empty shape/tex latent rows".to_string());
    }
    if shape_decoder.out_channels() < 7 || tex_decoder.out_channels() < 6 {
        return Err(format!(
            "decoder channel mismatch: shape_out={} tex_out={}",
            shape_decoder.out_channels(),
            tex_decoder.out_channels()
        ));
    }
    trellis_stage_log!("burn_trellis: stage decode.shape_decoder begin (rows={count})");
    if stage_debug {
        trellis_stage_log!("burn_trellis: decode runtime begin (rows={count})");
    }
    let conv_telemetry_debug = runtime_decoder_conv_telemetry_enabled();
    let shape_rows_host = if shape.features.len() >= count {
        Some(&shape.features[..count])
    } else {
        None
    };
    let tex_rows_host = if tex.features.len() >= count {
        Some(&tex.features[..count])
    } else {
        None
    };
    let shape_coords_host = if !shape.coords.is_empty() {
        Some(&shape.coords[..count])
    } else {
        None
    };
    reset_decoder_conv_telemetry();
    reset_decoder_op_telemetry();
    reset_neighbor_build_stats();
    let shape_decode_start = Instant::now();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_decode_result = if using_device_decode_inputs {
        let shape_coords_wgpu_for_decode = if let Some(coords_t) = shape_coords_wgpu.as_ref() {
            let [rows, cols] = coords_t.dims();
            if cols != 4 {
                return Err(format!(
                    "runtime decode shape coord tensor must have 4 columns, got {}",
                    cols
                ));
            }
            if rows < count {
                return Err(format!(
                    "runtime decode shape coord tensor rows {} smaller than requested count {}",
                    rows, count
                ));
            }
            if rows == count {
                coords_t.clone()
            } else {
                coords_t.clone().slice([0..count, 0..4])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device shape coords; host decode fallback is disabled"
                    .to_string(),
            );
        };
        let shape_rows_wgpu_for_decode = if let Some(rows_t) = shape_features_wgpu.as_ref() {
            let [rows, cols] = rows_t.dims();
            if cols != 32 {
                return Err(format!(
                    "runtime decode shape feature tensor must have 32 columns, got {}",
                    cols
                ));
            }
            if rows == count {
                rows_t.clone()
            } else {
                rows_t.clone().slice([0..count, 0..32])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device shape rows; host row tensorization fallback is disabled"
                    .to_string(),
            );
        };
        let rows_t = shape_rows_wgpu_for_decode.clone();
        match shape_guide_subdivisions {
            Some(guides) => shape_decoder.decode_with_guidance_result_with_tensors(
                shape_coords_wgpu_for_decode.clone(),
                rows_t,
                guides,
            ),
            None => shape_decoder
                .decode_sparse_result_with_tensors(shape_coords_wgpu_for_decode.clone(), rows_t),
        }
    } else if let Some(coords_host) = shape_coords_host {
        let shape_rows = shape_rows_host.ok_or_else(|| {
            "runtime decode missing shape host rows for host coord decode path".to_string()
        })?;
        match shape_guide_subdivisions {
            Some(guides) => shape_decoder.decode_with_guidance_result(coords_host, shape_rows, guides),
            None => shape_decoder.decode_sparse_result(coords_host, shape_rows),
        }
    } else {
        return Err("runtime decode missing shape coords for host decode path".to_string());
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let shape_decode_result = if let Some(coords_host) = shape_coords_host {
        let shape_rows = shape_rows_host.ok_or_else(|| {
            "runtime decode missing shape host rows for host coord decode path".to_string()
        })?;
        match shape_guide_subdivisions {
            Some(guides) => {
                shape_decoder.decode_with_guidance_result(coords_host, shape_rows, guides)
            }
            None => shape_decoder.decode_sparse_result(coords_host, shape_rows),
        }
    } else {
        return Err(
            "runtime decode missing shape coords and crate was built without runtime-model-wgpu"
                .to_string(),
        );
    };
    let shape_decode_result = match shape_decode_result {
        Ok(decoded) => decoded,
        Err(err) => {
            trellis_stage_log!("burn_trellis: stage decode.shape_decoder error ({err})");
            return Err(format!("shape runtime decoder failed: {err}"));
        }
    };
    #[cfg(feature = "runtime-model-wgpu")]
    runtime_decode_stage_boundary_sync(
        "shape_decoder",
        using_device_decode_inputs && decode_stage_fenced,
    )?;
    let shape_decoder_ms = shape_decode_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: stage decode.shape_decoder complete ({shape_decoder_ms:.2} ms, subs={}, coords={})",
        shape_decode_result.subdivisions.len(),
        shape_decode_result.rows()
    );
    let shape_conv_telemetry = decoder_conv_telemetry();
    let shape_op_telemetry = decoder_op_telemetry();
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime shape-decoder complete ({:.2} ms, subs={}, coords={})",
            shape_decoder_ms,
            shape_decode_result.subdivisions.len(),
            shape_decode_result.rows()
        );
    }
    if stage_debug || conv_telemetry_debug {
        log_decoder_conv_telemetry("shape_decoder", &shape_conv_telemetry);
        log_decoder_op_telemetry("shape_decoder", &shape_op_telemetry);
        log_neighbor_build_stats("shape_decoder");
    }

    reset_decoder_conv_telemetry();
    reset_decoder_op_telemetry();
    reset_neighbor_build_stats();
    let tex_decode_start = Instant::now();
    trellis_stage_log!(
        "burn_trellis: stage decode.tex_decoder begin (rows={} guides={})",
        count,
        shape_decode_result.subdivisions.len()
    );
    let tex_coords_host = if !tex.coords.is_empty() {
        Some(&tex.coords[..count])
    } else {
        None
    };
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_decode_result = if using_device_decode_inputs {
        let tex_coords_wgpu_for_decode = if let Some(coords_t) = tex_coords_wgpu.as_ref() {
            let [rows, cols] = coords_t.dims();
            if cols != 4 {
                return Err(format!(
                    "runtime decode tex coord tensor must have 4 columns, got {}",
                    cols
                ));
            }
            if rows < count {
                return Err(format!(
                    "runtime decode tex coord tensor rows {} smaller than requested count {}",
                    rows, count
                ));
            }
            if rows == count {
                coords_t.clone()
            } else {
                coords_t.clone().slice([0..count, 0..4])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device tex coords; shape-coord fallback is disabled"
                    .to_string(),
            );
        };
        let tex_rows_wgpu_for_decode = if let Some(rows_t) = tex_features_wgpu.as_ref() {
            let [rows, cols] = rows_t.dims();
            if cols != 32 {
                return Err(format!(
                    "runtime decode tex feature tensor must have 32 columns, got {}",
                    cols
                ));
            }
            if rows == count {
                rows_t.clone()
            } else {
                rows_t.clone().slice([0..count, 0..32])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device tex rows; host row tensorization fallback is disabled"
                    .to_string(),
            );
        };
        let rows_t = tex_rows_wgpu_for_decode.clone();
        match tex_decoder.decode_with_guidance_result_with_tensors(
            tex_coords_wgpu_for_decode.clone(),
            rows_t,
            shape_decode_result.subdivisions.as_slice(),
        ) {
            Ok(decoded) => decoded,
            Err(err) => {
                trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
                return Err(format!("tex runtime decoder failed: {err}"));
            }
        }
    } else {
        match if let Some(coords_host) = tex_coords_host {
            let tex_rows = tex_rows_host.ok_or_else(|| {
                "runtime decode missing tex host rows for host coord decode path".to_string()
            })?;
            tex_decoder.decode_with_guidance_result(
                coords_host,
                tex_rows,
                shape_decode_result.subdivisions.as_slice(),
            )
        } else {
            return Err("runtime decode missing tex coords for host decode path".to_string());
        } {
            Ok(decoded) => decoded,
            Err(err) => {
                trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
                return Err(format!("tex runtime decoder failed: {err}"));
            }
        }
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let tex_decode_result = match if let Some(coords_host) = tex_coords_host {
        let tex_rows = tex_rows_host.ok_or_else(|| {
            "runtime decode missing tex host rows for host coord decode path".to_string()
        })?;
        tex_decoder.decode_with_guidance_result(
            coords_host,
            tex_rows,
            shape_decode_result.subdivisions.as_slice(),
        )
    } else {
        return Err(
            "runtime decode missing tex coords and crate was built without runtime-model-wgpu"
                .to_string(),
        );
    } {
        Ok(decoded) => decoded,
        Err(err) => {
            trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
            return Err(format!("tex runtime decoder failed: {err}"));
        }
    };
    #[cfg(feature = "runtime-model-wgpu")]
    runtime_decode_stage_boundary_sync(
        "tex_decoder",
        using_device_decode_inputs && decode_stage_fenced,
    )?;
    let tex_decoder_ms = tex_decode_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: stage decode.tex_decoder complete ({tex_decoder_ms:.2} ms, coords={})",
        tex_decode_result.rows()
    );
    let tex_conv_telemetry = decoder_conv_telemetry();
    let tex_op_telemetry = decoder_op_telemetry();
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime tex-decoder complete ({:.2} ms, coords={})",
            tex_decoder_ms,
            tex_decode_result.rows()
        );
    }
    if stage_debug || conv_telemetry_debug {
        log_decoder_conv_telemetry("tex_decoder", &tex_conv_telemetry);
        log_decoder_op_telemetry("tex_decoder", &tex_op_telemetry);
        log_neighbor_build_stats("tex_decoder");
    }

    if final_resolution == 0 {
        return Err(format!(
            "runtime decode received invalid final resolution for pipeline '{pipeline_type}'"
        ));
    }
    let shape_decoded_coords_host = shape_decode_result
        .coords_host("runtime decode shape coord stage-boundary materialization")
        .map_err(|err| format!("shape runtime decoder coord readback failed: {err}"))?;
    let shape_decoded_feats_host = shape_decode_result
        .feats_host("runtime decode shape feat stage-boundary materialization")
        .map_err(|err| format!("shape runtime decoder feat readback failed: {err}"))?;
    let shape_decoded = decode_fdg_outputs_from_host(
        shape_decoded_coords_host,
        shape_decoded_feats_host.as_slice(),
        shape_decode_result.out_channels,
        shape_decode_result.subdivisions.as_slice(),
        shape_decoder.voxel_margin(),
    )
    .map_err(|err| format!("shape runtime decoder output decode failed: {err}"))?;
    let tex_decoded_rows = tex_decode_result.rows();
    if tex_decoded_rows != shape_decoded.coords.len() {
        return Err(format!(
            "tex runtime decoder row mismatch: expected_rows={} actual_rows={}",
            shape_decoded.coords.len(),
            tex_decoded_rows
        ));
    }
    let tex_decoded_feats_host = tex_decode_result
        .feats_host("runtime decode tex feat stage-boundary materialization")
        .map_err(|err| format!("tex runtime decoder feat readback failed: {err}"))?;
    let tex_attrs = decode_tex_attrs_from_host(
        tex_decoded_feats_host.as_slice(),
        tex_decode_result.out_channels,
        Some(shape_decoded.coords.len()),
    )
    .map_err(|err| format!("tex runtime decoder output decode failed: {err}"))?;
    let shape_subdivisions = shape_decoded.subdivisions;
    let coords = shape_decoded.coords;
    let shape_vertices = shape_decoded.vertices;
    let shape_intersected = shape_decoded.intersected;
    let _shape_intersection_logits = shape_decoded.intersection_logits;
    let shape_quad_lerp = shape_decoded.quad_lerp;
    trellis_stage_log!(
        "burn_trellis: stage decode.attr_merge begin (rows={})",
        coords.len()
    );
    let attr_merge_start = Instant::now();
    let voxel_attrs = merge_voxel_attrs_for_decode(
        coords.as_slice(),
        coords.as_slice(),
        tex_attrs.as_slice(),
        parity_strict,
    )?;
    let attr_merge_ms = attr_merge_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!("burn_trellis: stage decode.attr_merge complete ({attr_merge_ms:.2} ms)");
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime attr merge complete ({:.2} ms)",
            attr_merge_ms
        );
    }

    let grid_size = [
        final_resolution as u32,
        final_resolution as u32,
        final_resolution as u32,
    ];
    trellis_stage_log!(
        "burn_trellis: stage decode.mesh_extract begin (rows={} final_res={})",
        coords.len(),
        final_resolution
    );
    let mesh_start = Instant::now();
    let mut vertices;
    let mut faces;
    (vertices, faces) = flexible_dual_grid_to_mesh(
        &coords,
        shape_vertices.as_slice(),
        shape_intersected.as_slice(),
        Some(shape_quad_lerp.as_slice()),
        grid_size,
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
    );
    let mesh_ms = mesh_start.elapsed().as_secs_f64() * 1000.0;
    let (
        sanitized_vertices,
        sanitized_faces,
        dropped_vertices,
        dropped_invalid_faces,
        dropped_degenerate_faces,
    ) = sanitize_mesh_geometry(vertices, faces);
    vertices = sanitized_vertices;
    faces = sanitized_faces;
    if dropped_vertices > 0 || dropped_invalid_faces > 0 || dropped_degenerate_faces > 0 {
        trellis_stage_log!(
            "burn_trellis: decode runtime mesh sanitized (dropped_vertices={} dropped_invalid_faces={} dropped_degenerate_faces={})",
            dropped_vertices,
            dropped_invalid_faces,
            dropped_degenerate_faces
        );
    }
    trellis_stage_log!(
        "burn_trellis: stage decode.mesh_extract complete ({mesh_ms:.2} ms, vertices={}, faces={})",
        vertices.len(),
        faces.len()
    );
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime mesh complete ({:.2} ms, vertices={}, faces={})",
            mesh_ms,
            vertices.len(),
            faces.len()
        );
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err("runtime decode produced empty mesh".to_string());
    }
    if let Some(target_faces) = target_faces.filter(|limit| *limit > 0) {
        #[cfg(not(target_arch = "wasm32"))]
        if faces.len() > target_faces {
            let before_faces = faces.len();
            decimate_mesh_for_face_budget(&mut vertices, &mut faces, target_faces)?;
            trellis_stage_log!(
                "burn_trellis: runtime decode pre-pbr decimation complete (target_faces={} from_faces={} to_faces={})",
                target_faces,
                before_faces,
                faces.len()
            );
            if vertices.is_empty() || faces.is_empty() {
                return Err(
                    "runtime decode pre-pbr decimation produced empty mesh".to_string(),
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        if faces.len() > target_faces {
            return Err(format!(
                "runtime decode target face budget is unsupported on wasm (target_faces={} faces={})",
                target_faces,
                faces.len()
            ));
        }
    }
    trellis_stage_log!("burn_trellis: stage decode.pbr begin");
    let pbr_start = Instant::now();
    #[cfg(feature = "runtime-model-wgpu")]
    // Canonical device decode should keep decode/PBR sampling on-device whenever
    // tensor-native decode inputs are active; CPU PBR sampling remains available
    // only for explicit host decode mode.
    let prefer_wgpu_sampling = using_device_decode_inputs;
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let prefer_wgpu_sampling = false;
    let (uvs, pbr_textures, pbr_debug) = bake_pbr_from_voxels_with_options(
        vertices.as_slice(),
        faces.as_slice(),
        coords.as_slice(),
        voxel_attrs.as_slice(),
        final_resolution as u32,
        capture_debug_artifacts,
        prefer_wgpu_sampling,
    )
    .map_err(|err| format!("runtime decode pbr bake failed: {err}"))?;
    let pbr_ms = pbr_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!("burn_trellis: stage decode.pbr complete ({pbr_ms:.2} ms)");
    if stage_debug {
        trellis_stage_log!("burn_trellis: decode runtime pbr complete ({pbr_ms:.2} ms)");
    }
    if parity_strict && (pbr_textures.is_none() || uvs.len() != vertices.len()) {
        return Err(format!(
            "parity strict mode: runtime decode pbr mismatch (textures_present={} uvs={} vertices={})",
            pbr_textures.is_some(),
            uvs.len(),
            vertices.len()
        ));
    }
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mesh = Mesh {
        vertices,
        faces,
        uvs,
        material,
        pbr_textures,
    };

    let shape_subs = if capture_debug_artifacts {
        shape_subdivisions
            .iter()
            .map(runtime_subdivision_to_sample)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let tex_spatial = spatial_shape_from_sparse_coords(coords.as_slice());

    Ok(DecodedLatentOutput {
        source: DecodeStageSource::Runtime,
        mesh,
        shape_subs,
        tex_voxels: DecodeTexVoxelSample {
            coords,
            feats: voxel_attrs,
            spatial_shape: tex_spatial,
        },
        pbr: pbr_debug,
        timings: DecodeRuntimeTimings {
            stage_fenced: decode_stage_fenced,
            shape_decoder_ms,
            tex_decoder_ms,
            attr_merge_ms,
            mesh_ms,
            pbr_ms,
            shape_conv_calls: shape_conv_telemetry.conv_calls,
            tex_conv_calls: tex_conv_telemetry.conv_calls,
            shape_wgpu_dispatches: shape_conv_telemetry.dispatches,
            tex_wgpu_dispatches: tex_conv_telemetry.dispatches,
            shape_wgpu_chunked_calls: shape_conv_telemetry.chunked_calls,
            tex_wgpu_chunked_calls: tex_conv_telemetry.chunked_calls,
            shape_wgpu_input_bytes: shape_conv_telemetry.input_bytes,
            tex_wgpu_input_bytes: tex_conv_telemetry.input_bytes,
            shape_wgpu_output_bytes: shape_conv_telemetry.output_bytes,
            tex_wgpu_output_bytes: tex_conv_telemetry.output_bytes,
            shape_wgpu_max_chunk_rows: shape_conv_telemetry.max_chunk_rows,
            tex_wgpu_max_chunk_rows: tex_conv_telemetry.max_chunk_rows,
        },
    })
}

#[cfg(feature = "runtime-model-wgpu")]
fn runtime_decode_uses_device_inputs(
    shape_coords_wgpu: bool,
    tex_coords_wgpu: bool,
    shape_features_wgpu: bool,
    tex_features_wgpu: bool,
) -> bool {
    shape_coords_wgpu || tex_coords_wgpu || shape_features_wgpu || tex_features_wgpu
}

#[cfg(feature = "runtime-model-wgpu")]
fn runtime_decode_stage_boundary_sync(stage: &str, enabled: bool) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }
    // WGPU dispatch is asynchronous; fence here so per-stage decode timing includes
    // real GPU execution instead of spilling completion into later decode stages.
    <SparseFlowWgpuBackend as Backend>::sync(&WgpuDevice::default())
        .map_err(|err| format!("runtime decode {stage} device sync failed: {err}"))
}

#[cfg(all(test, feature = "runtime-model"))]
mod runtime_decode_tests {
    use super::sanitize_mesh_geometry;
    #[cfg(not(target_arch = "wasm32"))]
    use super::decimate_mesh_for_face_budget;

    #[cfg(feature = "runtime-model-wgpu")]
    use super::runtime_decode_uses_device_inputs;

    #[test]
    fn sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [f32::NAN, 0.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let (vertices, faces, dropped_vertices, dropped_invalid_faces, dropped_degenerate_faces) =
            sanitize_mesh_geometry(vertices, faces);
        assert_eq!(dropped_vertices, 1);
        assert_eq!(dropped_invalid_faces, 1);
        assert_eq!(dropped_degenerate_faces, 0);
        assert_eq!(vertices.len(), 3);
        assert_eq!(faces, vec![[0, 1, 2]]);
    }

    #[test]
    fn sanitize_mesh_geometry_drops_degenerate_faces() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 1], [0, 1, 2]];
        let (_vertices, faces, dropped_vertices, dropped_invalid_faces, dropped_degenerate_faces) =
            sanitize_mesh_geometry(vertices, faces);
        assert_eq!(dropped_vertices, 0);
        assert_eq!(dropped_invalid_faces, 0);
        assert_eq!(dropped_degenerate_faces, 1);
        assert_eq!(faces, vec![[0, 1, 2]]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_decode_pre_pbr_decimation_respects_face_budget() {
        let side = 32usize;
        let mut vertices = Vec::with_capacity((side + 1) * (side + 1));
        for y in 0..=side {
            for x in 0..=side {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        let idx = |x: usize, y: usize| -> u32 { (y * (side + 1) + x) as u32 };
        let mut faces = Vec::with_capacity(side * side * 2);
        for y in 0..side {
            for x in 0..side {
                let i0 = idx(x, y);
                let i1 = idx(x + 1, y);
                let i2 = idx(x, y + 1);
                let i3 = idx(x + 1, y + 1);
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }
        let original_faces = faces.len();
        decimate_mesh_for_face_budget(&mut vertices, &mut faces, 200)
            .expect("runtime decode pre-pbr decimation should succeed");
        assert!(faces.len() <= 200, "faces={} > 200", faces.len());
        assert!(!faces.is_empty());
        assert!(faces.len() < original_faces);
        assert!(!vertices.is_empty());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn runtime_decode_device_gate_allows_host_only_inputs() {
        assert!(!runtime_decode_uses_device_inputs(false, false, false, false));
        assert!(runtime_decode_uses_device_inputs(true, false, false, false));
        assert!(runtime_decode_uses_device_inputs(false, true, false, false));
        assert!(runtime_decode_uses_device_inputs(false, false, true, false));
        assert!(runtime_decode_uses_device_inputs(false, false, false, true));
    }
}
