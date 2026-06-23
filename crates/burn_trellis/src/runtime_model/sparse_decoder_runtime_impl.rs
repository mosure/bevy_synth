impl SparseUnetDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
    ) -> Result<Self, String> {
        Self::load_from_stem_with_config(
            weights_root,
            image_large_root,
            model_stem,
            DecoderRuntimeConfig::default(),
        )
    }

    pub fn load_from_stem_with_config(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        runtime_config: DecoderRuntimeConfig,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = crate::virtual_fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        let parsed: DecoderConfigFile = serde_json::from_slice(&config_bytes).map_err(|err| {
            format!(
                "failed to parse sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        if parsed.args.model_channels.is_empty() {
            return Err(format!(
                "sparse decoder config '{}' has empty model_channels",
                config_path.display()
            ));
        }
        if parsed.args.num_blocks.len() != parsed.args.model_channels.len() {
            return Err(format!(
                "sparse decoder config '{}' has mismatched num_blocks/model_channels lengths",
                config_path.display()
            ));
        }

        let weight_path =
            resolve_model_weight_candidates(model_stem, weights_root, image_large_root)
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format!("unable to resolve decoder weights for stem '{model_stem}'")
                })?;

        let weight_backing = load_weight_backing(&weight_path)?;
        let safetensors = SafeTensors::deserialize(weight_backing.as_slice()).map_err(|err| {
            format!(
                "failed to deserialize sparse decoder weights '{}' as safetensors: {err}",
                weight_path.display()
            )
        })?;

        let out_channels = parsed.args.out_channels.unwrap_or_else(|| {
            if parsed.name == "FlexiDualGridVaeDecoder" {
                7
            } else {
                6
            }
        });
        #[cfg(feature = "runtime-model-wgpu")]
        let decoder_kind =
            DecoderRuntimeKind::from_model_metadata(parsed.name.as_str(), out_channels);

        let from_latent = load_linear(
            &safetensors,
            "from_latent.weight",
            "from_latent.bias",
            parsed.args.latent_channels,
            parsed.args.model_channels[0],
        )?;
        let output_layer = load_linear(
            &safetensors,
            "output_layer.weight",
            "output_layer.bias",
            *parsed
                .args
                .model_channels
                .last()
                .expect("checked non-empty model_channels"),
            out_channels,
        )?;

        let mut stages = Vec::with_capacity(parsed.args.num_blocks.len());
        for stage_idx in 0..parsed.args.num_blocks.len() {
            let stage_channels = parsed.args.model_channels[stage_idx];
            let mut convnext_blocks = Vec::with_capacity(parsed.args.num_blocks[stage_idx]);
            for block_idx in 0..parsed.args.num_blocks[stage_idx] {
                let prefix = format!("blocks.{stage_idx}.{block_idx}");
                convnext_blocks.push(ConvNeXtBlock {
                    conv: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv.weight").as_str(),
                        format!("{prefix}.conv.bias").as_str(),
                        stage_channels,
                        stage_channels,
                    )?,
                    norm_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.weight").as_str(),
                        stage_channels,
                    )?,
                    norm_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_0: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.0.weight").as_str(),
                        format!("{prefix}.mlp.0.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_2: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.2.weight").as_str(),
                        format!("{prefix}.mlp.2.bias").as_str(),
                        0,
                    )?,
                });
            }

            let upsample_block = if stage_idx + 1 < parsed.args.model_channels.len() {
                let up_idx = parsed.args.num_blocks[stage_idx];
                let prefix = format!("blocks.{stage_idx}.{up_idx}");
                let in_channels = parsed.args.model_channels[stage_idx];
                let out_channels = parsed.args.model_channels[stage_idx + 1];
                let conv1_out = out_channels
                    .checked_mul(8)
                    .ok_or_else(|| "conv1_out channels overflow".to_string())?;
                let to_subdiv = match parsed.args.pred_subdiv.unwrap_or(true) {
                    true => Some(load_linear(
                        &safetensors,
                        format!("{prefix}.to_subdiv.weight").as_str(),
                        format!("{prefix}.to_subdiv.bias").as_str(),
                        in_channels,
                        8,
                    )?),
                    false => None,
                };

                Some(C2SUpsampleBlock {
                    in_channels,
                    out_channels,
                    norm1_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.weight").as_str(),
                        in_channels,
                    )?,
                    norm1_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.bias").as_str(),
                        in_channels,
                    )?,
                    to_subdiv,
                    conv1: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv1.weight").as_str(),
                        format!("{prefix}.conv1.bias").as_str(),
                        in_channels,
                        conv1_out,
                    )?,
                    conv2: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv2.weight").as_str(),
                        format!("{prefix}.conv2.bias").as_str(),
                        out_channels,
                        out_channels,
                    )?,
                })
            } else {
                None
            };

            stages.push(DecoderStage {
                convnext_blocks,
                upsample_block,
            });
        }

        Ok(Self {
            out_channels,
            pred_subdiv: parsed.args.pred_subdiv.unwrap_or(true),
            voxel_margin: parsed.args.voxel_margin.unwrap_or(0.5),
            compute_fp16: parsed.args.use_fp16.unwrap_or(false) && !runtime_config.force_fp32,
            model_channels: parsed.args.model_channels,
            runtime_config,
            from_latent,
            output_layer,
            stages,
            conv_cache: Arc::new(Mutex::new(DecoderConvCache::default())),
            #[cfg(feature = "runtime-model-wgpu")]
            wgpu_context: create_wgpu_decoder_context(decoder_kind),
        })
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn pred_subdiv(&self) -> bool {
        self.pred_subdiv
    }

    pub fn voxel_margin(&self) -> f32 {
        self.voxel_margin
    }

    fn compute_fp16_enabled(&self) -> bool {
        self.compute_fp16 && runtime_model_sparse_decoder_conv_f16_enabled()
    }

    pub fn decode(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.decode_internal(
                Some(coords),
                #[cfg(feature = "runtime-model-wgpu")]
                None,
                Some(rows),
                #[cfg(feature = "runtime-model-wgpu")]
                None,
                guide_subdivisions,
            ))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (coords, rows, guide_subdivisions);
            Err("decoder sync decode is unsupported on wasm; use the async runtime path".to_string())
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn decode_with_tensors(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.decode_internal(
                None,
                Some(coords_wgpu),
                None,
                Some(rows_wgpu),
                guide_subdivisions,
            ))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (coords_wgpu, rows_wgpu, guide_subdivisions);
            Err(
                "decoder sync tensor decode is unsupported on wasm; use decode_with_tensors_async"
                    .to_string(),
            )
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub async fn decode_with_tensors_async(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        self.decode_internal(
            None,
            Some(coords_wgpu),
            None,
            Some(rows_wgpu),
            guide_subdivisions,
        )
        .await
    }

    // The decoder serializes access to the shared WGPU sparse-conv context while
    // issuing async GPU readback/selection work; do not silently unlock and race
    // the cache/context until the runtime has an async-aware ownership model.
    #[allow(clippy::await_holding_lock)]
    async fn decode_internal(
        &self,
        coords: Option<&[[u32; 4]]>,
        #[cfg(feature = "runtime-model-wgpu")] coords_wgpu: Option<
            Tensor<DefaultWgpuBackend, 2, Int>,
        >,
        rows: Option<&[[f32; 32]]>,
        #[cfg(feature = "runtime-model-wgpu")] rows_wgpu: Option<Tensor<DefaultWgpuBackend, 2>>,
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        let using_device_coords = coords.is_none();
        let using_device_rows = rows.is_none();
        #[cfg(not(feature = "runtime-model-wgpu"))]
        if using_device_coords {
            return Err(
                "decoder device coord tensor input requires runtime-model-wgpu feature".to_string(),
            );
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        if using_device_rows {
            return Err(
                "decoder device row tensor input requires runtime-model-wgpu feature".to_string(),
            );
        }

        let row_count = if let Some(rows_host) = rows {
            rows_host.len()
        } else {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                let rows_t = rows_wgpu.as_ref().ok_or_else(|| {
                    "decoder requires host rows or device row tensor".to_string()
                })?;
                let [rows_device, row_cols] = rows_t.dims();
                if row_cols != 32 {
                    return Err(format!(
                        "decoder device row tensor must have 32 columns, got {}",
                        row_cols
                    ));
                }
                rows_device
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                0
            }
        };

        let count = if let Some(coords_host) = coords {
            coords_host.len().min(row_count)
        } else {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                let coords_t = coords_wgpu.as_ref().ok_or_else(|| {
                    "decoder requires host coords or device coord tensor".to_string()
                })?;
                let [coord_rows, coord_cols] = coords_t.dims();
                if coord_cols != 4 {
                    return Err(format!(
                        "decoder device coord tensor must have 4 columns, got {}",
                        coord_cols
                    ));
                }
                coord_rows.min(row_count)
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                0
            }
        };
        if count == 0 {
            return Ok(SparseDecodeResult::empty(self.out_channels));
        }
        decoder_wasm_log(
            format!(
                "burn_trellis: decoder.shape internal begin rows={count} stages={} device_coords={} device_rows={}",
                self.stages.len(),
                using_device_coords,
                using_device_rows
            )
            .as_str(),
        );

        let mut state_coords = if let Some(coords_host) = coords {
            coords_host[..count].to_vec()
        } else {
            Vec::new()
        };
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_coords_wgpu: Option<Tensor<DefaultWgpuBackend, 2, Int>> =
            if let Some(coords_t) = coords_wgpu {
                let [coord_rows, coord_cols] = coords_t.dims();
                if coord_cols != 4 {
                    return Err(format!(
                        "decoder device coord tensor must have 4 columns, got {}",
                        coord_cols
                    ));
                }
                if coord_rows == count {
                    Some(coords_t)
                } else {
                    Some(coords_t.slice([0..count, 0..4]))
                }
            } else {
                None
            };
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_spatial_shape = if using_device_coords {
            [1, 1, 1]
        } else {
            spatial_shape_from_coords(state_coords.as_slice())
        };
        let mut state_feats = if let Some(rows_host) = rows {
            flatten_rows_32(&rows_host[..count])
        } else {
            Vec::new()
        };
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_feats_wgpu: Option<Tensor<DefaultWgpuBackend, 2>> = if let Some(rows_t) = rows_wgpu {
            let [rows_device, row_cols] = rows_t.dims();
            if row_cols != 32 {
                return Err(format!(
                    "decoder device row tensor must have 32 columns, got {}",
                    row_cols
                ));
            }
            Some(if rows_device == count {
                rows_t
            } else {
                rows_t.slice([0..count, 0..32])
            })
        } else {
            None
        };
        let mut conv_cache = self
            .conv_cache
            .lock()
            .map_err(|_| "decoder conv cache lock poisoned".to_string())?;
        #[cfg(feature = "runtime-model-wgpu")]
        let mut wgpu_context = if let Some(context) = self.wgpu_context.as_ref() {
            Some(
                context
                    .lock()
                    .map_err(|_| "decoder wgpu context lock poisoned".to_string())?,
            )
        } else {
            None
        };
        #[cfg(feature = "runtime-model-wgpu")]
        let canonical_wgpu =
            decoder_wgpu_device_math_enabled() && (!self.compute_fp16_enabled() || decoder_wgpu_device_math_allow_fp16());
        #[cfg(feature = "runtime-model-wgpu")]
        let mut final_output_preprojected_wgpu = false;
        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu && wgpu_context.is_none() {
            return Err(
                "burn_trellis: canonical wgpu decoder path requires wgpu sparse conv context; host completion fallback is disabled"
                    .to_string(),
            );
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu && (!using_device_coords || !using_device_rows) {
            return Err(
                "burn_trellis: canonical wgpu decoder path requires device-backed coords+rows (decode_with_tensors); host completion fallback is disabled"
                    .to_string(),
            );
        }
        #[cfg(feature = "runtime-model-wgpu")]
        let mut from_latent_completed_on_device = false;
        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu
            && let Some(context_gpu) = wgpu_context.as_deref_mut()
            && let Some(state_t) = state_feats_wgpu.take()
        {
            decoder_wasm_log(
                format!("burn_trellis: decoder.shape from_latent begin rows={count}").as_str(),
            );
            let mut state_t = linear_forward_wgpu(
                context_gpu,
                state_t,
                &self.from_latent,
                false,
                "from_latent(wgpu_math)",
            )?;
            if self.compute_fp16_enabled() {
                state_t = quantize_f16_tensor_wgpu(state_t);
            }
            state_feats_wgpu = Some(state_t);
            from_latent_completed_on_device = true;
            state_feats.clear();
            decoder_wasm_log(
                format!("burn_trellis: decoder.shape from_latent complete rows={count}").as_str(),
            );
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if !from_latent_completed_on_device {
            if using_device_rows {
                return Err(
                    "burn_trellis: decoder canonical tensor path requires from_latent execution on device; host row fallback is disabled"
                        .to_string(),
                );
            }
            state_feats = linear_forward(
                state_feats.as_slice(),
                count,
                &self.from_latent,
                "from_latent",
            )?;
            if self.compute_fp16_enabled() {
                quantize_f16_inplace(state_feats.as_mut_slice());
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            state_feats = linear_forward(
                state_feats.as_slice(),
                count,
                &self.from_latent,
                "from_latent",
            )?;
            if self.compute_fp16_enabled() {
                quantize_f16_inplace(state_feats.as_mut_slice());
            }
        }

        let mut subdivisions = Vec::new();
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let stage_channels = self.model_channels[stage_idx];
            decoder_wasm_log(
                format!(
                    "burn_trellis: decoder.shape stage {stage_idx} begin channels={stage_channels} convnext_blocks={} has_upsample={}",
                    stage.convnext_blocks.len(),
                    stage.upsample_block.is_some()
                )
                .as_str(),
            );
            #[allow(unused_mut)]
            // Stages with no ConvNeXt blocks are a device no-op and should not
            // fall into host-completion error guards in canonical WGPU mode.
            let mut convnext_device_complete = stage.convnext_blocks.is_empty();
            #[cfg(feature = "runtime-model-wgpu")]
            if canonical_wgpu
                && !stage.convnext_blocks.is_empty()
                && let Some(context_gpu) = wgpu_context.as_deref_mut()
            {
                let row_count = if let Some(coords_t) = state_coords_wgpu.as_ref() {
                    let [rows_device, cols_device] = coords_t.dims();
                    if cols_device != 4 {
                        return Err(format!(
                            "burn_trellis: wgpu convnext coords tensor has invalid columns {} (expected 4)",
                            cols_device
                        ));
                    }
                    rows_device
                } else {
                    state_coords.len()
                };
                let state_bytes = row_count
                    .saturating_mul(stage_channels)
                    .saturating_mul(core::mem::size_of::<f32>());
                if state_bytes <= decoder_wgpu_device_math_max_state_bytes() {
                    decoder_wasm_log(
                        format!(
                            "burn_trellis: decoder.shape stage {stage_idx} convnext begin rows={row_count} bytes={state_bytes}"
                        )
                        .as_str(),
                    );
                    let state_coords_t = if let Some(coords_t) = state_coords_wgpu.as_ref() {
                        let [rows_device, cols_device] = coords_t.dims();
                        if rows_device == row_count && cols_device == 4 {
                            coords_t.clone()
                        } else {
                            if using_device_coords {
                                return Err(format!(
                                    "burn_trellis: decoder stage {} device coord tensor mismatch on canonical tensor path: got=[{},{}] expected=[{},4]",
                                    stage_idx, rows_device, cols_device, row_count
                                ));
                            }
                            let rebuilt = coords_tensor_from_u32_slice(
                                state_coords.as_slice(),
                                &context_gpu.device,
                            )?;
                            state_coords_wgpu = Some(rebuilt.clone());
                            rebuilt
                        }
                    } else {
                        if using_device_coords {
                            return Err(format!(
                                "burn_trellis: decoder stage {} missing device coord tensor on canonical tensor path",
                                stage_idx
                            ));
                        }
                        let built = coords_tensor_from_u32_slice(
                            state_coords.as_slice(),
                            &context_gpu.device,
                        )?;
                        state_coords_wgpu = Some(built.clone());
                        built
                    };
                    let state_t = if let Some(state_t) = state_feats_wgpu.take() {
                        let [rows_device, channels_device] = state_t.dims();
                        if rows_device == row_count && channels_device == stage_channels {
                            state_t
                        } else {
                            Tensor::<DefaultWgpuBackend, 1>::from_floats(
                                state_feats.as_slice(),
                                &context_gpu.device,
                            )
                            .reshape([row_count, stage_channels])
                        }
                    } else {
                        Tensor::<DefaultWgpuBackend, 1>::from_floats(
                            state_feats.as_slice(),
                            &context_gpu.device,
                        )
                        .reshape([row_count, stage_channels])
                    };
                    let convnext_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            convnext_blocks_forward_wgpu_tensor(
                                context_gpu,
                                state_coords_t,
                                state_t,
                                stage_idx,
                                stage_channels,
                                stage.convnext_blocks.as_slice(),
                                self.compute_fp16_enabled(),
                            )
                        }));
                    match convnext_result {
                        Ok(Ok(next_state_feats)) => {
                            state_feats_wgpu = Some(next_state_feats);
                            convnext_device_complete = true;
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape stage {stage_idx} convnext complete rows={row_count}"
                                )
                                .as_str(),
                            );
                        }
                        Ok(Err(err)) => {
                            if err.contains("BufferTooBig") {
                                context_gpu.wgpu_failed = true;
                            }
                            return Err(format!(
                                "burn_trellis: wgpu convnext stage failed stage={} reason={err}",
                                stage_idx
                            ));
                        }
                        Err(payload) => {
                            context_gpu.wgpu_failed = true;
                            let panic_message = panic_payload_to_string(payload);
                            return Err(format!(
                                "burn_trellis: wgpu convnext stage panicked stage={} panic={panic_message}",
                                stage_idx
                            ));
                        }
                    }
                } else {
                    return Err(format!(
                        "burn_trellis: wgpu convnext stage={} state_bytes={} exceeds max_state_bytes={}; refusing cpu fallback",
                        stage_idx,
                        state_bytes,
                        decoder_wgpu_device_math_max_state_bytes()
                    ));
                }
            }
            if !convnext_device_complete {
                #[cfg(feature = "runtime-model-wgpu")]
                if canonical_wgpu && !stage.convnext_blocks.is_empty()
                {
                    return Err(format!(
                        "burn_trellis: decoder stage {} convnext did not complete on wgpu; host completion path is disabled",
                        stage_idx
                    ));
                }
                #[cfg(feature = "runtime-model-wgpu")]
                if canonical_wgpu && state_feats_wgpu.is_some() {
                    return Err(format!(
                        "burn_trellis: decoder stage {} convnext produced device tensors but canonical path forbids host completion",
                        stage_idx
                    ));
                }
                let allow_host_convnext_completion = {
                    #[cfg(feature = "runtime-model-wgpu")]
                    {
                        !canonical_wgpu
                    }
                    #[cfg(not(feature = "runtime-model-wgpu"))]
                    {
                        true
                    }
                };
                if allow_host_convnext_completion {
                    for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
                        let row_count = state_coords.len();
                        if row_count == 0 {
                            break;
                        }
                        let residual = state_feats.clone();
                        let mut h = sparse_subm_conv_forward(
                            state_coords.as_slice(),
                            state_feats.as_slice(),
                            &block.conv,
                            format!("stage {stage_idx} block {block_idx} conv").as_str(),
                            &mut conv_cache,
                            #[cfg(feature = "runtime-model-wgpu")]
                            wgpu_context.as_deref_mut(),
                        )?;
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        layer_norm_inplace(
                            h.as_mut_slice(),
                            row_count,
                            stage_channels,
                            Some(block.norm_weight.as_slice()),
                            Some(block.norm_bias.as_slice()),
                            LAYER_NORM32_EPS,
                            format!("stage {stage_idx} block {block_idx} layer_norm").as_str(),
                        )?;
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        h = linear_forward(
                            h.as_slice(),
                            row_count,
                            &block.mlp_0,
                            format!("stage {stage_idx} block {block_idx} mlp_0").as_str(),
                        )?;
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        silu_inplace(
                            h.as_mut_slice(),
                            format!("stage {stage_idx} block {block_idx} silu").as_str(),
                        );
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        h = linear_forward(
                            h.as_slice(),
                            row_count,
                            &block.mlp_2,
                            format!("stage {stage_idx} block {block_idx} mlp_2").as_str(),
                        )?;
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        add_inplace(
                            h.as_mut_slice(),
                            residual.as_slice(),
                            format!("stage {stage_idx} block {block_idx} residual_add").as_str(),
                        );
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(h.as_mut_slice());
                        }
                        state_feats = h;
                    }
                }
            }

            if let Some(up) = stage.upsample_block.as_ref() {
                #[allow(unused_mut)]
                let mut upsample_device_complete = false;
                #[cfg(feature = "runtime-model-wgpu")]
                if canonical_wgpu
                    && let Some(context_gpu) = wgpu_context.as_deref_mut()
                {
                    decoder_wasm_log(
                        format!("burn_trellis: decoder.shape stage {stage_idx} upsample begin")
                            .as_str(),
                    );
                    let parent_coords_t = if let Some(coords_t) = state_coords_wgpu.take() {
                        coords_t
                    } else {
                        if using_device_coords {
                            return Err(format!(
                                "burn_trellis: decoder stage {} upsample missing device coord tensor on canonical tensor path",
                                stage_idx
                            ));
                        }
                        coords_tensor_from_u32_slice(state_coords.as_slice(), &context_gpu.device)?
                    };
                    let [parent_rows, coord_cols] = parent_coords_t.dims();
                    if coord_cols != 4 {
                        return Err(format!(
                            "burn_trellis: decoder stage {} parent coords tensor must have 4 columns, got {}",
                            stage_idx, coord_cols
                        ));
                    }
                    if parent_rows == 0 {
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} upsample empty-parent"
                            )
                            .as_str(),
                        );
                        state_feats_wgpu = Some(Tensor::<DefaultWgpuBackend, 2>::zeros(
                            [0, up.out_channels],
                            &context_gpu.device,
                        ));
                        state_coords_wgpu = Some(parent_coords_t);
                        upsample_device_complete = true;
                    } else {
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} upsample parent rows={parent_rows} in_channels={}",
                                up.in_channels
                            )
                            .as_str(),
                        );
                        let parent_feats_t = if let Some(state_t) = state_feats_wgpu.take() {
                            let [rows_device, channels_device] = state_t.dims();
                            if rows_device != parent_rows || channels_device != up.in_channels {
                                return Err(format!(
                                    "burn_trellis: decoder stage {} parent tensor mismatch for upsample: state=[{},{}] expected=[{},{}]",
                                    stage_idx,
                                    rows_device,
                                    channels_device,
                                    parent_rows,
                                    up.in_channels
                                ));
                            }
                            state_t
                        } else {
                            if using_device_coords {
                                return Err(format!(
                                    "burn_trellis: decoder stage {} upsample missing device feature tensor on canonical tensor path",
                                    stage_idx
                                ));
                            }
                            let expected = parent_rows.saturating_mul(up.in_channels);
                            if state_feats.len() != expected {
                                return Err(format!(
                                    "burn_trellis: decoder stage {} parent host tensor mismatch for upsample: len={} expected={}",
                                    stage_idx,
                                    state_feats.len(),
                                    expected
                                ));
                            }
                            Tensor::<DefaultWgpuBackend, 1>::from_floats(
                                state_feats.as_slice(),
                                &context_gpu.device,
                            )
                            .reshape([parent_rows, up.in_channels])
                        };

                        let guide = guide_subdivisions.and_then(|levels| levels.get(stage_idx));
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} subdivision logits begin guide={}",
                                guide.is_some()
                            )
                            .as_str(),
                        );
                        let mut subdiv_logits_t = if let Some(guide) = guide {
                            guide_subdivision_logits_tensor_for_parent_wgpu(
                                parent_coords_t.clone(),
                                guide,
                                &context_gpu.device,
                                stage_idx,
                            )?
                        } else if let Some(to_subdiv) = up.to_subdiv.as_ref() {
                            linear_forward_wgpu(
                                context_gpu,
                                parent_feats_t.clone(),
                                to_subdiv,
                                self.compute_fp16_enabled(),
                                format!("stage {stage_idx} to_subdiv(wgpu_math)").as_str(),
                            )?
                        } else {
                            return Err(format!(
                                "decoder stage {stage_idx} requires guide_subdivisions but none were provided"
                            ));
                        };
                        if self.compute_fp16_enabled() && guide.is_none() {
                            subdiv_logits_t = quantize_f16_tensor_wgpu(subdiv_logits_t);
                        }
                        if self.runtime_config.center_subdivision_logits {
                            let mean = subdiv_logits_t.clone().mean_dim(1);
                            subdiv_logits_t = subdiv_logits_t.sub(mean);
                        }
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} active-index begin"
                            )
                            .as_str(),
                        );
                        let active_indices_t = if let Some(guide) = guide {
                            guide_subdivision_active_indices_tensor_for_parent_wgpu(
                                parent_rows,
                                guide,
                                stage_idx,
                            )?
                        } else {
                            subdivision_active_indices_wgpu_async(
                                subdiv_logits_t.clone(),
                                false,
                                &self.runtime_config,
                            )
                            .await?
                        };
                        let [active_rows, active_cols] = active_indices_t.dims();
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} active-index complete rows={active_rows} cols={active_cols}"
                            )
                            .as_str(),
                        );
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} child-expand begin"
                            )
                            .as_str(),
                        );
                        let (child_coords_t, linear_idx_t) = if let Some(guide) = guide {
                            guide_subdivision_child_tensors_for_parent_wgpu(
                                parent_rows,
                                guide,
                                stage_idx,
                            )?
                        } else {
                            expand_subdivision_coords_and_linear_indices_wgpu(
                                parent_coords_t.clone(),
                                active_indices_t.clone(),
                            )?
                        };
                        if self.pred_subdiv {
                            subdivisions.push(
                                SparseSubdivisionLogits::from_device_tensors_with_active_and_children(
                                    state_spatial_shape,
                                    parent_coords_t.clone(),
                                    subdiv_logits_t.clone(),
                                    Some(active_indices_t.clone()),
                                    Some((child_coords_t.clone(), linear_idx_t.clone())),
                                )?,
                            );
                        }
                        let [child_rows, child_coord_cols] = child_coords_t.dims();
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} child-expand complete rows={child_rows}"
                            )
                            .as_str(),
                        );
                        if child_coord_cols != 4 {
                            return Err(format!(
                                "burn_trellis: decoder stage {} child coords tensor must have 4 columns, got {}",
                                stage_idx, child_coord_cols
                            ));
                        }
                        let child_state_bytes = child_rows
                            .saturating_mul(up.out_channels)
                            .saturating_mul(core::mem::size_of::<f32>());
                        if child_state_bytes > decoder_wgpu_device_math_max_state_bytes() {
                            if decoder_can_stream_final_stage_output(
                                self.stages.as_slice(),
                                stage_idx,
                            ) {
                                decoder_wasm_log(
                                    format!(
                                        "burn_trellis: decoder.shape stage {stage_idx} streamed-output begin child_rows={child_rows} bytes={child_state_bytes}"
                                    )
                                    .as_str(),
                                );
                                let streamed_output_t = upsample_stage_to_output_chunks_wgpu(
                                    context_gpu,
                                    stage_idx,
                                    up,
                                    parent_feats_t,
                                    parent_coords_t.clone(),
                                    child_coords_t.clone(),
                                    active_indices_t.clone(),
                                    linear_idx_t,
                                    self.compute_fp16_enabled(),
                                    &self.output_layer,
                                )
                                .await?;
                                state_feats_wgpu = Some(streamed_output_t);
                                state_coords_wgpu = Some(child_coords_t);
                                state_spatial_shape = if child_rows == 0 {
                                    [1, 1, 1]
                                } else {
                                    [
                                        state_spatial_shape[0].saturating_mul(2).max(1),
                                        state_spatial_shape[1].saturating_mul(2).max(1),
                                        state_spatial_shape[2].saturating_mul(2).max(1),
                                    ]
                                };
                                final_output_preprojected_wgpu = true;
                                decoder_wasm_log(
                                    format!(
                                        "burn_trellis: decoder.shape stage {stage_idx} streamed-output complete child_rows={child_rows}"
                                    )
                                    .as_str(),
                                );
                                continue;
                            }
                            return Err(format!(
                                "burn_trellis: wgpu upsample stage={} child_state_bytes={} exceeds max_state_bytes={}; refusing cpu fallback",
                                stage_idx,
                                child_state_bytes,
                                decoder_wgpu_device_math_max_state_bytes()
                            ));
                        }

                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} norm1_silu begin"
                            )
                            .as_str(),
                        );
                        let h_norm_t = layer_norm_silu_wgpu_with_fp16_fences(
                            context_gpu,
                            parent_feats_t.clone(),
                            Some(up.norm1_weight.as_slice()),
                            Some(up.norm1_bias.as_slice()),
                            LAYER_NORM32_EPS,
                            self.compute_fp16_enabled(),
                            format!("stage {stage_idx} up norm1_silu(wgpu_math)").as_str(),
                        )?;
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} conv1-select begin child_rows={child_rows}"
                            )
                            .as_str(),
                        );
                        let h_up_t = if child_rows == 0 {
                            Tensor::<DefaultWgpuBackend, 2>::zeros(
                                [0, up.out_channels],
                                &context_gpu.device,
                            )
                        } else {
                            let mut h_up_t = upsample_conv1_select_children_wgpu(
                                context_gpu,
                                up,
                                stage_idx,
                                h_norm_t,
                                parent_coords_t.clone(),
                                active_indices_t.clone(),
                                linear_idx_t.clone(),
                                child_rows,
                                self.compute_fp16_enabled(),
                            )
                            .await?;
                            if self.compute_fp16_enabled() {
                                h_up_t = quantize_f16_tensor_wgpu(h_up_t);
                            }
                            h_up_t
                        };
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} conv1-select complete child_rows={child_rows}"
                            )
                            .as_str(),
                        );

                        let skip_in_channels = up.in_channels / 8;
                        if skip_in_channels == 0 || up.out_channels % skip_in_channels != 0 {
                            return Err(format!(
                                "decoder stage {stage_idx} invalid skip channel ratio (in={}, out={})",
                                up.in_channels, up.out_channels
                            ));
                        }
                        let repeat_factor = up.out_channels / skip_in_channels;
                        let parent_feats_for_parity_t = parent_feats_t.clone();
                        let parent_flat = parent_feats_t.reshape([
                            parent_rows.checked_mul(8).ok_or_else(|| {
                                "decoder upsample parent_rows*8 overflow".to_string()
                            })?,
                            skip_in_channels,
                        ]);
                        let x_up_t = if child_rows == 0 {
                            Tensor::<DefaultWgpuBackend, 2>::zeros(
                                [0, skip_in_channels],
                                &context_gpu.device,
                            )
                        } else {
                            parent_flat.select(0, linear_idx_t)
                        };
                        let next_state_feats = if child_rows == 0 {
                            Tensor::<DefaultWgpuBackend, 2>::zeros(
                                [0, up.out_channels],
                                &context_gpu.device,
                            )
                        } else {
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape stage {stage_idx} conv2 norm begin child_rows={child_rows}"
                                )
                                .as_str(),
                            );
                            let h_up_t = layer_norm_silu_wgpu_with_fp16_fences(
                                context_gpu,
                                h_up_t,
                                None,
                                None,
                                LAYER_NORM32_EPS,
                                self.compute_fp16_enabled(),
                                format!("stage {stage_idx} up layer_norm_silu(wgpu_math)")
                                    .as_str(),
                            )?;
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape stage {stage_idx} conv2 sparse-conv begin child_rows={child_rows}"
                                )
                                .as_str(),
                            );
                            let conv2_config = flex_config_for_layer(&up.conv2);
                            let mut h_t = context_gpu.forward_with_coords_tensor_device(
                                &conv2_config,
                                &up.conv2,
                                h_up_t,
                                self.compute_fp16_enabled(),
                                format!("stage {stage_idx} up conv2(wgpu_math)").as_str(),
                                child_coords_t.clone(),
                            )?;
                            if self.compute_fp16_enabled() {
                                h_t = quantize_f16_tensor_wgpu(h_t);
                            }
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape stage {stage_idx} conv2 sparse-conv complete child_rows={child_rows}"
                                )
                                .as_str(),
                            );
                            // Preserve TRELLIS channel semantics without materializing a full
                            // repeated skip tensor: add skip channels via grouped broadcast.
                            let h_grouped = h_t.reshape([
                                child_rows,
                                skip_in_channels,
                                repeat_factor,
                            ]);
                            let skip_grouped = x_up_t.unsqueeze_dim::<3>(2);
                            let mut next_state_t = h_grouped
                                .add(skip_grouped)
                                .reshape([child_rows, up.out_channels]);
                            if self.compute_fp16_enabled() {
                                next_state_t = quantize_f16_tensor_wgpu(next_state_t);
                            }
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape stage {stage_idx} conv2+skip complete child_rows={child_rows}"
                                )
                                .as_str(),
                            );
                            next_state_t
                        };
                        decoder_wgpu_upsample_parity_check(
                            stage_idx,
                            up,
                            parent_feats_for_parity_t,
                            parent_coords_t.clone(),
                            active_indices_t.clone(),
                            next_state_feats.clone(),
                        )?;
                        state_feats_wgpu = Some(next_state_feats);
                        state_coords_wgpu = Some(child_coords_t);
                        state_spatial_shape = if child_rows == 0 {
                            [1, 1, 1]
                        } else {
                            [
                                state_spatial_shape[0].saturating_mul(2).max(1),
                                state_spatial_shape[1].saturating_mul(2).max(1),
                                state_spatial_shape[2].saturating_mul(2).max(1),
                            ]
                        };
                        upsample_device_complete = true;
                        decoder_wasm_log(
                            format!(
                                "burn_trellis: decoder.shape stage {stage_idx} upsample complete child_rows={child_rows}"
                            )
                            .as_str(),
                        );
                    }
                }

                if !upsample_device_complete {
                    #[cfg(feature = "runtime-model-wgpu")]
                    if canonical_wgpu {
                        if state_feats_wgpu.is_some() || state_coords_wgpu.is_some() {
                            return Err(format!(
                                "burn_trellis: decoder stage {} upsample produced device tensors but canonical path forbids host completion",
                                stage_idx
                            ));
                        }
                        return Err(format!(
                            "burn_trellis: decoder stage {} upsample did not complete on wgpu; host completion path is disabled",
                            stage_idx
                        ));
                    }
                    let parent_coords = state_coords.clone();
                    let parent_feats = state_feats.clone();
                    let parent_rows = parent_coords.len();
                    if parent_rows == 0 {
                        continue;
                    }

                    let subdiv_logits = if let Some(guide) =
                        guide_subdivisions.and_then(|levels| levels.get(stage_idx))
                    {
                        map_guide_subdivision_logits(parent_coords.as_slice(), guide)?
                    } else if let Some(to_subdiv) = up.to_subdiv.as_ref() {
                        let mut logits = linear_forward(
                            parent_feats.as_slice(),
                            parent_rows,
                            to_subdiv,
                            format!("stage {stage_idx} to_subdiv").as_str(),
                        )?;
                        if self.compute_fp16_enabled() {
                            quantize_f16_inplace(logits.as_mut_slice());
                        }
                        if self.runtime_config.center_subdivision_logits {
                            row_center_logits(
                                logits.as_mut_slice(),
                                parent_rows,
                                format!("stage {stage_idx} to_subdiv center_logits").as_str(),
                            );
                        }
                        logits
                    } else {
                        return Err(format!(
                            "decoder stage {stage_idx} requires guide_subdivisions but none were provided"
                        ));
                    };

                    let subdivision_mask = logits_to_mask(
                        subdiv_logits.as_slice(),
                        parent_rows,
                        false,
                        &self.runtime_config,
                    )?;
                    if self.pred_subdiv {
                        subdivisions.push(SparseSubdivisionLogits::from_host(
                            spatial_shape_from_coords(parent_coords.as_slice()),
                            parent_coords.clone(),
                            subdiv_logits.clone(),
                        )?);
                    }

                    let mut h_norm = parent_feats.clone();
                    layer_norm_inplace(
                        h_norm.as_mut_slice(),
                        parent_rows,
                        up.in_channels,
                        Some(up.norm1_weight.as_slice()),
                        Some(up.norm1_bias.as_slice()),
                        LAYER_NORM32_EPS,
                        format!("stage {stage_idx} up norm1").as_str(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h_norm.as_mut_slice());
                    }
                    silu_inplace(
                        h_norm.as_mut_slice(),
                        format!("stage {stage_idx} up silu1").as_str(),
                    );
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h_norm.as_mut_slice());
                    }
                    let h_conv1 = sparse_subm_conv_forward(
                        parent_coords.as_slice(),
                        h_norm.as_slice(),
                        &up.conv1,
                        format!("stage {stage_idx} up conv1").as_str(),
                        &mut conv_cache,
                        #[cfg(feature = "runtime-model-wgpu")]
                        wgpu_context.as_deref_mut(),
                    )?;
                    let mut h_conv1 = h_conv1;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h_conv1.as_mut_slice());
                    }
                    let (child_coords, mut h_up) = channel2spatial(
                        parent_coords.as_slice(),
                        h_conv1.as_slice(),
                        up.out_channels
                            .checked_mul(8)
                            .ok_or_else(|| "up.out_channels * 8 overflow".to_string())?,
                        subdivision_mask.as_slice(),
                    )?;
                    let (child_coords_skip, x_up) = channel2spatial(
                        parent_coords.as_slice(),
                        parent_feats.as_slice(),
                        up.in_channels,
                        subdivision_mask.as_slice(),
                    )?;
                    if child_coords != child_coords_skip {
                        return Err(format!(
                            "decoder stage {stage_idx} channel2spatial coord mismatch between conv and skip branches"
                        ));
                    }

                    let skip_in_channels = up.in_channels / 8;
                    if skip_in_channels == 0 || up.out_channels % skip_in_channels != 0 {
                        return Err(format!(
                            "decoder stage {stage_idx} invalid skip channel ratio (in={}, out={})",
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
                        format!("stage {stage_idx} up layer_norm").as_str(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h_up.as_mut_slice());
                    }
                    silu_inplace(
                        h_up.as_mut_slice(),
                        format!("stage {stage_idx} up silu2").as_str(),
                    );
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h_up.as_mut_slice());
                    }
                    let mut h = sparse_subm_conv_forward(
                        child_coords.as_slice(),
                        h_up.as_slice(),
                        &up.conv2,
                        format!("stage {stage_idx} up conv2").as_str(),
                        &mut conv_cache,
                        #[cfg(feature = "runtime-model-wgpu")]
                        wgpu_context.as_deref_mut(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    add_inplace(
                        h.as_mut_slice(),
                        skip.as_slice(),
                        format!("stage {stage_idx} up skip_add").as_str(),
                    );
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    state_feats = h;
                    #[cfg(feature = "runtime-model-wgpu")]
                    {
                        state_feats_wgpu = None;
                        state_coords_wgpu = None;
                    }
                    state_coords = child_coords;
                    #[cfg(feature = "runtime-model-wgpu")]
                    {
                        state_spatial_shape = spatial_shape_from_coords(state_coords.as_slice());
                    }
                } else {
                    state_feats.clear();
                    state_coords.clear();
                    #[cfg(feature = "runtime-model-wgpu")]
                    {
                        state_spatial_shape = [1, 1, 1];
                    }
                }
            }
        }

        #[cfg(feature = "runtime-model-wgpu")]
        let final_coords_tensor = state_coords_wgpu.take();
        decoder_wasm_log("burn_trellis: decoder.shape output-layer prepare");
        let rows_final = {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                if let Some(coords_t) = final_coords_tensor.as_ref() {
                    coords_t.dims()[0]
                } else {
                    state_coords.len()
                }
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                state_coords.len()
            }
        };
        let final_channels = *self
            .model_channels
            .last()
            .expect("checked non-empty model_channels");
        #[cfg(feature = "runtime-model-wgpu")]
        let mut final_feats_tensor: Option<Tensor<DefaultWgpuBackend, 2>> = None;
        let state_feats = {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                if final_output_preprojected_wgpu {
                    let Some(output_t) = state_feats_wgpu.take() else {
                        return Err(
                            "burn_trellis: streamed final wgpu decode path did not produce output tensor"
                                .to_string(),
                        );
                    };
                    final_feats_tensor = Some(output_t);
                    Vec::new()
                } else if let Some(state_t) = state_feats_wgpu.take() {
                    if canonical_wgpu {
                        if let Some(context_gpu) = wgpu_context.as_deref_mut() {
                            decoder_wasm_log(
                                format!(
                                    "burn_trellis: decoder.shape output-layer begin rows={rows_final} channels={final_channels}"
                                )
                                .as_str(),
                            );
                            let output_layer_start = crate::time::Instant::now();
                            if decoder_conv_debug_enabled() {
                                eprintln!(
                                    "burn_trellis: wgpu output_layer begin rows={} channels={}",
                                    rows_final, final_channels
                                );
                            }
                            let wgpu_result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                    || -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
                                        let state_t = layer_norm_wgpu(
                                            context_gpu,
                                            state_t.clone(),
                                            None,
                                            None,
                                            F_LAYER_NORM_EPS,
                                            "output_layer_norm(wgpu_math)",
                                        )?;
                                        let state_t = linear_forward_wgpu(
                                            context_gpu,
                                            state_t,
                                            &self.output_layer,
                                            false,
                                            "output_layer(wgpu_math)",
                                        )?;
                                        Ok(state_t)
                                    },
                                ));
                            match wgpu_result {
                                Ok(Ok(output_t)) => {
                                    let elapsed_ms =
                                        output_layer_start.elapsed().as_secs_f64() * 1000.0;
                                    if decoder_conv_debug_enabled() {
                                        eprintln!(
                                            "burn_trellis: wgpu output_layer complete rows={} channels={} elapsed_ms={:.2}",
                                            rows_final, final_channels, elapsed_ms
                                        );
                                    }
                                    final_feats_tensor = Some(output_t);
                                    decoder_wasm_log(
                                        format!(
                                            "burn_trellis: decoder.shape output-layer complete rows={rows_final}"
                                        )
                                        .as_str(),
                                    );
                                    Vec::new()
                                }
                                Ok(Err(err)) => {
                                    if err.contains("BufferTooBig") {
                                        context_gpu.wgpu_failed = true;
                                    }
                                    return Err(format!(
                                        "burn_trellis: wgpu output layer failed reason={err}"
                                    ));
                                }
                                Err(payload) => {
                                    context_gpu.wgpu_failed = true;
                                    let panic_message = panic_payload_to_string(payload);
                                    return Err(format!(
                                        "burn_trellis: wgpu output layer panicked panic={panic_message}"
                                    ));
                                }
                            }
                        } else {
                            return Err(
                                "burn_trellis: wgpu output layer tensor present without wgpu context"
                                    .to_string(),
                            );
                        }
                    } else {
                        let _ = state_t;
                        return Err(
                            "burn_trellis: wgpu output layer tensor requires device-math path; refusing host fallback"
                                .to_string(),
                        );
                    }
                } else {
                    if canonical_wgpu {
                        return Err(
                            "burn_trellis: decoder output layer missing device feature tensor on canonical wgpu path"
                                .to_string(),
                        );
                    }
                    if using_device_coords {
                        return Err(
                            "burn_trellis: decoder output layer missing device feature tensor on canonical tensor path"
                                .to_string(),
                        );
                    }
                    layer_norm_inplace(
                        state_feats.as_mut_slice(),
                        rows_final,
                        final_channels,
                        None,
                        None,
                        F_LAYER_NORM_EPS,
                        "output_layer_norm",
                    )?;
                    linear_forward(
                        state_feats.as_slice(),
                        rows_final,
                        &self.output_layer,
                        "output_layer",
                    )?
                }
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                layer_norm_inplace(
                    state_feats.as_mut_slice(),
                    rows_final,
                    final_channels,
                    None,
                    None,
                    F_LAYER_NORM_EPS,
                    "output_layer_norm",
                )?;
                linear_forward(
                    state_feats.as_slice(),
                    rows_final,
                    &self.output_layer,
                    "output_layer",
                )?
            }
        };

        #[cfg(feature = "runtime-model-wgpu")]
        if decoder_wgpu_clear_cache_after_decode()
            && let Some(context) = wgpu_context.as_deref_mut()
        {
            context.clear_caches();
        }

        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu && (final_coords_tensor.is_none() || final_feats_tensor.is_none()) {
            return Err(
                "burn_trellis: canonical wgpu decode must produce device-backed coords and features"
                    .to_string(),
            );
        }

        #[cfg(feature = "runtime-model-wgpu")]
        let (coords_host, feats_host) =
            if canonical_wgpu || final_coords_tensor.is_some() || final_feats_tensor.is_some() {
                (None, None)
            } else {
                (Some(state_coords), Some(state_feats))
            };
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let (coords_host, feats_host) = (Some(state_coords), Some(state_feats));

        Ok(SparseDecodeResult {
            coords: coords_host,
            feats: feats_host,
            out_channels: self.out_channels,
            subdivisions,
            #[cfg(feature = "runtime-model-wgpu")]
            coords_tensor: final_coords_tensor,
            #[cfg(feature = "runtime-model-wgpu")]
            feats_tensor: final_feats_tensor,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn upsample_coords_sparse(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        let count = coords.len().min(rows.len());
        let current_coords = coords[..count].to_vec();
        if upsample_times == 0 || current_coords.is_empty() {
            return Ok(SparseUpsampledCoords::from_host(current_coords));
        }
        #[cfg(feature = "runtime-model-wgpu")]
        {
            // Keep this host-input API as a narrow compatibility surface, but force all
            // actual upsample/decode semantics through the tensor-native canonical path.
            let device = {
                let context = self.wgpu_context.as_ref().ok_or_else(|| {
                    "burn_trellis: wgpu upsample requires decoder wgpu context; host completion fallback is disabled"
                        .to_string()
                })?;
                let guard = context
                    .lock()
                    .map_err(|_| "decoder wgpu context lock poisoned".to_string())?;
                guard.device.clone()
            };
            let coords_t = coords_tensor_from_u32_slice(&coords[..count], &device)?;
            let rows_flat = flatten_rows_32(&rows[..count]);
            let rows_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(rows_flat.as_slice(), &device)
                .reshape([count, 32]);
            return self.upsample_coords_result_with_tensors(coords_t, rows_t, upsample_times);
        }

        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let mut current_coords = current_coords;
            let decoded = self.decode(coords, rows, None)?;
            if decoded.subdivisions.len() < upsample_times {
                return Err(format!(
                    "decoder upsample requested {} levels but only {} subdivision levels are available",
                    upsample_times,
                    decoded.subdivisions.len()
                ));
            }
            for stage_idx in 0..upsample_times {
                let subdivision =
                    decoded.subdivisions.get(stage_idx).ok_or_else(|| {
                        format!("decoder upsample missing subdivision level {stage_idx}")
                    })?;
                let sub_coords = subdivision.coords_host(
                    format!("decoder upsample stage {stage_idx} coord materialization").as_str(),
                )?;
                if sub_coords != current_coords {
                    return Err(format!(
                        "decoder upsample stage {} coordinate mismatch (expected_rows={}, got_rows={})",
                        stage_idx,
                        current_coords.len(),
                        sub_coords.len()
                    ));
                }
                let sub_logits = subdivision.logits_host(
                    format!("decoder upsample stage {stage_idx} logits materialization").as_str(),
                )?;
                let mask = logits_to_mask(
                    sub_logits.as_slice(),
                    sub_coords.len(),
                    false,
                    &self.runtime_config,
                )?;
                current_coords = subdivide_coords_from_mask(sub_coords.as_slice(), mask.as_slice())?;
                if current_coords.is_empty() {
                    break;
                }
            }
            Ok(SparseUpsampledCoords::from_host(current_coords))
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn upsample_coords_result(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        self.upsample_coords_sparse(coords, rows, upsample_times)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn upsample_coords_result_with_tensors(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (coords_wgpu, rows_wgpu, upsample_times);
            return Err(
                "decoder sync tensor upsample is unsupported on wasm; use upsample_coords_result_with_tensors_async"
                    .to_string(),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.upsample_coords_result_with_tensors_async(
                coords_wgpu,
                rows_wgpu,
                upsample_times,
            ))
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn upsample_coords_result_with_tensors_async(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        let [coord_rows, coord_cols] = coords_wgpu.dims();
        if coord_cols != 4 {
            return Err(format!(
                "decoder upsample coord tensor must have 4 columns, got {}",
                coord_cols
            ));
        }
        let [row_rows, row_cols] = rows_wgpu.dims();
        if row_cols != 32 {
            return Err(format!(
                "decoder upsample row tensor must have 32 columns, got {}",
                row_cols
            ));
        }
        let count = coord_rows.min(row_rows);
        let mut current_coords_t = if coord_rows == count {
            coords_wgpu
        } else {
            coords_wgpu.slice([0..count, 0..4])
        };
        let rows_wgpu = if row_rows == count {
            rows_wgpu
        } else {
            rows_wgpu.slice([0..count, 0..32])
        };
        if upsample_times == 0 || count == 0 {
            return Ok(SparseUpsampledCoords::from_wgpu_tensor(current_coords_t));
        }

        let decoded = self
            .decode_with_tensors_async(current_coords_t.clone(), rows_wgpu, None)
            .await?;
        if decoded.subdivisions.len() < upsample_times {
            return Err(format!(
                "decoder upsample requested {} levels but only {} subdivision levels are available",
                upsample_times,
                decoded.subdivisions.len()
            ));
        }

        for stage_idx in 0..upsample_times {
            let subdivision = decoded
                .subdivisions
                .get(stage_idx)
                .ok_or_else(|| format!("decoder upsample missing subdivision level {stage_idx}"))?;
            let Some((sub_coords_t, _sub_logits_t)) = subdivision.device_tensors() else {
                return Err(format!(
                    "decoder upsample stage {stage_idx} requires tensor-native subdivision tensors on wgpu path; host completion fallback is disabled"
                ));
            };
            let [sub_rows, sub_cols] = sub_coords_t.dims();
            if sub_cols != 4 {
                return Err(format!(
                    "decoder upsample stage {} device coord tensor must have 4 columns, got {}",
                    stage_idx, sub_cols
                ));
            }
            let [curr_rows, curr_cols] = current_coords_t.dims();
            if curr_cols != 4 || curr_rows != sub_rows {
                return Err(format!(
                    "decoder upsample stage {} coord tensor mismatch: current=[{},{}] subdivision=[{},{}]",
                    stage_idx, curr_rows, curr_cols, sub_rows, sub_cols
                ));
            }
            let (child_coords_t, _child_linear_idx_t) = if let Some(child_tensors) =
                subdivision.child_tensors()
            {
                child_tensors
            } else {
                return Err(format!(
                    "decoder upsample stage {stage_idx} requires tensor-native child subdivision tensors on wgpu path; expansion fallback is disabled"
                ));
            };
            let [child_rows, child_cols] = child_coords_t.dims();
            if child_cols != 4 {
                return Err(format!(
                    "decoder upsample stage {} child coord tensor must have 4 columns, got {}",
                    stage_idx, child_cols
                ));
            }
            if child_rows > sub_rows.saturating_mul(8) {
                return Err(format!(
                    "decoder upsample stage {} child coord tensor has too many rows: child_rows={} max_rows={}",
                    stage_idx,
                    child_rows,
                    sub_rows.saturating_mul(8)
                ));
            }
            current_coords_t = child_coords_t;
        }
        Ok(SparseUpsampledCoords::from_wgpu_tensor(current_coords_t))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stage0_subdivision_logits(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseSubdivisionLogits, String> {
        if self.stages.is_empty() {
            return Err("decoder has no stages".to_string());
        }
        let stage = &self.stages[0];
        let up = stage
            .upsample_block
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no upsample block".to_string())?;
        let to_subdiv = up
            .to_subdiv
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no to_subdiv head".to_string())?;

        let count = coords.len().min(rows.len());
        if count == 0 {
            return SparseSubdivisionLogits::from_host([1, 1, 1], Vec::new(), Vec::new());
        }

        let state_coords = coords[..count].to_vec();
        let mut state_feats = flatten_rows_32(&rows[..count]);
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_feats_wgpu: Option<Tensor<DefaultWgpuBackend, 2>> = None;
        let mut conv_cache = self
            .conv_cache
            .lock()
            .map_err(|_| "decoder conv cache lock poisoned".to_string())?;
        #[cfg(feature = "runtime-model-wgpu")]
        let mut wgpu_context = if let Some(context) = self.wgpu_context.as_ref() {
            Some(
                context
                    .lock()
                    .map_err(|_| "decoder wgpu context lock poisoned".to_string())?,
            )
        } else {
            None
        };
        #[cfg(feature = "runtime-model-wgpu")]
        let canonical_wgpu =
            decoder_wgpu_device_math_enabled() && (!self.compute_fp16_enabled() || decoder_wgpu_device_math_allow_fp16());
        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu && wgpu_context.is_none() {
            return Err(
                "burn_trellis: canonical wgpu stage0 path requires wgpu sparse conv context; host completion fallback is disabled"
                    .to_string(),
            );
        }
        state_feats = linear_forward(
            state_feats.as_slice(),
            count,
            &self.from_latent,
            "from_latent(stage0)",
        )?;
        if self.compute_fp16_enabled() {
            quantize_f16_inplace(state_feats.as_mut_slice());
        }

        let stage_channels = self.model_channels[0];
        #[allow(unused_mut)]
        // Stage0 can legally have zero ConvNeXt blocks; treat that as completed
        // on-device to keep canonical WGPU flow in no-host-completion mode.
        let mut convnext_device_complete = stage.convnext_blocks.is_empty();
        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu
            && !stage.convnext_blocks.is_empty()
            && let Some(context_gpu) = wgpu_context.as_deref_mut()
        {
            let row_count = state_coords.len();
            let state_bytes = row_count
                .saturating_mul(stage_channels)
                .saturating_mul(core::mem::size_of::<f32>());
            if state_bytes <= decoder_wgpu_device_math_max_state_bytes() {
                let coords_t =
                    coords_tensor_from_u32_slice(state_coords.as_slice(), &context_gpu.device)?;
                let state_t = if let Some(state_t) = state_feats_wgpu.take() {
                    let [rows_device, channels_device] = state_t.dims();
                    if rows_device == row_count && channels_device == stage_channels {
                        state_t
                    } else {
                        Tensor::<DefaultWgpuBackend, 1>::from_floats(
                            state_feats.as_slice(),
                            &context_gpu.device,
                        )
                        .reshape([row_count, stage_channels])
                    }
                } else {
                    Tensor::<DefaultWgpuBackend, 1>::from_floats(
                        state_feats.as_slice(),
                        &context_gpu.device,
                    )
                    .reshape([row_count, stage_channels])
                };
                let convnext_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        convnext_blocks_forward_wgpu_tensor(
                            context_gpu,
                            coords_t,
                            state_t,
                            0,
                            stage_channels,
                            stage.convnext_blocks.as_slice(),
                            self.compute_fp16_enabled(),
                        )
                    }));
                match convnext_result {
                    Ok(Ok(next_state_feats)) => {
                        state_feats_wgpu = Some(next_state_feats);
                        convnext_device_complete = true;
                    }
                    Ok(Err(err)) => {
                        if err.contains("BufferTooBig") {
                            context_gpu.wgpu_failed = true;
                        }
                        return Err(format!(
                            "burn_trellis: wgpu stage0 convnext failed reason={err}"
                        ));
                    }
                    Err(payload) => {
                        context_gpu.wgpu_failed = true;
                        let panic_message = panic_payload_to_string(payload);
                        return Err(format!(
                            "burn_trellis: wgpu stage0 convnext panicked panic={panic_message}"
                        ));
                    }
                }
            } else {
                return Err(format!(
                    "burn_trellis: wgpu stage0 convnext state_bytes={} exceeds max_state_bytes={}; refusing cpu fallback",
                    state_bytes,
                    decoder_wgpu_device_math_max_state_bytes()
                ));
            }
        }

        if !convnext_device_complete {
            #[cfg(feature = "runtime-model-wgpu")]
            if canonical_wgpu && !stage.convnext_blocks.is_empty()
            {
                return Err(
                    "burn_trellis: decoder stage0 convnext did not complete on wgpu; host completion path is disabled"
                        .to_string(),
                );
            }
            #[cfg(feature = "runtime-model-wgpu")]
            if canonical_wgpu && state_feats_wgpu.is_some() {
                return Err(
                    "burn_trellis: decoder stage0 convnext produced device tensors but canonical path forbids host completion"
                        .to_string(),
                );
            }
            let allow_host_stage0_completion = {
                #[cfg(feature = "runtime-model-wgpu")]
                {
                    !canonical_wgpu
                }
                #[cfg(not(feature = "runtime-model-wgpu"))]
                {
                    true
                }
            };
            if allow_host_stage0_completion {
                for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
                    let row_count = state_coords.len();
                    if row_count == 0 {
                        break;
                    }
                    let residual = state_feats.clone();
                    let mut h = sparse_subm_conv_forward(
                        state_coords.as_slice(),
                        state_feats.as_slice(),
                        &block.conv,
                        format!("stage0 block {block_idx} conv(stage0)").as_str(),
                        &mut conv_cache,
                        #[cfg(feature = "runtime-model-wgpu")]
                        wgpu_context.as_deref_mut(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    layer_norm_inplace(
                        h.as_mut_slice(),
                        row_count,
                        stage_channels,
                        Some(block.norm_weight.as_slice()),
                        Some(block.norm_bias.as_slice()),
                        LAYER_NORM32_EPS,
                        format!("stage0 block {block_idx} layer_norm(stage0)").as_str(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    h = linear_forward(
                        h.as_slice(),
                        row_count,
                        &block.mlp_0,
                        format!("stage0 block {block_idx} mlp_0(stage0)").as_str(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    silu_inplace(
                        h.as_mut_slice(),
                        format!("stage0 block {block_idx} silu(stage0)").as_str(),
                    );
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    h = linear_forward(
                        h.as_slice(),
                        row_count,
                        &block.mlp_2,
                        format!("stage0 block {block_idx} mlp_2(stage0)").as_str(),
                    )?;
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    add_inplace(
                        h.as_mut_slice(),
                        residual.as_slice(),
                        format!("stage0 block {block_idx} residual_add(stage0)").as_str(),
                    );
                    if self.compute_fp16_enabled() {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    state_feats = h;
                }
            }
        }

        #[cfg(feature = "runtime-model-wgpu")]
        if canonical_wgpu && state_feats_wgpu.is_none() {
            return Err(
                "burn_trellis: decoder stage0 to_subdiv missing device feature tensor on canonical wgpu path"
                    .to_string(),
            );
        }

        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(state_t) = state_feats_wgpu.take() {
            if !canonical_wgpu {
                return Err(
                    "burn_trellis: wgpu stage0 to_subdiv tensor requires canonical device-math path"
                        .to_string(),
                );
            }
            let Some(context_gpu) = wgpu_context.as_deref_mut() else {
                return Err(
                    "burn_trellis: wgpu stage0 to_subdiv tensor present without wgpu context"
                        .to_string(),
                );
            };
            let wgpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
                    let mut logits_t = linear_forward_wgpu(
                        context_gpu,
                        state_t.clone(),
                        to_subdiv,
                        self.compute_fp16_enabled(),
                        "stage0 to_subdiv(wgpu_math)",
                    )?;
                    if self.compute_fp16_enabled() {
                        logits_t = quantize_f16_tensor_wgpu(logits_t);
                    }
                    if self.runtime_config.center_subdivision_logits {
                        let mean = logits_t.clone().mean_dim(1);
                        logits_t = logits_t.sub(mean);
                    }
                    Ok(logits_t)
                },
            ));
            let logits_t = match wgpu_result {
                Ok(Ok(logits_t)) => logits_t,
                Ok(Err(err)) => {
                    if err.contains("BufferTooBig") {
                        context_gpu.wgpu_failed = true;
                    }
                    return Err(format!(
                        "burn_trellis: wgpu stage0 to_subdiv failed reason={err}"
                    ));
                }
                Err(payload) => {
                    context_gpu.wgpu_failed = true;
                    let panic_message = panic_payload_to_string(payload);
                    return Err(format!(
                        "burn_trellis: wgpu stage0 to_subdiv panicked panic={panic_message}"
                    ));
                }
            };
            let coords_t =
                coords_tensor_from_u32_slice(state_coords.as_slice(), &context_gpu.device)?;
            let active_indices_t =
                subdivision_active_indices_wgpu(logits_t.clone(), false, &self.runtime_config)?;
            let (child_coords_t, child_linear_idx_t) = expand_subdivision_coords_and_linear_indices_wgpu(
                coords_t.clone(),
                active_indices_t.clone(),
            )?;
            #[cfg(feature = "runtime-model-wgpu")]
            if decoder_wgpu_clear_cache_after_decode() {
                context_gpu.clear_caches();
            }
            return SparseSubdivisionLogits::from_device_tensors_with_active_and_children(
                spatial_shape_from_coords(state_coords.as_slice()),
                coords_t,
                logits_t,
                Some(active_indices_t),
                Some((child_coords_t, child_linear_idx_t)),
            );
        }

        let mut subdiv_logits = linear_forward(
            state_feats.as_slice(),
            state_coords.len(),
            to_subdiv,
            "stage0 to_subdiv",
        )?;
        if self.compute_fp16_enabled() {
            quantize_f16_inplace(subdiv_logits.as_mut_slice());
        }
        if self.runtime_config.center_subdivision_logits {
            row_center_logits(
                subdiv_logits.as_mut_slice(),
                state_coords.len(),
                "stage0 to_subdiv center_logits",
            );
        }

        #[cfg(feature = "runtime-model-wgpu")]
        if decoder_wgpu_clear_cache_after_decode()
            && let Some(context) = wgpu_context.as_deref_mut()
        {
            context.clear_caches();
        }

        SparseSubdivisionLogits::from_host(
            spatial_shape_from_coords(state_coords.as_slice()),
            state_coords,
            subdiv_logits,
        )
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_can_stream_final_stage_output(stages: &[DecoderStage], stage_idx: usize) -> bool {
    // This shortcut used to stream sparse rows through the final upsample stage by
    // slicing parent/child coords before sparse convs. That changes TRELLIS.2
    // submanifold-conv semantics: sparse rows are compact storage for active
    // voxel cells, and neighbor lookup must see the full active coordinate set.
    //
    // Keep this disabled until the streamed path computes query chunks against a
    // full-coordinate child-neighbor map or an explicit halo.
    let _ = (stages, stage_idx);
    false
}

#[cfg(feature = "runtime-model-wgpu")]
async fn upsample_stage_to_output_chunks_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    stage_idx: usize,
    up: &C2SUpsampleBlock,
    parent_feats_t: Tensor<DefaultWgpuBackend, 2>,
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    child_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    active_indices_t: Tensor<DefaultWgpuBackend, 2, Int>,
    linear_idx_t: Tensor<DefaultWgpuBackend, 1, Int>,
    compute_fp16: bool,
    output_layer: &LinearLayer,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let _ = (
        context_gpu,
        stage_idx,
        up,
        parent_feats_t,
        parent_coords_t,
        child_coords_t,
        active_indices_t,
        linear_idx_t,
        compute_fp16,
        output_layer,
    );
    Err(
        "burn_trellis: streamed final-stage decoder output is disabled because sparse conv neighborhoods require full active-cell coordinate context"
            .to_string(),
    )
}

#[cfg(feature = "runtime-model-wgpu")]
async fn upsample_conv1_select_children_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    up: &C2SUpsampleBlock,
    stage_idx: usize,
    parent_feats_norm_t: Tensor<DefaultWgpuBackend, 2>,
    parent_coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    active_indices_t: Tensor<DefaultWgpuBackend, 2, Int>,
    linear_idx_t: Tensor<DefaultWgpuBackend, 1, Int>,
    child_rows: usize,
    compute_fp16: bool,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let [parent_rows, parent_channels] = parent_feats_norm_t.dims();
    if parent_channels != up.in_channels {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 parent feature channels mismatch: got={} expected={}",
            stage_idx, parent_channels, up.in_channels
        ));
    }
    let [coord_rows, coord_cols] = parent_coords_t.dims();
    if coord_cols != 4 || coord_rows != parent_rows {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 parent coord tensor mismatch: got=[{},{}] expected=[{},4]",
            stage_idx, coord_rows, coord_cols, parent_rows
        ));
    }
    let [linear_rows] = linear_idx_t.dims();
    if linear_rows != child_rows {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 linear index rows mismatch: got={} expected={}",
            stage_idx, linear_rows, child_rows
        ));
    }
    if child_rows == 0 {
        return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
            [0, up.out_channels],
            &context_gpu.device,
        ));
    }
    let conv1_config = flex_config_for_layer(&up.conv1);
    let bytes_per_parent_row = up
        .conv1
        .out_channels
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!(
                "burn_trellis: decoder stage {} up conv1 bytes-per-row overflow",
                stage_idx
            )
        })?;
    let conv1_output_bytes = parent_rows
        .checked_mul(bytes_per_parent_row)
        .ok_or_else(|| {
            format!(
                "burn_trellis: decoder stage {} up conv1 output-byte-size overflow",
                stage_idx
            )
        })?;
    let max_tensor_bytes = decoder_wgpu_max_tensor_bytes();
    let max_conv1_direct_bytes =
        decoder_wgpu_upsample_conv1_max_output_bytes().min(max_tensor_bytes);
    if conv1_output_bytes <= max_conv1_direct_bytes {
        let h_conv1_t = context_gpu.forward_with_coords_tensor_device(
            &conv1_config,
            &up.conv1,
            parent_feats_norm_t.clone(),
            compute_fp16,
            format!("stage {stage_idx} up conv1(wgpu_math)").as_str(),
            parent_coords_t.clone(),
        )?;
        let h_conv1_flat = h_conv1_t.reshape([
            parent_rows.checked_mul(8).ok_or_else(|| {
                "decoder upsample parent_rows*8 overflow".to_string()
            })?,
            up.out_channels,
        ]);
        let selected = h_conv1_flat.select(0, linear_idx_t.clone());
        decoder_wgpu_upsample_conv1_select_parity_check(
            stage_idx,
            up,
            parent_feats_norm_t,
            parent_coords_t,
            linear_idx_t,
            selected.clone(),
        )?;
        return Ok(selected);
    }

    let selected_output_bytes = child_rows
        .checked_mul(up.out_channels)
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!(
                "burn_trellis: decoder stage {} up conv1 selected output-byte-size overflow",
                stage_idx
            )
        })?;
    if selected_output_bytes > max_tensor_bytes {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 selected output_bytes={} exceeds max_tensor_bytes={}",
            stage_idx, selected_output_bytes, max_tensor_bytes
        ));
    }

    let [active_rows, active_cols] = active_indices_t.dims();
    if active_rows != child_rows || active_cols != 2 {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 active-index tensor mismatch: got=[{},{}] expected=[{},2]",
            stage_idx, active_rows, active_cols, child_rows
        ));
    }
    let idx_col = Tensor::<DefaultWgpuBackend, 1, Int>::from_ints([0], &context_gpu.device);
    let linear_parent_idx_t = active_indices_t
        .select(1, idx_col.clone())
        .squeeze_dim::<1>(1);
    #[cfg(target_arch = "wasm32")]
    let linear_parent_idx_host = tensor_to_vec_i32_async(
        linear_parent_idx_t.clone().reshape([child_rows, 1]),
        format!("stage {stage_idx} up conv1 streamed parent-index async readback").as_str(),
    )
    .await?;
    let parent_chunk_rows =
        decoder_wgpu_chunk_rows(parent_rows, bytes_per_parent_row, max_conv1_direct_bytes).max(1);
    eprintln!(
        "burn_trellis: decoder stage {} streaming up conv1 selected_rows={} parent_rows={} child_rows={} chunk_rows={} full_output_bytes={} selected_output_bytes={}",
        stage_idx,
        child_rows,
        parent_rows,
        child_rows,
        parent_chunk_rows,
        conv1_output_bytes,
        selected_output_bytes
    );

    let conv1_kernel_rows = kernel_rows(&conv1_config)?;
    let conv1_neighbor_t =
        neighbor_rows_tensor_from_coords_tensor(&conv1_config, parent_coords_t.clone())?;
    let mut selected_chunks: Vec<Tensor<DefaultWgpuBackend, 2>> = Vec::new();
    let mut start = 0usize;
    while start < parent_rows {
        let end = (start + parent_chunk_rows).min(parent_rows);
        if end <= start {
            return Err(format!(
                "burn_trellis: decoder stage {} up conv1 streamed selection produced invalid chunk bounds start={} end={}",
                stage_idx, start, end
            ));
        }
        let end_minus_one = end.saturating_sub(1);
        let start_i32 = i32::try_from(start).map_err(|_| {
            format!(
                "burn_trellis: decoder stage {} up conv1 streamed selection start exceeds i32 range ({})",
                stage_idx, start
            )
        })?;
        let end_minus_one_i32 = i32::try_from(end_minus_one).map_err(|_| {
            format!(
                "burn_trellis: decoder stage {} up conv1 streamed selection end exceeds i32 range ({})",
                stage_idx, end_minus_one
            )
        })?;
        #[cfg(target_arch = "wasm32")]
        let keep_idx_t = keep_indices_tensor_from_parent_index_host(
            linear_parent_idx_host.as_slice(),
            start_i32,
            end_minus_one_i32,
            &context_gpu.device,
            format!("stage {stage_idx} up conv1 streamed keep-index").as_str(),
        )?;
        #[cfg(not(target_arch = "wasm32"))]
        let keep_idx_t = {
            let keep_mask = linear_parent_idx_t
                .clone()
                .clamp(start_i32, end_minus_one_i32)
                .equal(linear_parent_idx_t.clone());
            let keep_idx_rows = keep_mask.argwhere_async().await;
            let [_keep_rows, keep_cols] = keep_idx_rows.dims();
            if keep_cols != 1 {
                return Err(format!(
                    "burn_trellis: decoder stage {} up conv1 streamed selection keep-index tensor must have 1 column, got {}",
                    stage_idx, keep_cols
                ));
            }
            keep_idx_rows.select(1, idx_col.clone()).squeeze_dim(1)
        };
        let [keep_rows] = keep_idx_t.dims();
        if keep_rows == 0 {
            start = end;
            continue;
        }
        let global_linear_t = linear_idx_t.clone().select(0, keep_idx_t);
        let chunk_linear_base = start.checked_mul(8).ok_or_else(|| {
            format!(
                "burn_trellis: decoder stage {} up conv1 streamed selection linear base overflow",
                stage_idx
            )
        })?;
        let chunk_linear_base_i32 = i32::try_from(chunk_linear_base).map_err(|_| {
            format!(
                "burn_trellis: decoder stage {} up conv1 streamed selection linear base exceeds i32 range ({})",
                stage_idx, chunk_linear_base
            )
        })?;
        let local_linear_t = global_linear_t.sub_scalar(chunk_linear_base_i32);

        let chunk_rows_count = end - start;
        let chunk_neighbor_t = conv1_neighbor_t
            .clone()
            .slice([start..end, 0..conv1_kernel_rows]);
        let h_conv1_chunk_t = context_gpu.forward_with_neighbor_tensor_tensor(
            &conv1_config,
            &up.conv1,
            parent_feats_norm_t.clone(),
            compute_fp16,
            format!("stage {stage_idx} up conv1(wgpu_math/selected_stream)").as_str(),
            chunk_rows_count,
            conv1_kernel_rows,
            chunk_neighbor_t,
        )?;
        let h_conv1_chunk_flat_t = h_conv1_chunk_t.reshape([
            chunk_rows_count.checked_mul(8).ok_or_else(|| {
                "decoder upsample streamed parent_rows*8 overflow".to_string()
            })?,
            up.out_channels,
        ]);
        selected_chunks.push(h_conv1_chunk_flat_t.select(0, local_linear_t));
        start = end;
    }

    if selected_chunks.is_empty() {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 streamed selection produced no chunks",
            stage_idx
        ));
    }
    let selected = if selected_chunks.len() == 1 {
        selected_chunks.remove(0)
    } else {
        Tensor::cat(selected_chunks, 0)
    };
    let [selected_rows, selected_cols] = selected.dims();
    if selected_rows != child_rows || selected_cols != up.out_channels {
        return Err(format!(
            "burn_trellis: decoder stage {} up conv1 streamed selection output mismatch: got=[{},{}] expected=[{},{}]",
            stage_idx, selected_rows, selected_cols, child_rows, up.out_channels
        ));
    }
    Ok(selected)
}

#[cfg(not(feature = "runtime-model-wgpu"))]
fn subdivide_coords_from_mask(
    coords: &[[u32; 4]],
    subdivision_mask: &[[bool; 8]],
) -> Result<Vec<[u32; 4]>, String> {
    if subdivision_mask.len() != coords.len() {
        return Err(format!(
            "subdivide_coords_from_mask: subdivision rows {} do not match coords rows {}",
            subdivision_mask.len(),
            coords.len()
        ));
    }
    let mut out_coords = Vec::new();
    for (row_idx, coord) in coords.iter().enumerate() {
        for (child, selected) in subdivision_mask[row_idx].iter().enumerate().take(8) {
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
        }
    }
    Ok(out_coords)
}
