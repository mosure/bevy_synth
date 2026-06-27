use crate::prelude::*;

pub fn run_stdio_server(config: ServerConfig) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut server = McpServer::new(config);

    while let Some(message) = read_framed_json(&mut reader).map_err(|err| err.to_string())? {
        let response = server.handle_message(message)?;
        if let Some(response) = response {
            write_framed_json(&mut writer, &response).map_err(|err| err.to_string())?;
        }
        if server.should_exit {
            break;
        }
    }

    Ok(())
}

pub(crate) struct McpServer {
    pub(crate) config: ServerConfig,
    pub(crate) runtime: SynthRuntime,
    pub(crate) grounding: SceneGroundingRuntime,
    pub(crate) should_exit: bool,
}

struct NoopSceneProvider;

impl SceneAiProvider for NoopSceneProvider {
    fn plan_objects(&self, _request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan objects".to_string(),
        ))
    }

    fn generate_object_images(
        &self,
        _request: &burn_synth_scene::ObjectImageRequest,
    ) -> SceneResult<Vec<Vec<u8>>> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot generate images".to_string(),
        ))
    }

    fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan BSN".to_string(),
        ))
    }
}

impl McpServer {
    pub(crate) fn new(config: ServerConfig) -> Self {
        let runtime = SynthRuntime::new(config.runtime_config());
        Self {
            config,
            runtime,
            grounding: SceneGroundingRuntime::default(),
            should_exit: false,
        }
    }

    fn handle_message(&mut self, message: Value) -> Result<Option<Value>, String> {
        let request: RpcRequest = serde_json::from_value(message)
            .map_err(|err| format!("invalid JSON-RPC request: {err}"))?;
        self.handle_request(request)
    }

    fn handle_request(&mut self, request: RpcRequest) -> Result<Option<Value>, String> {
        match request.method.as_str() {
            "initialize" => {
                let params: InitializeParams = request
                    .params
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|err| format!("invalid initialize params: {err}"))?
                    .unwrap_or_default();
                let protocol_version = params
                    .protocol_version
                    .unwrap_or_else(|| DEFAULT_PROTOCOL_VERSION.to_string());
                let result = json!({
                    "protocolVersion": protocol_version,
                    "serverInfo": {
                        "name": "burn_synth_mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    }
                });
                Ok(Some(success_response(request.id, result)))
            }
            "notifications/initialized" => Ok(None),
            "tools/list" => Ok(Some(success_response(
                request.id,
                json!({ "tools": tool_defs() }),
            ))),
            "tools/call" => {
                let params: ToolsCallParams = request
                    .params
                    .ok_or_else(|| "missing tools/call params".to_string())
                    .and_then(|value| {
                        serde_json::from_value(value)
                            .map_err(|err| format!("invalid tools/call params: {err}"))
                    })?;
                let result = self.dispatch_tool_call(params);
                Ok(Some(success_response(request.id, result)))
            }
            "shutdown" => Ok(Some(success_response(request.id, Value::Null))),
            "exit" => {
                self.should_exit = true;
                Ok(None)
            }
            _ => {
                if request.id.is_none() {
                    return Ok(None);
                }
                Ok(Some(error_response(
                    request.id,
                    -32601,
                    format!("method '{}' not found", request.method),
                )))
            }
        }
    }

    fn dispatch_tool_call(&mut self, params: ToolsCallParams) -> Value {
        match params.name.as_str() {
            "image_to_foreground" => {
                let args: Result<ForegroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_foreground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for image_to_foreground: {err}"
                    )),
                }
            }
            "image_to_mesh" => {
                let args: Result<MeshToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_mesh(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_mesh: {err}"))
                    }
                }
            }
            "image_to_splat" => {
                let args: Result<SplatToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_splat(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_splat: {err}"))
                    }
                }
            }
            "images_to_assets" => {
                let args: Result<ImagesToAssetsToolArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_images_to_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for images_to_assets: {err}"))
                    }
                }
            }
            "scene_prepare_build" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_prepare_build(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_prepare_build: {err}"
                    )),
                }
            }
            "scene_plan_objects" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_objects(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_plan_objects: {err}"
                    )),
                }
            }
            "scene_generate_object_images" => {
                let args: Result<SceneGenerateObjectImagesArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_generate_object_images(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_generate_object_images: {err}"
                    )),
                }
            }
            "scene_build_from_image" => {
                let args: Result<SceneBuildFromImageArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_build_from_image(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_build_from_image: {err}"
                    )),
                }
            }
            "scene_ground" => {
                let args: Result<SceneGroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_ground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_ground: {err}"))
                    }
                }
            }
            "scene_plan_bsn" => {
                let args: Result<ScenePlanBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_plan_bsn: {err}"))
                    }
                }
            }
            "scene_apply_bsn" => {
                let args: Result<SceneApplyBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_apply_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_apply_bsn: {err}"))
                    }
                }
            }
            "scene_status" => match self.call_scene_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_project_status" => match self.call_scene_project_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_list_assets" => match self.call_scene_list_assets() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_spawn_cached" => {
                let args: Result<SceneSpawnCachedArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_cached(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_spawn_cached: {err}"
                    )),
                }
            }
            "scene_spawn_path" => {
                let args: Result<SceneSpawnPathArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_path(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_spawn_path: {err}"))
                    }
                }
            }
            "scene_delete" => {
                let args: Result<SceneDeleteArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_delete(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_delete: {err}"))
                    }
                }
            }
            "scene_clear" => match self.call_scene_clear() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_set_camera" => {
                let args: Result<SceneSetCameraArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_set_camera(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_set_camera: {err}"))
                    }
                }
            }
            "scene_save" => match self.send_scene_commands(vec![json!({ "type": "save_cache" })]) {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_capture" => {
                let args: Result<SceneCaptureArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_capture(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_capture: {err}"))
                    }
                }
            }
            "scene_compose_assets" => {
                let args: Result<SceneComposeArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_compose_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_compose_assets: {err}"
                    )),
                }
            }
            "scene_validate_layout" => {
                let args: Result<SceneValidateArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_validate_layout(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_validate_layout: {err}"
                    )),
                }
            }
            other => error_tool_result(format!("unknown tool '{other}'")),
        }
    }

    fn call_image_to_foreground(&mut self, args: ForegroundToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        let output_path = args
            .output_image_path
            .unwrap_or_else(|| default_output_path(&input_path, "_foreground", "png"));
        ensure_parent_dir(&output_path).map_err(|err| err.to_string())?;

        let selected_model = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let dry_run = args.dry_run;

        let (width, height) = if dry_run {
            let passthrough = image::open(&input_path)
                .map_err(|err| {
                    format!("failed to open input image {}: {err}", input_path.display())
                })?
                .to_rgba8();
            let dims = passthrough.dimensions();
            passthrough.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        } else {
            let output = self
                .runtime
                .extract_foreground(ForegroundRequest {
                    image: ImageSource::from_path(input_path.clone()),
                    model: Some(selected_model.into()),
                })
                .map_err(|err| err.to_string())?;
            let dims = (output.width, output.height);
            output.image.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        };

        Ok(json!({
            "tool": "image_to_foreground",
            "input_image_path": input_path.display().to_string(),
            "output_image_path": output_path.display().to_string(),
            "width": width,
            "height": height,
            "rmbg_model": selected_model.as_str(),
            "dry_run": dry_run,
        }))
    }

    fn call_image_to_mesh(&mut self, args: MeshToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        if let Some(output_format) = args.output_format
            && !matches!(output_format, MeshOutputFormat::Glb)
        {
            return Err(format!(
                "only glb output is supported; requested {}",
                output_format.as_str()
            ));
        }
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![input_path],
            output_dir: None,
            output_paths: args.output_mesh_path.map(|path| vec![path]),
            output_format: Some(AssetOutputFormat::Glb),
            rmbg_model: args.rmbg_model,
            synthesis_models: args.synthesis_models,
            backend: args.backend,
            target_faces: args.target_faces,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_mesh produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_mesh",
            "input_image_path": item["input_image_path"].clone(),
            "output_mesh_path": item["output_path"].clone(),
            "output_format": "glb",
            "vertices": item["vertices"].clone(),
            "faces": item["faces"].clone(),
            "local_aabb": item["local_aabb"].clone(),
            "target_faces": item["target_faces"].clone(),
            "material": item["material"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    fn call_image_to_splat(&mut self, args: SplatToolArgs) -> Result<Value, String> {
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![args.input_image_path],
            output_dir: None,
            output_paths: args.output_splat_path.map(|path| vec![path]),
            output_format: args.output_format.or(Some(AssetOutputFormat::Splat)),
            rmbg_model: args.rmbg_model,
            synthesis_models: Some(vec![SynthesisModel::Triposplat]),
            backend: args.backend,
            target_faces: None,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_splat produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_splat",
            "input_image_path": item["input_image_path"].clone(),
            "output_splat_path": item["output_path"].clone(),
            "output_format": item["output_format"].clone(),
            "gaussians": item["gaussians"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    fn call_images_to_assets(&mut self, args: ImagesToAssetsToolArgs) -> Result<Value, String> {
        if args.input_image_paths.is_empty() {
            return Err("input_image_paths must not be empty".to_string());
        }
        for input in &args.input_image_paths {
            if !input.exists() {
                return Err(format!("input image does not exist: {}", input.display()));
            }
        }
        if let Some(output_paths) = args.output_paths.as_ref()
            && output_paths.len() != args.input_image_paths.len()
        {
            return Err(format!(
                "output_paths length ({}) must match input_image_paths length ({})",
                output_paths.len(),
                args.input_image_paths.len()
            ));
        }

        let selected_rmbg = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let selected_backend = args.backend.unwrap_or(self.config.default_backend);
        let selected_synthesis_models = args
            .synthesis_models
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.default_synthesis_models.clone());
        let policy = RuntimeBatchPolicy {
            max_items: args.batch_size.or(self.config.batch_size),
            vram_budget_mb: args.batch_vram_mb.or(self.config.batch_vram_mb),
            ..RuntimeBatchPolicy::default()
        };
        let previous_trellis_pbr_enabled = self.runtime.config().trellis_pbr_enabled;
        let previous_trellis_pbr_texture_size = self.runtime.config().trellis_pbr_texture_size;
        let previous_target_faces = self.runtime.config().target_faces;
        let effective_trellis_pbr_enabled =
            args.trellis_pbr.unwrap_or(previous_trellis_pbr_enabled);
        let effective_trellis_pbr_texture_size = args
            .trellis_pbr_texture_size
            .or(previous_trellis_pbr_texture_size);
        let effective_target_faces = match args.target_faces {
            Some(0) => None,
            Some(value) => Some(value),
            None => previous_target_faces,
        };
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = effective_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = effective_trellis_pbr_texture_size;
            config.target_faces = effective_target_faces;
        }

        let batch_result = self.runtime.synthesize_assets_batch(AssetBatchRequest {
            items: args
                .input_image_paths
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    AssetBatchItem::new(
                        format!("asset_{index}"),
                        ImageSource::from_path(input.clone()),
                    )
                })
                .collect(),
            foreground_model: Some(selected_rmbg.into()),
            synthesis_models: Some(
                selected_synthesis_models
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect(),
            ),
            backend: Some(selected_backend.into()),
            dry_run: args.dry_run,
            policy,
        });
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = previous_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = previous_trellis_pbr_texture_size;
            config.target_faces = previous_target_faces;
        }
        let batch = batch_result.map_err(|err| err.to_string())?;

        let mut items = Vec::with_capacity(batch.items.len());
        let mut catalog_cache = if args.promote_to_catalog {
            Some(self.open_catalog_cache()?)
        } else {
            None
        };
        for batch_item in batch.items {
            let input_path = args
                .input_image_paths
                .get(batch_item.item_index)
                .ok_or_else(|| {
                    format!(
                        "asset batch item index {} out of range for {} input images",
                        batch_item.item_index,
                        args.input_image_paths.len()
                    )
                })?;
            let output = batch_item.output.map_err(|err| err.to_string())?;
            let item = write_asset_output(
                input_path,
                args.output_dir.as_deref(),
                args.output_paths
                    .as_ref()
                    .and_then(|paths| paths.get(batch_item.item_index).cloned()),
                args.output_format.unwrap_or(AssetOutputFormat::Auto),
                output.asset,
                effective_target_faces,
                catalog_cache.as_mut(),
            )?;
            items.push(json!({
                "id": batch_item.id,
                "input_image_path": input_path.display().to_string(),
                "chunk_index": batch_item.chunk_index,
                "item_index": batch_item.item_index,
                "elapsed_ms": batch_item.elapsed_ms,
                "foreground_model": runtime_foreground_model_str(output.foreground_model),
                "synthesis_backend": runtime_synthesis_model_str(output.synthesis_backend),
                "backend": runtime_backend_str(output.backend),
                "output_path": item.output_path.display().to_string(),
                "output_format": item.output_format.as_str(),
                "asset_kind": item.asset_kind,
                "vertices": item.vertices,
                "faces": item.faces,
                "gaussians": item.gaussians,
                "local_aabb": item.local_aabb,
                "target_faces": effective_target_faces,
                "material": item.material,
                "mesh_quality": item.mesh_quality,
                "mesh_quality_failures": item.mesh_quality_failures,
                "cache_key": item.catalog_entry.as_ref().map(|entry| entry.cache_key.clone()),
                "catalog_entry": item.catalog_entry,
            }));
        }

        Ok(json!({
            "tool": "images_to_assets",
            "items": items,
            "stats": {
                "total_items": batch.stats.total_items,
                "chunk_size": batch.stats.chunk_size,
                "chunks": batch.stats.chunks,
                "execution_mode": batch.stats.execution_mode.as_str(),
                "vram_budget_mb": batch.stats.vram_budget_mb,
                "estimated_item_mb": batch.stats.estimated_item_mb,
                "elapsed_ms": batch.stats.elapsed_ms,
            },
            "rmbg_model": selected_rmbg.as_str(),
            "synthesis_models": selected_synthesis_models.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            "backend": selected_backend.as_str(),
            "trellis_pbr_enabled": effective_trellis_pbr_enabled,
            "trellis_pbr_backend": if effective_trellis_pbr_enabled { Some("rust-ovoxel") } else { None },
            "trellis_pbr_texture_size": effective_trellis_pbr_texture_size,
            "target_faces": effective_target_faces,
            "promote_to_catalog": args.promote_to_catalog,
            "dry_run": args.dry_run,
        }))
    }

    fn call_scene_prepare_build(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = NoopSceneProvider;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        serde_json::to_value(preparation).map_err(|err| err.to_string())
    }

    fn call_scene_plan_objects(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let manifest = pipeline.plan_objects().map_err(|err| err.to_string())?;
        serde_json::to_value(manifest).map_err(|err| err.to_string())
    }

    fn call_scene_generate_object_images(
        &self,
        args: SceneGenerateObjectImagesArgs,
    ) -> Result<Value, String> {
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args.object_reference_image_path,
            output_dir: args.output_dir,
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: false,
        };
        let config = self.scene_build_config(prepare_args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let requests = pipeline
            .prepare_object_image_requests(&args.manifest)
            .map_err(|err| err.to_string())?;
        let candidates = pipeline
            .generate_object_candidates(&requests)
            .map_err(|err| err.to_string())?;
        Ok(json!({
            "tool": "scene_generate_object_images",
            "requests": requests,
            "candidates": candidates,
        }))
    }

    pub(crate) fn call_scene_build_from_image(
        &mut self,
        args: SceneBuildFromImageArgs,
    ) -> Result<Value, String> {
        let e2e_started = Instant::now();
        let mut stage_report = Vec::new();
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path.clone(),
            object_reference_image_path: args.object_reference_image_path,
            output_dir: args.output_dir,
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: args.allow_catalog_reuse,
        };
        let stage_started = Instant::now();
        let config = self.scene_build_config(prepare_args)?;
        let output_dir = config.output_dir.clone();
        let requested_candidate_count = args
            .candidate_count
            .unwrap_or(config.candidate_count)
            .max(1);
        let candidates_per_attempt = args
            .candidate_batch_size
            .unwrap_or(requested_candidate_count)
            .max(1);
        let max_attempts_per_object = args
            .candidate_retry_attempts
            .unwrap_or(if args.candidate_batch_size.is_some() {
                requested_candidate_count
            } else {
                1
            })
            .max(1);
        let candidate_policy = ObjectImageGenerationPolicy {
            min_score: args
                .min_reconstruction_score
                .unwrap_or(DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE),
            max_attempts_per_object,
            candidates_per_attempt,
        };
        let provider = self.openai_provider()?;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "prepare_openai_inputs", stage_started);
        let stage_started = Instant::now();
        let manifest = pipeline.plan_objects().map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "plan_objects", stage_started);
        let stage_started = Instant::now();
        let requests = pipeline
            .prepare_object_image_requests(&manifest)
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "prepare_object_image_requests",
            stage_started,
        );
        let stage_started = Instant::now();
        let candidate_report = pipeline
            .generate_object_candidates_with_policy(&requests, candidate_policy)
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "generate_object_candidates",
            stage_started,
        );
        let mut selected = if candidate_report.rejected_objects.is_empty() {
            candidate_report.selected_candidates.clone()
        } else {
            Vec::new()
        };
        let mut selected_values = selected_candidates_to_values(&selected);

        let mut response = json!({
            "tool": "scene_build_from_image",
            "preparation": preparation,
            "provider_metadata": pipeline.provider_metadata(),
            "manifest": manifest,
            "object_image_requests": requests,
            "candidate_generation": candidate_report.clone(),
            "candidates": candidate_report.candidates.clone(),
            "selected_candidates": selected_values.clone(),
            "lift_assets": args.lift_assets,
        });
        if !candidate_report.rejected_objects.is_empty() {
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            let message = candidate_report
                .rejected_objects
                .first()
                .map(|rejection| rejection.message.clone())
                .unwrap_or_else(|| "scene candidate generation failed guardrails".to_string());
            return Err(message);
        }
        if !args.lift_assets {
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Ok(response);
        }

        let stage_started = Instant::now();
        let mut excluded_asset_candidates = HashSet::new();
        let mut cached_asset_outputs = HashMap::<(String, usize), Value>::new();
        let mut asset_attempts = Vec::new();
        let asset_outputs = loop {
            let missing_selected = selected_values
                .iter()
                .filter(|selected| {
                    scene_selected_candidate_key(selected)
                        .is_none_or(|key| !cached_asset_outputs.contains_key(&key))
                })
                .cloned()
                .collect::<Vec<_>>();
            if !missing_selected.is_empty() {
                let input_image_paths = missing_selected
                    .iter()
                    .map(|candidate| {
                        candidate
                            .get("image_path")
                            .and_then(Value::as_str)
                            .map(PathBuf::from)
                            .ok_or_else(|| "selected candidate missing image_path".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let new_outputs = self.call_images_to_assets(ImagesToAssetsToolArgs {
                    input_image_paths,
                    output_dir: Some(output_dir.join("assets")),
                    output_paths: None,
                    output_format: Some(AssetOutputFormat::Glb),
                    rmbg_model: Some(ForegroundModel::Rmbg2),
                    synthesis_models: Some(vec![SynthesisModel::Trellis]),
                    backend: Some(self.config.default_backend),
                    target_faces: args
                        .target_faces
                        .or(Some(DEFAULT_SCENE_TRELLIS_TARGET_FACES)),
                    batch_size: args.batch_size,
                    batch_vram_mb: args.batch_vram_mb,
                    trellis_pbr: Some(args.trellis_pbr.unwrap_or(true)),
                    trellis_pbr_texture_size: args
                        .trellis_pbr_texture_size
                        .or(Some(DEFAULT_SCENE_TRELLIS_PBR_TEXTURE_SIZE)),
                    promote_to_catalog: args.promote_to_catalog,
                    dry_run: false,
                })?;
                cache_scene_asset_outputs(
                    &mut cached_asset_outputs,
                    &missing_selected,
                    &new_outputs,
                )?;
            }
            let outputs =
                scene_cached_asset_outputs_for_selected(&selected_values, &cached_asset_outputs)?;
            let attempt_failures =
                scene_asset_quality_failures_with_selected(&outputs, &selected_values);
            asset_attempts.push(json!({
                "attempt_index": asset_attempts.len(),
                "selected_candidates": selected_values.clone(),
                "lifted_candidates": missing_selected,
                "mesh_quality_failures": attempt_failures.iter().map(SceneAssetQualityFailure::message).collect::<Vec<_>>(),
            }));
            if attempt_failures.is_empty() {
                break outputs;
            }

            for failure in &attempt_failures {
                excluded_asset_candidates
                    .insert((failure.object_id.clone(), failure.candidate_index));
            }
            let next_selected = select_object_image_candidates_with_exclusions(
                &manifest,
                &candidate_report.candidates,
                candidate_policy.min_score,
                &excluded_asset_candidates,
            );
            let Ok(next_selected) = next_selected else {
                break outputs;
            };
            if next_selected == selected {
                break outputs;
            }
            selected = next_selected;
            selected_values = selected_candidates_to_values(&selected);
        };
        record_stage(&mut stage_report, "images_to_assets", stage_started);
        let mesh_quality_failures =
            scene_asset_quality_failures_with_selected(&asset_outputs, &selected_values)
                .into_iter()
                .map(|failure| failure.message())
                .collect::<Vec<_>>();
        response["selected_candidates"] = json!(selected_values.clone());
        response["asset_lift_attempts"] = json!(asset_attempts);
        if !mesh_quality_failures.is_empty() {
            response["asset_outputs"] = asset_outputs;
            response["mesh_quality_failures"] = json!(mesh_quality_failures);
            response["failed_stage"] = json!("images_to_assets.mesh_quality_gate");
            response["next_action"] = json!({
                "kind": "regenerate_failed_assets",
                "recommendation": "Generate another isolated object-image candidate for each failed asset and rerun TRELLIS lifting before scene grounding/composition.",
                "reason": "Bad mesh topology should not be allowed to propagate into scene placement, feedback, or catalog reuse.",
            });
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Err(response["mesh_quality_failures"]
                .as_array()
                .and_then(|failures| failures.first())
                .and_then(Value::as_str)
                .unwrap_or("scene mesh quality gate failed")
                .to_string());
        }
        let asset_bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected_values, &asset_outputs)?;
        let stage_started = Instant::now();
        let (grounding_source, mut grounding_evidence) =
            if args.composition_mode == SceneCompositionMode::CvGrounded {
                if args.locator == SceneLocatorProvider::LocateAnything {
                    let backend = args
                        .locate_anything_backend
                        .unwrap_or(self.config.locate_anything_backend);
                    let evidence = self.locate_anything_grounding_evidence(
                        backend,
                        &manifest,
                        &args.source_scene_path,
                        &output_dir,
                    )?;
                    let LocateAnythingBackend::BurnNative = backend;
                    ("locate_anything_burn_native", evidence)
                } else {
                    ("manifest_fallback", manifest_grounding_evidence(&manifest))
                }
            } else {
                ("disabled", manifest_grounding_evidence(&manifest))
            };
        record_stage(&mut stage_report, "load_grounding_evidence", stage_started);
        if args.composition_mode == SceneCompositionMode::CvGrounded
            && args.depth_provider == SceneDepthProvider::DepthPro
            && grounding_evidence.depth.is_none()
        {
            let stage_started = Instant::now();
            self.depth_pro_grounding_evidence(
                &mut grounding_evidence,
                &args.source_scene_path,
                &output_dir,
            )?;
            record_stage(
                &mut stage_report,
                "depth_pro_grounding_evidence",
                stage_started,
            );
        }
        let stage_started = Instant::now();
        let composition_candidates = scene_composition_candidates(
            args.composition_mode,
            args.feedback && args.lift_assets,
            &manifest,
            &asset_bindings,
            &grounding_evidence,
            args.clear_existing,
        )?;
        let mut selected_composition = composition_candidates
            .first()
            .cloned()
            .ok_or_else(|| "scene composition produced no candidates".to_string())?;
        let mut commands = selected_composition.commands.clone();
        let mut feedback_candidate_reports = Vec::new();
        record_stage(&mut stage_report, "plan_grounded_scene", stage_started);
        if args.feedback && args.lift_assets {
            let stage_started = Instant::now();
            let selection = self.run_scene_composition_feedback_selection(
                &output_dir,
                &manifest,
                &asset_bindings,
                composition_candidates.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                    rotation_selector: args.feedback_rotation_selector,
                },
            )?;
            selected_composition = selection.candidate;
            commands = selection.commands;
            feedback_candidate_reports = selection.candidate_reports;
            let feedback = selection.feedback;
            if feedback
                .get("accepted")
                .and_then(Value::as_bool)
                .is_some_and(|accepted| !accepted)
                && response.get("failed_stage").is_none()
            {
                response["failed_stage"] = json!("render_capture_feedback.quality_gate");
                response["next_action"] = json!({
                    "reason": "Render-capture-feedback did not find a composition that satisfied projection and physical layout thresholds.",
                    "inspect": feedback_report_path_from_result(&feedback)
                        .unwrap_or_else(|| output_dir.join("iterations/feedback_report.md").display().to_string()),
                });
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
        }
        let grounded_layout = selected_composition.layout;
        let plan = selected_composition.plan;
        let bsn = feedback_bsn_from_commands(&asset_bindings, &grounded_layout, &commands)?;
        response["asset_outputs"] = asset_outputs;
        response["asset_bindings"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["requested_composition_mode"] = json!(args.composition_mode);
        response["composition_mode"] = json!(selected_composition.mode);
        response["pose_fit"] = json!(args.pose_fit);
        response["canonical_pose"] = json!(args.canonical_pose);
        response["max_pose_candidates"] = json!(args.max_pose_candidates);
        response["save_pose_debug"] = json!(args.save_pose_debug);
        if !feedback_candidate_reports.is_empty() {
            response["composition_candidate_reports"] = json!(feedback_candidate_reports);
        }
        response["depth_provider"] = json!(args.depth_provider);
        response["locator"] = json!(args.locator);
        response["grounding_source"] = json!(grounding_source);
        response["grounding_evidence"] =
            serde_json::to_value(&grounding_evidence).map_err(|err| err.to_string())?;
        response["bsn"] = json!(bsn);
        response["plan"] = serde_json::to_value(&plan).map_err(|err| err.to_string())?;
        response["grounded_layout"] =
            serde_json::to_value(&grounded_layout).map_err(|err| err.to_string())?;
        response["commands"] = json!(commands);
        response["clear_existing"] = json!(args.clear_existing);
        response["apply"] = json!(args.apply);
        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            match self
                .send_scene_commands(response["commands"].as_array().cloned().unwrap_or_default())
            {
                Ok(acknowledgement) => {
                    response["acknowledgement"] = acknowledgement;
                }
                Err(err) => {
                    record_stage(&mut stage_report, "apply_scene_commands", stage_started);
                    response["apply_error"] = json!(err);
                    response["stage_report"] = json!(stage_report);
                    response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
                    if args.write_artifacts {
                        write_scene_build_artifacts(&output_dir, &response)?;
                    }
                    return Err(response["apply_error"]
                        .as_str()
                        .unwrap_or("scene apply failed")
                        .to_string());
                }
            }
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
        }
        response["stage_report"] = json!(stage_report);
        response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
        if args.write_artifacts {
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        Ok(response)
    }

    fn call_scene_plan_bsn(&self, args: ScenePlanBsnArgs) -> Result<Value, String> {
        let grounded_layout =
            grounded_scene_layout_for_manifest(&args.manifest, &args.asset_bindings)
                .map_err(|err| err.to_string())?;
        let bsn = grounded_layout.bsn.clone();
        let plan = match parse_scene_bsn(&bsn, &args.asset_bindings) {
            Ok(plan) => plan,
            Err(err) => {
                return Ok(json!({
                    "tool": "scene_plan_bsn",
                    "valid": false,
                    "bsn": bsn,
                    "validation_error": err.to_string(),
                    "asset_bindings": args.asset_bindings,
                    "clear_existing": args.clear_existing,
                    "apply": false,
                }));
            }
        };
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_plan_bsn",
            "valid": true,
            "bsn": bsn,
            "plan": plan,
            "grounded_layout": grounded_layout,
            "commands": commands,
            "asset_bindings": args.asset_bindings,
            "clear_existing": args.clear_existing,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    pub(crate) fn call_scene_ground(&mut self, args: SceneGroundToolArgs) -> Result<Value, String> {
        let started = Instant::now();
        let output_dir = args.output_dir.unwrap_or_else(default_scene_output_dir);
        fs::create_dir_all(&output_dir).map_err(|err| {
            format!(
                "failed to create scene-ground output directory {}: {err}",
                output_dir.display()
            )
        })?;
        let mut stage_report = Vec::new();
        let stage_started = Instant::now();
        let mut manifest = args.manifest;
        manifest.source_scene_path = args.source_scene_path.display().to_string();
        let asset_bindings = args.asset_bindings;
        let (grounding_source, mut evidence) = if let Some(evidence) = args.grounding_evidence {
            ("provided", evidence)
        } else if args.locator == SceneLocatorProvider::LocateAnything {
            let backend = args
                .locate_anything_backend
                .unwrap_or(self.config.locate_anything_backend);
            let evidence = self.locate_anything_grounding_evidence(
                backend,
                &manifest,
                &args.source_scene_path,
                &output_dir,
            )?;
            let LocateAnythingBackend::BurnNative = backend;
            let source = "locate_anything_burn_native";
            (source, evidence)
        } else {
            ("manifest_fallback", manifest_grounding_evidence(&manifest))
        };
        record_stage(&mut stage_report, "load_grounding_evidence", stage_started);

        if args.depth_provider == SceneDepthProvider::DepthPro && evidence.depth.is_none() {
            let stage_started = Instant::now();
            self.depth_pro_grounding_evidence(&mut evidence, &args.source_scene_path, &output_dir)?;
            record_stage(
                &mut stage_report,
                "depth_pro_grounding_evidence",
                stage_started,
            );
        }

        let stage_started = Instant::now();
        let composition_candidates = scene_composition_candidates(
            args.composition_mode,
            args.feedback,
            &manifest,
            &asset_bindings,
            &evidence,
            args.clear_existing,
        )?;
        let mut selected_composition = composition_candidates
            .first()
            .cloned()
            .ok_or_else(|| "scene composition produced no candidates".to_string())?;
        let mut commands = selected_composition.commands.clone();
        let mut feedback_candidate_reports = Vec::new();
        record_stage(&mut stage_report, "solve_grounded_scene", stage_started);

        let mut response = json!({
            "tool": "scene_ground",
            "source_scene_path": args.source_scene_path,
            "requested_composition_mode": args.composition_mode,
            "pose_fit": args.pose_fit,
            "canonical_pose": args.canonical_pose,
            "max_pose_candidates": args.max_pose_candidates,
            "save_pose_debug": args.save_pose_debug,
            "depth_provider": args.depth_provider,
            "locator": args.locator,
            "grounding_source": grounding_source,
            "manifest": manifest.clone(),
            "asset_bindings": asset_bindings.clone(),
            "grounding_evidence": evidence.clone(),
            "clear_existing": args.clear_existing,
            "apply": args.apply,
        });

        if args.feedback {
            let stage_started = Instant::now();
            let selection = self.run_scene_composition_feedback_selection(
                &output_dir,
                &manifest,
                &asset_bindings,
                composition_candidates.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                    rotation_selector: args.feedback_rotation_selector,
                },
            )?;
            selected_composition = selection.candidate;
            commands = selection.commands;
            feedback_candidate_reports = selection.candidate_reports;
            let feedback = selection.feedback;
            if feedback
                .get("accepted")
                .and_then(Value::as_bool)
                .is_some_and(|accepted| !accepted)
            {
                response["failed_stage"] = json!("render_capture_feedback.quality_gate");
                response["next_action"] = json!({
                    "reason": "Render-capture-feedback did not find a composition that satisfied projection and physical layout thresholds.",
                    "inspect": feedback_report_path_from_result(&feedback)
                        .unwrap_or_else(|| output_dir.join("iterations/feedback_report.md").display().to_string()),
                });
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
        }

        let grounded_layout = selected_composition.layout;
        let plan = selected_composition.plan;
        let bsn = feedback_bsn_from_commands(&asset_bindings, &grounded_layout, &commands)?;
        response["composition_mode"] = json!(selected_composition.mode);
        if !feedback_candidate_reports.is_empty() {
            response["composition_candidate_reports"] = json!(feedback_candidate_reports);
        }
        response["grounded_layout"] =
            serde_json::to_value(&grounded_layout).map_err(|err| err.to_string())?;
        response["bsn"] = json!(bsn);
        response["plan"] = serde_json::to_value(&plan).map_err(|err| err.to_string())?;
        response["commands"] = json!(commands.clone());

        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            let acknowledgement = self.send_scene_commands(commands)?;
            response["acknowledgement"] = acknowledgement;
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
        }

        response["stage_report"] = json!(stage_report);
        response["e2e_summary"] = json!({
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "objects": response["grounded_layout"]["placements"].as_array().map(Vec::len).unwrap_or_default(),
            "composition_mode": response["composition_mode"],
            "grounding_source": response["grounding_source"],
        });
        write_scene_ground_artifacts(&output_dir, &response)?;
        Ok(response)
    }

    fn call_scene_apply_bsn(&self, args: SceneApplyBsnArgs) -> Result<Value, String> {
        let plan =
            parse_scene_bsn(&args.bsn, &args.asset_bindings).map_err(|err| err.to_string())?;
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_apply_bsn",
            "plan": plan,
            "commands": commands,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    fn scene_build_config(&self, args: ScenePrepareBuildArgs) -> Result<SceneBuildConfig, String> {
        Ok(SceneBuildConfig {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args
                .object_reference_image_path
                .unwrap_or_else(|| self.config.scene_object_reference_image.clone()),
            output_dir: args.output_dir.unwrap_or_else(default_scene_output_dir),
            candidate_count: args.candidate_count.unwrap_or(3).max(1),
            quality_profile: args.quality_profile.unwrap_or(SceneQualityProfile::Quality),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            allow_catalog_reuse: args.allow_catalog_reuse,
        })
    }

    fn openai_provider(&self) -> Result<OpenAiSceneProvider, String> {
        OpenAiSceneProvider::from_env(OpenAiProviderConfig {
            api_key: self.config.openai_api_key.clone().unwrap_or_default(),
            base_url: self
                .config
                .openai_base_url
                .clone()
                .unwrap_or_else(|| OpenAiProviderConfig::default().base_url),
            project_id: self.config.openai_project_id.clone(),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            ..OpenAiProviderConfig::default()
        })
        .map_err(|err| err.to_string())
    }

    fn open_catalog_cache(&self) -> Result<MeshCache, String> {
        if let Some(root) = self.config.catalog_cache_root.as_ref() {
            MeshCache::load_from_root(root.clone())
        } else {
            MeshCache::load_default()
        }
        .map_err(|err| format!("failed to open shared asset cache: {err}"))
    }

    fn call_scene_status(&self) -> Result<Value, String> {
        let status_path = self
            .config
            .scene_status_path
            .as_ref()
            .ok_or_else(|| "scene_status_path is not configured".to_string())?;
        read_scene_status(status_path)
    }

    fn call_scene_project_status(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_project_status",
            "camera": status.get("camera").cloned().unwrap_or(Value::Null),
            "world_items": status.get("world_items").cloned().unwrap_or(Value::Null),
            "projected_items": status.get("projected_items").cloned().unwrap_or(Value::Null),
            "screenshots": status.get("screenshots").cloned().unwrap_or(Value::Null),
            "status": status,
        }))
    }

    fn call_scene_list_assets(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_list_assets",
            "cache_entries": status["cache_entries"].clone(),
            "world_items": status["world_items"].clone(),
        }))
    }

    fn call_scene_spawn_cached(&self, args: SceneSpawnCachedArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_cached",
            "cache_key": args.cache_key,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_spawn_path(&self, args: SceneSpawnPathArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_path",
            "path": args.path,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_delete(&self, args: SceneDeleteArgs) -> Result<Value, String> {
        if let Some(cache_key) = args.cache_key {
            return self.send_scene_commands(vec![json!({
                "type": "delete_by_cache_key",
                "cache_key": cache_key,
            })]);
        }
        if args.selected {
            return self.send_scene_commands(vec![json!({ "type": "delete_selected" })]);
        }
        self.send_scene_commands(vec![json!({ "type": "clear_selection" })])
    }

    fn call_scene_clear(&self) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({ "type": "clear_scene" })])
    }

    fn call_scene_set_camera(&self, args: SceneSetCameraArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "set_camera",
            "translation": args.translation,
            "rotation": args.rotation,
            "focus": args.focus,
            "yaw": args.yaw,
            "pitch": args.pitch,
            "radius": args.radius,
            "vertical_fov": args.vertical_fov,
        })])
    }

    fn call_scene_capture(&self, args: SceneCaptureArgs) -> Result<Value, String> {
        let path = args.output_path;
        let response = self.send_scene_commands(vec![json!({
            "type": "capture_screenshot",
            "path": path.display().to_string(),
        })])?;
        let timeout = self.config.scene_timeout;
        let started = Instant::now();
        while started.elapsed() < timeout {
            if path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
            {
                return Ok(json!({
                    "tool": "scene_capture",
                    "output_path": path.display().to_string(),
                    "acknowledgement": response,
                }));
            }
            thread::sleep(Duration::from_millis(25));
        }
        Err(format!(
            "scene_capture timed out waiting for screenshot {}",
            path.display()
        ))
    }

    fn call_scene_compose_assets(&self, args: SceneComposeArgs) -> Result<Value, String> {
        let plan = compose_scene_layout(args)?;
        let mut response =
            serde_json::to_value(&plan).map_err(|err| format!("serialize layout plan: {err}"))?;
        if plan.apply {
            let acknowledgement = self.send_scene_commands(scene_commands_from_plan(&plan)?)?;
            response["acknowledgement"] = acknowledgement;
        }
        Ok(response)
    }

    fn call_scene_validate_layout(&self, mut args: SceneValidateArgs) -> Result<Value, String> {
        if args.scene_status.is_none() {
            args.scene_status = Some(self.call_scene_status()?);
        }
        validate_scene_layout(args)
    }

    fn run_scene_composition_feedback_selection(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        candidates: Vec<SceneCompositionCandidate>,
        options: SceneFeedbackOptions,
    ) -> Result<SceneCompositionFeedbackSelection, String> {
        if candidates.is_empty() {
            return Err("composition feedback selection requires candidates".to_string());
        }
        let compare_candidates = candidates.len() > 1;
        let mut reports = Vec::new();
        let mut best: Option<(f64, SceneCompositionCandidate, Vec<Value>, Value)> = None;
        for candidate in candidates {
            let capture_dir = if compare_candidates {
                let base = options
                    .capture_dir
                    .clone()
                    .unwrap_or_else(|| output_dir.join("iterations"));
                Some(base.join(scene_composition_mode_label(candidate.mode)))
            } else {
                options.capture_dir.clone()
            };
            let feedback = self.run_scene_feedback(
                output_dir,
                manifest,
                asset_bindings,
                &candidate.layout,
                candidate.commands.clone(),
                SceneFeedbackOptions {
                    max_iters: options.max_iters,
                    keep_viewer: options.keep_viewer,
                    capture_dir,
                    threshold_profile: options.threshold_profile,
                    rotation_selector: options.rotation_selector,
                },
            );
            match feedback {
                Ok(mut feedback) => {
                    let selection_score = feedback_result_selection_score(&feedback);
                    feedback["composition_mode"] = json!(candidate.mode);
                    feedback["selection_score"] =
                        finite_json_f64(selection_score).unwrap_or(Value::Null);
                    let final_commands = feedback
                        .get("final_commands")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_else(|| candidate.commands.clone());
                    reports.push(json!({
                        "mode": candidate.mode,
                        "accepted": feedback.get("accepted").and_then(Value::as_bool).unwrap_or(false),
                        "best_iteration": feedback.get("best_iteration").cloned().unwrap_or(Value::Null),
                        "best_score": feedback.get("best_score").cloned().unwrap_or(Value::Null),
                        "selection_score": finite_json_f64(selection_score).unwrap_or(Value::Null),
                        "capture_dir": feedback.get("capture_dir").cloned().unwrap_or(Value::Null),
                        "feedback_report": feedback_report_path_from_result(&feedback).unwrap_or_default(),
                    }));
                    if best
                        .as_ref()
                        .is_none_or(|(best_score, _, _, _)| selection_score > *best_score)
                    {
                        best = Some((selection_score, candidate, final_commands, feedback));
                    }
                }
                Err(err) => {
                    reports.push(json!({
                        "mode": candidate.mode,
                        "accepted": false,
                        "selection_score": Value::Null,
                        "error": err,
                    }));
                }
            }
        }
        let Some((selection_score, candidate, commands, mut feedback)) = best else {
            return Err(format!(
                "all composition feedback candidates failed: {}",
                serde_json::to_string(&reports).unwrap_or_else(|_| "<unserializable>".to_string())
            ));
        };
        feedback["candidate_selection"] = json!({
            "selected_mode": candidate.mode,
            "selected_score": finite_json_f64(selection_score).unwrap_or(Value::Null),
            "compared_candidates": compare_candidates,
            "candidates": reports.clone(),
        });
        Ok(SceneCompositionFeedbackSelection {
            candidate,
            commands,
            feedback,
            candidate_reports: reports,
        })
    }

    pub(crate) fn run_scene_feedback(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        grounded_layout: &GroundedSceneLayout,
        initial_commands: Vec<Value>,
        options: SceneFeedbackOptions,
    ) -> Result<Value, String> {
        let original_control_path = self.config.scene_control_path.clone();
        let original_status_path = self.config.scene_status_path.clone();
        let original_timeout = self.config.scene_timeout;
        let capture_root = options
            .capture_dir
            .clone()
            .unwrap_or_else(|| output_dir.join("iterations"));
        fs::create_dir_all(&capture_root).map_err(|err| {
            format!(
                "failed to create feedback capture directory {}: {err}",
                capture_root.display()
            )
        })?;

        let mut spawned_viewer = None;
        if self.config.scene_control_path.is_none() {
            let bridge_dir = output_dir.join("feedback_viewer");
            fs::create_dir_all(&bridge_dir).map_err(|err| {
                format!(
                    "failed to create feedback viewer directory {}: {err}",
                    bridge_dir.display()
                )
            })?;
            let control_path = bridge_dir.join("scene_commands.json");
            let status_path = bridge_dir.join("scene_commands.status.json");
            let log_path = bridge_dir.join("viewer.log");
            spawned_viewer = Some(spawn_feedback_viewer(&control_path, &log_path)?);
            self.config.scene_control_path = Some(control_path);
            self.config.scene_status_path = Some(status_path);
            self.config.scene_timeout = self.config.scene_timeout.max(Duration::from_secs(60));
        }

        let lock_result = self.send_scene_commands(vec![scene_interaction_lock_command(
            true,
            "iterative scene composition",
        )]);
        let feedback_result = match lock_result {
            Ok(lock_ack) => {
                let _ = write_json_file(&capture_root.join("interaction_lock_ack.json"), &lock_ack);
                let mut result =
                    self.run_scene_feedback_iterations(SceneFeedbackIterationContext {
                        capture_root: &capture_root,
                        manifest,
                        asset_bindings,
                        grounded_layout,
                        initial_commands,
                        max_iters: options.max_iters.max(1),
                        threshold_profile: options.threshold_profile,
                        rotation_selector: options.rotation_selector,
                    });
                if let Ok(value) = &mut result {
                    value["interaction_lock_ack"] = lock_ack;
                }
                result
            }
            Err(err) => Err(format!("failed to lock scene interaction: {err}")),
        };
        let unlock_result =
            self.send_scene_commands(vec![scene_interaction_lock_command(false, "")]);
        let feedback_result = match (feedback_result, unlock_result) {
            (Ok(mut value), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                value["interaction_unlock_ack"] = unlock_ack;
                Ok(value)
            }
            (Ok(_), Err(unlock_err)) => Err(format!(
                "feedback completed but failed to unlock scene interaction: {unlock_err}"
            )),
            (Err(err), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                Err(err)
            }
            (Err(err), Err(unlock_err)) => Err(format!(
                "{err}; additionally failed to unlock scene interaction: {unlock_err}"
            )),
        };

        if let Some(mut child) = spawned_viewer
            && !options.keep_viewer
        {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.config.scene_control_path = original_control_path;
        self.config.scene_status_path = original_status_path;
        self.config.scene_timeout = original_timeout;

        feedback_result
    }

    fn run_scene_feedback_iterations(
        &self,
        context: SceneFeedbackIterationContext<'_>,
    ) -> Result<Value, String> {
        let SceneFeedbackIterationContext {
            capture_root,
            manifest,
            asset_bindings,
            grounded_layout,
            initial_commands,
            max_iters,
            threshold_profile,
            rotation_selector,
        } = context;
        let mut commands = initial_commands;
        let thresholds = threshold_profile.thresholds();
        let mut iterations = Vec::new();
        let mut accepted_iteration = None;
        let mut best_iteration = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_commands = commands.clone();
        let mut previous_iteration_snapshot: Option<Value> = None;
        let mut previous_commands: Option<Vec<Value>> = None;
        for iteration_index in 0..max_iters {
            let iteration_dir = capture_root.join(format!("iter_{iteration_index:02}"));
            fs::create_dir_all(&iteration_dir).map_err(|err| {
                format!(
                    "failed to create feedback iteration directory {}: {err}",
                    iteration_dir.display()
                )
            })?;
            write_json_file(&iteration_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            let iteration_bsn =
                feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
            fs::write(iteration_dir.join("scene.bsn"), iteration_bsn).map_err(|err| {
                format!(
                    "failed to write feedback BSN {}: {err}",
                    iteration_dir.join("scene.bsn").display()
                )
            })?;

            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&iteration_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            thread::sleep(Duration::from_millis(250));
            let screenshot_path = iteration_dir.join("screenshot.png");
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.clone(),
            })?;
            write_json_file(&iteration_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            let status = Self::feedback_capture_status(&apply_ack, &capture);
            write_json_file(&iteration_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            let mut metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            let object_crops = feedback_object_crops(
                &iteration_dir,
                Path::new(&manifest.source_scene_path),
                &screenshot_path,
                &metrics,
            );
            if !object_crops.is_null() {
                metrics["object_crops"] = object_crops.clone();
                write_json_file(&iteration_dir.join("object_crops.json"), &object_crops)
                    .map_err(|err| err.to_string())?;
            }
            let rotation_selection_task = feedback_rotation_selection_task(&metrics, &object_crops);
            if !rotation_selection_task.is_null() {
                write_json_file(
                    &iteration_dir.join("rotation_selection_task.json"),
                    &rotation_selection_task,
                )
                .map_err(|err| err.to_string())?;
            }
            let rotation_selection_report = self.apply_feedback_rotation_selector(
                rotation_selector,
                &iteration_dir,
                &rotation_selection_task,
                &mut metrics,
            )?;
            if !rotation_selection_report.is_null() {
                metrics["rotation_selector"] = rotation_selection_report.clone();
                write_json_file(
                    &iteration_dir.join("rotation_selection_report.json"),
                    &rotation_selection_report,
                )
                .map_err(|err| err.to_string())?;
            }
            write_json_file(&iteration_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let selection_score = feedback_selection_score(&metrics);
            if selection_score > best_score {
                best_score = selection_score;
                best_iteration = Some(iteration_index);
                best_commands = commands.clone();
            }
            let deltas = feedback_layout_deltas(&metrics);
            write_json_file(&iteration_dir.join("layout_delta.json"), &deltas)
                .map_err(|err| err.to_string())?;
            let iteration_context = feedback_iteration_context(
                iteration_index,
                previous_iteration_snapshot.as_ref(),
                previous_commands.as_deref(),
                &commands,
                &screenshot_path,
                &metrics,
                &deltas,
                &object_crops,
            );
            write_json_file(
                &iteration_dir.join("iteration_context.json"),
                &iteration_context,
            )
            .map_err(|err| err.to_string())?;
            iterations.push(json!({
                "iteration": iteration_index,
                "dir": iteration_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics.clone(),
                "layout_delta": deltas.clone(),
                "object_crops": object_crops.clone(),
                "iteration_context": iteration_context.clone(),
                "passed": passed,
                "selection_score": selection_score,
            }));
            if passed {
                accepted_iteration = Some(iteration_index);
                break;
            }
            previous_commands = Some(commands.clone());
            previous_iteration_snapshot = iterations.last().cloned();
            commands = apply_feedback_deltas_to_commands(&commands, &deltas)?;
        }
        if accepted_iteration.is_none() && best_iteration.is_some() {
            commands = best_commands;
        }
        let final_bsn = feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
        fs::write(capture_root.join("scene.feedback.bsn"), &final_bsn).map_err(|err| {
            format!(
                "failed to write final feedback BSN {}: {err}",
                capture_root.join("scene.feedback.bsn").display()
            )
        })?;
        write_json_file(
            &capture_root.join("commands.feedback.json"),
            &json!(commands),
        )
        .map_err(|err| err.to_string())?;
        let mut final_evidence = Value::Null;
        if accepted_iteration.is_none() && max_iters > 0 {
            let final_dir = capture_root.join("final");
            fs::create_dir_all(&final_dir).map_err(|err| {
                format!(
                    "failed to create final feedback directory {}: {err}",
                    final_dir.display()
                )
            })?;
            write_json_file(&final_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            fs::write(final_dir.join("scene.bsn"), &final_bsn).map_err(|err| {
                format!(
                    "failed to write final feedback BSN {}: {err}",
                    final_dir.join("scene.bsn").display()
                )
            })?;
            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&final_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            thread::sleep(Duration::from_millis(250));
            let screenshot_path = final_dir.join("screenshot.png");
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.clone(),
            })?;
            write_json_file(&final_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            let status = Self::feedback_capture_status(&apply_ack, &capture);
            write_json_file(&final_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            let metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            write_json_file(&final_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            final_evidence = json!({
                "dir": final_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics,
                "passed": passed,
            });
        }
        let report = feedback_markdown_report(
            capture_root,
            threshold_profile,
            accepted_iteration,
            &iterations,
        );
        fs::write(capture_root.join("feedback_report.md"), report).map_err(|err| {
            format!(
                "failed to write feedback markdown report {}: {err}",
                capture_root.join("feedback_report.md").display()
            )
        })?;
        Ok(json!({
            "tool": "scene_render_capture_feedback",
            "enabled": true,
            "threshold_profile": threshold_profile,
            "rotation_selector": rotation_selector,
            "max_iters": max_iters,
            "accepted": accepted_iteration.is_some(),
            "accepted_iteration": accepted_iteration,
            "best_iteration": best_iteration,
            "best_score": if best_score.is_finite() { Value::from(best_score) } else { Value::Null },
            "capture_dir": capture_root.display().to_string(),
            "iterations": iterations,
            "final_evidence": final_evidence,
            "final_bsn_path": capture_root.join("scene.feedback.bsn").display().to_string(),
            "final_commands_path": capture_root.join("commands.feedback.json").display().to_string(),
            "final_commands": commands,
        }))
    }

    fn apply_feedback_rotation_selector(
        &self,
        selector: FeedbackRotationSelector,
        iteration_dir: &Path,
        task: &Value,
        metrics: &mut Value,
    ) -> Result<Value, String> {
        if task.is_null() {
            return Ok(Value::Null);
        }
        match selector {
            FeedbackRotationSelector::Deterministic => Ok(json!({
                "selector": "deterministic",
                "applied_count": 0,
                "reason": "using deterministic geometry-selected rotation candidates",
            })),
            FeedbackRotationSelector::Openai => {
                let image_paths = feedback_rotation_selection_image_paths(task);
                let prompt = feedback_rotation_selection_prompt(task);
                write_json_file(
                    &iteration_dir.join("rotation_selection_request.json"),
                    &json!({
                        "prompt": prompt,
                        "task": task,
                        "image_paths": image_paths
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>(),
                    }),
                )
                .map_err(|err| err.to_string())?;
                match self.openai_provider().and_then(|provider| {
                    provider
                        .select_rotation_candidates(&SceneRotationSelectionRequest {
                            prompt: feedback_rotation_selection_prompt(task),
                            task: task.clone(),
                            image_paths,
                        })
                        .map_err(|err| err.to_string())
                }) {
                    Ok(response) => {
                        write_json_file(
                            &iteration_dir.join("rotation_selection_response.json"),
                            &serde_json::to_value(&response).map_err(|err| err.to_string())?,
                        )
                        .map_err(|err| err.to_string())?;
                        let applied =
                            apply_feedback_rotation_selection_response(metrics, &response);
                        Ok(json!({
                            "selector": "openai",
                            "fallback": false,
                            "applied": applied,
                        }))
                    }
                    Err(err) => Ok(json!({
                        "selector": "openai",
                        "fallback": true,
                        "error": err,
                        "reason": "keeping deterministic geometry-selected rotation candidates",
                    })),
                }
            }
        }
    }

    pub(crate) fn feedback_capture_status(apply_ack: &Value, capture_ack: &Value) -> Value {
        capture_ack
            .get("acknowledgement")
            .and_then(|ack| ack.get("status"))
            .cloned()
            .or_else(|| apply_ack.get("status").cloned())
            .unwrap_or(Value::Null)
    }

    pub(crate) fn send_scene_commands(&self, commands: Vec<Value>) -> Result<Value, String> {
        if commands.is_empty() {
            return Err("scene command list must not be empty".to_string());
        }
        let control_path = self
            .config
            .scene_control_path
            .as_ref()
            .ok_or_else(|| "scene_control_path is not configured".to_string())?;
        let sequence = next_scene_sequence();
        let session_id = format!("burn_synth_mcp-{}", std::process::id());
        let envelope = json!({
            "session_id": session_id,
            "sequence": sequence,
            "commands": commands,
        });
        atomic_write_json(control_path, &envelope)?;

        let Some(status_path) = self.config.scene_status_path.as_ref() else {
            return Ok(json!({
                "tool": "scene_command",
                "command_path": control_path.display().to_string(),
                "sequence": sequence,
                "acknowledged": false,
            }));
        };
        let status = wait_scene_status(status_path, sequence, self.config.scene_timeout)?;
        Ok(json!({
            "tool": "scene_command",
            "command_path": control_path.display().to_string(),
            "status_path": status_path.display().to_string(),
            "sequence": sequence,
            "acknowledged": true,
            "status": status,
        }))
    }
}
