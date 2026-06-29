use crate::prelude::*;
use crate::server::McpServer;
use crate::server::run_stdio_server;

pub fn run_from_args(args: ServerArgs) -> Result<(), String> {
    configure_cubecl_runtime(&args)?;
    let command = args.command.clone();
    let config = ServerConfig::from_args(args);
    match command {
        Some(ServerCommand::SceneBuild(args)) => run_scene_build_command(config, args),
        Some(ServerCommand::SceneGround(args)) => run_scene_ground_command(config, args),
        Some(ServerCommand::SceneGroundingReport(args)) => {
            run_scene_grounding_report_command(config, args)
        }
        Some(ServerCommand::SceneFeedbackReplay(args)) => {
            run_scene_feedback_replay_command(config, args)
        }
        None => run_stdio_server(config),
    }
}

fn configure_cubecl_runtime(args: &ServerArgs) -> Result<(), String> {
    if args.cubecl_autotune_level == CubeClAutotuneLevelSetting::Default
        && args.cubecl_autotune_cache == CubeClAutotuneCacheSetting::Default
    {
        return Ok(());
    }

    let mut config = cubecl::config::CubeClRuntimeConfig::default();
    config.autotune.level = match args.cubecl_autotune_level {
        CubeClAutotuneLevelSetting::Default => config.autotune.level,
        CubeClAutotuneLevelSetting::Minimal => cubecl::config::autotune::AutotuneLevel::Minimal,
        CubeClAutotuneLevelSetting::Balanced => cubecl::config::autotune::AutotuneLevel::Balanced,
        CubeClAutotuneLevelSetting::Extensive => cubecl::config::autotune::AutotuneLevel::Extensive,
        CubeClAutotuneLevelSetting::Full => cubecl::config::autotune::AutotuneLevel::Full,
    };
    config.autotune.cache = match args.cubecl_autotune_cache {
        CubeClAutotuneCacheSetting::Default => config.autotune.cache,
        CubeClAutotuneCacheSetting::Local => cubecl::config::cache::CacheConfig::Local,
        CubeClAutotuneCacheSetting::Target => cubecl::config::cache::CacheConfig::Target,
        CubeClAutotuneCacheSetting::Global => cubecl::config::cache::CacheConfig::Global,
    };

    std::panic::catch_unwind(move || {
        <cubecl::config::CubeClRuntimeConfig as cubecl::config::RuntimeConfig>::set(config);
    })
    .map_err(|_| {
        "CubeCL runtime configuration was already initialized before MCP startup; pass \
         --cubecl-autotune-* flags earlier or use Burn.toml/cubecl.toml configuration"
            .to_string()
    })
}

fn run_scene_build_command(config: ServerConfig, args: SceneBuildCliArgs) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let response = server.call_scene_build_from_image(SceneBuildFromImageArgs {
        source_scene_path: args.source_scene_path,
        object_reference_image_path: args.object_reference_image_path,
        output_dir: args.output_dir,
        candidate_count: args.candidate_count,
        candidate_retry_attempts: args.candidate_retry_attempts,
        candidate_batch_size: args.candidate_batch_size,
        min_reconstruction_score: args.min_reconstruction_score,
        quality_profile: args.quality_profile,
        allow_catalog_reuse: args.allow_catalog_reuse,
        lift_assets: args.lift_assets,
        synthesis_models: args.synthesis_models,
        target_faces: args.target_faces,
        batch_size: args.batch_size.filter(|value| *value > 0),
        batch_vram_mb: args.batch_vram_mb,
        trellis_pbr: Some(args.trellis_pbr),
        trellis_pbr_texture_size: args.trellis_pbr_texture_size,
        promote_to_catalog: args.promote_to_catalog,
        composition_mode: args.composition_mode,
        pose_fit: args.pose_fit,
        canonical_pose: args.canonical_pose,
        scale_policy: args.scale_policy,
        max_pose_candidates: args.max_pose_candidates,
        save_pose_debug: args.save_pose_debug,
        ground_calibration: args.ground_calibration,
        instance_generation: args.instance_generation,
        depth_provider: args.depth_provider,
        locator: args.locator,
        locate_anything_backend: args.locate_anything_backend,
        segmentation_provider: args.segmentation_provider,
        segmentation_precision: args.segmentation_precision,
        segmentation_quantization: args.segmentation_quantization,
        write_artifacts: args.write_artifacts,
        apply: args.apply,
        clear_existing: args.clear_existing,
        feedback: args.feedback,
        feedback_iters: args.feedback_iters,
        feedback_keep_viewer: args.feedback_keep_viewer,
        feedback_capture_dir: args.feedback_capture_dir,
        feedback_threshold_profile: args.feedback_threshold_profile,
        feedback_rotation_selector: args.feedback_rotation_selector,
        rotation_fit: args.rotation_fit,
        rotation_fit_max_gpt_rounds: args.rotation_fit_max_gpt_rounds,
        rotation_fit_min_mask_iou: args.rotation_fit_min_mask_iou,
        rotation_fit_max_depth_error_m: args.rotation_fit_max_depth_error_m,
        rotation_fit_write_artifacts: args.rotation_fit_write_artifacts,
        object_pose_refinement: args.object_pose_refinement,
        object_pose_refinement_set: args.object_pose_refinement_set,
        feedback_rubric_scorer: args.feedback_rubric_scorer,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-build response: {err}"))?
    );
    Ok(())
}

fn run_scene_ground_command(config: ServerConfig, args: SceneGroundCliArgs) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let response = server.call_scene_ground(SceneGroundToolArgs {
        source_scene_path: args.source_scene_path,
        manifest: read_json_path(&args.manifest)?,
        asset_bindings: read_json_path(&args.asset_bindings)?,
        grounding_evidence: args
            .grounding_evidence
            .as_ref()
            .map(|path| read_json_path::<SceneGroundingEvidence>(path.as_path()))
            .transpose()?,
        output_dir: args.output_dir,
        composition_mode: args.composition_mode,
        pose_fit: args.pose_fit,
        canonical_pose: args.canonical_pose,
        scale_policy: args.scale_policy,
        max_pose_candidates: args.max_pose_candidates,
        save_pose_debug: args.save_pose_debug,
        ground_calibration: args.ground_calibration,
        depth_provider: args.depth_provider,
        locator: args.locator,
        locate_anything_backend: args.locate_anything_backend,
        segmentation_provider: args.segmentation_provider,
        segmentation_precision: args.segmentation_precision,
        segmentation_quantization: args.segmentation_quantization,
        clear_existing: args.clear_existing,
        apply: args.apply,
        feedback: args.feedback,
        feedback_iters: args.feedback_iters,
        feedback_keep_viewer: args.feedback_keep_viewer,
        feedback_capture_dir: args.feedback_capture_dir,
        feedback_threshold_profile: args.feedback_threshold_profile,
        feedback_rotation_selector: args.feedback_rotation_selector,
        rotation_fit: args.rotation_fit,
        rotation_fit_max_gpt_rounds: args.rotation_fit_max_gpt_rounds,
        rotation_fit_min_mask_iou: args.rotation_fit_min_mask_iou,
        rotation_fit_max_depth_error_m: args.rotation_fit_max_depth_error_m,
        rotation_fit_write_artifacts: args.rotation_fit_write_artifacts,
        object_pose_refinement: args.object_pose_refinement,
        object_pose_refinement_set: args.object_pose_refinement_set,
        feedback_rubric_scorer: args.feedback_rubric_scorer,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-ground response: {err}"))?
    );
    Ok(())
}

fn run_scene_grounding_report_command(
    config: ServerConfig,
    args: SceneGroundingReportCliArgs,
) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let response = server.call_scene_grounding_report(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-grounding-report response: {err}"))?
    );
    Ok(())
}

fn run_scene_feedback_replay_command(
    config: ServerConfig,
    args: SceneFeedbackReplayCliArgs,
) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let manifest_path = args
        .manifest_path
        .unwrap_or_else(|| args.output_dir.join("manifest.json"));
    let asset_bindings_path = args
        .asset_bindings_path
        .unwrap_or_else(|| args.output_dir.join("asset_bindings.json"));
    let grounded_layout_path = args
        .grounded_layout_path
        .unwrap_or_else(|| args.output_dir.join("grounded_layout.json"));
    let commands_path = args
        .commands_path
        .unwrap_or_else(|| args.output_dir.join("commands.json"));
    let capture_dir = args
        .feedback_capture_dir
        .unwrap_or_else(|| args.output_dir.join("iterations_replay"));
    let manifest = read_json_path::<SceneObjectManifest>(&manifest_path)?;
    let asset_bindings = read_json_path::<Vec<SceneAssetBinding>>(&asset_bindings_path)?;
    let grounded_layout = read_json_path::<GroundedSceneLayout>(&grounded_layout_path)?;
    let grounding_evidence_path = args.output_dir.join("grounding_evidence.json");
    let grounding_evidence = grounding_evidence_path
        .exists()
        .then(|| read_json_path::<SceneGroundingEvidence>(&grounding_evidence_path))
        .transpose()?;
    let commands = if args.rebuild_commands_from_grounded_layout {
        let plan = parse_scene_bsn(&grounded_layout.bsn, &asset_bindings)
            .map_err(|err| err.to_string())?;
        scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &asset_bindings, true)
                .map_err(|err| err.to_string())?,
        )
    } else {
        read_json_path::<Vec<Value>>(&commands_path)?
    };
    let response = server.run_scene_feedback(
        &args.output_dir,
        &manifest,
        &asset_bindings,
        &grounded_layout,
        commands,
        SceneFeedbackOptions {
            max_iters: args.feedback_iters,
            keep_viewer: args.feedback_keep_viewer,
            capture_dir: Some(capture_dir),
            threshold_profile: args.feedback_threshold_profile,
            rotation_selector: args.feedback_rotation_selector,
            rotation_fit: args.rotation_fit,
            rotation_fit_max_gpt_rounds: args.rotation_fit_max_gpt_rounds,
            rotation_fit_min_mask_iou: args.rotation_fit_min_mask_iou,
            rotation_fit_max_depth_error_m: args.rotation_fit_max_depth_error_m,
            rotation_fit_write_artifacts: args.rotation_fit_write_artifacts,
            rubric_scorer: args.feedback_rubric_scorer,
            scale_policy: SceneScalePolicy::AssetPreserving,
            grounding_evidence,
        },
        None,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-feedback-replay response: {err}"))?
    );
    Ok(())
}

pub(crate) fn read_json_path<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read JSON {}: {err}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| format!("failed to parse JSON {}: {err}", path.display()))
}
