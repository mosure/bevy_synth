use crate::prelude::*;

pub(crate) fn atomic_write_json(path: &Path, value: &Value) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize scene command: {err}"))?;
    fs::write(&tmp, bytes).map_err(|err| format!("failed to write {}: {err}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|err| {
        format!(
            "failed to atomically replace scene command file {}: {err}",
            path.display()
        )
    })
}

pub(crate) fn read_scene_status(path: &Path) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|err| format!("failed to parse scene status {}: {err}", path.display()))
}

pub(crate) fn wait_scene_status(
    path: &Path,
    sequence: u64,
    timeout: Duration,
) -> Result<Value, String> {
    let started = Instant::now();
    let mut last_error = None;
    while started.elapsed() < timeout {
        match read_scene_status(path) {
            Ok(status) => {
                let acknowledged = status
                    .get("last_sequence")
                    .and_then(Value::as_u64)
                    .map(|last| last >= sequence)
                    .unwrap_or(false);
                if acknowledged {
                    return Ok(status);
                }
            }
            Err(err) => last_error = Some(err),
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!(
        "timed out waiting for scene status {} to acknowledge sequence {sequence}{}",
        path.display(),
        last_error
            .map(|err| format!("; last read error: {err}"))
            .unwrap_or_default()
    ))
}

pub(crate) fn success_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "result": result,
    })
}

pub(crate) fn error_response(id: Option<Value>, code: i32, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "error": {
            "code": code,
            "message": message,
        }
    })
}

pub(crate) fn success_tool_result(payload: Value) -> Value {
    let text = serde_json::to_string_pretty(&payload)
        .unwrap_or_else(|_| "{\"error\":\"failed to render tool payload\"}".to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": payload,
    })
}

pub(crate) fn error_tool_result(message: String) -> Value {
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ]
    })
}

pub(crate) fn tool_defs() -> Vec<Value> {
    vec![
        json!({
            "name": "image_to_foreground",
            "description": "Extract foreground alpha from an input image and write a PNG with transparency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_image_path": { "type": "string", "description": "Optional output path (defaults to *_foreground.png)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and just write a pass-through output image." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_mesh",
            "description": "Run image-to-mesh synthesis and write a GLB mesh output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_mesh_path": { "type": "string", "description": "Optional output GLB path (defaults to *_mesh.glb)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis"] }, "description": "Optional mesh synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical cube mesh." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_splat",
            "description": "Run TripoSplat image-to-Gaussian-splat synthesis and write a .splat or .ply output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_splat_path": { "type": "string", "description": "Optional output path (defaults to *_splat.splat)." },
                    "output_format": { "type": "string", "enum": ["splat", "ply"], "description": "Optional splat output format." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical debug splat cloud." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "images_to_assets",
            "description": "Run batched image-to-asset synthesis over multiple images with shared model loading and chunk planning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_paths": { "type": "array", "items": { "type": "string" }, "description": "Input image paths to process in one batch request." },
                    "output_dir": { "type": "string", "description": "Optional output directory for per-input output names." },
                    "output_paths": { "type": "array", "items": { "type": "string" }, "description": "Optional explicit output path per input." },
                    "output_format": { "type": "string", "enum": ["auto", "glb", "splat", "ply"], "description": "Optional output format. Auto writes GLB for meshes and .splat for Gaussian splats." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis", "triposplat"] }, "description": "Optional synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "batch_size": { "type": "integer", "description": "Optional explicit chunk size; omit for server default/auto." },
                    "batch_vram_mb": { "type": "integer", "description": "Optional VRAM budget in MB for auto chunking." },
                    "trellis_pbr": { "type": "boolean", "description": "Enable TRELLIS UV/material texture baking through the Rust/Burn o_voxel export path for lifted GLB assets." },
                    "trellis_pbr_texture_size": { "type": "integer", "description": "TRELLIS PBR texture size." },
                    "promote_to_catalog": { "type": "boolean", "description": "Also add generated assets to the shared Bevy catalog/cache for later reuse. Defaults to false for direct batch conversion." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit canonical debug assets." }
                },
                "required": ["input_image_paths"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_prepare_build",
            "description": "Prepare a formal OpenAI scene-builder run offline: validate paths and return strict schemas/prompts without calling OpenAI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string", "description": "Source scene image path." },
                    "object_reference_image_path": { "type": "string", "description": "Optional isolated-object style reference image; defaults to docs/input_chair.jpg." },
                    "output_dir": { "type": "string", "description": "Run output directory under tmp/runs." },
                    "candidate_count": { "type": "integer", "description": "Object image candidates per reusable object." },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean", "description": "Whether the planner may consider existing catalog assets." }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_plan_objects",
            "description": "Use the raw OpenAI API to create a strict object manifest from a source scene image. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean" }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_generate_object_images",
            "description": "Use the raw OpenAI Image API to generate isolated object-image candidates from a scene manifest, source crop, source scene image, and docs/input_chair.jpg-style reference. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "manifest": { "type": "object", "description": "SceneObjectManifest returned by scene_plan_objects." },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] }
                },
                "required": ["source_scene_path", "manifest"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_build_from_image",
            "description": "Bare-bones explicit scene build: locate objects and DepthPro camera/depth/floor evidence, crop each object, generate isolated object images with gpt-image-2 plus the reference image, lift selected candidates through RMBG+TRELLIS, then compose generated 3D objects with source-camera geometric constraints. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "candidate_retry_attempts": { "type": "integer", "description": "Maximum guarded image-generation attempts per object. Defaults to candidate_count." },
                    "candidate_batch_size": { "type": "integer", "description": "Generated image candidates requested per retry attempt. Defaults to 1 so weak candidates can be retried without overwriting artifacts." },
                    "min_reconstruction_score": { "type": "number", "description": "Minimum isolated-object reconstruction suitability score before TRELLIS lifting. Defaults to the canonical scene threshold." },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean" },
                    "lift_assets": { "type": "boolean", "description": "When false, stop after object image generation." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis", "triposplat"] }, "description": "Optional image-to-3D asset model list override for the scene asset-lift stage. The first model is used for generated scene objects." },
                    "target_faces": { "type": "integer" },
                    "batch_size": { "type": "integer" },
                    "batch_vram_mb": { "type": "integer" },
                    "trellis_pbr": { "type": "boolean", "description": "Enable TRELLIS UV/material texture baking through the Rust/Burn o_voxel export path for lifted GLB assets." },
                    "trellis_pbr_texture_size": { "type": "integer", "description": "TRELLIS PBR texture size." },
                    "promote_to_catalog": { "type": "boolean", "description": "Add lifted objects to the shared Bevy catalog/cache for later reuse. Defaults to true; fresh scene mode still does not read existing catalog assets while planning." },
                    "composition_mode": { "type": "string", "enum": ["heuristic", "cv-grounded"], "description": "Scene composition path after lifting. Full scene-build defaults to cv-grounded." },
                    "pose_fit": { "type": "string", "enum": ["projected-aabb", "rendered-silhouette"], "description": "Pose fitting strategy used inside cv-grounded composition. rendered-silhouette fits bounded X/Z, uniform scale, and Y-axis yaw with source masks, DepthPro dense depth sidecar crops, depth distributions, and projected GLB visible-surface point masks." },
                    "canonical_pose": { "type": "string", "enum": ["off", "heuristic", "render-sweep", "openai", "auto"], "description": "Canonical asset orientation strategy for cv-grounded composition." },
                    "scale_policy": { "type": "string", "enum": ["asset-preserving", "bounded-anisotropic", "free-anisotropic"], "description": "Generated asset scale policy. Defaults to asset-preserving to avoid skinny/distorted lifted tables." },
                    "max_pose_candidates": { "type": "integer", "description": "Maximum deterministic pose candidates per object." },
                    "save_pose_debug": { "type": "boolean", "description": "Write canonical pose, pose-fit candidate, and camera grounding artifacts." },
                    "ground_calibration": { "type": "string", "enum": ["depth-heuristic", "gpt"], "description": "Camera/floor calibration source after depth evidence is available. gpt asks the reasoning model for camera height/FOV/floor sanity, then writes the same EstimatedCamera/EstimatedFloorPlane evidence consumed by geometry." },
                    "instance_generation": { "type": "string", "enum": ["category-representative", "fine-grained-types"], "description": "Reusable-object generation granularity. category-representative generates one reusable asset per semantic category/reuse group; fine-grained-types uses GPT to split repeated same-category chairs into visually distinct asset types before image generation." },
                    "depth_provider": { "type": "string", "enum": ["none", "depth-pro"], "description": "Depth provider for CV-grounded scene composition." },
                    "locator": { "type": "string", "enum": ["manifest", "locate-anything"], "description": "Object locator for CV-grounded scene composition. Full scene-build defaults to locate-anything." },
                    "locate_anything_backend": { "type": "string", "enum": ["burn-native"], "description": "Optional backend override when locator is locate-anything." },
                    "segmentation_provider": { "type": "string", "enum": ["none", "bbox-prompt", "sam2", "sam3"], "description": "Optional mask provider for CV-grounded composition. bbox-prompt writes deterministic bbox masks; SAM modes require Burn-native support." },
                    "segmentation_precision": { "type": "string", "enum": ["f32", "f16", "bf16"], "description": "Segmentation artifact precision override." },
                    "segmentation_quantization": { "type": "string", "enum": ["none", "q8", "q4"], "description": "Segmentation artifact quantization override." },
                    "write_artifacts": { "type": "boolean", "description": "Write structured e2e artifacts such as selected candidates, asset outputs, grounded layout, commands, summary, and scene.bsn to output_dir. Defaults to true." },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean" },
                    "feedback": { "type": "boolean", "description": "Opt-in render-capture-feedback placement validation/refinement. The default flow relies on deterministic LocateAnything/SAM/DepthPro/visible-surface geometry." },
                    "feedback_iters": { "type": "integer", "description": "Maximum feedback iterations when feedback is enabled." },
                    "feedback_keep_viewer": { "type": "boolean", "description": "Leave the temporary feedback viewer running after completion." },
                    "feedback_capture_dir": { "type": "string", "description": "Optional feedback artifact directory. Defaults to output_dir/iterations." },
                    "feedback_threshold_profile": { "type": "string", "enum": ["loose", "standard", "strict"] },
                    "feedback_rotation_selector": { "type": "string", "enum": ["deterministic", "rendered-sweep", "openai"], "description": "Rotation candidate selector. deterministic uses geometry feedback; rendered-sweep renders per-object yaw candidates; openai asks the reasoning model to pick candidate_index values from source/render/candidate crops." },
                    "rotation_fit": { "type": "string", "enum": ["off", "depth-mask-ransac", "gpt-refine"], "description": "Extra pre-feedback object yaw fitting strategy. Defaults to off; rendered-silhouette pose fit is the normal geometric yaw/position/scale solver." },
                    "rotation_fit_max_gpt_rounds": { "type": "integer", "description": "Maximum GPT refinement rounds when rotation_fit is gpt-refine." },
                    "rotation_fit_min_mask_iou": { "type": "number", "description": "Minimum source-mask IoU required before applying a depth/mask/surface rotation candidate." },
                    "rotation_fit_max_depth_error_m": { "type": "number", "description": "Maximum accepted median visible-surface depth error in meters; depth-distribution loss is also reported." },
                    "rotation_fit_write_artifacts": { "type": "boolean", "description": "Write rotation-fit candidate overlays, report JSON, and HTML review artifacts." },
                    "object_pose_refinement": { "type": "string", "enum": ["off", "geometry", "gated-gpt", "always-gpt"], "description": "Object-set pose refinement after the generic visible-surface fit. geometry reruns deterministic mask/depth fitting on the selected object set; gated-gpt additionally marks failed or ambiguous fits for bounded GPT candidate selection; always-gpt marks every selected object." },
                    "object_pose_refinement_set": { "type": "string", "enum": ["tables", "large-seating", "tables-and-large-seating", "all-furniture"], "description": "Object set targeted by object_pose_refinement. Defaults to tables-and-large-seating." },
                    "final_yaw_refinement": { "type": "string", "enum": ["off", "gated-gpt", "always-gpt"], "description": "Final full-scene contextual yaw selector. It renders all scene objects while sweeping only one target object's Y-axis yaw, then applies candidate_index selections without changing translation or scale." },
                    "final_yaw_refinement_set": { "type": "string", "enum": ["tables", "large-seating", "tables-and-large-seating", "all-furniture"], "description": "Object set targeted by final_yaw_refinement." },
                    "final_yaw_confidence_threshold": { "type": "number", "description": "Minimum bounded GPT confidence required before a final yaw candidate may apply." },
                    "final_yaw_max_candidates": { "type": "integer", "description": "Maximum final contextual yaw candidates per target object." },
                    "feedback_rubric_scorer": { "type": "string", "enum": ["off", "openai"], "description": "Optional scene-level source-vs-render rubric scorer. openai writes strict JSON diagnostics and contributes to feedback candidate selection." }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_ground",
            "description": "Recompute source-scene composition from an existing object manifest and asset bindings without regenerating object images or TRELLIS assets. When grounding evidence is omitted, the default path reruns LocateAnything, SAM, and DepthPro from the source image.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "manifest": { "type": "object", "description": "SceneObjectManifest from scene_build_from_image or scene_plan_objects." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path/cache_key/local_aabb."
                    },
                    "grounding_evidence": { "type": "object", "description": "Optional SceneGroundingEvidence. When omitted, cv-grounded mode refreshes LocateAnything/SAM/DepthPro evidence unless those providers are explicitly disabled." },
                    "output_dir": { "type": "string" },
                    "composition_mode": { "type": "string", "enum": ["heuristic", "cv-grounded"] },
                    "pose_fit": { "type": "string", "enum": ["projected-aabb", "rendered-silhouette"], "description": "Pose fitting strategy used inside cv-grounded composition. rendered-silhouette fits bounded X/Z, uniform scale, and Y-axis yaw with source masks, DepthPro dense depth sidecar crops, depth distributions, and projected GLB visible-surface point masks." },
                    "canonical_pose": { "type": "string", "enum": ["off", "heuristic", "render-sweep", "openai", "auto"] },
                    "scale_policy": { "type": "string", "enum": ["asset-preserving", "bounded-anisotropic", "free-anisotropic"], "description": "Generated asset scale policy. Defaults to asset-preserving." },
                    "max_pose_candidates": { "type": "integer" },
                    "save_pose_debug": { "type": "boolean" },
                    "ground_calibration": { "type": "string", "enum": ["depth-heuristic", "gpt"], "description": "Camera/floor calibration source when grounding evidence is refreshed." },
                    "depth_provider": { "type": "string", "enum": ["none", "depth-pro"] },
                    "locator": { "type": "string", "enum": ["manifest", "locate-anything"] },
                    "locate_anything_backend": { "type": "string", "enum": ["burn-native"], "description": "Optional backend override when locator is locate-anything. Defaults to the server --locate-anything-backend setting." },
                    "segmentation_provider": { "type": "string", "enum": ["none", "bbox-prompt", "sam2", "sam3"], "description": "Optional mask provider for CV-grounded composition." },
                    "segmentation_precision": { "type": "string", "enum": ["f32", "f16", "bf16"] },
                    "segmentation_quantization": { "type": "string", "enum": ["none", "q8", "q4"] },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean" },
                    "feedback": { "type": "boolean" },
                    "feedback_iters": { "type": "integer" },
                    "feedback_keep_viewer": { "type": "boolean" },
                    "feedback_capture_dir": { "type": "string" },
                    "promote_to_catalog": { "type": "boolean", "description": "Add the recomposed scene snapshot to the shared Bevy scene catalog/cache. Defaults to true so composition-only reruns appear in the scene catalog." },
                    "feedback_threshold_profile": { "type": "string", "enum": ["loose", "standard", "strict"] },
                    "feedback_rotation_selector": { "type": "string", "enum": ["deterministic", "rendered-sweep", "openai"], "description": "Rotation candidate selector. deterministic uses geometry feedback; rendered-sweep renders per-object yaw candidates; openai asks the reasoning model to pick candidate_index values from source/render/candidate crops." },
                    "rotation_fit": { "type": "string", "enum": ["off", "depth-mask-ransac", "gpt-refine"], "description": "Extra pre-feedback object yaw fitting strategy. Defaults to off; rendered-silhouette pose fit is the normal geometric yaw/position/scale solver." },
                    "rotation_fit_max_gpt_rounds": { "type": "integer", "description": "Maximum GPT refinement rounds when rotation_fit is gpt-refine." },
                    "rotation_fit_min_mask_iou": { "type": "number", "description": "Minimum source-mask IoU required before applying a depth/mask/surface rotation candidate." },
                    "rotation_fit_max_depth_error_m": { "type": "number", "description": "Maximum accepted median visible-surface depth error in meters; depth-distribution loss is also reported." },
                    "rotation_fit_write_artifacts": { "type": "boolean", "description": "Write rotation-fit candidate overlays, report JSON, and HTML review artifacts." },
                    "object_pose_refinement": { "type": "string", "enum": ["off", "geometry", "gated-gpt", "always-gpt"], "description": "Object-set pose refinement after the generic visible-surface fit. geometry reruns deterministic mask/depth fitting on the selected object set; gated-gpt additionally marks failed or ambiguous fits for bounded GPT candidate selection; always-gpt marks every selected object." },
                    "object_pose_refinement_set": { "type": "string", "enum": ["tables", "large-seating", "tables-and-large-seating", "all-furniture"], "description": "Object set targeted by object_pose_refinement. Defaults to tables-and-large-seating." },
                    "final_yaw_refinement": { "type": "string", "enum": ["off", "gated-gpt", "always-gpt"], "description": "Final full-scene contextual yaw selector. It renders all scene objects while sweeping only one target object's Y-axis yaw, then applies candidate_index selections without changing translation or scale." },
                    "final_yaw_refinement_set": { "type": "string", "enum": ["tables", "large-seating", "tables-and-large-seating", "all-furniture"], "description": "Object set targeted by final_yaw_refinement." },
                    "final_yaw_confidence_threshold": { "type": "number", "description": "Minimum bounded GPT confidence required before a final yaw candidate may apply." },
                    "final_yaw_max_candidates": { "type": "integer", "description": "Maximum final contextual yaw candidates per target object." },
                    "feedback_rubric_scorer": { "type": "string", "enum": ["off", "openai"], "description": "Optional scene-level source-vs-render rubric scorer. openai writes strict JSON diagnostics and contributes to feedback candidate selection." }
                },
                "required": ["source_scene_path", "manifest", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_plan_bsn",
            "description": "Plan a grounded restricted synth_scene_v1 BSN from an existing object manifest and generated asset bindings, using source-image bbox contact points, class scale priors, and asset AABBs; then validate commands before optional Bevy apply.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "manifest": { "type": "object", "description": "SceneObjectManifest returned by scene_plan_objects or scene_build_from_image." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path or cache_key. local_aabb is used for ground-plane bottom-fit and scale when present."
                    },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean", "description": "When true, send commands to Bevy scene bridge." }
                },
                "required": ["manifest", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_apply_bsn",
            "description": "Validate restricted synth_scene_v1 BSN against explicit generated asset bindings and optionally apply it to the Bevy scene bridge.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bsn": { "type": "string", "description": "Restricted synth_scene_v1 text." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path or cache_key."
                    },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean", "description": "When true, send commands to Bevy scene bridge." }
                },
                "required": ["bsn", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_status",
            "description": "Read the latest Bevy scene bridge status, including cache entries, world items, camera, and screenshots.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_project_status",
            "description": "Read camera/world status plus per-object projected screen-space evidence from the Bevy scene bridge.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_list_assets",
            "description": "List cached assets and spawned world items from the latest Bevy scene bridge status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_cached",
            "description": "Spawn an asset already present in the Bevy mesh/splat cache.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["cache_key"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_path",
            "description": "Spawn a GLB mesh asset file directly into the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "GLB mesh path to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_delete",
            "description": "Delete a spawned cached asset by cache key, delete the selection, or clear selection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to delete." },
                    "selected": { "type": "boolean", "description": "Delete the current selection when true; clear selection when false and no cache key is provided." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_clear",
            "description": "Clear all spawned cache-backed scene items from the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_set_camera",
            "description": "Set the Bevy scene camera transform and optional orbit state.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "focus": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "yaw": { "type": "number" },
                    "pitch": { "type": "number" },
                    "radius": { "type": "number" },
                    "vertical_fov": { "type": "number" }
                },
                "required": ["translation", "rotation"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_save",
            "description": "Flush the Bevy scene cache/world state.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_capture",
            "description": "Capture a screenshot from the Bevy primary window and wait for the image file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_path": { "type": "string", "description": "Screenshot path to write." }
                },
                "required": ["output_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_compose_assets",
            "description": "Create deterministic Bevy placements from source-image object boxes and generated asset bindings; optionally apply them to the live scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4, "description": "Normalized source-image box [x_min, y_min, x_max, y_max]." }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "assets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reference_id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "path": { "type": "string" },
                                "cache_key": { "type": "string" },
                                "local_aabb": { "type": "object", "description": "Optional asset local bounds {min:[x,y,z], max:[x,y,z]} for ground-fit scaling." },
                                "select": { "type": "boolean" }
                            },
                            "additionalProperties": false
                        }
                    },
            "apply": { "type": "boolean", "description": "When true, send spawn commands to the configured Bevy scene bridge." },
                    "clear_existing": { "type": "boolean", "description": "When true, clear existing scene instances before placing generated assets." },
                    "layout_width": { "type": "number" },
                    "layout_depth": { "type": "number" },
                    "y": { "type": "number" },
                    "min_scale": { "type": "number" },
                    "scale_multiplier": { "type": "number" }
                },
                "required": ["reference_objects", "assets"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_validate_layout",
            "description": "Validate a composed Bevy scene against source-image object boxes using semantic label matching, object counts, normalized layout, and optional screenshot image similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "scene_status": { "type": "object", "description": "Optional scene status JSON. Omit to read the configured scene_status_path." },
                    "source_image_path": { "type": "string" },
                    "rendered_image_path": { "type": "string" },
                    "thresholds": {
                        "type": "object",
                        "properties": {
                            "min_semantic_score": { "type": "number" },
                            "min_layout_score": { "type": "number" },
                            "min_overall_score": { "type": "number" },
                            "max_extra_objects": { "type": "integer" },
                            "min_image_similarity": { "type": "number" }
                        },
                        "additionalProperties": false
                    }
                },
                "required": ["reference_objects"],
                "additionalProperties": false
            }
        }),
    ]
}

pub(crate) fn read_framed_json<R: BufRead + Read>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length = None;
    let mut saw_header = false;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            if !saw_header {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading MCP headers",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        saw_header = true;
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("Content-Length")
        {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid Content-Length header: {err}"),
                )
            })?;
            content_length = Some(parsed);
        }
    }

    let content_length = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "missing Content-Length header in MCP message",
        )
    })?;
    let mut payload = vec![0u8; content_length];
    reader.read_exact(&mut payload)?;
    let value = serde_json::from_slice::<Value>(&payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid MCP JSON payload: {err}"),
        )
    })?;
    Ok(Some(value))
}

pub(crate) fn write_framed_json<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(value).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize MCP JSON payload: {err}"),
        )
    })?;
    write!(writer, "Content-Length: {}\r\n\r\n", payload.len())?;
    writer.write_all(&payload)?;
    writer.flush()
}

#[derive(Debug, Deserialize)]
pub(crate) struct RpcRequest {
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ForegroundToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_image_path: Option<PathBuf>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MeshToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_mesh_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<MeshOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SplatToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_splat_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ImagesToAssetsToolArgs {
    #[serde(default, alias = "image_paths")]
    pub input_image_paths: Vec<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub output_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub batch_vram_mb: Option<u64>,
    #[serde(default)]
    pub trellis_pbr: Option<bool>,
    #[serde(default)]
    pub trellis_pbr_texture_size: Option<usize>,
    #[serde(default)]
    pub promote_to_catalog: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScenePrepareBuildArgs {
    pub source_scene_path: PathBuf,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
    #[serde(default)]
    pub allow_catalog_reuse: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneGenerateObjectImagesArgs {
    pub source_scene_path: PathBuf,
    pub manifest: SceneObjectManifest,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
}

#[derive(Debug, Deserialize)]
pub struct SceneBuildFromImageArgs {
    pub source_scene_path: PathBuf,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub candidate_retry_attempts: Option<usize>,
    #[serde(default)]
    pub candidate_batch_size: Option<usize>,
    #[serde(default)]
    pub min_reconstruction_score: Option<f32>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
    #[serde(default)]
    pub allow_catalog_reuse: bool,
    #[serde(default = "default_scene_lift_assets")]
    pub lift_assets: bool,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub batch_vram_mb: Option<u64>,
    #[serde(default)]
    pub trellis_pbr: Option<bool>,
    #[serde(default)]
    pub trellis_pbr_texture_size: Option<usize>,
    #[serde(default = "default_scene_promote_to_catalog")]
    pub promote_to_catalog: bool,
    #[serde(default = "default_scene_composition_mode")]
    pub composition_mode: SceneCompositionMode,
    #[serde(default = "default_scene_pose_fit_mode")]
    pub pose_fit: ScenePoseFitMode,
    #[serde(default = "default_scene_canonical_pose_mode")]
    pub canonical_pose: SceneCanonicalPoseMode,
    #[serde(default = "default_scene_scale_policy")]
    pub scale_policy: SceneScalePolicy,
    #[serde(default = "default_scene_max_pose_candidates")]
    pub max_pose_candidates: usize,
    #[serde(default = "default_scene_write_artifacts")]
    pub save_pose_debug: bool,
    #[serde(default = "default_scene_ground_calibration_mode")]
    pub ground_calibration: SceneGroundCalibrationMode,
    #[serde(default = "default_scene_instance_generation_mode")]
    pub instance_generation: SceneInstanceGenerationMode,
    #[serde(default = "default_scene_depth_provider")]
    pub depth_provider: SceneDepthProvider,
    #[serde(default = "default_scene_locator_provider")]
    pub locator: SceneLocatorProvider,
    #[serde(default)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,
    #[serde(default)]
    pub segmentation_provider: Option<SceneSegmentationProvider>,
    #[serde(default)]
    pub segmentation_precision: Option<SceneSegmentationPrecision>,
    #[serde(default)]
    pub segmentation_quantization: Option<SceneSegmentationQuantization>,
    #[serde(default = "default_scene_write_artifacts")]
    pub write_artifacts: bool,
    #[serde(default)]
    pub apply: bool,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
    #[serde(default = "default_scene_feedback")]
    pub feedback: bool,
    #[serde(default = "default_scene_feedback_iters")]
    pub feedback_iters: usize,
    #[serde(default)]
    pub feedback_keep_viewer: bool,
    #[serde(default)]
    pub feedback_capture_dir: Option<PathBuf>,
    #[serde(default = "default_scene_feedback_threshold_profile")]
    pub feedback_threshold_profile: FeedbackThresholdProfile,
    #[serde(default = "default_feedback_rotation_selector")]
    pub feedback_rotation_selector: FeedbackRotationSelector,
    #[serde(default = "default_scene_rotation_fit_mode")]
    pub rotation_fit: SceneRotationFitMode,
    #[serde(default = "default_scene_rotation_fit_max_gpt_rounds")]
    pub rotation_fit_max_gpt_rounds: usize,
    #[serde(default = "default_scene_rotation_fit_min_mask_iou")]
    pub rotation_fit_min_mask_iou: f32,
    #[serde(default = "default_scene_rotation_fit_max_depth_error_m")]
    pub rotation_fit_max_depth_error_m: f32,
    #[serde(default = "default_scene_rotation_fit_write_artifacts")]
    pub rotation_fit_write_artifacts: bool,
    #[serde(default = "default_scene_object_pose_refinement")]
    pub object_pose_refinement: SceneObjectPoseRefinementMode,
    #[serde(default = "default_scene_object_pose_refinement_set")]
    pub object_pose_refinement_set: SceneObjectPoseRefinementSet,
    #[serde(default = "default_scene_final_yaw_refinement")]
    pub final_yaw_refinement: SceneFinalYawRefinementMode,
    #[serde(default = "default_scene_final_yaw_refinement_set")]
    pub final_yaw_refinement_set: SceneObjectPoseRefinementSet,
    #[serde(default = "default_scene_final_yaw_confidence_threshold")]
    pub final_yaw_confidence_threshold: f32,
    #[serde(default = "default_scene_final_yaw_max_candidates")]
    pub final_yaw_max_candidates: usize,
    #[serde(default = "default_feedback_rubric_scorer")]
    pub feedback_rubric_scorer: FeedbackRubricScorer,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneGroundToolArgs {
    pub source_scene_path: PathBuf,
    pub manifest: SceneObjectManifest,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub grounding_evidence: Option<SceneGroundingEvidence>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default = "default_scene_composition_mode")]
    pub composition_mode: SceneCompositionMode,
    #[serde(default = "default_scene_pose_fit_mode")]
    pub pose_fit: ScenePoseFitMode,
    #[serde(default = "default_scene_canonical_pose_mode")]
    pub canonical_pose: SceneCanonicalPoseMode,
    #[serde(default = "default_scene_scale_policy")]
    pub scale_policy: SceneScalePolicy,
    #[serde(default = "default_scene_max_pose_candidates")]
    pub max_pose_candidates: usize,
    #[serde(default = "default_scene_write_artifacts")]
    pub save_pose_debug: bool,
    #[serde(default = "default_scene_ground_calibration_mode")]
    pub ground_calibration: SceneGroundCalibrationMode,
    #[serde(default = "default_scene_depth_provider")]
    pub depth_provider: SceneDepthProvider,
    #[serde(default = "default_scene_locator_provider")]
    pub locator: SceneLocatorProvider,
    #[serde(default)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,
    #[serde(default)]
    pub segmentation_provider: Option<SceneSegmentationProvider>,
    #[serde(default)]
    pub segmentation_precision: Option<SceneSegmentationPrecision>,
    #[serde(default)]
    pub segmentation_quantization: Option<SceneSegmentationQuantization>,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
    #[serde(default)]
    pub apply: bool,
    #[serde(default)]
    pub feedback: bool,
    #[serde(default = "default_scene_feedback_iters")]
    pub feedback_iters: usize,
    #[serde(default)]
    pub feedback_keep_viewer: bool,
    #[serde(default)]
    pub feedback_capture_dir: Option<PathBuf>,
    #[serde(default = "default_scene_promote_to_catalog")]
    pub promote_to_catalog: bool,
    #[serde(default = "default_scene_feedback_threshold_profile")]
    pub feedback_threshold_profile: FeedbackThresholdProfile,
    #[serde(default = "default_feedback_rotation_selector")]
    pub feedback_rotation_selector: FeedbackRotationSelector,
    #[serde(default = "default_scene_rotation_fit_mode")]
    pub rotation_fit: SceneRotationFitMode,
    #[serde(default = "default_scene_rotation_fit_max_gpt_rounds")]
    pub rotation_fit_max_gpt_rounds: usize,
    #[serde(default = "default_scene_rotation_fit_min_mask_iou")]
    pub rotation_fit_min_mask_iou: f32,
    #[serde(default = "default_scene_rotation_fit_max_depth_error_m")]
    pub rotation_fit_max_depth_error_m: f32,
    #[serde(default = "default_scene_rotation_fit_write_artifacts")]
    pub rotation_fit_write_artifacts: bool,
    #[serde(default = "default_scene_object_pose_refinement")]
    pub object_pose_refinement: SceneObjectPoseRefinementMode,
    #[serde(default = "default_scene_object_pose_refinement_set")]
    pub object_pose_refinement_set: SceneObjectPoseRefinementSet,
    #[serde(default = "default_scene_final_yaw_refinement")]
    pub final_yaw_refinement: SceneFinalYawRefinementMode,
    #[serde(default = "default_scene_final_yaw_refinement_set")]
    pub final_yaw_refinement_set: SceneObjectPoseRefinementSet,
    #[serde(default = "default_scene_final_yaw_confidence_threshold")]
    pub final_yaw_confidence_threshold: f32,
    #[serde(default = "default_scene_final_yaw_max_candidates")]
    pub final_yaw_max_candidates: usize,
    #[serde(default = "default_feedback_rubric_scorer")]
    pub feedback_rubric_scorer: FeedbackRubricScorer,
}

#[derive(Clone, Debug)]
pub(crate) struct SceneFeedbackOptions {
    pub(crate) max_iters: usize,
    pub(crate) keep_viewer: bool,
    pub(crate) capture_dir: Option<PathBuf>,
    pub(crate) threshold_profile: FeedbackThresholdProfile,
    pub(crate) rotation_selector: FeedbackRotationSelector,
    pub(crate) rotation_fit: SceneRotationFitMode,
    pub(crate) rotation_fit_max_gpt_rounds: usize,
    pub(crate) rotation_fit_min_mask_iou: f32,
    pub(crate) rotation_fit_max_depth_error_m: f32,
    pub(crate) rotation_fit_write_artifacts: bool,
    pub(crate) rubric_scorer: FeedbackRubricScorer,
    pub(crate) scale_policy: SceneScalePolicy,
    pub(crate) grounding_evidence: Option<SceneGroundingEvidence>,
}

pub(crate) struct SceneFeedbackIterationContext<'a> {
    pub(crate) capture_root: &'a Path,
    pub(crate) manifest: &'a SceneObjectManifest,
    pub(crate) asset_bindings: &'a [SceneAssetBinding],
    pub(crate) grounded_layout: &'a GroundedSceneLayout,
    pub(crate) initial_commands: Vec<Value>,
    pub(crate) max_iters: usize,
    pub(crate) threshold_profile: FeedbackThresholdProfile,
    pub(crate) rotation_selector: FeedbackRotationSelector,
    pub(crate) rotation_fit: SceneRotationFitMode,
    pub(crate) rotation_fit_max_gpt_rounds: usize,
    pub(crate) rotation_fit_min_mask_iou: f32,
    pub(crate) rotation_fit_max_depth_error_m: f32,
    pub(crate) rotation_fit_write_artifacts: bool,
    pub(crate) rubric_scorer: FeedbackRubricScorer,
    pub(crate) scale_policy: SceneScalePolicy,
    pub(crate) grounding_evidence: Option<&'a SceneGroundingEvidence>,
}

#[derive(Clone, Debug)]
pub(crate) struct SceneCompositionCandidate {
    pub(crate) mode: SceneCompositionMode,
    pub(crate) layout: GroundedSceneLayout,
    pub(crate) plan: ScenePlan,
    pub(crate) commands: Vec<Value>,
}

#[derive(Clone, Debug)]
pub(crate) struct SceneCompositionFeedbackSelection {
    pub(crate) candidate: SceneCompositionCandidate,
    pub(crate) commands: Vec<Value>,
    pub(crate) feedback: Value,
    pub(crate) candidate_reports: Vec<Value>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneFeedbackThresholds {
    pub(crate) max_center_error: f32,
    pub(crate) max_contact_error: f32,
    pub(crate) max_area_log2_error: f32,
    pub(crate) min_overall_score: f32,
    pub(crate) max_seating_table_overlap_fraction: f32,
    pub(crate) max_seating_table_penetration_m: f32,
    pub(crate) max_seating_seating_overlap_fraction: f32,
    pub(crate) max_seating_seating_penetration_m: f32,
}

impl FeedbackThresholdProfile {
    pub(crate) fn thresholds(self) -> SceneFeedbackThresholds {
        match self {
            Self::Loose => SceneFeedbackThresholds {
                max_center_error: 0.18,
                max_contact_error: 0.22,
                max_area_log2_error: 1.20,
                min_overall_score: 0.55,
                max_seating_table_overlap_fraction: 0.45,
                max_seating_table_penetration_m: 0.30,
                max_seating_seating_overlap_fraction: 0.16,
                max_seating_seating_penetration_m: 0.08,
            },
            Self::Standard => SceneFeedbackThresholds {
                max_center_error: 0.10,
                max_contact_error: 0.14,
                max_area_log2_error: 0.65,
                min_overall_score: 0.65,
                max_seating_table_overlap_fraction: 0.35,
                max_seating_table_penetration_m: 0.25,
                max_seating_seating_overlap_fraction: 0.10,
                max_seating_seating_penetration_m: 0.05,
            },
            Self::Strict => SceneFeedbackThresholds {
                max_center_error: 0.06,
                max_contact_error: 0.09,
                max_area_log2_error: 0.35,
                min_overall_score: 0.82,
                max_seating_table_overlap_fraction: 0.25,
                max_seating_table_penetration_m: 0.18,
                max_seating_seating_overlap_fraction: 0.06,
                max_seating_seating_penetration_m: 0.03,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScenePlanBsnArgs {
    pub manifest: SceneObjectManifest,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub apply: bool,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneApplyBsnArgs {
    pub bsn: String,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub clear_existing: bool,
    #[serde(default)]
    pub apply: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneSpawnCachedArgs {
    pub cache_key: String,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneSpawnPathArgs {
    pub path: PathBuf,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct SceneDeleteArgs {
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub selected: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneSetCameraArgs {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    #[serde(default)]
    pub focus: Option<[f32; 3]>,
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub vertical_fov: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneCaptureArgs {
    #[serde(alias = "path")]
    pub output_path: PathBuf,
}
