use crate::prelude::*;
use crate::server::McpServer;

pub(crate) fn default_output_path(input: &Path, suffix: &str, ext: &str) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{suffix}.{ext}"))
}

#[derive(Debug)]
pub(crate) struct WrittenAsset {
    pub(crate) output_path: PathBuf,
    pub(crate) output_format: AssetOutputFormat,
    pub(crate) asset_kind: &'static str,
    pub(crate) vertices: Option<usize>,
    pub(crate) faces: Option<usize>,
    pub(crate) gaussians: Option<usize>,
    pub(crate) local_aabb: Option<SceneAssetAabb>,
    pub(crate) material: Option<Value>,
    pub(crate) mesh_quality: Option<Value>,
    pub(crate) mesh_quality_failures: Vec<String>,
    pub(crate) catalog_entry: Option<CachedMeshMetadata>,
}

pub(crate) fn write_asset_output(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    requested_format: AssetOutputFormat,
    asset: SynthesisAsset,
    target_faces: Option<usize>,
    catalog_cache: Option<&mut MeshCache>,
) -> Result<WrittenAsset, String> {
    match asset {
        SynthesisAsset::Mesh(mesh) => {
            if matches!(
                requested_format,
                AssetOutputFormat::Splat | AssetOutputFormat::Ply
            ) {
                return Err(format!(
                    "mesh synthesis cannot be written as {}",
                    requested_format.as_str()
                ));
            }
            let mesh = apply_mesh_decimation(mesh, target_faces)
                .map_err(|err| format!("mesh decimation failed: {err}"))?;
            let quality = mesh_quality_metrics(&mesh);
            let quality_failures = mesh_quality_failures(&quality);
            let mesh_quality = Some(
                serde_json::to_value(&quality)
                    .map_err(|err| format!("failed to serialize mesh quality metrics: {err}"))?,
            );
            let catalog_mesh = catalog_cache
                .as_ref()
                .map(|_| cached_mesh_from_runtime_mesh(&mesh));
            let output_path =
                resolve_asset_output_path(input_path, output_dir, explicit_output, "_mesh", "glb");
            write_glb_mesh(output_path.as_path(), &mesh)?;
            let material = mesh.material.map(|value| {
                json!({
                    "base_color": value.base_color,
                    "metallic": value.metallic,
                    "roughness": value.roughness,
                    "alpha": value.alpha,
                })
            });
            let catalog_entry = match (catalog_cache, catalog_mesh.as_ref()) {
                (Some(cache), Some(cached_mesh)) => Some(
                    cache
                        .upsert_mesh_for_image(input_path, cached_mesh)
                        .map_err(|err| {
                            format!("failed to promote mesh to shared catalog: {err}")
                        })?,
                ),
                _ => None,
            };
            Ok(WrittenAsset {
                output_path,
                output_format: AssetOutputFormat::Glb,
                asset_kind: "mesh",
                vertices: Some(mesh.vertices.len()),
                faces: Some(mesh.faces.len()),
                gaussians: None,
                local_aabb: mesh_scene_aabb(&mesh),
                material,
                mesh_quality,
                mesh_quality_failures: quality_failures,
                catalog_entry,
            })
        }
        SynthesisAsset::GaussianSplat(splats) => {
            if matches!(requested_format, AssetOutputFormat::Glb) {
                return Err("Gaussian splats cannot be written as glb".to_string());
            }
            let output_format = match requested_format {
                AssetOutputFormat::Ply => AssetOutputFormat::Ply,
                _ => AssetOutputFormat::Splat,
            };
            let output_path = resolve_asset_output_path(
                input_path,
                output_dir,
                explicit_output,
                "_splat",
                output_format.as_str(),
            );
            write_splat_asset(output_path.as_path(), &splats, output_format)?;
            let catalog_entry = match catalog_cache {
                Some(cache) => Some(
                    cache
                        .upsert_gaussian_splat_for_image(input_path, &splats)
                        .map_err(|err| {
                            format!("failed to promote Gaussian splat to shared catalog: {err}")
                        })?,
                ),
                None => None,
            };
            Ok(WrittenAsset {
                output_path,
                output_format,
                asset_kind: "gaussian_splat",
                vertices: None,
                faces: None,
                gaussians: Some(splats.len()),
                local_aabb: None,
                material: None,
                mesh_quality: None,
                mesh_quality_failures: Vec::new(),
                catalog_entry,
            })
        }
    }
}

pub(crate) fn cached_mesh_from_runtime_mesh(mesh: &Mesh) -> CachedSynthMesh {
    CachedSynthMesh {
        mesh: CachedTripoMesh {
            vertices: mesh.vertices.clone(),
            faces: mesh.faces.clone(),
        },
        uvs: mesh.uvs.clone(),
        normals: mesh.normals.clone(),
        material: mesh.material.map(|material| CachedSynthMeshMaterial {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            alpha: material.alpha,
        }),
        pbr_textures: mesh
            .pbr_textures
            .clone()
            .map(|textures| CachedSynthMeshPbrTextures {
                base_color: cached_texture_from_runtime_texture(textures.base_color),
                metallic_roughness: cached_texture_from_runtime_texture(
                    textures.metallic_roughness,
                ),
                normal: textures.normal.map(cached_texture_from_runtime_texture),
                emissive: textures.emissive.map(cached_texture_from_runtime_texture),
                occlusion: textures.occlusion.map(cached_texture_from_runtime_texture),
            }),
    }
}

pub(crate) fn mesh_scene_aabb(mesh: &Mesh) -> Option<SceneAssetAabb> {
    let mut iter = mesh.vertices.iter();
    let first = *iter.next()?;
    let mut min = first;
    let mut max = first;
    for vertex in iter {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Some(SceneAssetAabb { min, max })
}

pub(crate) fn cached_aabb_to_scene(value: CachedAssetAabb) -> SceneAssetAabb {
    SceneAssetAabb {
        min: value.min,
        max: value.max,
    }
}

pub(crate) fn inferred_scene_asset_frame(
    label: &str,
    aliases: &[String],
    local_aabb: Option<SceneAssetAabb>,
    target_footprint_m: Option<[f32; 2]>,
) -> SceneAssetFrame {
    let descriptor = format!("{} {}", label, aliases.join(" ")).to_ascii_lowercase();
    let is_table = descriptor.contains("table") || descriptor.contains("desk");
    let is_seating = descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("bench");
    let x_major = local_aabb
        .map(|aabb| aabb.max[0] - aabb.min[0] > (aabb.max[2] - aabb.min[2]) * 1.15)
        .unwrap_or(false);
    let yaw_offset_degrees = if is_table && x_major { 90.0 } else { 0.0 };
    let symmetry = if is_table {
        SceneAssetSymmetry::Axis180
    } else if is_seating {
        SceneAssetSymmetry::Bilateral
    } else {
        SceneAssetSymmetry::Unknown
    };
    let source = if is_table && local_aabb.is_some() {
        SceneAssetFrameSource::AabbHeuristic
    } else {
        SceneAssetFrameSource::DescriptorHeuristic
    };
    let confidence = if is_table && local_aabb.is_some() {
        0.70
    } else if is_seating {
        0.55
    } else {
        0.35
    };
    SceneAssetFrame {
        yaw_offset_degrees,
        footprint_m: target_footprint_m,
        symmetry: Some(symmetry),
        confidence: Some(confidence),
        source: Some(source),
    }
}

pub(crate) fn cached_texture_from_runtime_texture(
    texture: burn_synth::MeshTexture,
) -> CachedSynthMeshTexture {
    CachedSynthMeshTexture {
        width: texture.width,
        height: texture.height,
        rgba8: texture.rgba8,
    }
}

pub(crate) fn resolve_asset_output_path(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    if let Some(path) = explicit_output {
        if path.extension().is_none() || path.is_dir() {
            let stem = input_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("asset");
            return path.join(format!("{stem}{suffix}.{ext}"));
        }
        return path;
    }
    if let Some(dir) = output_dir {
        let stem = input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("asset");
        return dir.join(format!("{stem}{suffix}.{ext}"));
    }
    default_output_path(input_path, suffix, ext)
}

pub(crate) fn write_splat_asset(
    path: &Path,
    splats: &burn_synth::triposplat::GaussianSplatCloud,
    format: AssetOutputFormat,
) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    match format {
        AssetOutputFormat::Ply => splats.write_ply(path),
        AssetOutputFormat::Splat | AssetOutputFormat::Auto => splats.write_splat(path),
        AssetOutputFormat::Glb => Err("Gaussian splats cannot be written as glb".to_string()),
    }
}

pub(crate) fn apply_mesh_decimation(
    mesh: Mesh,
    target_faces: Option<usize>,
) -> Result<Mesh, String> {
    let target_faces = target_faces.filter(|value| *value > 0);
    let Some(target) = target_faces else {
        return Ok(mesh);
    };
    if mesh.pbr_textures.is_some() {
        return Ok(mesh);
    }
    if mesh.faces.len() <= target {
        return Ok(mesh);
    }
    decimate_mesh(&mesh, target)
}

pub(crate) fn decimate_mesh(mesh: &Mesh, target_faces: usize) -> Result<Mesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(mesh.vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        1.0,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(&indices, &adapter, target_index_count, 1.0, None);
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(mesh.vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(mesh.vertices.as_slice(), vertex_count, &remap);
    let uvs = if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        meshopt::remap_vertex_buffer(mesh.uvs.as_slice(), vertex_count, &remap)
    } else {
        Vec::new()
    };
    let normals = if mesh.normals.len() == mesh.vertices.len() && !mesh.normals.is_empty() {
        meshopt::remap_vertex_buffer(mesh.normals.as_slice(), vertex_count, &remap)
    } else {
        Vec::new()
    };
    let indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();
    Ok(Mesh {
        vertices,
        faces,
        uvs,
        normals,
        material: mesh.material,
        pbr_textures: mesh.pbr_textures.clone(),
    })
}

pub(crate) fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub(crate) fn next_scene_sequence() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let mut current = NEXT_SCENE_SEQUENCE.load(Ordering::Relaxed);
    loop {
        let next = now.max(current.saturating_add(1));
        match NEXT_SCENE_SEQUENCE.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return next,
            Err(value) => current = value,
        }
    }
}

pub(crate) fn default_scene_output_dir() -> PathBuf {
    PathBuf::from("tmp/runs").join(format!("{}_scene_openai_mcp", next_scene_sequence()))
}

pub(crate) fn default_scene_lift_assets() -> bool {
    true
}

pub(crate) fn default_scene_clear_existing() -> bool {
    true
}

pub(crate) fn default_scene_promote_to_catalog() -> bool {
    true
}

pub(crate) fn default_scene_write_artifacts() -> bool {
    true
}

pub(crate) fn default_scene_feedback() -> bool {
    true
}

pub(crate) fn default_scene_feedback_iters() -> usize {
    DEFAULT_SCENE_FEEDBACK_ITERS
}

pub(crate) fn default_scene_feedback_threshold_profile() -> FeedbackThresholdProfile {
    FeedbackThresholdProfile::Standard
}

pub(crate) fn default_feedback_rotation_selector() -> FeedbackRotationSelector {
    FeedbackRotationSelector::Deterministic
}

pub(crate) fn default_scene_composition_mode() -> SceneCompositionMode {
    SceneCompositionMode::CvGrounded
}

pub(crate) fn default_scene_pose_fit_mode() -> ScenePoseFitMode {
    ScenePoseFitMode::ProjectedAabb
}

pub(crate) fn default_scene_canonical_pose_mode() -> SceneCanonicalPoseMode {
    SceneCanonicalPoseMode::RenderSweep
}

pub(crate) fn default_scene_scale_policy() -> SceneScalePolicy {
    SceneScalePolicy::AssetPreserving
}

pub(crate) fn default_scene_max_pose_candidates() -> usize {
    32
}

pub(crate) fn default_scene_depth_provider() -> SceneDepthProvider {
    SceneDepthProvider::DepthPro
}

pub(crate) fn default_scene_locator_provider() -> SceneLocatorProvider {
    SceneLocatorProvider::Manifest
}

pub(crate) fn default_scene_build_locator_provider() -> SceneLocatorProvider {
    SceneLocatorProvider::LocateAnything
}

#[cfg(test)]
pub(crate) fn select_scene_candidates(
    manifest: &SceneObjectManifest,
    candidates: &[burn_synth_scene::ObjectImageCandidate],
) -> Result<Vec<Value>, String> {
    let selected = burn_synth_scene::select_object_image_candidates(
        manifest,
        candidates,
        DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE,
    )
    .map_err(|err| err.to_string())?;
    Ok(selected_candidates_to_values(&selected))
}

pub(crate) fn selected_candidates_to_values(
    selected: &[burn_synth_scene::SelectedObjectImageCandidate],
) -> Vec<Value> {
    selected
        .iter()
        .map(|candidate| {
            json!({
                "object_id": candidate.object_id,
                "reuse_group": candidate.reuse_group,
                "label": candidate.label,
                "image_path": candidate.image_path,
                "candidate_index": candidate.candidate_index,
                "score": candidate.score,
                "prompt_hash": candidate.prompt_hash,
            })
        })
        .collect()
}

pub(crate) fn record_stage(stage_report: &mut Vec<Value>, stage: &str, started: Instant) {
    stage_report.push(json!({
        "stage": stage,
        "elapsed_ms": elapsed_ms(started.elapsed()),
    }));
}

pub(crate) fn append_scene_progress_event(
    output_dir: &Path,
    event: &SceneBuildProgressEvent,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "create scene progress artifact directory {}: {err}",
            output_dir.display()
        )
    })?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("progress_events.jsonl"))
        .map_err(|err| {
            format!(
                "open scene progress artifact {}: {err}",
                output_dir.join("progress_events.jsonl").display()
            )
        })?;
    let line = serde_json::to_string(event)
        .map_err(|err| format!("serialize scene progress event: {err}"))?;
    writeln!(file, "{line}").map_err(|err| {
        format!(
            "write scene progress artifact {}: {err}",
            output_dir.join("progress_events.jsonl").display()
        )
    })
}

pub(crate) fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[derive(Clone, Debug, Default)]
struct TokenUsageAccumulator {
    requests: usize,
    reported_requests: usize,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    image_tokens: u64,
    text_tokens: u64,
}

impl TokenUsageAccumulator {
    fn record(&mut self, usage: Option<&Value>) {
        self.requests = self.requests.saturating_add(1);
        let Some(usage) = usage else {
            return;
        };
        if usage.is_null() {
            return;
        }
        self.reported_requests = self.reported_requests.saturating_add(1);
        let input_tokens = usage_u64(usage, &["input_tokens", "prompt_tokens"]);
        let output_tokens = usage_u64(usage, &["output_tokens", "completion_tokens"]);
        let total_tokens = usage_u64(usage, &["total_tokens"]).unwrap_or_else(|| {
            input_tokens
                .unwrap_or(0)
                .saturating_add(output_tokens.unwrap_or(0))
        });
        self.input_tokens = self.input_tokens.saturating_add(input_tokens.unwrap_or(0));
        self.output_tokens = self
            .output_tokens
            .saturating_add(output_tokens.unwrap_or(0));
        self.total_tokens = self.total_tokens.saturating_add(total_tokens);
        self.image_tokens = self.image_tokens.saturating_add(
            usage_pointer_u64(usage, "/input_tokens_details/image_tokens")
                .or_else(|| usage_pointer_u64(usage, "/prompt_tokens_details/image_tokens"))
                .unwrap_or(0),
        );
        self.text_tokens = self.text_tokens.saturating_add(
            usage_pointer_u64(usage, "/input_tokens_details/text_tokens")
                .or_else(|| usage_pointer_u64(usage, "/prompt_tokens_details/text_tokens"))
                .unwrap_or(0),
        );
    }

    fn as_value(&self) -> Value {
        json!({
            "requests": self.requests,
            "reported_requests": self.reported_requests,
            "unreported_requests": self.requests.saturating_sub(self.reported_requests),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
        })
    }
}

pub(crate) fn scene_token_usage_summary(provider_metadata: &Value) -> Value {
    let mut total = TokenUsageAccumulator::default();
    let mut by_stage = std::collections::BTreeMap::<String, TokenUsageAccumulator>::new();
    if let Some(requests) = provider_metadata.get("requests").and_then(Value::as_array) {
        for request in requests {
            let stage = request
                .get("operation")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let usage = request.get("usage");
            total.record(usage);
            by_stage.entry(stage).or_default().record(usage);
        }
    }
    let by_stage = by_stage
        .into_iter()
        .map(|(stage, usage)| {
            let mut value = usage.as_value();
            value["stage"] = json!(stage);
            value
        })
        .collect::<Vec<_>>();
    json!({
        "provider": provider_metadata.get("provider").cloned().unwrap_or(Value::Null),
        "total": total.as_value(),
        "by_stage": by_stage,
    })
}

pub(crate) fn attach_scene_token_usage(response: &mut Value) -> Value {
    let summary =
        scene_token_usage_summary(response.get("provider_metadata").unwrap_or(&Value::Null));
    response["token_usage"] = summary.clone();
    summary
}

fn usage_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(value_as_u64))
}

fn usage_pointer_u64(value: &Value, pointer: &str) -> Option<u64> {
    value.pointer(pointer).and_then(value_as_u64)
}

fn value_as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_f64().map(|value| value.max(0.0).round() as u64))
}

pub(crate) fn scene_build_summary(response: &Value, elapsed: Duration) -> Value {
    let rejected = response
        .pointer("/candidate_generation/rejected_objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mesh_quality_failures = response
        .get("mesh_quality_failures")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let failed_stage = response.get("failed_stage").cloned().unwrap_or(Value::Null);
    let selected_count = response
        .get("selected_candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let candidate_count = response
        .get("candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let assets = response
        .pointer("/asset_outputs/items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    json!({
                        "input_image_path": item.get("input_image_path").cloned().unwrap_or(Value::Null),
                        "output_path": item.get("output_path").cloned().unwrap_or(Value::Null),
                        "cache_key": item.get("cache_key").cloned().unwrap_or(Value::Null),
                        "vertices": item.get("vertices").cloned().unwrap_or(Value::Null),
                        "faces": item.get("faces").cloned().unwrap_or(Value::Null),
                        "local_aabb": item.get("local_aabb").cloned().unwrap_or(Value::Null),
                        "mesh_quality": item.get("mesh_quality").cloned().unwrap_or(Value::Null),
                        "mesh_quality_failures": item.get("mesh_quality_failures").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let placements = response
        .pointer("/grounded_layout/placements")
        .and_then(Value::as_array)
        .map(|placements| {
            placements
                .iter()
                .map(|placement| {
                    json!({
                        "object_id": placement.get("object_id").cloned().unwrap_or(Value::Null),
                        "asset_id": placement.get("asset_id").cloned().unwrap_or(Value::Null),
                        "translation": placement.get("translation").cloned().unwrap_or(Value::Null),
                        "scale": placement.get("scale").cloned().unwrap_or(Value::Null),
                        "target_footprint_m": placement.get("target_footprint_m").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let feedback_enabled = response
        .pointer("/feedback/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let feedback_accepted = response
        .pointer("/feedback/accepted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let feedback_gate_passed = !feedback_enabled || feedback_accepted;
    json!({
        "ok": rejected.is_empty()
            && mesh_quality_failures.is_empty()
            && failed_stage.is_null()
            && feedback_gate_passed,
        "elapsed_ms": elapsed_ms(elapsed),
        "source_scene_path": response.pointer("/manifest/source_scene_path").cloned().unwrap_or(Value::Null),
        "failed_stage": failed_stage,
        "next_action": response.get("next_action").cloned().unwrap_or(Value::Null),
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "rejected_objects": rejected,
        "mesh_quality_failures": mesh_quality_failures,
        "asset_lift_attempts": response.get("asset_lift_attempts").cloned().unwrap_or(Value::Null),
        "asset_count": assets.len(),
        "assets": assets,
        "placement_count": placements.len(),
        "placements": placements,
        "feedback": response.get("feedback").map(|feedback| json!({
            "enabled": feedback.get("enabled").cloned().unwrap_or(Value::Null),
            "accepted": feedback.get("accepted").cloned().unwrap_or(Value::Null),
            "accepted_iteration": feedback.get("accepted_iteration").cloned().unwrap_or(Value::Null),
            "capture_dir": feedback.get("capture_dir").cloned().unwrap_or(Value::Null),
            "gate_passed": feedback_gate_passed,
        })).unwrap_or(Value::Null),
        "stage_report": response.get("stage_report").cloned().unwrap_or(Value::Null),
        "token_usage": response.get("token_usage").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn attach_scene_grounding_contracts(
    response: &mut Value,
    args: &SceneBuildFromImageArgs,
    grounding_source: &str,
    evidence: &SceneGroundingEvidence,
    segmentation_provider: SceneSegmentationProvider,
) -> Result<(), String> {
    let contract = scene_grounding_contract_report(
        response,
        args,
        grounding_source,
        evidence,
        segmentation_provider,
    );
    let decision_log = scene_decision_log(response, args, grounding_source, evidence);
    response["grounding_contract"] = serde_json::to_value(contract)
        .map_err(|err| format!("serialize grounding contract: {err}"))?;
    response["decision_log"] = serde_json::to_value(decision_log)
        .map_err(|err| format!("serialize decision log: {err}"))?;
    Ok(())
}

fn scene_grounding_contract_report(
    response: &Value,
    args: &SceneBuildFromImageArgs,
    grounding_source: &str,
    evidence: &SceneGroundingEvidence,
    segmentation_provider: SceneSegmentationProvider,
) -> GroundingContractReport {
    let valid_detections = evidence
        .detections
        .iter()
        .filter(|detection| detection_bbox_is_sane(detection.bbox))
        .count();
    let invalid_detections = evidence.detections.len().saturating_sub(valid_detections);
    let object_count = evidence.objects.len();
    let grounded_objects = evidence
        .objects
        .iter()
        .filter(|object| object.detection.is_some())
        .count();
    let detected_status = if args.locator == SceneLocatorProvider::LocateAnything {
        if valid_detections > 0 && invalid_detections == 0 {
            GroundingVerificationStatus::Verified
        } else if valid_detections > 0 {
            GroundingVerificationStatus::Fallback
        } else {
            GroundingVerificationStatus::Invalid
        }
    } else {
        GroundingVerificationStatus::Fallback
    };
    let camera_status = if source_camera_is_sane(evidence) {
        GroundingVerificationStatus::Verified
    } else if evidence.depth.is_some() {
        GroundingVerificationStatus::Invalid
    } else if args.depth_provider == SceneDepthProvider::None {
        GroundingVerificationStatus::Absent
    } else {
        GroundingVerificationStatus::Invalid
    };
    let floor_status = if floor_is_sane(evidence) {
        GroundingVerificationStatus::Verified
    } else if evidence.depth.is_some() {
        GroundingVerificationStatus::Invalid
    } else {
        GroundingVerificationStatus::Absent
    };
    let crop_count = response
        .get("object_image_requests")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let selected_count = response
        .get("selected_candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let rejected_count = response
        .pointer("/candidate_generation/rejected_objects")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let asset_items = response
        .pointer("/asset_outputs/items")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let mesh_quality_failure_count = response
        .get("mesh_quality_failures")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let (canonical_frame_count, verified_canonical_frame_count) =
        canonical_frame_counts(response.get("asset_bindings").unwrap_or(&Value::Null));
    let feedback_enabled = response
        .pointer("/feedback/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(args.feedback && args.lift_assets);
    let feedback_accepted = response
        .pointer("/feedback/accepted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let projection_applied = response
        .pointer("/grounded_layout/projection_fit/applied")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let projection_final_score = response
        .pointer("/grounded_layout/projection_fit/final_score")
        .and_then(Value::as_f64);
    let segmentation_status = match segmentation_provider {
        SceneSegmentationProvider::None => GroundingVerificationStatus::Absent,
        _ => response
            .get("segmentation_grounding")
            .and_then(|report| report.get("mask_count"))
            .and_then(Value::as_u64)
            .filter(|count| *count > 0)
            .map(|_| GroundingVerificationStatus::Verified)
            .unwrap_or(GroundingVerificationStatus::Invalid),
    };

    GroundingContractReport {
        schema_version: 1,
        source_scene_path: args.source_scene_path.display().to_string(),
        composition_mode: serde_json::to_value(args.composition_mode)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_string()),
        entries: vec![
            contract_entry(
                "object_manifest",
                "openai_reasoning",
                "plan_objects",
                if response.get("manifest").is_some() {
                    GroundingVerificationStatus::Fallback
                } else {
                    GroundingVerificationStatus::Absent
                },
                GptDelegationRole::Hypothesis,
                ["object_image_requests", "asset_reuse_groups"],
                json!({
                    "object_count": response.pointer("/manifest/objects").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                    "allow_catalog_reuse": args.allow_catalog_reuse,
                }),
                ["openai_object_plan"],
                Some("object manifest is a planning prior and must be checked against visual grounding".to_string()),
            ),
            contract_entry(
                "detections",
                grounding_source,
                "load_grounding_evidence",
                detected_status,
                GptDelegationRole::None,
                ["object_count_binding", "source_crops", "projection_fit", "contact_estimation"],
                json!({
                    "total_detections": evidence.detections.len(),
                    "valid_detections": valid_detections,
                    "invalid_detections": invalid_detections,
                    "grounded_objects": grounded_objects,
                    "objects": object_count,
                    "locator": args.locator,
                }),
                [grounding_source],
                detection_reason(args.locator, valid_detections, invalid_detections),
            ),
            contract_entry(
                "source_camera_depth",
                args.depth_provider_label(),
                "depth_pro_grounding_evidence",
                camera_status,
                GptDelegationRole::None,
                ["camera_rays", "projection_fit", "floor_fit"],
                json!({
                    "focal_length_px": evidence.camera.focal_length_px,
                    "principal_point": evidence.camera.principal_point,
                    "image_size": evidence.camera.image_size,
                    "vertical_fov_degrees": evidence.camera.vertical_fov_degrees,
                }),
                ["depth_pro"],
                camera_reason(evidence),
            ),
            contract_entry(
                "floor_plane",
                args.depth_provider_label(),
                "depth_pro_grounding_evidence",
                floor_status,
                GptDelegationRole::None,
                ["floor_contact", "object_y", "projection_fit"],
                json!({
                    "normal": evidence.floor.normal,
                    "distance_m": evidence.floor.distance_m,
                    "residual_m": evidence.floor.residual_m,
                    "confidence": evidence.floor.confidence,
                    "floor_sample_count": evidence.depth.as_ref().and_then(|depth| depth.floor_sample_count),
                    "residual_threshold_m": 0.18,
                    "sample_threshold": 64,
                }),
                ["depth_pro_floor_samples_excluding_object_bboxes"],
                floor_reason(evidence),
            ),
            contract_entry(
                "segmentation_masks",
                segmentation_provider_label(segmentation_provider),
                "segmentation_grounding_evidence",
                segmentation_status,
                GptDelegationRole::None,
                ["future_visible_surface_fit"],
                json!({
                    "provider": segmentation_provider,
                    "mask_count": response.pointer("/segmentation_grounding/mask_count").cloned().unwrap_or(Value::Null),
                }),
                ["segmentation_provider"],
                (segmentation_provider == SceneSegmentationProvider::None)
                    .then(|| "mask grounding is optional and disabled by default".to_string()),
            ),
            contract_entry(
                "source_crops",
                "deterministic_cropper",
                "prepare_object_image_requests",
                if crop_count > 0 {
                    GroundingVerificationStatus::Verified
                } else {
                    GroundingVerificationStatus::Absent
                },
                GptDelegationRole::None,
                ["openai_image_generation", "rotation_feedback"],
                json!({ "crop_request_count": crop_count }),
                ["source_bboxes"],
                None,
            ),
            contract_entry(
                "generated_object_images",
                "openai_image",
                "generate_object_candidates",
                if selected_count > 0 && rejected_count == 0 {
                    GroundingVerificationStatus::Verified
                } else if selected_count > 0 {
                    GroundingVerificationStatus::Fallback
                } else {
                    GroundingVerificationStatus::Invalid
                },
                GptDelegationRole::ImageSynthesis,
                ["trellis_asset_lifting"],
                json!({
                    "selected_count": selected_count,
                    "rejected_count": rejected_count,
                    "min_reconstruction_score": args.min_reconstruction_score.unwrap_or(DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE),
                }),
                ["openai_image_candidates", "reconstruction_score_guard"],
                None,
            ),
            contract_entry(
                "trellis_assets",
                "trellis2",
                "images_to_assets",
                if asset_items > 0 && mesh_quality_failure_count == 0 {
                    GroundingVerificationStatus::Verified
                } else if asset_items > 0 {
                    GroundingVerificationStatus::Invalid
                } else {
                    GroundingVerificationStatus::Absent
                },
                GptDelegationRole::None,
                ["canonical_pose", "projection_fit", "scene_catalog"],
                json!({
                    "asset_count": asset_items,
                    "mesh_quality_failure_count": mesh_quality_failure_count,
                    "trellis_pbr": args.trellis_pbr,
                    "target_faces": args.target_faces,
                }),
                ["mesh_quality_gate"],
                None,
            ),
            contract_entry(
                "canonical_asset_frame",
                "canonical_pose_calibration",
                "canonical_pose_calibration",
                if verified_canonical_frame_count > 0 && verified_canonical_frame_count == canonical_frame_count {
                    GroundingVerificationStatus::Verified
                } else if canonical_frame_count > 0 {
                    GroundingVerificationStatus::Fallback
                } else {
                    GroundingVerificationStatus::Absent
                },
                if args.canonical_pose == SceneCanonicalPoseMode::Openai {
                    GptDelegationRole::BoundedCandidateSelection
                } else {
                    GptDelegationRole::None
                },
                ["layout_yaw_offsets", "feedback_rotation_candidates"],
                json!({
                    "canonical_frame_count": canonical_frame_count,
                    "verified_canonical_frame_count": verified_canonical_frame_count,
                    "confidence_threshold": 0.55,
                    "mode": args.canonical_pose,
                    "calibration_report_count": response
                        .get("canonical_pose_calibration")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or_default(),
                    "selection": response
                        .get("canonical_pose_selection")
                        .cloned()
                        .unwrap_or(Value::Null),
                }),
                ["asset_aabb_descriptor_heuristics", "source_crops", "generated_object_images"],
                (canonical_frame_count > verified_canonical_frame_count)
                    .then(|| "some asset frames are low-confidence; rotation feedback must treat them as priors".to_string()),
            ),
            contract_entry(
                "projected_layout_fit",
                "deterministic_cv_optimizer",
                "plan_grounded_scene",
                if projection_final_score.is_some_and(|score| score.is_finite()) {
                    GroundingVerificationStatus::Verified
                } else {
                    GroundingVerificationStatus::Absent
                },
                GptDelegationRole::None,
                ["bsn_scene", "render_feedback"],
                json!({
                    "applied": projection_applied,
                    "initial_score": response.pointer("/grounded_layout/projection_fit/initial_score").cloned().unwrap_or(Value::Null),
                    "final_score": response.pointer("/grounded_layout/projection_fit/final_score").cloned().unwrap_or(Value::Null),
                    "final_loss": response.pointer("/grounded_layout/projection_fit/final_loss").cloned().unwrap_or(Value::Null),
                    "fit_mode": response.pointer("/grounded_layout/projection_fit/fit_mode").cloned().unwrap_or(Value::Null),
                }),
                ["detections", "source_camera_depth", "floor_plane", "asset_aabbs"],
                None,
            ),
            contract_entry(
                "render_feedback",
                "bevy_headless_capture",
                "render_capture_feedback",
                if feedback_enabled && feedback_accepted {
                    GroundingVerificationStatus::Verified
                } else if feedback_enabled {
                    GroundingVerificationStatus::Invalid
                } else {
                    GroundingVerificationStatus::Absent
                },
                if args.feedback_rotation_selector == FeedbackRotationSelector::Openai {
                    GptDelegationRole::BoundedCandidateSelection
                } else {
                    GptDelegationRole::None
                },
                ["accepted_scene_selection", "scene_catalog"],
                json!({
                    "enabled": feedback_enabled,
                    "accepted": feedback_accepted,
                    "accepted_iteration": response.pointer("/feedback/accepted_iteration").cloned().unwrap_or(Value::Null),
                    "rotation_selector": args.feedback_rotation_selector,
                    "threshold_profile": args.feedback_threshold_profile,
                }),
                ["screenshots", "screen_bboxes", "physical_overlap_metrics"],
                (!feedback_enabled).then(|| "feedback disabled; scene relies on deterministic projected fit".to_string()),
            ),
        ],
    }
}

fn scene_decision_log(
    response: &Value,
    args: &SceneBuildFromImageArgs,
    grounding_source: &str,
    evidence: &SceneGroundingEvidence,
) -> SceneDecisionLog {
    let feedback_enabled = response
        .pointer("/feedback/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(args.feedback && args.lift_assets);
    let feedback_accepted = response
        .pointer("/feedback/accepted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let scene_ok = response.pointer("/e2e_summary/ok").and_then(Value::as_bool);
    SceneDecisionLog {
        schema_version: 1,
        source_scene_path: args.source_scene_path.display().to_string(),
        entries: vec![
            decision_entry(
                "object_manifest",
                "plan_objects",
                "openai_reasoning_hypothesis",
                response
                    .pointer("/manifest/objects")
                    .and_then(Value::as_array)
                    .filter(|objects| !objects.is_empty())
                    .map(|_| GroundingVerificationStatus::Fallback)
                    .unwrap_or(GroundingVerificationStatus::Absent),
                GptDelegationRole::Hypothesis,
                json!({
                    "manifest_objects": response.pointer("/manifest/objects").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                    "allow_catalog_reuse": args.allow_catalog_reuse,
                }),
                ["locate_anything_detections", "manifest_bbox_fallback"],
                Some("GPT proposes object hypotheses; visual grounding owns count and placement when available".to_string()),
            ),
            decision_entry(
                "object_count_and_bbox_binding",
                "load_grounding_evidence",
                grounding_source,
                if evidence.objects.iter().any(|object| object.detection.is_some()) {
                    GroundingVerificationStatus::Verified
                } else {
                    GroundingVerificationStatus::Fallback
                },
                GptDelegationRole::None,
                json!({
                    "detections": evidence.detections.len(),
                    "objects": evidence.objects.len(),
                    "objects_with_detection": evidence.objects.iter().filter(|object| object.detection.is_some()).count(),
                }),
                ["manifest_bbox_fallback"],
                None,
            ),
            decision_entry(
                "floor_and_contact_source",
                "depth_pro_grounding_evidence",
                "depth_pro_excluded_floor_fit",
                if floor_is_sane(evidence) {
                    GroundingVerificationStatus::Verified
                } else {
                    GroundingVerificationStatus::Invalid
                },
                GptDelegationRole::None,
                json!({
                    "floor_residual_m": evidence.floor.residual_m,
                    "floor_sample_count": evidence.depth.as_ref().and_then(|depth| depth.floor_sample_count),
                    "metric_contact_objects": evidence.objects.iter().filter(|object| object.metric_contact_point_m.is_some()).count(),
                }),
                ["bbox_bottom_center_depth", "manifest_floor_y"],
                None,
            ),
            decision_entry(
                "asset_selection",
                "images_to_assets",
                "reconstruction_score_and_mesh_quality_gate",
                if response
                    .get("mesh_quality_failures")
                    .and_then(Value::as_array)
                    .is_none_or(|failures| failures.is_empty())
                {
                    GroundingVerificationStatus::Verified
                } else {
                    GroundingVerificationStatus::Invalid
                },
                GptDelegationRole::ImageSynthesis,
                json!({
                    "selected_candidates": response.get("selected_candidates").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                    "asset_count": response.pointer("/asset_outputs/items").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                    "mesh_quality_failures": response.get("mesh_quality_failures").cloned().unwrap_or(Value::Null),
                }),
                ["retry_next_image_candidate"],
                None,
            ),
            decision_entry(
                "scene_pose_fit",
                "plan_grounded_scene",
                "deterministic_projected_aabb_contact_depth_fit",
                response
                    .pointer("/grounded_layout/projection_fit/final_score")
                    .and_then(Value::as_f64)
                    .filter(|score| score.is_finite())
                    .map(|_| GroundingVerificationStatus::Verified)
                    .unwrap_or(GroundingVerificationStatus::Fallback),
                GptDelegationRole::None,
                json!({
                    "composition_mode": args.composition_mode,
                    "pose_fit": args.pose_fit,
                    "initial_score": response.pointer("/grounded_layout/projection_fit/initial_score").cloned().unwrap_or(Value::Null),
                    "final_score": response.pointer("/grounded_layout/projection_fit/final_score").cloned().unwrap_or(Value::Null),
                    "candidate_reports": response.get("composition_candidate_reports").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                }),
                ["heuristic_candidate"],
                None,
            ),
            decision_entry(
                "feedback_acceptance",
                "render_capture_feedback",
                "bevy_render_metrics",
                if feedback_enabled && feedback_accepted {
                    GroundingVerificationStatus::Verified
                } else if feedback_enabled {
                    GroundingVerificationStatus::Invalid
                } else {
                    GroundingVerificationStatus::Absent
                },
                if args.feedback_rotation_selector == FeedbackRotationSelector::Openai {
                    GptDelegationRole::BoundedCandidateSelection
                } else {
                    GptDelegationRole::None
                },
                json!({
                    "enabled": feedback_enabled,
                    "accepted": feedback_accepted,
                    "accepted_iteration": response.pointer("/feedback/accepted_iteration").cloned().unwrap_or(Value::Null),
                    "threshold_profile": args.feedback_threshold_profile,
                    "rotation_selector": args.feedback_rotation_selector,
                }),
                ["deterministic_best_candidate"],
                None,
            ),
            decision_entry(
                "scene_catalog_promotion",
                "write_scene_build_artifacts",
                "shared_mesh_cache_scene_snapshot",
                if args.promote_to_catalog && scene_ok.unwrap_or(false) {
                    GroundingVerificationStatus::Verified
                } else if args.promote_to_catalog {
                    GroundingVerificationStatus::Fallback
                } else {
                    GroundingVerificationStatus::Absent
                },
                GptDelegationRole::None,
                json!({
                    "requested": args.promote_to_catalog,
                    "catalog_entry": response.get("scene_catalog_entry").cloned().unwrap_or(Value::Null),
                }),
                ["artifact_only_scene"],
                None,
            ),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn contract_entry(
    name: impl Into<String>,
    producer: impl Into<String>,
    pipeline_stage: impl Into<String>,
    status: GroundingVerificationStatus,
    gpt_role: GptDelegationRole,
    consumers: impl IntoIterator<Item = impl Into<String>>,
    metrics: Value,
    provenance: impl IntoIterator<Item = impl Into<String>>,
    reason: Option<String>,
) -> GroundingContractEntry {
    GroundingContractEntry {
        name: name.into(),
        producer: producer.into(),
        pipeline_stage: pipeline_stage.into(),
        status,
        gpt_role,
        consumers: consumers.into_iter().map(Into::into).collect(),
        metrics,
        provenance: provenance.into_iter().map(Into::into).collect(),
        reason,
    }
}

#[allow(clippy::too_many_arguments)]
fn decision_entry(
    decision: impl Into<String>,
    pipeline_stage: impl Into<String>,
    source_of_truth: impl Into<String>,
    status: GroundingVerificationStatus,
    gpt_role: GptDelegationRole,
    metrics: Value,
    alternatives: impl IntoIterator<Item = impl Into<String>>,
    reason: Option<String>,
) -> SceneDecisionLogEntry {
    SceneDecisionLogEntry {
        decision: decision.into(),
        pipeline_stage: pipeline_stage.into(),
        source_of_truth: source_of_truth.into(),
        status,
        gpt_role,
        metrics,
        alternatives: alternatives.into_iter().map(Into::into).collect(),
        reason,
    }
}

trait SceneBuildDepthProviderLabel {
    fn depth_provider_label(&self) -> &'static str;
}

impl SceneBuildDepthProviderLabel for SceneBuildFromImageArgs {
    fn depth_provider_label(&self) -> &'static str {
        match self.depth_provider {
            SceneDepthProvider::None => "none",
            SceneDepthProvider::DepthPro => "depth-pro",
        }
    }
}

fn segmentation_provider_label(provider: SceneSegmentationProvider) -> &'static str {
    match provider {
        SceneSegmentationProvider::None => "none",
        SceneSegmentationProvider::BboxPrompt => "bbox-prompt",
        SceneSegmentationProvider::Sam2 => "sam2",
        SceneSegmentationProvider::Sam3 => "sam3",
    }
}

fn detection_bbox_is_sane(bbox: [f32; 4]) -> bool {
    if !bbox.iter().all(|value| value.is_finite()) {
        return false;
    }
    let x0 = bbox[0].min(bbox[2]);
    let x1 = bbox[0].max(bbox[2]);
    let y0 = bbox[1].min(bbox[3]);
    let y1 = bbox[1].max(bbox[3]);
    if x0 < -0.001 || y0 < -0.001 || x1 > 1.001 || y1 > 1.001 {
        return false;
    }
    let area = ((x1 - x0).max(0.0) * (y1 - y0).max(0.0)).clamp(0.0, 1.0);
    (0.001..=0.90).contains(&area)
}

fn source_camera_is_sane(evidence: &SceneGroundingEvidence) -> bool {
    let Some(depth) = evidence.depth.as_ref() else {
        return false;
    };
    let Some([width, height]) = evidence.camera.image_size.or(depth.image_size) else {
        return false;
    };
    let Some(focal_length) = evidence.camera.focal_length_px.or(depth.focal_length_px) else {
        return false;
    };
    width > 0
        && height > 0
        && focal_length.is_finite()
        && focal_length > 1.0
        && evidence
            .camera
            .principal_point
            .is_some_and(|point| point.iter().all(|value| value.is_finite()))
}

fn floor_is_sane(evidence: &SceneGroundingEvidence) -> bool {
    let sample_count = evidence
        .depth
        .as_ref()
        .and_then(|depth| depth.floor_sample_count)
        .unwrap_or_default();
    let normal_len_sq = evidence
        .floor
        .normal
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    evidence.floor.normal.iter().all(|value| value.is_finite())
        && evidence.floor.distance_m.is_finite()
        && normal_len_sq.is_finite()
        && normal_len_sq > 0.25
        && evidence.floor.normal[1].abs() >= 0.45
        && evidence
            .floor
            .residual_m
            .is_some_and(|residual| residual.is_finite() && residual <= 0.18)
        && sample_count >= 64
}

fn detection_reason(
    locator: SceneLocatorProvider,
    valid_detections: usize,
    invalid_detections: usize,
) -> Option<String> {
    if locator == SceneLocatorProvider::Manifest {
        Some(
            "manifest locator is an explicit fallback, not independent visual grounding"
                .to_string(),
        )
    } else if valid_detections == 0 {
        Some("LocateAnything produced no sane bboxes; downstream layout must not trust visual object count".to_string())
    } else if invalid_detections > 0 {
        Some("some LocateAnything bboxes failed finite/range/area sanity checks".to_string())
    } else {
        None
    }
}

fn camera_reason(evidence: &SceneGroundingEvidence) -> Option<String> {
    (!source_camera_is_sane(evidence)).then(|| {
        "source camera requires finite focal length, principal point, and image size from depth evidence"
            .to_string()
    })
}

fn floor_reason(evidence: &SceneGroundingEvidence) -> Option<String> {
    (!floor_is_sane(evidence)).then(|| {
        "floor contact grounding requires object-excluded depth samples, residual <= 0.18m, and at least 64 samples"
            .to_string()
    })
}

fn canonical_frame_counts(asset_bindings: &Value) -> (usize, usize) {
    let Some(bindings) = asset_bindings.as_array() else {
        return (0, 0);
    };
    let mut frame_count = 0usize;
    let mut verified_count = 0usize;
    for binding in bindings {
        let Some(frame) = binding.get("canonical_frame") else {
            continue;
        };
        frame_count += 1;
        let confidence = frame
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        if confidence >= 0.55 {
            verified_count += 1;
        }
    }
    (frame_count, verified_count)
}

pub(crate) fn promote_scene_build_scene_to_catalog(
    catalog_cache: &mut MeshCache,
    source_scene_path: &Path,
    output_dir: &Path,
    bsn: &str,
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
    response: &Value,
) -> Result<Value, String> {
    let source_image_bytes = fs::read(source_scene_path).map_err(|err| {
        format!(
            "failed to read source scene image for scene catalog promotion {}: {err}",
            source_scene_path.display()
        )
    })?;
    let payload = CachedScenePayload {
        world_items: scene_world_items_from_layout(asset_bindings, grounded_layout),
        camera: Some(cached_camera_from_scene_camera(&grounded_layout.camera)),
        bsn: Some(bsn.to_string()),
        asset_bindings: Some(
            serde_json::to_value(asset_bindings)
                .map_err(|err| format!("serialize scene asset bindings: {err}"))?,
        ),
        e2e_summary: response.get("e2e_summary").cloned(),
        response_summary: Some(json!({
            "grounding_contract": response.get("grounding_contract").cloned().unwrap_or(Value::Null),
            "decision_log": response.get("decision_log").cloned().unwrap_or(Value::Null),
            "stage_report": response.get("stage_report").cloned().unwrap_or(Value::Null),
            "token_usage": response.get("token_usage").cloned().unwrap_or(Value::Null),
        })),
    };
    let metadata = catalog_cache
        .upsert_scene_snapshot(
            source_scene_path,
            Some(&source_image_bytes),
            scene_catalog_label(output_dir, source_scene_path),
            "explicit",
            &payload,
            Some(output_dir.display().to_string()),
            Some(scene_cache_metrics_from_response(response)),
        )
        .map_err(|err| format!("failed to promote scene to shared catalog: {err}"))?;
    serde_json::to_value(metadata).map_err(|err| format!("serialize scene catalog metadata: {err}"))
}

fn scene_world_items_from_layout(
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
) -> Vec<CachedWorldItem> {
    let cache_keys = asset_bindings
        .iter()
        .filter_map(|binding| {
            binding
                .cache_key
                .as_ref()
                .map(|cache_key| (binding.asset_id.as_str(), cache_key.as_str()))
        })
        .collect::<HashMap<_, _>>();
    grounded_layout
        .placements
        .iter()
        .filter_map(|placement| {
            let cache_key = cache_keys.get(placement.asset_id.as_str())?;
            Some(CachedWorldItem {
                cache_key: (*cache_key).to_string(),
                translation: placement.translation,
                rotation: yaw_degrees_to_quat_y(placement.rotation_y_degrees),
                scale: placement.scale,
            })
        })
        .collect()
}

fn cached_camera_from_scene_camera(camera: &burn_synth_scene::SceneCamera) -> CachedCameraState {
    let dx = camera.translation[0] - camera.focus[0];
    let dy = camera.translation[1] - camera.focus[1];
    let dz = camera.translation[2] - camera.focus[2];
    let radius = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-5);
    let yaw = camera.yaw.unwrap_or_else(|| dx.atan2(dz));
    let pitch = camera
        .pitch
        .unwrap_or_else(|| (dy / radius).clamp(-1.0, 1.0).asin());
    CachedCameraState {
        translation: camera.translation,
        rotation: [0.0, 0.0, 0.0, 1.0],
        focus: camera.focus,
        yaw,
        pitch,
        radius: camera.radius.unwrap_or(radius),
        vertical_fov_degrees: camera.vertical_fov_degrees,
    }
}

fn yaw_degrees_to_quat_y(yaw_degrees: f32) -> [f32; 4] {
    let half = yaw_degrees.to_radians() * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
}

fn scene_catalog_label(output_dir: &Path, source_scene_path: &Path) -> String {
    output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .or_else(|| {
            source_scene_path
                .file_stem()
                .and_then(|name| name.to_str())
                .map(|stem| format!("{stem}_scene"))
        })
        .unwrap_or_else(|| "generated_scene".to_string())
}

fn scene_cache_metrics_from_response(response: &Value) -> CachedSceneMetrics {
    CachedSceneMetrics::from_scene_build_response(response)
}

#[derive(Clone, Debug)]
pub(crate) struct SceneAssetQualityFailure {
    pub(crate) object_id: String,
    pub(crate) candidate_index: usize,
    pub(crate) output_path: String,
    pub(crate) failure: String,
}

impl SceneAssetQualityFailure {
    pub(crate) fn message(&self) -> String {
        format!("{}: {}", self.output_path, self.failure)
    }
}

pub(crate) fn scene_selected_candidate_key(selected: &Value) -> Option<(String, usize)> {
    let object_id = selected.get("object_id").and_then(Value::as_str)?;
    let candidate_index = selected.get("candidate_index").and_then(Value::as_u64)? as usize;
    Some((object_id.to_string(), candidate_index))
}

pub(crate) fn scene_asset_lift_chunk_size(
    batch_size: Option<usize>,
    selected_count: usize,
) -> usize {
    let selected_count = selected_count.max(1);
    match batch_size.filter(|size| *size > 0) {
        Some(size) => size.min(selected_count),
        None => 1,
    }
}

pub(crate) fn cache_scene_asset_outputs(
    cache: &mut HashMap<(String, usize), Value>,
    selected_candidates: &[Value],
    asset_outputs: &Value,
) -> Result<(), String> {
    let items = asset_outputs
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "images_to_assets response missing items array".to_string())?;
    if items.len() != selected_candidates.len() {
        return Err(format!(
            "asset output count ({}) did not match selected candidate count ({})",
            items.len(),
            selected_candidates.len()
        ));
    }
    for (selected, item) in selected_candidates.iter().zip(items.iter()) {
        let key = scene_selected_candidate_key(selected)
            .ok_or_else(|| "selected candidate missing object_id or candidate_index".to_string())?;
        cache.insert(key, item.clone());
    }
    Ok(())
}

pub(crate) fn scene_cached_asset_outputs_for_selected(
    selected_candidates: &[Value],
    cache: &HashMap<(String, usize), Value>,
) -> Result<Value, String> {
    let mut items = Vec::with_capacity(selected_candidates.len());
    for selected in selected_candidates {
        let key = scene_selected_candidate_key(selected)
            .ok_or_else(|| "selected candidate missing object_id or candidate_index".to_string())?;
        let item = cache.get(&key).ok_or_else(|| {
            format!(
                "selected candidate `{}` index {} has no cached asset output",
                key.0, key.1
            )
        })?;
        items.push(item.clone());
    }
    Ok(json!({
        "tool": "images_to_assets_cached_merge",
        "items": items,
    }))
}

pub(crate) fn scene_asset_quality_failures_with_selected(
    asset_outputs: &Value,
    selected_candidates: &[Value],
) -> Vec<SceneAssetQualityFailure> {
    let Some(items) = asset_outputs.get("items").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut failures = Vec::new();
    for (item, selected) in items.iter().zip(selected_candidates.iter()) {
        if item.get("asset_kind").and_then(Value::as_str) != Some("mesh") {
            continue;
        }
        if item.get("synthesis_backend").and_then(Value::as_str) != Some("trellis") {
            continue;
        }
        let output_path = item
            .get("output_path")
            .and_then(Value::as_str)
            .unwrap_or("<unknown output>")
            .to_string();
        let object_id = selected
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown object>")
            .to_string();
        let candidate_index = selected
            .get("candidate_index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        for failure in item
            .get("mesh_quality_failures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if scene_mesh_quality_failure_is_waived(item, selected, failure) {
                continue;
            }
            failures.push(SceneAssetQualityFailure {
                object_id: object_id.clone(),
                candidate_index,
                output_path: output_path.clone(),
                failure: failure.to_string(),
            });
        }
    }
    failures
}

pub(crate) fn scene_mesh_quality_failure_is_waived(
    item: &Value,
    selected: &Value,
    failure: &str,
) -> bool {
    if !failure.starts_with("position-welded boundary edge ratio ") {
        return false;
    }
    if !scene_selected_candidate_is_legged_furniture(selected) {
        return false;
    }
    let Some(boundary_edge_ratio) = item
        .pointer("/mesh_quality/position_welded_connectivity/boundary_edge_ratio")
        .and_then(Value::as_f64)
    else {
        return false;
    };
    let non_manifold_edges = item
        .pointer("/mesh_quality/position_welded_connectivity/non_manifold_edges")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let tiny_component_faces = item
        .pointer("/mesh_quality/position_welded_connectivity/tiny_components_le_16_faces")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    boundary_edge_ratio <= 0.10 && non_manifold_edges == 0 && tiny_component_faces <= 8
}

pub(crate) fn scene_selected_candidate_is_legged_furniture(selected: &Value) -> bool {
    ["label", "object_id", "reuse_group"]
        .iter()
        .filter_map(|key| selected.get(*key).and_then(Value::as_str))
        .any(|value| {
            let value = value.to_ascii_lowercase();
            value.contains("chair")
                || value.contains("stool")
                || value.contains("bench")
                || value.contains("seat")
        })
}

#[cfg(test)]
pub(crate) fn scene_asset_quality_failures(asset_outputs: &Value) -> Vec<String> {
    let Some(items) = asset_outputs.get("items").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut failures = Vec::new();
    for item in items {
        if item.get("asset_kind").and_then(Value::as_str) != Some("mesh") {
            continue;
        }
        if item.get("synthesis_backend").and_then(Value::as_str) != Some("trellis") {
            continue;
        }
        let output_path = item
            .get("output_path")
            .and_then(Value::as_str)
            .unwrap_or("<unknown output>");
        for failure in item
            .get("mesh_quality_failures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            failures.push(format!("{output_path}: {failure}"));
        }
    }
    failures
}

pub(crate) fn write_scene_build_artifacts(
    output_dir: &Path,
    response: &Value,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "create scene build artifact directory {}: {err}",
            output_dir.display()
        )
    })?;
    for (key, file_name) in [
        ("preparation", "preparation.json"),
        ("manifest_initial", "manifest_initial.json"),
        ("manifest", "manifest.json"),
        (
            "manifest_grounded_for_crops",
            "manifest_grounded_for_crops.json",
        ),
        (
            "pre_generation_grounding_evidence",
            "pre_generation_grounding_evidence.json",
        ),
        (
            "pre_generation_locate_anything_report",
            "pre_generation_locate_anything_report.json",
        ),
        ("object_image_requests", "object_image_requests.json"),
        ("provider_metadata", "provider_metadata.json"),
        ("token_usage", "token_usage.json"),
        ("candidate_generation", "candidate_generation.json"),
        ("candidates", "candidates.json"),
        ("selected_candidates", "selected_candidates.json"),
        ("asset_outputs", "asset_outputs.json"),
        ("asset_lift_attempts", "asset_lift_attempts.json"),
        ("mesh_quality_failures", "mesh_quality_failures.json"),
        ("asset_bindings_initial", "asset_bindings_initial.json"),
        ("asset_bindings", "asset_bindings.json"),
        (
            "asset_bindings_calibrated",
            "asset_bindings_calibrated.json",
        ),
        (
            "canonical_pose_calibration",
            "canonical_pose_calibration_report.json",
        ),
        ("canonical_pose_selection", "canonical_pose_selection.json"),
        (
            "canonical_pose_selection_task",
            "canonical_pose_selection_task.json",
        ),
        (
            "canonical_pose_verification",
            "canonical_pose_verification.json",
        ),
        ("plan", "plan.json"),
        ("grounded_layout", "grounded_layout.json"),
        ("commands", "commands.json"),
        ("feedback", "feedback_report.json"),
        ("grounding_contract", "grounding_contract.json"),
        ("decision_log", "decision_log.json"),
        (
            "composition_candidate_reports",
            "composition_candidate_reports.json",
        ),
        ("stage_report", "stage_report.json"),
        ("e2e_summary", "summary.json"),
    ] {
        if let Some(value) = response.get(key) {
            write_json_file(&output_dir.join(file_name), value).map_err(|err| err.to_string())?;
        }
    }
    if let Some(bsn) = response.get("bsn").and_then(Value::as_str) {
        fs::write(output_dir.join("scene.bsn"), bsn).map_err(|err| {
            format!(
                "write scene BSN artifact {}: {err}",
                output_dir.join("scene.bsn").display()
            )
        })?;
    }
    write_pose_fit_artifacts(output_dir, response)?;
    write_json_file(
        &output_dir.join("scene_build_response_structured.json"),
        response,
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

impl McpServer {
    pub(crate) fn depth_pro_grounding_evidence(
        &mut self,
        evidence: &mut SceneGroundingEvidence,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<DepthProGroundingReport, String> {
        let config = DepthProGroundingConfig {
            cache_dir: self.config.depth_cache_dir.clone(),
            precision: self.config.depth_precision.into(),
            allow_download: self.config.depth_allow_download,
            require_gpu: true,
        };
        self.grounding
            .depth_pro_grounding_evidence(evidence, source_scene_path, output_dir, config)
    }

    pub(crate) fn segmentation_grounding_evidence(
        &mut self,
        provider: SceneSegmentationProvider,
        precision: Option<SceneSegmentationPrecision>,
        quantization: Option<SceneSegmentationQuantization>,
        evidence: &mut SceneGroundingEvidence,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<Option<SegmentationGroundingReport>, String> {
        let precision = precision.unwrap_or(self.config.scene_segmentation_precision);
        let quantization = quantization.unwrap_or(self.config.scene_segmentation_quantization);
        let config = match provider {
            SceneSegmentationProvider::None => return Ok(None),
            SceneSegmentationProvider::BboxPrompt => SegmentationGroundingConfig {
                model: SegmentationModelKind::BboxPrompt,
                backend: SegmentationRuntimeBackend::BboxPrompt,
                model_root: self.config.scene_segmentation_model_root.clone(),
                cache_dir: self.config.scene_segmentation_cache_dir.clone(),
                cdn_base_url: self.config.scene_segmentation_cdn_base_url.clone(),
                precision: precision.into(),
                quantization: quantization.into(),
                allow_download: self.config.scene_segmentation_allow_download,
                require_gpu: false,
            },
            SceneSegmentationProvider::Sam2 => SegmentationGroundingConfig {
                model: SegmentationModelKind::Sam2,
                backend: SegmentationRuntimeBackend::BurnNative,
                model_root: self.config.scene_segmentation_model_root.clone(),
                cache_dir: self.config.scene_segmentation_cache_dir.clone(),
                cdn_base_url: self.config.scene_segmentation_cdn_base_url.clone(),
                precision: precision.into(),
                quantization: quantization.into(),
                allow_download: self.config.scene_segmentation_allow_download,
                require_gpu: true,
            },
            SceneSegmentationProvider::Sam3 => SegmentationGroundingConfig {
                model: SegmentationModelKind::Sam3,
                backend: SegmentationRuntimeBackend::BurnNative,
                model_root: self.config.scene_segmentation_model_root.clone(),
                cache_dir: self.config.scene_segmentation_cache_dir.clone(),
                cdn_base_url: self.config.scene_segmentation_cdn_base_url.clone(),
                precision: precision.into(),
                quantization: quantization.into(),
                allow_download: self.config.scene_segmentation_allow_download,
                require_gpu: true,
            },
        };
        self.grounding
            .segmentation_grounding_evidence(evidence, source_scene_path, output_dir, config)
            .map(Some)
    }
}

impl McpServer {
    pub(crate) fn locate_anything_grounding_evidence(
        &mut self,
        backend: LocateAnythingBackend,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<SceneGroundingEvidence, String> {
        self.locate_anything_grounding_evidence_with_report(
            backend,
            manifest,
            source_scene_path,
            output_dir,
        )
        .map(|(evidence, _report)| evidence)
    }

    pub(crate) fn locate_anything_grounding_evidence_with_report(
        &mut self,
        backend: LocateAnythingBackend,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<(SceneGroundingEvidence, LocateAnythingGroundingReport), String> {
        let LocateAnythingBackend::BurnNative = backend;
        self.locate_anything_burn_native_grounding_evidence(manifest, source_scene_path, output_dir)
    }

    pub(crate) fn locate_anything_burn_native_grounding_evidence(
        &mut self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<(SceneGroundingEvidence, LocateAnythingGroundingReport), String> {
        let config = LocateAnythingGroundingConfig {
            model_root: self.config.locate_anything_model_root.clone(),
            cache_dir: self.config.locate_anything_cache_dir.clone(),
            cdn_base_url: self.config.locate_anything_cdn_base_url.clone(),
            allow_download: self.config.locate_anything_allow_download,
            precision: self.config.locate_anything_precision.into(),
            in_token_limit: self.config.locate_anything_in_token_limit,
            ..LocateAnythingGroundingConfig::default()
        };
        self.grounding
            .locate_anything_burn_native_grounding_evidence(
                manifest,
                source_scene_path,
                output_dir,
                config,
            )
    }
}

pub(crate) fn write_scene_ground_artifacts(
    output_dir: &Path,
    response: &Value,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create scene-ground artifact directory {}: {err}",
            output_dir.display()
        )
    })?;
    for (key, file_name) in [
        ("manifest", "manifest.json"),
        ("asset_bindings_initial", "asset_bindings_initial.json"),
        ("asset_bindings", "asset_bindings.json"),
        (
            "asset_bindings_calibrated",
            "asset_bindings_calibrated.json",
        ),
        (
            "canonical_pose_calibration",
            "canonical_pose_calibration_report.json",
        ),
        ("canonical_pose_selection", "canonical_pose_selection.json"),
        (
            "canonical_pose_selection_task",
            "canonical_pose_selection_task.json",
        ),
        (
            "canonical_pose_verification",
            "canonical_pose_verification.json",
        ),
        ("grounding_evidence", "grounding_evidence.json"),
        ("grounded_layout", "grounded_layout.json"),
        ("commands", "commands.json"),
        (
            "composition_candidate_reports",
            "composition_candidate_reports.json",
        ),
        ("stage_report", "stage_report.json"),
        ("e2e_summary", "summary.json"),
    ] {
        if let Some(value) = response.get(key) {
            write_json_file(&output_dir.join(file_name), value).map_err(|err| err.to_string())?;
        }
    }
    if let Some(bsn) = response.get("bsn").and_then(Value::as_str) {
        fs::write(output_dir.join("scene.bsn"), bsn).map_err(|err| {
            format!(
                "failed to write scene-ground BSN {}: {err}",
                output_dir.join("scene.bsn").display()
            )
        })?;
    }
    write_pose_fit_artifacts(output_dir, response)?;
    Ok(())
}

fn write_pose_fit_artifacts(output_dir: &Path, response: &Value) -> Result<(), String> {
    if response
        .get("save_pose_debug")
        .and_then(Value::as_bool)
        .is_some_and(|enabled| !enabled)
    {
        return Ok(());
    }
    write_canonical_pose_artifacts(output_dir, response)?;
    write_projection_fit_artifacts(output_dir, response)
}

fn write_canonical_pose_artifacts(output_dir: &Path, response: &Value) -> Result<(), String> {
    let Some(asset_bindings) = response
        .get("asset_bindings")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<SceneAssetBinding>>(value).ok())
    else {
        return Ok(());
    };
    let evidence = canonical_pose_evidence_for_assets(&asset_bindings);
    write_json_file(&output_dir.join("canonical_pose_evidence.json"), &evidence)
        .map_err(|err| err.to_string())
}

fn write_projection_fit_artifacts(output_dir: &Path, response: &Value) -> Result<(), String> {
    let Some(report) = response
        .get("grounded_layout")
        .and_then(|layout| layout.get("projection_fit"))
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    write_json_file(&output_dir.join("projection_fit_report.json"), report)
        .map_err(|err| err.to_string())?;
    write_json_file(&output_dir.join("pose_fit_report.json"), report)
        .map_err(|err| err.to_string())?;
    if let Some(candidates) = report.get("candidates") {
        write_json_file(&output_dir.join("pose_fit_candidates.json"), candidates)
            .map_err(|err| err.to_string())?;
    }
    let camera_grounding = json!({
        "stage": "camera_grounding",
        "source_scene_path": projection_fit_source_path(response)
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        "depth": response.pointer("/grounding_evidence/depth").cloned().unwrap_or(Value::Null),
        "estimated_camera": response.pointer("/grounding_evidence/camera").cloned().unwrap_or(Value::Null),
        "estimated_floor": response.pointer("/grounding_evidence/floor").cloned().unwrap_or(Value::Null),
        "fit_camera": report.get("camera").cloned().unwrap_or(Value::Null),
    });
    write_json_file(
        &output_dir.join("camera_grounding_report.json"),
        &camera_grounding,
    )
    .map_err(|err| err.to_string())?;
    let initial = json!({
        "stage": "projection_fit_initial",
        "loss": report.get("initial_loss").cloned().unwrap_or(Value::Null),
        "score": report.get("initial_score").cloned().unwrap_or(Value::Null),
        "camera": report.get("camera").cloned().unwrap_or(Value::Null),
        "objects": report.get("initial_objects").cloned().unwrap_or(Value::Null),
    });
    write_json_file(&output_dir.join("projection_fit_initial.json"), &initial)
        .map_err(|err| err.to_string())?;
    let final_fit = json!({
        "stage": "projection_fit_final",
        "applied": report.get("applied").cloned().unwrap_or(Value::Null),
        "iteration_count": report.get("iteration_count").cloned().unwrap_or(Value::Null),
        "loss": report.get("final_loss").cloned().unwrap_or(Value::Null),
        "score": report.get("final_score").cloned().unwrap_or(Value::Null),
        "camera": report.get("camera").cloned().unwrap_or(Value::Null),
        "objects": report.get("objects").cloned().unwrap_or(Value::Null),
    });
    write_json_file(&output_dir.join("projection_fit_final.json"), &final_fit)
        .map_err(|err| err.to_string())?;
    if let Some(source_path) = projection_fit_source_path(response)
        && source_path.exists()
    {
        let overlay_path = output_dir.join("projection_fit_overlay.png");
        if let Err(err) = write_projection_fit_overlay(&source_path, report, &overlay_path) {
            let warning = json!({
                "projection_fit_overlay": overlay_path.display().to_string(),
                "warning": err,
            });
            write_json_file(
                &output_dir.join("projection_fit_overlay_error.json"),
                &warning,
            )
            .map_err(|err| err.to_string())?;
        }
    }
    write_visible_surface_fit_artifacts(output_dir, response, report)?;
    Ok(())
}

fn projection_fit_source_path(response: &Value) -> Option<PathBuf> {
    response
        .get("source_scene_path")
        .and_then(Value::as_str)
        .or_else(|| {
            response
                .get("grounding_evidence")
                .and_then(|evidence| evidence.get("source_image_path"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            response
                .get("manifest")
                .and_then(|manifest| manifest.get("source_scene_path"))
                .and_then(Value::as_str)
        })
        .map(PathBuf::from)
}

fn write_projection_fit_overlay(
    source_path: &Path,
    report: &Value,
    output_path: &Path,
) -> Result<(), String> {
    let image = image::open(source_path)
        .map_err(|err| {
            format!(
                "failed to open projection-fit source {}: {err}",
                source_path.display()
            )
        })?
        .resize(1800, 1800, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let mut image = image;
    let objects = report
        .get("objects")
        .and_then(Value::as_array)
        .ok_or_else(|| "projection fit report missing objects array".to_string())?;
    for object in objects {
        if let Some(source_bbox) = object.get("source_bbox").and_then(json_array4) {
            draw_normalized_rect(&mut image, source_bbox, image::Rgba([36, 220, 74, 255]));
        }
        if let Some(projected_bbox) = object.get("projected_bbox").and_then(json_array4) {
            draw_normalized_rect(&mut image, projected_bbox, image::Rgba([255, 76, 216, 255]));
        }
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write projection-fit overlay {}: {err}",
            output_path.display()
        )
    })
}

fn draw_normalized_rect(image: &mut image::RgbaImage, bbox: [f32; 4], color: image::Rgba<u8>) {
    let width = image.width().saturating_sub(1).max(1);
    let height = image.height().saturating_sub(1).max(1);
    let x0 = (bbox[0].clamp(0.0, 1.0) * width as f32).round() as u32;
    let y0 = (bbox[1].clamp(0.0, 1.0) * height as f32).round() as u32;
    let x1 = (bbox[2].clamp(0.0, 1.0) * width as f32).round() as u32;
    let y1 = (bbox[3].clamp(0.0, 1.0) * height as f32).round() as u32;
    for thickness in 0..3 {
        let x0 = x0.saturating_sub(thickness);
        let y0 = y0.saturating_sub(thickness);
        let x1 = (x1 + thickness).min(width);
        let y1 = (y1 + thickness).min(height);
        for x in x0..=x1 {
            image.put_pixel(x, y0, color);
            image.put_pixel(x, y1, color);
        }
        for y in y0..=y1 {
            image.put_pixel(x0, y, color);
            image.put_pixel(x1, y, color);
        }
    }
}

pub(crate) fn scene_asset_bindings_from_outputs(
    manifest: &SceneObjectManifest,
    selected_candidates: &[Value],
    asset_outputs: &Value,
) -> Result<Vec<SceneAssetBinding>, String> {
    let items = asset_outputs["items"]
        .as_array()
        .ok_or_else(|| "images_to_assets response missing items array".to_string())?;
    if items.len() != selected_candidates.len() {
        return Err(format!(
            "asset output count ({}) did not match selected candidate count ({})",
            items.len(),
            selected_candidates.len()
        ));
    }
    let objects_by_id = manifest
        .objects
        .iter()
        .map(|object| (object.id.as_str(), object))
        .collect::<HashMap<_, _>>();
    let mut bindings = Vec::with_capacity(manifest.objects.len().max(items.len()));
    let mut bindings_by_reuse_group = HashMap::new();
    for (item, selected) in items.iter().zip(selected_candidates.iter()) {
        let object_id = selected["object_id"]
            .as_str()
            .ok_or_else(|| "selected candidate missing object_id".to_string())?;
        let object = objects_by_id
            .get(object_id)
            .ok_or_else(|| format!("selected candidate references unknown object `{object_id}`"))?;
        let output_path = item["output_path"]
            .as_str()
            .ok_or_else(|| "asset output item missing output_path".to_string())?;
        let cache_key = item["cache_key"].as_str().map(ToOwned::to_owned);
        let local_aabb = item
            .get("local_aabb")
            .filter(|value| !value.is_null())
            .cloned()
            .and_then(|value| serde_json::from_value::<SceneAssetAabb>(value).ok())
            .or_else(|| {
                item.get("catalog_entry")
                    .and_then(|entry| entry.get("local_aabb"))
                    .filter(|value| !value.is_null())
                    .cloned()
                    .and_then(|value| serde_json::from_value::<CachedAssetAabb>(value).ok())
                    .map(cached_aabb_to_scene)
            });
        let binding = SceneAssetBinding {
            asset_id: sanitize_scene_identifier(&format!("{object_id}_asset")),
            object_id: object.id.clone(),
            label: object.label.clone(),
            aliases: object.aliases.clone(),
            path: Some(output_path.to_string()),
            cache_key: cache_key.clone(),
            reusable: cache_key.is_some()
                || object.instance_count > 1
                || object.instances.len() > 1
                || object.reuse_group.is_some(),
            source_image_path: selected["image_path"].as_str().map(ToOwned::to_owned),
            pipeline: item["synthesis_backend"].as_str().map(ToOwned::to_owned),
            local_aabb,
            canonical_frame: Some(inferred_scene_asset_frame(
                &object.label,
                &object.aliases,
                local_aabb,
                object.target_footprint_m,
            )),
            provenance: Some(burn_synth_scene::SceneAssetProvenance {
                run_id: "scene_build_from_image".to_string(),
                source_scene_path: manifest.source_scene_path.clone(),
                source_object_id: object.id.clone(),
                generated_by: "scene_build_from_image".to_string(),
            }),
        };
        let reuse_group = object
            .reuse_group
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(object.id.as_str())
            .to_string();
        bindings_by_reuse_group
            .entry(reuse_group)
            .or_insert_with(|| binding.clone());
        bindings.push(binding);
    }

    for object in &manifest.objects {
        if bindings
            .iter()
            .any(|binding| binding.object_id.as_str() == object.id.as_str())
        {
            continue;
        }
        let reuse_group = object
            .reuse_group
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(object.id.as_str());
        let Some(source_binding) = bindings_by_reuse_group.get(reuse_group) else {
            return Err(format!(
                "scene object `{}` was not selected and has no reusable asset binding for reuse group `{reuse_group}`",
                object.id
            ));
        };
        let mut binding = source_binding.clone();
        binding.asset_id = sanitize_scene_identifier(&format!("{}_asset", object.id));
        binding.object_id = object.id.clone();
        binding.label = object.label.clone();
        binding.aliases = object.aliases.clone();
        binding.reusable = true;
        if let Some(provenance) = binding.provenance.as_mut() {
            provenance.source_object_id = object.id.clone();
        }
        bindings.push(binding);
    }
    Ok(bindings)
}

pub(crate) fn sanitize_scene_identifier(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "asset".to_string()
    } else {
        output
    }
}
