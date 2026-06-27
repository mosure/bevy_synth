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
    3
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
    SceneCanonicalPoseMode::Auto
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
        ("manifest", "manifest.json"),
        ("object_image_requests", "object_image_requests.json"),
        ("provider_metadata", "provider_metadata.json"),
        ("token_usage", "token_usage.json"),
        ("candidate_generation", "candidate_generation.json"),
        ("candidates", "candidates.json"),
        ("selected_candidates", "selected_candidates.json"),
        ("asset_outputs", "asset_outputs.json"),
        ("asset_lift_attempts", "asset_lift_attempts.json"),
        ("mesh_quality_failures", "mesh_quality_failures.json"),
        ("asset_bindings", "asset_bindings.json"),
        ("plan", "plan.json"),
        ("grounded_layout", "grounded_layout.json"),
        ("commands", "commands.json"),
        ("feedback", "feedback_report.json"),
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
        let LocateAnythingBackend::BurnNative = backend;
        self.locate_anything_burn_native_grounding_evidence(manifest, source_scene_path, output_dir)
    }

    pub(crate) fn locate_anything_burn_native_grounding_evidence(
        &mut self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<SceneGroundingEvidence, String> {
        let config = LocateAnythingGroundingConfig {
            model_root: self.config.locate_anything_model_root.clone(),
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
            .map(|(evidence, _report)| evidence)
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
        ("asset_bindings", "asset_bindings.json"),
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
