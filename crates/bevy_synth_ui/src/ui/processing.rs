use super::*;

#[allow(clippy::type_complexity)]
pub(super) fn tick_processing_elapsed(mut state: ResMut<SceneProcessingState>) {
    state.tick();
}

#[allow(clippy::type_complexity)]
pub(super) fn sync_processing_panel(
    state: Res<SceneProcessingState>,
    mut roots: Query<&mut Visibility, With<ProcessingPanelRoot>>,
    mut text_queries: ParamSet<(
        Query<&mut Text, With<ProcessingCurrentText>>,
        Query<&mut Text, With<ProcessingTimelineText>>,
        Query<&mut Text, With<ProcessingArtifactText>>,
        Query<&mut Text, With<ProcessingErrorText>>,
    )>,
) {
    let visible = state.is_visible();
    for mut visibility in &mut roots {
        *visibility = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if !visible {
        return;
    }

    let status = if state.active {
        "running"
    } else {
        &state.current_phase
    };
    let source = state
        .source_label
        .as_deref()
        .map(ellipsize_processing_text)
        .unwrap_or_else(|| "scene".to_string());
    let elapsed = format_elapsed_ms(state.elapsed_ms);
    let mut current_rows = vec![
        format!(
            "{status} | {elapsed} | {}",
            ellipsize_text(&state.current_stage, 32)
        ),
        source,
        format!(
            "{} | {}",
            state.current_phase,
            ellipsize_text(&state.current_execution, 16)
        ),
        ellipsize_text(&state.current_message, 64),
    ];
    if let Some(token_usage) = state.token_usage_summary.as_ref() {
        current_rows.push(ellipsize_text(token_usage, 64));
    }
    let current_text = current_rows.join("\n");
    for mut text in &mut text_queries.p0() {
        text.0 = current_text.clone();
    }

    let rows = state
        .recent_events
        .iter()
        .take(2)
        .map(format_processing_event)
        .collect::<Vec<_>>();
    let timeline_text = if rows.is_empty() {
        String::new()
    } else {
        rows.join("\n")
    };
    for mut text in &mut text_queries.p1() {
        text.0 = timeline_text.clone();
    }

    let artifact_text = state
        .recent_artifacts
        .iter()
        .take(1)
        .map(|path| format!("artifact: {}", ellipsize_text(path, 48)))
        .collect::<Vec<_>>()
        .join("\n");
    for mut text in &mut text_queries.p2() {
        text.0 = artifact_text.clone();
    }

    let error_text = state
        .last_error
        .as_deref()
        .map(|error| format!("error: {}", ellipsize_text(error, 96)))
        .unwrap_or_default();
    for mut text in &mut text_queries.p3() {
        text.0 = error_text.clone();
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn sync_settings_developer_panel(
    state: Res<SceneProcessingState>,
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut text_queries: ParamSet<(
        Query<&mut Text, With<SettingsDeveloperCurrentText>>,
        Query<&mut Text, With<SettingsDeveloperTokenText>>,
        Query<&mut Text, With<SettingsDeveloperEventsText>>,
        Query<&mut Text, With<SettingsDeveloperArtifactText>>,
        Query<&mut Text, With<SettingsDeveloperVisualText>>,
    )>,
) {
    let current_text = format_developer_current_block(&state);
    for mut text in &mut text_queries.p0() {
        text.0 = current_text.clone();
    }

    let token_text = format_developer_token_block(&state);
    for mut text in &mut text_queries.p1() {
        text.0 = token_text.clone();
    }

    let event_text = format_developer_event_block(&state);
    for mut text in &mut text_queries.p2() {
        text.0 = event_text.clone();
    }

    let artifact_text = format_developer_artifact_block(&state);
    for mut text in &mut text_queries.p3() {
        text.0 = artifact_text.clone();
    }

    let visual_text = format_developer_visual_block(&state, &artifact_previews);
    for mut text in &mut text_queries.p4() {
        text.0 = visual_text.clone();
    }
}

pub(super) fn format_developer_current_block(state: &SceneProcessingState) -> String {
    let active = if state.active { "active" } else { "idle" };
    let last_event = state
        .last_event_age_ms()
        .map(format_elapsed_ms)
        .unwrap_or_else(|| "none".to_string());
    let error = state
        .last_error
        .as_deref()
        .map(|value| format!("\nerror: {}", ellipsize_text(value, 92)))
        .unwrap_or_default();
    format!(
        "state: {active}\nrun: {}\nsource: {}\nstage: {} / {} / {}\nelapsed: {} | last event: {last_event}\nmessage: {}{}",
        state.run_id.as_deref().unwrap_or("none"),
        ellipsize_text(state.source_label.as_deref().unwrap_or("none"), 74),
        ellipsize_text(&state.current_stage, 40),
        state.current_phase,
        state.current_execution,
        format_elapsed_ms(state.elapsed_ms),
        ellipsize_text(&state.current_message, 92),
        error
    )
}

pub(super) fn format_developer_token_block(state: &SceneProcessingState) -> String {
    state
        .token_usage_summary
        .as_deref()
        .map(|summary| ellipsize_text(summary, 104))
        .unwrap_or_else(|| {
            if state.active {
                "waiting for provider token usage; local GPU stages do not emit token counts"
                    .to_string()
            } else {
                "no token usage reported yet".to_string()
            }
        })
}

pub(super) fn format_developer_event_block(state: &SceneProcessingState) -> String {
    let rows = state
        .recent_events
        .iter()
        .take(DEVELOPER_EVENT_ROWS)
        .map(format_developer_event)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return if state.active {
            "worker is active; waiting for first progress event".to_string()
        } else {
            "no scene build events yet".to_string()
        };
    }
    rows.join("\n")
}

pub(super) fn format_developer_artifact_block(state: &SceneProcessingState) -> String {
    let rows = state
        .recent_artifacts
        .iter()
        .take(DEVELOPER_ARTIFACT_ROWS)
        .map(|path| format!("{} {}", artifact_kind_label(path), ellipsize_text(path, 92)))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return if state.active {
            "waiting for first artifact path; generated files appear under tmp/runs/<run_id>"
                .to_string()
        } else {
            "no artifacts yet".to_string()
        };
    }
    rows.join("\n")
}

pub(super) fn format_developer_visual_block(
    state: &SceneProcessingState,
    artifact_previews: &ProcessingArtifactPreviewCache,
) -> String {
    if artifact_previews.total_count == 0 {
        if state.active {
            "waiting for locate/depth/crop/canonical/feedback images".to_string()
        } else {
            "no visual artifacts yet".to_string()
        }
    } else {
        format!(
            "{} visual artifact(s) | latest first | page {}/{}",
            artifact_previews.total_count,
            artifact_previews.page + 1,
            artifact_previews.page_count.max(1)
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn sync_processing_artifact_previews(
    state: Res<SceneProcessingState>,
    mut developer: ResMut<DeveloperPanelState>,
    mut cache: ResMut<ProcessingArtifactPreviewCache>,
    mut images: ResMut<Assets<Image>>,
) {
    let discovered = discover_processing_visual_artifacts(&state);
    let total_count = discovered.len();
    let page_count = total_count.div_ceil(DEVELOPER_VISUAL_ROWS);
    let max_page = page_count.saturating_sub(1);
    if developer.visual_page > max_page {
        developer.visual_page = max_page;
    }
    let page = developer.visual_page;
    let page_start = page.saturating_mul(DEVELOPER_VISUAL_ROWS);
    let signature = discovered
        .iter()
        .map(|(path, kind)| format!("{}:{}", kind.label(), path.display()))
        .collect::<Vec<_>>()
        .join("|");
    let signature = format!("page={page};total={total_count};{signature}");
    if cache.signature == signature {
        return;
    }

    cache.signature = signature;
    cache.total_count = total_count;
    cache.page = page;
    cache.page_count = page_count;
    cache.previews.clear();
    for (path, kind) in discovered
        .into_iter()
        .skip(page_start)
        .take(DEVELOPER_VISUAL_ROWS)
    {
        if let Some(image) = load_processing_artifact_preview(&path, &mut images) {
            cache.previews.push(ProcessingArtifactPreview {
                path: path.display().to_string(),
                kind,
                image,
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn sync_processing_artifact_previews(mut cache: ResMut<ProcessingArtifactPreviewCache>) {
    if !cache.signature.is_empty() || !cache.previews.is_empty() || cache.total_count != 0 {
        cache.signature.clear();
        cache.previews.clear();
        cache.total_count = 0;
        cache.page = 0;
        cache.page_count = 0;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn discover_processing_visual_artifacts(
    state: &SceneProcessingState,
) -> Vec<(PathBuf, ProcessingArtifactVisualKind)> {
    let mut roots = Vec::new();
    if let Some(run_id) = state.run_id.as_deref()
        && !run_id.trim().is_empty()
    {
        roots.push(PathBuf::from("tmp").join("runs").join(run_id));
    }
    for path in &state.recent_artifacts {
        roots.push(PathBuf::from(path));
    }

    let mut discovered = Vec::new();
    for root in roots {
        collect_visual_artifacts(&root, 0, &mut discovered);
        if discovered.len() >= 96 {
            break;
        }
    }

    sort_visual_artifacts_for_display(&mut discovered);
    discovered.dedup_by(|(left_path, _), (right_path, _)| left_path == right_path);
    discovered
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn visual_artifact_modified_ms(path: &Path) -> u128 {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn visual_artifact_generation_order(path: &Path) -> Vec<u64> {
    let mut values = Vec::new();
    let mut current = String::new();
    for ch in path.to_string_lossy().chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            values.push(current.parse::<u64>().unwrap_or(u64::MAX));
            current.clear();
        }
    }
    if !current.is_empty() {
        values.push(current.parse::<u64>().unwrap_or(u64::MAX));
    }
    values
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn sort_visual_artifacts_for_display(
    discovered: &mut [(PathBuf, ProcessingArtifactVisualKind)],
) {
    discovered.sort_by_cached_key(|(path, kind)| {
        (
            Reverse(visual_artifact_modified_ms(path)),
            Reverse(visual_artifact_generation_order(path)),
            kind.priority(),
            visual_artifact_score(path, kind),
            Reverse(path.clone()),
        )
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn collect_visual_artifacts(
    path: &Path,
    depth: usize,
    out: &mut Vec<(PathBuf, ProcessingArtifactVisualKind)>,
) {
    if out.len() >= 128 || depth > 6 {
        return;
    }
    if path.is_file() {
        if let Some(kind) = visual_artifact_kind(path) {
            out.push((path.to_path_buf(), kind));
        }
        return;
    }
    if !path.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    let mut entries = entries.filter_map(Result::ok).collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        collect_visual_artifacts(&entry.path(), depth + 1, out);
        if out.len() >= 128 {
            break;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn visual_artifact_kind(path: &Path) -> Option<ProcessingArtifactVisualKind> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp") {
        return None;
    }

    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.contains("detections_overlay") || lower.contains("locate") {
        Some(ProcessingArtifactVisualKind::Locate)
    } else if lower.contains("masks_overlay")
        || lower.contains("segmentation")
        || lower.contains("/mask")
        || lower.contains("\\mask")
    {
        Some(ProcessingArtifactVisualKind::Segmentation)
    } else if lower.contains("depth") || lower.contains("floor") {
        Some(ProcessingArtifactVisualKind::Depth)
    } else if lower.contains("current_isolated_full_frame")
        || lower.contains("isolated_render_full_frame")
        || (lower.contains("rotation_candidates") && lower.ends_with("_screenshot.png"))
    {
        Some(ProcessingArtifactVisualKind::IsolatedRender)
    } else if lower.contains("/crops/") || lower.contains("\\crops\\") || lower.contains("_crop") {
        Some(ProcessingArtifactVisualKind::Crop)
    } else if lower.contains("/generated/")
        || lower.contains("\\generated\\")
        || lower.contains("candidate")
    {
        Some(ProcessingArtifactVisualKind::Generated)
    } else if lower.contains("canonical") || lower.contains("yaw") {
        Some(ProcessingArtifactVisualKind::Canonical)
    } else if lower.contains("projection_fit")
        || lower.contains("visible_surface")
        || lower.contains("silhouette")
    {
        Some(ProcessingArtifactVisualKind::Projection)
    } else if lower.contains("/iterations")
        || lower.contains("\\iterations")
        || lower.contains("feedback")
        || lower.ends_with("screenshot.png")
    {
        Some(ProcessingArtifactVisualKind::Feedback)
    } else if lower.contains("source") || lower.contains("input") {
        Some(ProcessingArtifactVisualKind::Source)
    } else {
        Some(ProcessingArtifactVisualKind::Other)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn visual_artifact_score(path: &Path, kind: &ProcessingArtifactVisualKind) -> usize {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    match kind {
        ProcessingArtifactVisualKind::Locate if lower.contains("detections_overlay") => 0,
        ProcessingArtifactVisualKind::Segmentation if lower.contains("masks_overlay") => 0,
        ProcessingArtifactVisualKind::Depth if lower.contains("depth_overlay") => 0,
        ProcessingArtifactVisualKind::IsolatedRender
            if lower.contains("current_isolated_full_frame") =>
        {
            0
        }
        ProcessingArtifactVisualKind::IsolatedRender => 1,
        ProcessingArtifactVisualKind::Projection if lower.contains("projection_fit_overlay") => 0,
        ProcessingArtifactVisualKind::Feedback if lower.ends_with("screenshot.png") => 0,
        ProcessingArtifactVisualKind::Canonical if lower.contains("selection") => 0,
        ProcessingArtifactVisualKind::Crop => 1,
        ProcessingArtifactVisualKind::Generated => 1,
        _ => 2,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_processing_artifact_preview(
    path: &Path,
    images: &mut Assets<Image>,
) -> Option<Handle<Image>> {
    let bytes = fs::read(path).ok()?;
    let decoded = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (width, height) = decoded.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    let image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    Some(images.add(image))
}

pub(super) fn format_processing_event(event: &SceneProcessingEvent) -> String {
    format_event_row(event, 72, true)
}

pub(super) fn format_developer_event(event: &SceneProcessingEvent) -> String {
    format_event_row(event, 96, false)
}

pub(super) fn format_event_row(
    event: &SceneProcessingEvent,
    max_message_chars: usize,
    compact: bool,
) -> String {
    let item = match (event.item_index, event.item_count) {
        (Some(index), Some(total)) => format!(" [{index}/{total}]"),
        (None, Some(total)) => format!(" [{total}]"),
        _ => String::new(),
    };
    let marker = if event.is_failure { "!" } else { "-" };
    let artifact = event
        .artifact_path
        .as_deref()
        .map(|path| format!(" -> {}", ellipsize_text(path, 40)))
        .unwrap_or_default();
    if compact {
        format!(
            "{marker} {} {} {}{}: {}",
            format_elapsed_ms(event.elapsed_ms),
            event.phase,
            event.stage,
            item,
            ellipsize_text(&event.message, max_message_chars)
        )
    } else {
        format!(
            "{marker} {} [{}] {} / {}{}: {}{}",
            format_elapsed_ms(event.elapsed_ms),
            event.execution,
            event.phase,
            event.stage,
            item,
            ellipsize_text(&event.message, max_message_chars),
            artifact
        )
    }
}

pub(super) fn artifact_kind_label(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".glb") || lower.ends_with(".gltf") || lower.ends_with(".splat") {
        "asset"
    } else if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
    {
        "image"
    } else if lower.ends_with(".bsn") {
        "bsn  "
    } else if lower.ends_with(".json") || lower.ends_with(".jsonl") {
        "json "
    } else if lower.contains("/assets") || lower.ends_with("assets") {
        "dir  "
    } else {
        "file "
    }
}

pub(super) fn compact_worker_status_text(message: &str) -> String {
    let normalized = message.trim();
    if let Some(rest) = normalized.strip_prefix("scene ") {
        let (phase, stage_and_message) = rest.split_once(": ").unwrap_or((rest, ""));
        let (stage, _) = stage_and_message
            .split_once(" - ")
            .unwrap_or((stage_and_message, ""));
        if !stage.is_empty() {
            let label = format!("scene {phase}: {stage}");
            return ellipsize_text(&label, 34);
        }
    }
    ellipsize_text(normalized, 34)
}

pub(super) fn format_elapsed_ms(elapsed_ms: u64) -> String {
    let seconds = elapsed_ms as f64 / 1000.0;
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        let minutes = (seconds / 60.0).floor() as u64;
        let seconds = (seconds as u64) % 60;
        format!("{minutes}:{seconds:02}")
    }
}

pub(super) fn ellipsize_processing_text(text: &str) -> String {
    ellipsize_text(text, 64)
}
