use crate::{
    CanonicalPoseEvidence, SceneAssetAabb, SceneAssetBinding, SceneAssetFrame,
    SceneAssetFrameSource, SceneAssetSymmetry,
};

pub fn canonical_spawn_yaw_degrees(
    instance_yaw_degrees: f32,
    asset_yaw_offset_degrees: f32,
    feedback_delta_degrees: f32,
) -> f32 {
    normalize_degrees(instance_yaw_degrees - asset_yaw_offset_degrees + feedback_delta_degrees)
}

pub fn canonical_pose_evidence_for_assets(
    assets: &[SceneAssetBinding],
) -> Vec<CanonicalPoseEvidence> {
    assets
        .iter()
        .map(canonical_pose_evidence_for_asset)
        .collect()
}

pub fn canonical_pose_evidence_for_asset(asset: &SceneAssetBinding) -> CanonicalPoseEvidence {
    let descriptor = format!("{} {}", asset.label, asset.aliases.join(" ")).to_ascii_lowercase();
    let frame = canonical_frame_for_asset(asset, &descriptor);
    let symmetry = frame
        .symmetry
        .unwrap_or_else(|| symmetry_for_descriptor(&descriptor));
    let confidence = frame
        .confidence
        .unwrap_or_else(|| frame_confidence(&descriptor, asset.local_aabb));
    CanonicalPoseEvidence {
        asset_id: asset.asset_id.clone(),
        object_id: asset.object_id.clone(),
        label: asset.label.clone(),
        frame,
        local_aabb: asset.local_aabb,
        source_image_path: asset.source_image_path.clone(),
        descriptor,
        method: frame
            .source
            .unwrap_or(SceneAssetFrameSource::Unknown)
            .to_string(),
        confidence,
        symmetry,
    }
}

pub fn canonical_frame_for_asset(asset: &SceneAssetBinding, descriptor: &str) -> SceneAssetFrame {
    if let Some(mut frame) = asset.canonical_frame {
        if frame.symmetry.is_none() {
            frame.symmetry = Some(symmetry_for_descriptor(descriptor));
        }
        if frame.confidence.is_none() {
            frame.confidence = Some(frame_confidence(descriptor, asset.local_aabb));
        }
        if frame.source.is_none() {
            frame.source = Some(SceneAssetFrameSource::Explicit);
        }
        return frame;
    }

    let yaw_offset_degrees =
        if is_table_like_descriptor(descriptor) && aabb_x_major(asset.local_aabb) {
            90.0
        } else {
            0.0
        };
    let mut frame = SceneAssetFrame::heuristic(yaw_offset_degrees, None);
    frame.symmetry = Some(symmetry_for_descriptor(descriptor));
    frame.confidence = Some(frame_confidence(descriptor, asset.local_aabb));
    frame.source = Some(if asset.local_aabb.is_some() {
        SceneAssetFrameSource::AabbHeuristic
    } else {
        SceneAssetFrameSource::DescriptorHeuristic
    });
    frame
}

pub fn symmetry_for_descriptor(descriptor: &str) -> SceneAssetSymmetry {
    if descriptor.contains("round table")
        || descriptor.contains("circular table")
        || descriptor.contains("stool")
    {
        SceneAssetSymmetry::Radial
    } else if is_table_like_descriptor(descriptor) {
        SceneAssetSymmetry::Axis180
    } else if descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("bench")
    {
        SceneAssetSymmetry::Bilateral
    } else {
        SceneAssetSymmetry::Unknown
    }
}

fn frame_confidence(descriptor: &str, local_aabb: Option<SceneAssetAabb>) -> f32 {
    if is_table_like_descriptor(descriptor) && local_aabb.is_some() {
        0.70
    } else if descriptor.contains("chair") || descriptor.contains("seat") {
        0.58
    } else if descriptor.contains("sofa") || descriptor.contains("couch") {
        0.50
    } else {
        0.35
    }
}

fn is_table_like_descriptor(descriptor: &str) -> bool {
    descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
}

fn aabb_x_major(local_aabb: Option<SceneAssetAabb>) -> bool {
    local_aabb
        .map(|aabb| aabb.size()[0] > aabb.size()[2] * 1.15)
        .unwrap_or(false)
}

impl std::fmt::Display for SceneAssetFrameSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Explicit => "explicit",
            Self::AabbHeuristic => "aabb_heuristic",
            Self::DescriptorHeuristic => "descriptor_heuristic",
            Self::PoseFitHeuristic => "pose_fit_heuristic",
            Self::VisualRenderSweep => "visual_render_sweep",
            Self::GptVisualSelection => "gpt_visual_selection",
            Self::AmbiguousFallback => "ambiguous_fallback",
            Self::Unknown => "unknown",
        };
        f.write_str(value)
    }
}

fn normalize_degrees(mut degrees: f32) -> f32 {
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    degrees
}
