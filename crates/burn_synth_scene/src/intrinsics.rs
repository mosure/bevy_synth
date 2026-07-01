use crate::{SceneCameraIntrinsics, SceneGroundingEvidence};

pub fn source_camera_intrinsics_from_evidence(
    evidence: &SceneGroundingEvidence,
) -> Option<SceneCameraIntrinsics> {
    evidence
        .camera
        .intrinsics
        .filter(|intrinsics| intrinsics.is_valid())
        .or_else(|| {
            evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.intrinsics)
                .filter(|intrinsics| intrinsics.is_valid())
        })
        .or_else(|| source_camera_intrinsics_from_legacy_evidence(evidence))
}

fn source_camera_intrinsics_from_legacy_evidence(
    evidence: &SceneGroundingEvidence,
) -> Option<SceneCameraIntrinsics> {
    let [width, height] = evidence
        .camera
        .image_size
        .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))?;
    let principal = evidence.camera.principal_point;
    evidence
        .camera
        .focal_length_px
        .or_else(|| {
            evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.focal_length_px)
        })
        .filter(|value| value.is_finite() && *value > 1.0)
        .and_then(|focal| {
            SceneCameraIntrinsics::from_focal_length_px(focal, width, height, principal)
        })
        .or_else(|| {
            evidence
                .camera
                .vertical_fov_degrees
                .or_else(|| {
                    evidence
                        .depth
                        .as_ref()
                        .and_then(|depth| depth.vertical_fov_degrees)
                })
                .filter(|value| value.is_finite() && *value > 1.0)
                .and_then(|vertical_fov| {
                    SceneCameraIntrinsics::from_vertical_fov_degrees(
                        vertical_fov,
                        width,
                        height,
                        principal,
                    )
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DepthEvidenceRef, EstimatedCamera, EstimatedFloorPlane, SceneGroundingEvidence};

    #[test]
    fn source_camera_intrinsics_preserve_wide_depthpro_fovx_and_fovy() {
        let width = 3839;
        let height = 2157;
        let fy = 1794.9255;
        let intrinsics =
            SceneCameraIntrinsics::from_fx_fy(fy, fy, 1919.0, 1078.0, width, height).unwrap();
        let evidence = SceneGroundingEvidence {
            source_image_path: "curry.jpg".to_string(),
            depth: Some(DepthEvidenceRef {
                provider: "depth-pro".to_string(),
                model: Some("depth-pro".to_string()),
                precision: Some("f16".to_string()),
                artifact_path: None,
                intrinsics: Some(intrinsics),
                focal_length_px: Some(fy),
                vertical_fov_degrees: Some(62.0),
                image_size: Some([width, height]),
                depth_map_size: Some([width, height]),
                floor_sample_count: Some(1),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: EstimatedCamera {
                intrinsics: None,
                focal_length_px: Some(3200.0),
                principal_point: Some([1919.0, 1078.0]),
                image_size: Some([width, height]),
                vertical_fov_degrees: Some(35.0),
                confidence: Some(0.5),
            },
            floor: EstimatedFloorPlane::default(),
            objects: Vec::new(),
        };

        let resolved = source_camera_intrinsics_from_evidence(&evidence).unwrap();

        assert!((resolved.fov_y_degrees - 62.0).abs() < 0.01);
        assert!((resolved.fov_x_degrees - 93.84).abs() < 0.02);
        assert_eq!(resolved.fx, resolved.fy);
    }
}
