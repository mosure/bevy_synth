use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::*;

#[derive(Clone, Debug, Serialize)]
pub struct SceneGroundCalibrationRequest {
    pub prompt: String,
    pub source_scene_path: PathBuf,
    pub image_paths: Vec<PathBuf>,
    pub context: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneGroundCalibrationResponse {
    pub camera_height_m: f32,
    pub vertical_fov_degrees: f32,
    pub floor_confidence: f32,
    pub floor_residual_m: f32,
    pub scene_calibration: Option<SceneCalibration>,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneGroundCalibrationReport {
    pub schema_version: u32,
    pub source_scene_path: String,
    pub previous_camera: EstimatedCamera,
    pub previous_floor: EstimatedFloorPlane,
    pub response: SceneGroundCalibrationResponse,
    pub applied_camera: EstimatedCamera,
    pub applied_floor: EstimatedFloorPlane,
}

pub fn ground_calibration_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "camera_height_m",
            "vertical_fov_degrees",
            "floor_confidence",
            "floor_residual_m",
            "scene_calibration",
            "rationale"
        ],
        "properties": {
            "camera_height_m": {
                "type": "number",
                "minimum": 0.8,
                "maximum": 4.5,
                "description": "Estimated metric height of the source camera above the room floor."
            },
            "vertical_fov_degrees": {
                "type": "number",
                "minimum": 25.0,
                "maximum": 115.0,
                "description": "Estimated source-camera vertical field of view."
            },
            "floor_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence that the selected floor/camera relation is usable for furniture grounding."
            },
            "floor_residual_m": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.35,
                "description": "Expected residual error of the calibrated floor relation in meters."
            },
            "scene_calibration": {
                "type": ["object", "null"],
                "additionalProperties": false,
                "required": [
                    "table_center",
                    "table_axis_degrees",
                    "table_size_m",
                    "camera_yaw_degrees",
                    "camera_pitch_degrees",
                    "camera_radius_m",
                    "vertical_fov_degrees"
                ],
                "properties": {
                    "table_center": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
                    "table_axis_degrees": { "type": ["number", "null"] },
                    "table_size_m": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
                    "camera_yaw_degrees": { "type": ["number", "null"] },
                    "camera_pitch_degrees": { "type": ["number", "null"] },
                    "camera_radius_m": { "type": ["number", "null"] },
                    "vertical_fov_degrees": { "type": ["number", "null"] }
                }
            },
            "rationale": {
                "type": "string",
                "description": "Concise reason for the camera/floor estimate and any visible ambiguities."
            }
        }
    })
}

pub fn prepare_ground_calibration_request(
    source_scene_path: &Path,
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
    extra_image_paths: &[PathBuf],
) -> SceneGroundCalibrationRequest {
    let object_summary = manifest
        .objects
        .iter()
        .map(|object| {
            format!(
                "{}: label='{}', bbox=[{:.3},{:.3},{:.3},{:.3}], instances={}",
                object.id,
                object.label,
                object.bbox[0],
                object.bbox[1],
                object.bbox[2],
                object.bbox[3],
                object.instance_count.max(object.instances.len())
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let detection_summary = evidence
        .detections
        .iter()
        .map(|detection| {
            format!(
                "{} query='{}' bbox=[{:.3},{:.3},{:.3},{:.3}] point={:?} confidence={:?}",
                detection.label,
                detection.source_query,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3],
                detection.point,
                detection.confidence
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        "Estimate the source camera and floor relation for 3D furniture grounding.\n\
Use the first image as the source scene. Optional following images are diagnostic depth/floor visualizations, not object crops.\n\
Return a practical camera height above the visible floor and vertical field of view. The downstream geometry will convert camera_height_m to an upright floor plane with normal [0,1,0] in camera space.\n\
Prefer room-scale plausible values over overfitting noisy depth: camera_height_m usually lies between 1.1 and 3.8 meters, vertical_fov_degrees usually lies between 35 and 95 degrees. If the image is wide-angle or panorama-like, choose a wider FOV and explain it.\n\
Do not invent object counts; boxes from LocateAnything remain the object cardinality source of truth. Only estimate camera/floor calibration and optional table/camera hints.\n\n\
Objects planned by the scene pipeline:\n{object_summary}\n\n\
Grounding detections:\n{detection_summary}",
    );
    let mut image_paths = Vec::with_capacity(extra_image_paths.len() + 1);
    image_paths.push(source_scene_path.to_path_buf());
    image_paths.extend(
        extra_image_paths
            .iter()
            .filter(|path| path.exists())
            .cloned(),
    );
    SceneGroundCalibrationRequest {
        prompt,
        source_scene_path: source_scene_path.to_path_buf(),
        image_paths,
        context: json!({
            "manifest_objects": manifest.objects.len(),
            "detections": evidence.detections.len(),
            "depth": evidence.depth,
            "camera_before": evidence.camera,
            "floor_before": evidence.floor,
        }),
    }
}

pub fn apply_ground_calibration_response(
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
    response: &SceneGroundCalibrationResponse,
) -> SceneResult<(
    SceneObjectManifest,
    SceneGroundingEvidence,
    SceneGroundCalibrationReport,
)> {
    let camera_height_m = finite_range(response.camera_height_m, 1.1, 3.8, "camera_height_m")?;
    let vertical_fov_degrees = finite_range(
        response.vertical_fov_degrees,
        35.0,
        95.0,
        "vertical_fov_degrees",
    )?;
    let floor_confidence = finite_range(response.floor_confidence, 0.0, 1.0, "floor_confidence")?;
    let floor_residual_m = finite_range(response.floor_residual_m, 0.0, 0.35, "floor_residual_m")?;

    let mut next_manifest = manifest.clone();
    if let Some(mut calibration) = response.scene_calibration {
        calibration.vertical_fov_degrees = calibration
            .vertical_fov_degrees
            .or(Some(vertical_fov_degrees));
        next_manifest.scene_calibration = Some(calibration);
    } else {
        next_manifest.scene_calibration = Some(SceneCalibration {
            vertical_fov_degrees: Some(vertical_fov_degrees),
            ..manifest.scene_calibration.unwrap_or(SceneCalibration {
                table_center: None,
                table_axis_degrees: None,
                table_size_m: None,
                camera_yaw_degrees: None,
                camera_pitch_degrees: None,
                camera_radius_m: None,
                vertical_fov_degrees: None,
            })
        });
    }

    let mut next_evidence = evidence.clone();
    let image_size = next_evidence
        .camera
        .image_size
        .or_else(|| {
            next_evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.image_size)
        })
        .or_else(|| {
            image::image_dimensions(&next_evidence.source_image_path)
                .ok()
                .map(|(w, h)| [w, h])
        });
    next_evidence.camera.image_size = image_size;
    next_evidence.camera.vertical_fov_degrees = Some(vertical_fov_degrees);
    next_evidence.camera.confidence = Some(floor_confidence);
    if let Some([width, height]) = image_size {
        next_evidence.camera.principal_point = next_evidence
            .camera
            .principal_point
            .or(Some([width as f32 * 0.5, height as f32 * 0.5]));
        next_evidence.camera.focal_length_px = Some(
            (height.max(1) as f32 * 0.5)
                / (vertical_fov_degrees.to_radians() * 0.5).tan().max(1.0e-5),
        );
    }
    next_evidence.floor = EstimatedFloorPlane {
        normal: [0.0, 1.0, 0.0],
        distance_m: -camera_height_m,
        residual_m: Some(floor_residual_m),
        confidence: Some(floor_confidence),
    };
    if let Some(depth) = next_evidence.depth.as_mut() {
        depth.vertical_fov_degrees = Some(vertical_fov_degrees);
        depth.focal_length_px = next_evidence.camera.focal_length_px;
        depth.image_size = depth.image_size.or(image_size);
    }

    let report = SceneGroundCalibrationReport {
        schema_version: 1,
        source_scene_path: next_evidence.source_image_path.clone(),
        previous_camera: evidence.camera,
        previous_floor: evidence.floor,
        response: response.clone(),
        applied_camera: next_evidence.camera,
        applied_floor: next_evidence.floor,
    };
    Ok((next_manifest, next_evidence, report))
}

fn finite_range(value: f32, min: f32, max: f32, name: &str) -> SceneResult<f32> {
    if !value.is_finite() {
        return Err(SceneError::Provider(format!(
            "ground calibration returned non-finite {name}"
        )));
    }
    Ok(value.clamp(min, max))
}
