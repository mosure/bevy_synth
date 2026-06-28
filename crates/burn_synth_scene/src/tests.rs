use super::*;
use crate::bsn::{image_data_url, representative_crop_bbox};
use crate::layout::{
    MetricSceneFrame, bsn_yaw_toward_point_degrees, floor_contact_point_from_evidence,
};
use crate::object_images::{
    ObjectImageMatteStats, generated_shape_consistency_score, generated_source_crop_edge_mismatch,
    matte_generated_object_rgb, object_reconstruction_min_score, score_generated_object_rgb,
};
use crate::openai::image_error_is_retryable;
use serde_json::json;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

struct RetryImageProvider {
    images: RefCell<VecDeque<Vec<u8>>>,
}

impl RetryImageProvider {
    fn new(images: Vec<Vec<u8>>) -> Self {
        Self {
            images: RefCell::new(images.into()),
        }
    }
}

impl SceneAiProvider for RetryImageProvider {
    fn plan_objects(&self, _request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        Err(SceneError::Provider(
            "plan_objects is not used by retry image tests".to_string(),
        ))
    }

    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>> {
        let mut images = self.images.borrow_mut();
        let mut output = Vec::new();
        for _ in 0..request.candidate_count {
            output.push(images.pop_front().ok_or_else(|| {
                SceneError::Provider("test image provider exhausted".to_string())
            })?);
        }
        Ok(output)
    }

    fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
        Err(SceneError::Provider(
            "plan_scene_bsn is not used by retry image tests".to_string(),
        ))
    }
}

struct ParallelImageProvider {
    image: Vec<u8>,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

impl ParallelImageProvider {
    fn new(image: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            image,
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        })
    }

    fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
}

impl SceneAiProvider for Arc<ParallelImageProvider> {
    fn plan_objects(&self, _request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        Err(SceneError::Provider(
            "plan_objects is not used by parallel image tests".to_string(),
        ))
    }

    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(40));
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok((0..request.candidate_count)
            .map(|_| self.image.clone())
            .collect())
    }

    fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
        Err(SceneError::Provider(
            "plan_scene_bsn is not used by parallel image tests".to_string(),
        ))
    }
}

#[test]
fn canonical_pose_evidence_marks_table_axis_and_chair_symmetry() {
    let assets = vec![
        SceneAssetBinding {
            asset_id: "table_asset".to_string(),
            object_id: "table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: None,
            reusable: true,
            source_image_path: None,
            pipeline: None,
            local_aabb: Some(SceneAssetAabb {
                min: [-1.5, 0.0, -0.4],
                max: [1.5, 0.2, 0.4],
            }),
            canonical_frame: None,
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "chair_asset".to_string(),
            object_id: "chair".to_string(),
            label: "mesh chair".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: None,
            reusable: true,
            source_image_path: None,
            pipeline: None,
            local_aabb: None,
            canonical_frame: None,
            provenance: None,
        },
    ];
    let evidence = canonical_pose_evidence_for_assets(&assets);
    assert_eq!(evidence[0].frame.yaw_offset_degrees, 90.0);
    assert_eq!(evidence[0].symmetry, SceneAssetSymmetry::Axis180);
    assert_eq!(evidence[1].symmetry, SceneAssetSymmetry::Bilateral);
}

#[test]
fn canonical_spawn_yaw_subtracts_asset_offset_then_applies_feedback_delta() {
    assert!((canonical_spawn_yaw_degrees(20.0, 180.0, -10.0) + 170.0).abs() <= 1.0e-5);
    assert!((canonical_spawn_yaw_degrees(-175.0, 20.0, 0.0) - 165.0).abs() <= 1.0e-5);
}

fn png_bytes(image: image::RgbImage) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .unwrap();
    cursor.into_inner()
}

fn scene_object(id: &str, reuse_group: Option<&str>) -> SceneObjectSpec {
    SceneObjectSpec {
        id: id.to_string(),
        label: id.replace('_', " "),
        aliases: Vec::new(),
        bbox: [0.1, 0.2, 0.4, 0.7],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: reuse_group.map(str::to_string),
        instance_count: 1,
        object_prompt: format!("isolated {id}"),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    }
}

#[test]
fn candidate_selection_can_exclude_failed_asset_candidate() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![scene_object("chair", None)],
    };
    let candidates = vec![
        ObjectImageCandidate {
            object_id: "chair".to_string(),
            candidate_index: 0,
            image_path: "/tmp/chair_bad.png".to_string(),
            raw_image_path: None,
            prompt_hash: "bad".to_string(),
            score: 0.99,
            provider_request_id: None,
        },
        ObjectImageCandidate {
            object_id: "chair".to_string(),
            candidate_index: 1,
            image_path: "/tmp/chair_retry.png".to_string(),
            raw_image_path: None,
            prompt_hash: "retry".to_string(),
            score: 0.91,
            provider_request_id: None,
        },
    ];
    let excluded = std::collections::HashSet::from([("chair".to_string(), 0usize)]);

    let selected =
        select_object_image_candidates_with_exclusions(&manifest, &candidates, 0.45, &excluded)
            .unwrap();

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[0].image_path, "/tmp/chair_retry.png");
}

#[test]
fn candidate_selection_deduplicates_reuse_groups_with_exclusions() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            scene_object("chair_left", Some("chair_group")),
            scene_object("chair_right", Some("chair_group")),
            scene_object("table", None),
        ],
    };
    let candidates = vec![
        ObjectImageCandidate {
            object_id: "chair_left".to_string(),
            candidate_index: 0,
            image_path: "/tmp/chair_bad.png".to_string(),
            raw_image_path: None,
            prompt_hash: "bad".to_string(),
            score: 0.99,
            provider_request_id: None,
        },
        ObjectImageCandidate {
            object_id: "chair_left".to_string(),
            candidate_index: 1,
            image_path: "/tmp/chair_good.png".to_string(),
            raw_image_path: None,
            prompt_hash: "good".to_string(),
            score: 0.92,
            provider_request_id: None,
        },
        ObjectImageCandidate {
            object_id: "table".to_string(),
            candidate_index: 0,
            image_path: "/tmp/table.png".to_string(),
            raw_image_path: None,
            prompt_hash: "table".to_string(),
            score: 0.88,
            provider_request_id: None,
        },
    ];
    let excluded = std::collections::HashSet::from([("chair_left".to_string(), 0usize)]);

    let selected =
        select_object_image_candidates_with_exclusions(&manifest, &candidates, 0.45, &excluded)
            .unwrap();

    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].object_id, "chair_left");
    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[1].object_id, "table");
}

#[test]
fn grounding_evidence_overrides_manifest_bbox_and_contact() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chair".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.3, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let mut evidence = manifest_grounding_evidence(&manifest);
    let object = evidence.objects.first_mut().unwrap();
    object.detection.as_mut().unwrap().bbox = [0.2, 0.3, 0.4, 0.9];
    object.contact_pixel = Some([0.33, 0.88]);

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);

    assert_eq!(adjusted.objects[0].bbox, [0.2, 0.3, 0.4, 0.9]);
    assert_eq!(evidence.detections.len(), 1);
    assert_eq!(evidence.objects[0].provenance, ["manifest_fallback"]);
}

#[test]
fn object_image_requests_use_grounded_manifest_bbox_for_crop() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("scene.png");
    image::RgbImage::from_pixel(100, 100, image::Rgb([32, 32, 32]))
        .save(&source_path)
        .unwrap();
    let manifest = SceneObjectManifest {
        source_scene_path: source_path.display().to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "table".to_string(),
            label: "white table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.0, 0.0, 1.0, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "white table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.objects[0].detection.as_mut().unwrap().bbox = [0.45, 0.35, 0.55, 0.60];
    evidence.objects[0].contact_pixel = Some([0.50, 0.60]);
    let grounded_manifest = manifest_with_grounding_evidence(&manifest, &evidence);
    let pipeline = ScenePipeline::new(
        SceneBuildConfig {
            source_scene_path: source_path.clone(),
            object_reference_image_path: source_path,
            output_dir: dir.path().join("run"),
            candidate_count: 1,
            quality_profile: SceneQualityProfile::Draft,
            reasoning_model: "test-reasoning".to_string(),
            image_model: "test-image".to_string(),
            allow_catalog_reuse: false,
        },
        RetryImageProvider::new(Vec::new()),
    );

    let requests = pipeline
        .prepare_object_image_requests(&grounded_manifest)
        .expect("prepare grounded object image requests");

    assert_eq!(requests[0].object.bbox, [0.45, 0.35, 0.55, 0.60]);
    let crop = image::open(&requests[0].source_crop_path).expect("grounded crop should exist");
    assert_eq!((crop.width(), crop.height()), (15, 31));
}

#[test]
fn object_image_requests_use_grounded_single_instance_bbox_for_reuse_group_crop() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("scene.png");
    image::RgbImage::from_pixel(100, 100, image::Rgb([32, 32, 32]))
        .save(&source_path)
        .unwrap();
    let manifest = SceneObjectManifest {
        source_scene_path: source_path.display().to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chair_group".to_string(),
            label: "reusable chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.0, 0.0, 1.0, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "one chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let evidence = SceneGroundingEvidence {
        source_image_path: source_path.display().to_string(),
        depth: None,
        segmentation: None,
        detections: Vec::new(),
        camera: EstimatedCamera::default(),
        floor: EstimatedFloorPlane::default(),
        objects: vec![
            ObjectGroundingEvidence {
                object_id: "chair_group".to_string(),
                instance_id: Some("chair_left".to_string()),
                reuse_group: Some("chair".to_string()),
                detection: Some(Detection {
                    label: "chair".to_string(),
                    bbox: [0.10, 0.20, 0.22, 0.55],
                    point: Some([0.16, 0.55]),
                    confidence: Some(0.9),
                    source_query: "chair".to_string(),
                }),
                mask: None,
                asset_id: None,
                contact_pixel: Some([0.16, 0.55]),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: None,
                provenance: vec!["locate_anything".to_string()],
            },
            ObjectGroundingEvidence {
                object_id: "chair_group".to_string(),
                instance_id: Some("chair_right".to_string()),
                reuse_group: Some("chair".to_string()),
                detection: Some(Detection {
                    label: "chair".to_string(),
                    bbox: [0.62, 0.24, 0.86, 0.78],
                    point: Some([0.74, 0.78]),
                    confidence: Some(0.92),
                    source_query: "chair".to_string(),
                }),
                mask: None,
                asset_id: None,
                contact_pixel: Some([0.74, 0.78]),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: None,
                provenance: vec!["locate_anything".to_string()],
            },
        ],
    };
    let grounded_manifest = manifest_with_grounding_evidence(&manifest, &evidence);
    let pipeline = ScenePipeline::new(
        SceneBuildConfig {
            source_scene_path: source_path.clone(),
            object_reference_image_path: source_path,
            output_dir: dir.path().join("run"),
            candidate_count: 1,
            quality_profile: SceneQualityProfile::Draft,
            reasoning_model: "test-reasoning".to_string(),
            image_model: "test-image".to_string(),
            allow_catalog_reuse: false,
        },
        RetryImageProvider::new(Vec::new()),
    );

    let requests = pipeline
        .prepare_object_image_requests(&grounded_manifest)
        .expect("prepare grounded repeated-object image requests");

    assert_eq!(requests[0].object.bbox, [0.62, 0.24, 0.86, 0.78]);
    let crop =
        image::open(&requests[0].source_crop_path).expect("single-instance crop should exist");
    assert_eq!((crop.width(), crop.height()), (30, 66));
}

#[test]
fn grounding_evidence_prefers_mask_bbox_when_available() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chair".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.3, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let mut evidence = manifest_grounding_evidence(&manifest);
    let object = evidence.objects.first_mut().unwrap();
    object.detection.as_mut().unwrap().bbox = [0.10, 0.20, 0.60, 0.90];
    object.mask = Some(ObjectMaskEvidence {
        provider: "bbox-prompt".to_string(),
        model: "bbox-prompt".to_string(),
        bbox: [0.20, 0.30, 0.40, 0.80],
        score: 1.0,
        area_px: 100,
        image_size: [100, 100],
        mask_rle: Vec::new(),
        center_pixel: Some([0.30, 0.55]),
        contact_pixel: Some([0.30, 0.80]),
        coverage: Some(0.01),
        artifact_path: None,
        mask_png_path: None,
    });

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);

    assert_eq!(adjusted.objects[0].bbox, [0.20, 0.30, 0.40, 0.80]);
}

#[test]
fn grounding_evidence_can_add_detected_repeated_instances() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chairs".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.8, 0.8],
            instances: vec![SceneObjectInstanceSpec {
                id: Some("chair_existing".to_string()),
                bbox: [0.1, 0.2, 0.3, 0.8],
                contact: Some([0.2, 0.8]),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Left),
                slot_index: None,
                target_footprint_m: Some([0.6, 0.6]),
            }],
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.6]),
        }],
    };
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.objects.push(ObjectGroundingEvidence {
        object_id: "chairs".to_string(),
        instance_id: Some("locate_01".to_string()),
        reuse_group: Some("chair".to_string()),
        detection: Some(Detection {
            label: "chair".to_string(),
            bbox: [0.55, 0.3, 0.75, 0.85],
            point: None,
            confidence: None,
            source_query: "chair".to_string(),
        }),
        mask: None,
        asset_id: None,
        contact_pixel: Some([0.65, 0.85]),
        depth_stats: None,
        candidate_floor_contact_rays: Vec::new(),
        metric_contact_point_m: Some([1.0, 0.0, 2.0]),
        target_footprint_m: Some([0.62, 0.64]),
        provenance: vec!["locate_anything_extra_instance".to_string()],
    });

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);

    assert_eq!(adjusted.objects[0].instances.len(), 2);
    assert_eq!(adjusted.objects[0].instance_count, 2);
    let added = adjusted.objects[0]
        .instances
        .iter()
        .find(|instance| instance.id.as_deref() == Some("locate_01"))
        .unwrap();
    assert_eq!(added.bbox, [0.55, 0.3, 0.75, 0.85]);
    assert_eq!(added.contact, Some([0.65, 0.85]));
    assert_eq!(added.target_footprint_m, Some([0.62, 0.64]));
}

#[test]
fn grounding_evidence_deduplicates_detected_instances_by_bbox() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chairs".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.8, 0.8],
            instances: vec![SceneObjectInstanceSpec {
                id: Some("chair_left".to_string()),
                bbox: [0.12, 0.30, 0.30, 0.82],
                contact: Some([0.21, 0.82]),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Left),
                slot_index: None,
                target_footprint_m: Some([0.6, 0.6]),
            }],
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.6]),
        }],
    };
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.objects.push(ObjectGroundingEvidence {
        object_id: "chairs".to_string(),
        instance_id: Some("locate_01".to_string()),
        reuse_group: Some("chair".to_string()),
        detection: Some(Detection {
            label: "chair".to_string(),
            bbox: [0.121, 0.301, 0.299, 0.821],
            point: None,
            confidence: Some(0.91),
            source_query: "chair".to_string(),
        }),
        mask: None,
        asset_id: None,
        contact_pixel: Some([0.21, 0.82]),
        depth_stats: None,
        candidate_floor_contact_rays: Vec::new(),
        metric_contact_point_m: None,
        target_footprint_m: Some([0.6, 0.6]),
        provenance: vec!["locate_anything_duplicate".to_string()],
    });

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);

    assert_eq!(adjusted.objects[0].instances.len(), 1);
    assert_eq!(adjusted.objects[0].instance_count, 1);
    assert_eq!(
        adjusted.objects[0].instances[0].id.as_deref(),
        Some("chair_left")
    );
}

#[test]
fn grounding_evidence_deduplicates_manifest_instances_by_bbox() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chairs".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.8, 0.8],
            instances: vec![
                SceneObjectInstanceSpec {
                    id: Some("chair_right_far_01".to_string()),
                    bbox: [0.612, 0.515, 0.673, 0.744],
                    contact: Some([0.6425, 0.744]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: Some(1),
                    target_footprint_m: Some([0.6, 0.6]),
                },
                SceneObjectInstanceSpec {
                    id: Some("chair_right_02".to_string()),
                    bbox: [0.612, 0.515, 0.673, 0.744],
                    contact: Some([0.6425, 0.744]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: Some(2),
                    target_footprint_m: Some([0.6, 0.6]),
                },
            ],
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.6]),
        }],
    };
    let evidence = manifest_grounding_evidence(&manifest);

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);

    assert_eq!(adjusted.objects[0].instances.len(), 1);
    assert_eq!(adjusted.objects[0].instance_count, 1);
    assert_eq!(
        adjusted.objects[0].instances[0].id.as_deref(),
        Some("chair_right_far_01")
    );
}

#[test]
fn grounding_evidence_preserves_table_embedded_seating_instances() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.30, 0.45, 0.65, 1.0],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("table".to_string()),
                instance_count: 1,
                object_prompt: "table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.0, 1.2]),
            },
            SceneObjectSpec {
                id: "chairs".to_string(),
                label: "conference chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.1, 0.2, 0.8, 0.9],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("valid_side_chair".to_string()),
                        bbox: [0.12, 0.55, 0.32, 0.95],
                        contact: Some([0.22, 0.95]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Left),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.6, 0.6]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("embedded_table_artifact".to_string()),
                        bbox: [0.41, 0.72, 0.56, 0.99],
                        contact: Some([0.48, 0.99]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Foot),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.6, 0.6]),
                    },
                ],
                representative_instance_id: None,
                reuse_group: Some("chair".to_string()),
                instance_count: 2,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.6, 0.6]),
            },
        ],
    };
    let evidence = manifest_grounding_evidence(&manifest);

    let adjusted = manifest_with_grounding_evidence(&manifest, &evidence);
    let chairs = adjusted
        .objects
        .iter()
        .find(|object| object.id == "chairs")
        .unwrap();

    assert_eq!(chairs.instances.len(), 2);
    assert_eq!(chairs.instance_count, 2);
    assert!(
        chairs
            .instances
            .iter()
            .any(|instance| instance.id.as_deref() == Some("valid_side_chair"))
    );
    assert!(
        chairs
            .instances
            .iter()
            .any(|instance| instance.id.as_deref() == Some("embedded_table_artifact"))
    );
}

#[test]
fn depth_grounding_evidence_drives_metric_contact_points_and_scale() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/depth_scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.35, 0.40, 0.65, 0.74],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "large conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.68, 0.50, 0.82, 0.92],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("conference_chair".to_string()),
                instance_count: 1,
                object_prompt: "conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 0.35, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        chair_asset(),
    ];
    let mut evidence = manifest_grounding_evidence(&manifest);
    for object in &mut evidence.objects {
        match object.object_id.as_str() {
            "conference_table" => {
                object.metric_contact_point_m = Some([2.0, 0.4, 5.0]);
                object.target_footprint_m = Some([2.4, 1.0]);
                object.provenance.push("depth_pro".to_string());
            }
            "chair_group" => {
                object.metric_contact_point_m = Some([3.4, 0.8, 6.6]);
                object.target_footprint_m = Some([0.72, 0.76]);
                object.provenance.push("depth_pro".to_string());
            }
            _ => {}
        }
    }

    let layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence).unwrap();
    let table = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "conference_table")
        .unwrap();
    let chair = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair_group")
        .unwrap();

    assert_eq!(table.ground_point, [0.0, 0.0, 0.0]);
    assert!(
        (chair.ground_point[0] - 1.4).abs() < 0.08,
        "chair ground point {:?}, camera {:?}",
        chair.ground_point,
        layout.camera
    );
    assert!(
        (chair.ground_point[2] + 1.6).abs() < 0.08,
        "chair ground point {:?}, camera {:?}",
        chair.ground_point,
        layout.camera
    );
    assert_eq!(chair.target_footprint_m, [0.72, 0.76]);
    assert!(chair.scale[0] > 0.7);
    assert!(
        layout
            .bsn
            .contains("asset conference_table_asset = \"cache:table\";")
    );
    assert!(
        layout
            .bsn
            .contains("asset chair_asset = \"path:/tmp/chair.glb\";")
    );
    parse_scene_bsn(&layout.bsn, &assets).expect("depth-grounded BSN parses");
}

#[test]
fn depth_grounding_uses_table_bbox_center_as_scene_origin() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/depth_scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.40, 0.40, 0.60, 0.60],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "large conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([2.4, 1.0]),
            },
            SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.68, 0.50, 0.82, 0.92],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("conference_chair".to_string()),
                instance_count: 1,
                object_prompt: "conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.72, 0.76]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 0.35, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        chair_asset(),
    ];
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.camera = EstimatedCamera {
        focal_length_px: Some(100.0),
        principal_point: Some([50.0, 50.0]),
        image_size: Some([101, 101]),
        vertical_fov_degrees: None,
        confidence: Some(1.0),
    };
    for object in &mut evidence.objects {
        match object.object_id.as_str() {
            "conference_table" => {
                object.metric_contact_point_m = Some([2.0, 0.4, 5.0]);
                object.depth_stats = Some(ObjectDepthStats {
                    median_m: 4.0,
                    min_m: 4.0,
                    max_m: 4.0,
                    contact_m: Some(5.0),
                    sample_count: Some(64),
                });
                object.target_footprint_m = Some([2.4, 1.0]);
                object.provenance.push("depth_pro".to_string());
            }
            "chair_group" => {
                object.metric_contact_point_m = Some([3.0, 0.8, 6.0]);
                object.target_footprint_m = Some([0.72, 0.76]);
                object.provenance.push("depth_pro".to_string());
            }
            _ => {}
        }
    }

    let layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence).unwrap();
    let table = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "conference_table")
        .unwrap();
    let chair = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair_group")
        .unwrap();

    assert_eq!(table.ground_point, [0.0, 0.0, 0.0]);
    assert!((chair.ground_point[0] - 3.0).abs() < 1.0e-4);
    assert!((chair.ground_point[2] + 2.0).abs() < 1.0e-4);
}

#[test]
fn depth_grounding_preserves_source_camera_frame_with_yaw_hint() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/depth_scene.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.5, 0.5]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([2.4, 1.0]),
            camera_yaw_degrees: Some(180.0),
            camera_pitch_degrees: Some(45.0),
            camera_radius_m: Some(5.0),
            vertical_fov_degrees: Some(78.0),
        }),
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.40, 0.40, 0.60, 0.60],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "large conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([2.4, 1.0]),
            },
            SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.68, 0.50, 0.82, 0.92],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("conference_chair".to_string()),
                instance_count: 1,
                object_prompt: "conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.72, 0.76]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 0.35, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        chair_asset(),
    ];
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.depth = Some(DepthEvidenceRef {
        provider: "synthetic_depth".to_string(),
        model: None,
        precision: None,
        artifact_path: None,
        focal_length_px: Some(100.0),
        vertical_fov_degrees: Some(60.0),
        image_size: Some([101, 101]),
        depth_map_size: Some([101, 101]),
        floor_sample_count: Some(32),
    });
    evidence.camera = EstimatedCamera {
        focal_length_px: Some(100.0),
        principal_point: Some([50.0, 50.0]),
        image_size: Some([101, 101]),
        vertical_fov_degrees: Some(60.0),
        confidence: Some(1.0),
    };
    evidence.floor = EstimatedFloorPlane {
        normal: [0.0, 1.0, 0.0],
        distance_m: -1.5,
        residual_m: Some(0.01),
        confidence: Some(0.99),
    };
    for object in &mut evidence.objects {
        match object.object_id.as_str() {
            "conference_table" => object.metric_contact_point_m = Some([2.0, 0.0, 5.0]),
            "chair_group" => object.metric_contact_point_m = Some([3.4, 0.0, 6.6]),
            _ => {}
        }
    }

    let layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence).unwrap();
    let chair = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair_group")
        .unwrap();

    assert!(
        (chair.ground_point[0] - 1.4).abs() < 0.08,
        "chair ground point {:?}, camera {:?}",
        chair.ground_point,
        layout.camera
    );
    assert!(
        (chair.ground_point[2] + 1.6).abs() < 0.08,
        "chair ground point {:?}, camera {:?}",
        chair.ground_point,
        layout.camera
    );
    assert_eq!(layout.camera.translation, [-2.0, 1.5, 5.0]);
    assert_eq!(layout.camera.focus, [-2.0, 1.5, 4.0]);
    assert_eq!(layout.camera.yaw, None);
    assert_eq!(layout.camera.pitch, None);
    assert_eq!(layout.camera.radius, None);
    assert!(layout.bsn.contains("vertical_fov 60"));
    assert!(!layout.bsn.contains(" yaw "));
    assert!(!layout.bsn.contains(" pitch "));
    assert!(!layout.bsn.contains(" radius "));
}

#[test]
fn depth_grounding_prefers_floor_ray_intersections_over_raw_depth_points() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/floor_ray_scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "table".to_string(),
                label: "table".to_string(),
                aliases: Vec::new(),
                bbox: [0.4, 0.4, 0.6, 0.7],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([1.0, 1.0]),
            },
            SceneObjectSpec {
                id: "chair".to_string(),
                label: "chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.65, 0.5, 0.78, 0.85],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.6, 0.6]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "table_asset".to_string(),
            object_id: "table".to_string(),
            label: "table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 0.3, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "chair_asset".to_string(),
            object_id: "chair".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
    ];
    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.floor = EstimatedFloorPlane {
        normal: [0.0, 1.0, 0.0],
        distance_m: 2.0,
        residual_m: Some(0.01),
        confidence: Some(0.99),
    };
    for object in &mut evidence.objects {
        object.metric_contact_point_m = Some([100.0, 0.0, 100.0]);
        object.candidate_floor_contact_rays = match object.object_id.as_str() {
            "table" => vec![[1.0, -1.0, 2.0]],
            "chair" => vec![[2.0, -1.0, 3.0]],
            _ => Vec::new(),
        };
    }

    let layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence).unwrap();
    let chair = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair")
        .unwrap();

    assert!((chair.ground_point[0] - 2.0).abs() < 1.0e-4);
    assert!((chair.ground_point[2] + 2.0).abs() < 1.0e-4);
}

#[test]
fn depth_grounding_ignores_high_residual_floor_ray_intersections() {
    let object = ObjectGroundingEvidence {
        object_id: "chair".to_string(),
        instance_id: None,
        reuse_group: None,
        detection: None,
        mask: None,
        asset_id: None,
        contact_pixel: None,
        depth_stats: None,
        candidate_floor_contact_rays: vec![[2.0, -1.0, 3.0]],
        metric_contact_point_m: Some([7.0, 0.0, 11.0]),
        target_footprint_m: None,
        provenance: vec!["test".to_string()],
    };
    let floor = EstimatedFloorPlane {
        normal: [0.0, 1.0, 0.0],
        distance_m: 2.0,
        residual_m: Some(0.25),
        confidence: Some(0.99),
    };

    assert_eq!(floor_contact_point_from_evidence(&object, &floor), None);
}

#[test]
fn depth_metric_contact_overrides_calibrated_side_slot_layout() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/depth_calibrated_scene.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.5, 0.62]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([1.2, 3.4]),
            camera_yaw_degrees: Some(0.0),
            camera_pitch_degrees: Some(-30.0),
            camera_radius_m: Some(5.0),
            vertical_fov_degrees: Some(78.0),
        }),
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.35, 0.35, 0.65, 0.85],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "large rectangular conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([1.2, 3.4]),
            },
            SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.70, 0.40, 0.82, 0.82],
                instances: vec![SceneObjectInstanceSpec {
                    id: Some("right_01".to_string()),
                    bbox: [0.70, 0.40, 0.82, 0.82],
                    contact: Some([0.76, 0.82]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: Some(0),
                    target_footprint_m: Some([0.58, 0.62]),
                }],
                representative_instance_id: Some("right_01".to_string()),
                reuse_group: Some("conference_chair".to_string()),
                instance_count: 1,
                object_prompt: "conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.58, 0.62]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.35, 0.0, -1.0],
                max: [0.35, 0.32, 1.0],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([1.2, 3.4]))),
            provenance: None,
        },
        chair_asset(),
    ];
    let slot_layout =
        grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
            .expect("slot-only grounded layout");
    let slot_chair = slot_layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair_group")
        .unwrap();
    assert!(slot_chair.ground_point[0] > 0.9);
    assert!(slot_chair.ground_point[2].abs() < 0.1);

    let mut evidence = manifest_grounding_evidence(&manifest);
    for object in &mut evidence.objects {
        match (object.object_id.as_str(), object.instance_id.as_deref()) {
            ("conference_table", None) => {
                object.metric_contact_point_m = Some([2.0, 0.4, 5.0]);
                object.target_footprint_m = Some([1.2, 3.4]);
                object.provenance.push("depth_pro".to_string());
            }
            ("chair_group", Some("right_01")) => {
                object.metric_contact_point_m = Some([5.0, 0.8, 9.0]);
                object.target_footprint_m = Some([0.58, 0.62]);
                object.provenance.push("depth_pro".to_string());
            }
            _ => {}
        }
    }

    let depth_layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence).unwrap();
    let table = depth_layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "conference_table")
        .unwrap();
    let chair = depth_layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "chair_group")
        .unwrap();

    assert_eq!(table.ground_point, [0.0, 0.0, 0.0]);
    assert!((chair.ground_point[0] - 3.0).abs() < 1.0e-4);
    assert!((chair.ground_point[2] + 4.0).abs() < 1.0e-4);
    assert!((chair.ground_point[0] - slot_chair.ground_point[0]).abs() > 1.0);
    assert!((chair.ground_point[2] - slot_chair.ground_point[2]).abs() > 1.0);
    parse_scene_bsn(&depth_layout.bsn, &assets).expect("depth-calibrated BSN parses");
}

#[test]
fn preparation_records_configured_openai_models() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("scene.jpg");
    let reference = dir.path().join("input_chair.jpg");
    fs::write(&source, b"source").unwrap();
    fs::write(&reference, b"reference").unwrap();
    let config = SceneBuildConfig {
        source_scene_path: source,
        object_reference_image_path: reference,
        output_dir: dir.path().join("out"),
        candidate_count: 1,
        quality_profile: SceneQualityProfile::Quality,
        reasoning_model: "gpt-5.5".to_string(),
        image_model: "gpt-image-2".to_string(),
        allow_catalog_reuse: false,
    };
    let provider = RetryImageProvider::new(Vec::new());
    let mut pipeline = ScenePipeline::new(config, provider);

    let preparation = pipeline.prepare_openai_inputs().unwrap();

    assert_eq!(preparation.provider, "openai");
    assert_eq!(preparation.reasoning_model, "gpt-5.5");
    assert_eq!(preparation.image_model, "gpt-image-2");
}

fn low_contrast_candidate_png() -> Vec<u8> {
    let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([222, 222, 222]));
    for y in 50..72 {
        for x in 20..108 {
            image.put_pixel(x, y, image::Rgb([234, 234, 232]));
        }
    }
    png_bytes(image)
}

fn high_contrast_candidate_png() -> Vec<u8> {
    let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
    for y in 32..96 {
        for x in 24..104 {
            image.put_pixel(x, y, image::Rgb([238, 238, 232]));
        }
    }
    png_bytes(image)
}

fn chair_asset() -> SceneAssetBinding {
    SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_group".to_string(),
        label: "chair".to_string(),
        aliases: vec!["conference chair".to_string()],
        path: Some("/tmp/chair.glb".to_string()),
        cache_key: None,
        reusable: true,
        source_image_path: Some("/tmp/chair.png".to_string()),
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        }),
        canonical_frame: None,
        provenance: None,
    }
}

#[test]
fn bsn_parser_accepts_restricted_scene_and_emits_commands() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
spawn chair_right uses chair_asset translation [1.0,0.0,2.0] rotation_y -25.0 scale [1.0,1.0,1.0];
environment rug translation [0.0,0.0,0.0] scale [4.0,1.0,3.0];
camera translation [0.0,4.0,6.0] focus [0.0,0.0,0.0] yaw 0.0 pitch -0.5 radius 6.0;
}
"#;
    let plan = parse_scene_bsn(bsn, &[chair_asset()]).expect("valid bsn");
    assert_eq!(plan.placements.len(), 2);
    assert!(plan.camera.is_some());
    let commands = scene_plan_to_mcp_commands(&plan, &[chair_asset()], true).unwrap();
    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "spawn_path");
    assert_eq!(commands[1]["cache_key"], "chair_asset");
    assert_eq!(commands[1]["local_aabb"]["min"], json!([-0.5, 0.0, -0.5]));
    assert_eq!(commands[1]["local_aabb"]["max"], json!([0.5, 1.0, 0.5]));
}

#[test]
fn bsn_to_mcp_envelope_preserves_commands_and_sequence() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
}
"#;
    let envelope =
        scene_bsn_to_mcp_command_envelope(bsn, &[chair_asset()], true, Some("viewer"), Some(7))
            .expect("valid envelope");
    assert_eq!(envelope["session_id"], json!("viewer"));
    assert_eq!(envelope["sequence"], json!(7));
    let commands = envelope["commands"].as_array().unwrap();
    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "spawn_path");
}

#[test]
fn bsn_to_mcp_envelope_resolves_cache_asset_without_sidecar() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "cache:central-chair-cache-key";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
}
"#;
    let envelope = scene_bsn_to_mcp_command_envelope(bsn, &[], true, Some("viewer"), Some(7))
        .expect("self-contained cache BSN");
    let commands = envelope["commands"].as_array().unwrap();
    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "spawn_cached");
    assert_eq!(commands[1]["cache_key"], "central-chair-cache-key");
}

#[test]
fn bsn_to_mcp_envelope_resolves_path_asset_without_sidecar() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "path:/tmp/chair.glb";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
}
"#;
    let envelope = scene_bsn_to_mcp_command_envelope(bsn, &[], false, None, None)
        .expect("self-contained path BSN");
    let commands = envelope["commands"].as_array().unwrap();
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0]["type"], "spawn_path");
    assert_eq!(commands[0]["path"], "/tmp/chair.glb");
    assert_eq!(commands[0]["cache_key"], "chair_asset");
}

#[test]
fn generated_bsn_asset_requires_sidecar_binding() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
}
"#;
    let err = scene_bsn_to_mcp_command_envelope(bsn, &[], true, None, None).unwrap_err();
    assert!(err.to_string().contains("without a matching asset binding"));
}

#[test]
fn bsn_parser_rejects_proxy_furniture() {
    let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn debug_cube_chair uses chair_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];
}
"#;
    let err = parse_scene_bsn(bsn, &[chair_asset()]).unwrap_err();
    assert!(err.to_string().contains("proxy/debug"));
}

#[test]
fn scene_bsn_prompt_requires_single_line_restricted_grammar() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![],
    };
    let prompt = scene_bsn_prompt(&manifest, &[chair_asset()]);
    assert!(prompt.contains("Every statement must be on exactly one line"));
    assert!(prompt.contains("asset <asset_id> = \"generated:<asset_id>\";"));
    assert!(prompt.contains("spawn <entity_id> uses <asset_id>"));
}

#[test]
fn grounded_scene_layout_uses_asset_aabb_scale_and_bottom_fit() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "curved_sofa_001".to_string(),
                label: "curved sofa".to_string(),
                aliases: vec!["sectional".to_string()],
                bbox: [0.0, 0.2, 1.0, 0.95],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "tan sectional sofa".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            SceneObjectSpec {
                id: "coffee_table_001".to_string(),
                label: "coffee table".to_string(),
                aliases: Vec::new(),
                bbox: [0.4, 0.4, 0.65, 0.65],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "white coffee table scaled below the sofa".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "curved_sofa_001_asset".to_string(),
            object_id: "curved_sofa_001".to_string(),
            label: "curved sofa".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("sofa".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, -0.25, -0.5],
                max: [0.5, 0.75, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "coffee_table_001_asset".to_string(),
            object_id: "coffee_table_001".to_string(),
            label: "coffee table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, -0.1, -0.5],
                max: [0.5, 0.4, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
    ];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");

    let sofa = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "curved_sofa_001")
        .unwrap();
    let table = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "coffee_table_001")
        .unwrap();
    assert!(sofa.scale[0] > table.scale[0]);
    assert_eq!(table.target_footprint_m, [1.8, 0.95]);
    let sofa_bottom = sofa.translation[1] + sofa.local_aabb.min[1] * sofa.scale[1];
    assert!(sofa_bottom.abs() < 1.0e-4);
    assert!(
        layout
            .bsn
            .contains("spawn curved_sofa_001 uses curved_sofa_001_asset")
    );
    parse_scene_bsn(&layout.bsn, &assets).expect("grounded BSN parses");
}

#[test]
fn grounded_scene_layout_uses_explicit_repeated_instances() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.35, 0.35, 0.65, 0.7],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "rectangular table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            SceneObjectSpec {
                id: "black_mesh_chair_group".to_string(),
                label: "black mesh conference chair group".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.1, 0.2, 0.9, 0.85],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("left_back".to_string()),
                        bbox: [0.1, 0.2, 0.22, 0.62],
                        contact: Some([0.16, 0.62]),
                        rotation_hint_degrees: Some(-35.0),
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                    SceneObjectInstanceSpec {
                        id: Some("right_front".to_string()),
                        bbox: [0.72, 0.46, 0.9, 0.85],
                        contact: Some([0.81, 0.85]),
                        rotation_hint_degrees: Some(42.0),
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                ],
                representative_instance_id: None,
                reuse_group: Some("black_mesh_conference_chair".to_string()),
                instance_count: 2,
                object_prompt: "one reusable black mesh conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 0.4, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "black_mesh_chair_group_asset".to_string(),
            object_id: "black_mesh_chair_group".to_string(),
            label: "black mesh conference chair group".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.4, 0.0, -0.4],
                max: [0.4, 1.1, 0.4],
            }),
            canonical_frame: None,
            provenance: None,
        },
    ];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");

    let chairs = layout
        .placements
        .iter()
        .filter(|placement| placement.object_id == "black_mesh_chair_group")
        .collect::<Vec<_>>();
    assert_eq!(chairs.len(), 2);
    assert_eq!(chairs[0].instance_id.as_deref(), Some("left_back"));
    assert_eq!(chairs[0].source_bbox, [0.1, 0.2, 0.22, 0.62]);
    assert_eq!(chairs[0].contact_pixel, [0.16, 0.62]);
    assert_eq!(chairs[0].rotation_y_degrees, -35.0);
    assert_eq!(chairs[1].instance_id.as_deref(), Some("right_front"));
    assert_eq!(chairs[1].source_bbox, [0.72, 0.46, 0.9, 0.85]);
    assert_eq!(chairs[1].contact_pixel, [0.81, 0.85]);
    assert_eq!(chairs[1].rotation_y_degrees, 42.0);
    assert!(
        layout
            .bsn
            .contains("spawn black_mesh_chair_group_left_back")
    );
    assert!(
        layout
            .bsn
            .contains("spawn black_mesh_chair_group_right_front")
    );
}

#[test]
fn grounded_scene_layout_keeps_reused_asset_instances_same_scale() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chair_group".to_string(),
            label: "chair group".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.1, 0.2, 0.9, 0.9],
            instances: vec![
                SceneObjectInstanceSpec {
                    id: Some("near_large".to_string()),
                    bbox: [0.1, 0.55, 0.3, 0.95],
                    contact: Some([0.2, 0.95]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: None,
                    slot_index: None,
                    target_footprint_m: Some([0.9, 0.9]),
                },
                SceneObjectInstanceSpec {
                    id: Some("far_small".to_string()),
                    bbox: [0.6, 0.25, 0.75, 0.55],
                    contact: Some([0.675, 0.55]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: None,
                    slot_index: None,
                    target_footprint_m: Some([0.45, 0.45]),
                },
            ],
            representative_instance_id: Some("near_large".to_string()),
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "one reusable chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let assets = vec![SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_group".to_string(),
        label: "chair".to_string(),
        aliases: Vec::new(),
        path: None,
        cache_key: Some("chair".to_string()),
        reusable: true,
        source_image_path: None,
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        }),
        canonical_frame: None,
        provenance: None,
    }];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");
    let chairs = layout
        .placements
        .iter()
        .filter(|placement| placement.asset_id == "chair_asset")
        .collect::<Vec<_>>();

    assert_eq!(chairs.len(), 2);
    assert!((chairs[0].scale[0] - chairs[1].scale[0]).abs() <= 1.0e-6);
    assert!((chairs[0].translation[1] - chairs[1].translation[1]).abs() <= 1.0e-6);
}

#[test]
fn grounded_scene_layout_uses_calibrated_table_slots_and_source_camera() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/galaxy.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.50, 0.60]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([1.2, 3.4]),
            camera_yaw_degrees: Some(0.0),
            camera_pitch_degrees: Some(-30.0),
            camera_radius_m: Some(5.2),
            vertical_fov_degrees: Some(78.0),
        }),
        objects: vec![
            SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.35, 0.35, 0.65, 0.85],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "large rectangular conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            SceneObjectSpec {
                id: "gray_chair_group".to_string(),
                label: "gray conference chair group".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.12, 0.30, 0.92, 0.95],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("left_01".to_string()),
                        bbox: [0.20, 0.38, 0.30, 0.72],
                        contact: Some([0.25, 0.72]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Left),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("right_01".to_string()),
                        bbox: [0.74, 0.42, 0.86, 0.78],
                        contact: Some([0.80, 0.78]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Right),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("near_01".to_string()),
                        bbox: [0.10, 0.66, 0.26, 0.98],
                        contact: Some([0.18, 0.98]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Near),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("far_01".to_string()),
                        bbox: [0.48, 0.22, 0.57, 0.48],
                        contact: Some([0.52, 0.48]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Far),
                        slot_index: Some(0),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                ],
                representative_instance_id: Some("right_01".to_string()),
                reuse_group: Some("gray_conference_chair".to_string()),
                instance_count: 4,
                object_prompt: "one gray conference chair with mesh back".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.58, 0.62]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.35, 0.0, -1.0],
                max: [0.35, 0.32, 1.0],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([1.2, 3.4]))),
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "gray_chair_group_asset".to_string(),
            object_id: "gray_chair_group".to_string(),
            label: "gray conference chair".to_string(),
            aliases: vec!["chair".to_string()],
            path: None,
            cache_key: Some("chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.3, 0.0, -0.3],
                max: [0.3, 1.0, 0.3],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.58, 0.62]))),
            provenance: None,
        },
    ];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");
    let table = layout
        .placements
        .iter()
        .find(|placement| placement.object_id == "conference_table")
        .unwrap();
    let chairs = layout
        .placements
        .iter()
        .filter(|placement| placement.object_id == "gray_chair_group")
        .collect::<Vec<_>>();
    assert_eq!(chairs.len(), 4);
    assert_eq!(table.ground_point, [0.0, 0.0, 0.0]);
    assert_eq!(table.target_footprint_m, [1.2, 3.4]);
    assert!(table.scale[0] > chairs[0].scale[0]);
    let left = chairs
        .iter()
        .find(|placement| placement.instance_id.as_deref() == Some("left_01"))
        .unwrap();
    let right = chairs
        .iter()
        .find(|placement| placement.instance_id.as_deref() == Some("right_01"))
        .unwrap();
    let near = chairs
        .iter()
        .find(|placement| placement.instance_id.as_deref() == Some("near_01"))
        .unwrap();
    let far = chairs
        .iter()
        .find(|placement| placement.instance_id.as_deref() == Some("far_01"))
        .unwrap();
    assert!(left.ground_point[0] < -0.9);
    assert!(right.ground_point[0] > 0.9);
    assert!(near.ground_point[2] > 1.9);
    assert!(far.ground_point[2] < -1.9);
    assert!((left.rotation_y_degrees - 90.0).abs() < 1.0e-3);
    assert!((right.rotation_y_degrees + 90.0).abs() < 1.0e-3);
    assert!(near.rotation_y_degrees.abs() >= 179.0);
    assert!(far.rotation_y_degrees.abs() < 1.0e-3);
    assert!(layout.camera.translation[1] > 2.0);
    assert_eq!(layout.camera.pitch, Some(30.0));
    assert_eq!(layout.camera.vertical_fov_degrees, Some(78.0));
    assert!(layout.bsn.contains("vertical_fov 78.0"));
    parse_scene_bsn(&layout.bsn, &assets).expect("calibrated BSN parses");

    let mut evidence = manifest_grounding_evidence(&manifest);
    evidence.depth = Some(DepthEvidenceRef {
        provider: "synthetic_depth".to_string(),
        model: None,
        precision: None,
        artifact_path: None,
        focal_length_px: Some(900.0),
        vertical_fov_degrees: Some(62.0),
        image_size: Some([1600, 900]),
        depth_map_size: Some([1600, 900]),
        floor_sample_count: Some(128),
    });
    evidence.camera.image_size = Some([1600, 900]);
    evidence.camera.focal_length_px = Some(900.0);
    evidence.camera.vertical_fov_degrees = Some(62.0);
    for object in &mut evidence.objects {
        object.depth_stats = Some(ObjectDepthStats {
            median_m: 4.0,
            min_m: 3.8,
            max_m: 4.2,
            contact_m: Some(4.0),
            sample_count: Some(16),
        });
    }
    let depth_layout = grounded_scene_layout_with_evidence(&manifest, &assets, &evidence)
        .expect("depth grounded layout");
    assert_eq!(
        depth_layout
            .projection_fit
            .as_ref()
            .expect("projection fit report")
            .camera
            .basis,
        "source-depth-intrinsics"
    );
}

#[test]
fn grounded_scene_layout_defaults_to_asset_preserving_table_scale() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/cslewis.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.49, 0.64]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([4.2, 1.4]),
            camera_yaw_degrees: Some(180.0),
            camera_pitch_degrees: Some(34.0),
            camera_radius_m: Some(5.4),
            vertical_fov_degrees: Some(82.0),
        }),
        objects: vec![SceneObjectSpec {
            id: "conference_table".to_string(),
            label: "white rectangular conference table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.30, 0.47, 0.65, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "long white conference table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: Some(0.0),
            target_footprint_m: Some([4.2, 1.4]),
        }],
    };
    let assets = vec![SceneAssetBinding {
        asset_id: "conference_table_asset".to_string(),
        object_id: "conference_table".to_string(),
        label: "white rectangular conference table".to_string(),
        aliases: vec!["table".to_string()],
        path: None,
        cache_key: Some("table".to_string()),
        reusable: true,
        source_image_path: None,
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.50, -0.15, -0.28],
            max: [0.50, 0.15, 0.28],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(90.0, Some([4.2, 1.4]))),
        provenance: None,
    }];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");
    let table = &layout.placements[0];

    assert_eq!(table.rotation_y_degrees, -90.0);
    assert!((table.scale[0] - table.scale[1]).abs() < 1.0e-5);
    assert!((table.scale[1] - table.scale[2]).abs() < 1.0e-5);
    assert!((table.scale[0] - 4.2).abs() < 0.05);
    let bottom_y = table.translation[1] + table.local_aabb.min[1] * table.scale[1];
    assert!(bottom_y.abs() < 1.0e-4);
}

#[test]
fn grounded_scene_layout_supports_explicit_free_anisotropic_table_scale() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/cslewis.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.49, 0.64]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([4.2, 1.4]),
            camera_yaw_degrees: Some(180.0),
            camera_pitch_degrees: Some(34.0),
            camera_radius_m: Some(5.4),
            vertical_fov_degrees: Some(82.0),
        }),
        objects: vec![SceneObjectSpec {
            id: "conference_table".to_string(),
            label: "white rectangular conference table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.30, 0.47, 0.65, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "long white conference table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: Some(0.0),
            target_footprint_m: Some([4.2, 1.4]),
        }],
    };
    let assets = vec![SceneAssetBinding {
        asset_id: "conference_table_asset".to_string(),
        object_id: "conference_table".to_string(),
        label: "white rectangular conference table".to_string(),
        aliases: vec!["table".to_string()],
        path: None,
        cache_key: Some("table".to_string()),
        reusable: true,
        source_image_path: None,
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.50, -0.15, -0.28],
            max: [0.50, 0.15, 0.28],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(90.0, Some([4.2, 1.4]))),
        provenance: None,
    }];
    let config = GroundedSceneLayoutConfig {
        scale_policy: SceneScalePolicy::FreeAnisotropic,
        ..GroundedSceneLayoutConfig::default()
    };

    let layout = grounded_scene_layout(&manifest, &assets, config).expect("grounded layout");
    let table = &layout.placements[0];

    assert_eq!(table.rotation_y_degrees, -90.0);
    assert!(table.scale[2] > table.scale[0] * 4.0);
    assert!((table.scale[0] - 1.4).abs() < 0.05);
    assert!((table.scale[2] - 7.5).abs() < 0.10);
}

#[test]
fn metric_frame_maps_source_sides_through_camera_yaw() {
    let frame = MetricSceneFrame {
        table_axis_degrees: 0.0,
        table_size_m: [1.2, 3.4],
        seating_clearance_m: 0.18,
        camera_yaw_degrees: Some(180.0),
        camera_pitch_degrees: Some(24.0),
        camera_radius_m: Some(4.2),
        vertical_fov_degrees: Some(74.0),
    };

    let left = frame.side_point(SceneInstanceSide::Left, 0, 1, [0.58, 0.62]);
    let right = frame.side_point(SceneInstanceSide::Right, 0, 1, [0.58, 0.62]);
    let near = frame.side_point(SceneInstanceSide::Near, 0, 1, [0.58, 0.62]);
    let far = frame.side_point(SceneInstanceSide::Far, 0, 1, [0.58, 0.62]);

    assert!(left[0] > 0.9);
    assert!(right[0] < -0.9);
    assert!(near[2] < -1.9);
    assert!(far[2] > 1.9);
}

#[test]
fn bsn_yaw_convention_faces_plus_z_at_zero_degrees() {
    let from = [0.0, 0.0, 0.0];

    assert!((bsn_yaw_toward_point_degrees(from, [0.0, 0.0, 1.0]).unwrap() - 0.0).abs() < 1.0e-6);
    assert!((bsn_yaw_toward_point_degrees(from, [1.0, 0.0, 0.0]).unwrap() - 90.0).abs() < 1.0e-6);
    assert!((bsn_yaw_toward_point_degrees(from, [-1.0, 0.0, 0.0]).unwrap() + 90.0).abs() < 1.0e-6);
    assert!(
        bsn_yaw_toward_point_degrees(from, [0.0, 0.0, -1.0])
            .unwrap()
            .abs()
            >= 179.999
    );
    assert!(bsn_yaw_toward_point_degrees(from, from).is_none());
}

#[test]
fn representative_crop_bbox_prefers_requested_single_instance() {
    let object = SceneObjectSpec {
        id: "chair_group".to_string(),
        label: "chair group".to_string(),
        aliases: vec!["chair".to_string()],
        bbox: [0.1, 0.2, 0.9, 0.9],
        instances: vec![
            SceneObjectInstanceSpec {
                id: Some("left".to_string()),
                bbox: [0.1, 0.3, 0.25, 0.75],
                contact: Some([0.18, 0.75]),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Left),
                slot_index: Some(0),
                target_footprint_m: None,
            },
            SceneObjectInstanceSpec {
                id: Some("right".to_string()),
                bbox: [0.72, 0.42, 0.9, 0.86],
                contact: Some([0.81, 0.86]),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Right),
                slot_index: Some(0),
                target_footprint_m: None,
            },
        ],
        representative_instance_id: Some("right".to_string()),
        reuse_group: Some("chair".to_string()),
        instance_count: 2,
        object_prompt: "one chair".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };

    assert_eq!(representative_crop_bbox(&object), [0.72, 0.42, 0.9, 0.86]);
}

#[test]
fn grounded_scene_layout_re_ranks_global_slots_per_table_side() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: Some(SceneCalibration {
            table_center: Some([0.5, 0.6]),
            table_axis_degrees: Some(0.0),
            table_size_m: Some([3.2, 1.25]),
            camera_yaw_degrees: Some(180.0),
            camera_pitch_degrees: Some(24.0),
            camera_radius_m: Some(4.2),
            vertical_fov_degrees: Some(74.0),
        }),
        objects: vec![
            SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.3, 0.4, 0.7, 0.9],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.2, 1.25]),
            },
            SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "chair group".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.2, 0.2, 0.8, 0.8],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("far_left_global_02".to_string()),
                        bbox: [0.38, 0.45, 0.46, 0.7],
                        contact: Some([0.42, 0.7]),
                        rotation_hint_degrees: Some(135.0),
                        facing_yaw_degrees: Some(135.0),
                        side: Some(SceneInstanceSide::Far),
                        slot_index: Some(2),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("far_right_global_04".to_string()),
                        bbox: [0.58, 0.45, 0.66, 0.7],
                        contact: Some([0.62, 0.7]),
                        rotation_hint_degrees: Some(-135.0),
                        facing_yaw_degrees: Some(-135.0),
                        side: Some(SceneInstanceSide::Far),
                        slot_index: Some(4),
                        target_footprint_m: Some([0.58, 0.62]),
                    },
                ],
                representative_instance_id: Some("far_right_global_04".to_string()),
                reuse_group: Some("chair".to_string()),
                instance_count: 2,
                object_prompt: "one conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.58, 0.62]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "table_asset".to_string(),
            object_id: "table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.2],
                max: [0.5, 0.2, 0.2],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(90.0, Some([3.2, 1.25]))),
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "chair_asset".to_string(),
            object_id: "chair_group".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.35, 0.0, -0.35],
                max: [0.35, 1.0, 0.35],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.58, 0.62]))),
            provenance: None,
        },
    ];

    let layout = grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
        .expect("grounded layout");
    let chairs = layout
        .placements
        .iter()
        .filter(|placement| placement.object_id == "chair_group")
        .collect::<Vec<_>>();

    assert_eq!(chairs.len(), 2);
    assert!(
        (chairs[0].ground_point[0] - chairs[1].ground_point[0]).abs() > 0.5,
        "global slot indices should be converted to unique side-local positions: {chairs:?}"
    );
    assert!((chairs[0].ground_point[2] - chairs[1].ground_point[2]).abs() < 1.0e-4);
}

#[test]
fn sofa_shape_score_rejects_source_aspect_drift() {
    let object = SceneObjectSpec {
        id: "straight_sofa_001".to_string(),
        label: "straight sofa".to_string(),
        aliases: Vec::new(),
        bbox: [0.0, 0.155, 1.0, 1.0],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "tan sectional sofa".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.36,
        alpha_bbox: Some([65, 187, 971, 897]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 2.0);

    assert_eq!(score, 0.0);
}

#[test]
fn sofa_shape_score_rejects_curry_wraparound_candidate() {
    let object = SceneObjectSpec {
        id: "tan_open_sectional_sofa_01".to_string(),
        label: "tan open sectional sofa".to_string(),
        aliases: Vec::new(),
        bbox: [0.0, 0.105, 1.0, 1.0],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "wide low tan open sectional with gentle right bend".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.3846,
        alpha_bbox: Some([27, 254, 1005, 923]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 1.78087);

    assert_eq!(score, 0.0);
}

#[test]
fn sofa_shape_score_flags_generated_product_sofa_that_loses_source_crop_edge() {
    let object = SceneObjectSpec {
        id: "tan_open_sectional_sofa".to_string(),
        label: "tan open crescent sectional sofa".to_string(),
        aliases: Vec::new(),
        bbox: [0.13, 0.12, 0.871, 1.0],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "large low tan crescent sectional with source-cropped foreground extent"
            .to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.4185,
        alpha_bbox: Some([43, 60, 969, 975]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 1.78087);

    assert!(
        score > 0.48,
        "open sofa crop-edge mismatch should be reviewable evidence, not an automatic reconstruction blocker"
    );
    assert!(generated_source_crop_edge_mismatch(&object, &matte));
}

#[test]
fn scene_candidate_selection_uses_object_specific_anchor_thresholds() {
    let object = SceneObjectSpec {
        id: "tan_open_sectional_sofa".to_string(),
        label: "tan open crescent sectional sofa".to_string(),
        aliases: Vec::new(),
        bbox: [0.13, 0.12, 0.871, 1.0],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "large low tan crescent sectional with source-cropped foreground extent"
            .to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/curry.jpg".to_string(),
        scene_calibration: None,
        objects: vec![object.clone()],
    };
    let candidates = vec![ObjectImageCandidate {
        object_id: object.id.clone(),
        candidate_index: 0,
        image_path: "/tmp/sofa.png".to_string(),
        raw_image_path: None,
        prompt_hash: "test".to_string(),
        score: DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE + 0.01,
        provider_request_id: None,
    }];

    let min_score =
        object_reconstruction_min_score(&object, DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE);
    assert!(min_score > DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE);
    let err = select_object_image_candidates(
        &manifest,
        &candidates,
        DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE,
    )
    .unwrap_err();

    assert!(err.to_string().contains("min=0.480"));
}

#[test]
fn crescent_sofa_shape_score_keeps_source_like_curved_candidate_selectable() {
    let object = SceneObjectSpec {
        id: "tan_open_crescent_sectional_sofa".to_string(),
        label: "tan open crescent sectional sofa".to_string(),
        aliases: vec!["semicircular banquette sofa".to_string()],
        bbox: [0.0, 0.105, 1.0, 1.0],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "large low tan crescent banquette sectional".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.4368,
        alpha_bbox: Some([37, 90, 997, 990]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 1.78087);

    assert!(score >= 0.48, "score {score}");
}

#[test]
fn crescent_sofa_shape_score_accepts_square_perspective_candidate() {
    let object = SceneObjectSpec {
        id: "tan_open_crescent_sectional_sofa".to_string(),
        label: "tan open crescent sectional sofa".to_string(),
        aliases: vec!["semicircular banquette sofa".to_string()],
        bbox: [0.08, 0.38, 0.92, 0.94],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "large low tan crescent banquette sectional".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.4696,
        alpha_bbox: Some([54, 44, 985, 984]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 2.0);

    assert!(score >= 0.20, "score {score}");
}

#[test]
fn sofa_shape_score_keeps_wide_source_aligned_sectional_selectable() {
    let object = SceneObjectSpec {
        id: "tan_open_sectional_sofa_001".to_string(),
        label: "tan open sectional sofa".to_string(),
        aliases: Vec::new(),
        bbox: [0.0, 0.155, 1.0, 0.95],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "wide low tan open sectional with gentle right bend".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let matte = ObjectImageMatteStats {
        alpha_coverage: 0.29,
        alpha_bbox: Some([16, 270, 1005, 760]),
        image_size: [1024, 1024],
    };

    let score = generated_shape_consistency_score(&object, &matte, 1.779);

    assert!(score >= 0.82, "score {score}");
}

#[test]
fn image_retry_classification_skips_client_status_errors() {
    assert!(image_error_is_retryable(&SceneError::Http(
        "decode image response body: unexpected eof".to_string()
    )));
    assert!(!image_error_is_retryable(&SceneError::Http(
        "status 400 Bad Request: invalid image".to_string()
    )));
    assert!(!image_error_is_retryable(&SceneError::Provider(
        "image response missing data array".to_string()
    )));
}

#[test]
fn object_image_prompt_includes_style_reference_and_exclusion_rules() {
    let object = SceneObjectSpec {
        id: "sofa_curved".to_string(),
        label: "curved sofa".to_string(),
        aliases: vec![],
        bbox: [0.2, 0.3, 0.9, 0.95],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "A tan curved upholstered sectional sofa.".to_string(),
        camera_hint: Some("high oblique".to_string()),
        rotation_hint_degrees: Some(35.0),
        target_footprint_m: None,
    };
    let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
    assert!(prompt.contains("docs/input_chair.jpg"));
    assert!(prompt.contains("Source-preserving edit requirement"));
    assert!(prompt.contains("image 1 is the source object crop"));
    assert!(prompt.contains("whole-scene context only"));
    assert!(prompt.contains("style reference only"));
    assert!(prompt.contains("clean isolated"));
    assert!(prompt.contains("Do not include the room"));
    assert!(prompt.contains("preserve that partial visible source shape"));
    assert!(prompt.contains("curved sofa"));
    assert!(prompt.contains("solid matte cobalt-blue background"));
    assert!(prompt.contains("curved crescent"));
    assert!(prompt.contains("conventional straight L-sectional"));
    assert!(prompt.contains("Target yaw/rotation hint: 35.0 degrees"));
}

#[test]
fn object_image_prompt_preserves_source_crop_edges_for_curry_sectional() {
    let object = SceneObjectSpec {
        id: "tan_open_crescent_sectional_sofa".to_string(),
        label: "tan open crescent sectional sofa".to_string(),
        aliases: vec!["semicircular sectional sofa".to_string()],
        bbox: [0.0, 0.13, 1.0, 0.98],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "large low tan crescent banquette sofa with source-cropped foreground"
            .to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };

    let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);

    assert!(prompt.contains("touches the source image left, right, bottom edge(s)"));
    assert!(prompt.contains("must continue to the same left, right, bottom edge(s)"));
    assert!(prompt.contains("no blue/background margin"));
    assert!(prompt.contains("Do not center the sofa with padding"));
    assert!(prompt.contains("do not complete hidden left/right/bottom ends"));
    assert!(prompt.contains("finished showroom product sofa"));
}

#[test]
fn object_image_prompt_protects_thin_white_table_geometry() {
    let object = SceneObjectSpec {
        id: "table".to_string(),
        label: "white coffee table".to_string(),
        aliases: vec!["low white table".to_string()],
        bbox: [0.3, 0.3, 0.7, 0.6],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: None,
        instance_count: 1,
        object_prompt: "A glossy white rectangular coffee table with slim white metal frame."
            .to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };
    let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
    assert!(prompt.contains("solid matte cobalt-blue background"));
    assert!(prompt.contains("Do not omit thin legs"));
    assert!(prompt.contains("Do not merge the tabletop into the background"));
    assert!(prompt.contains("no floor plane"));
}

#[test]
fn object_image_prompt_prioritizes_chair_geometry_over_table_context() {
    let object = SceneObjectSpec {
        id: "black_mesh_conference_chair_group".to_string(),
        label: "black mesh conference chair".to_string(),
        aliases: vec!["chair".to_string()],
        bbox: [0.1, 0.2, 0.3, 0.7],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: Some("black_mesh_conference_chair".to_string()),
        instance_count: 6,
        object_prompt:
            "one reusable chair observed around the conference table; scale smaller than tabletop"
                .to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    };

    let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);

    assert!(prompt.contains("preserve one complete source-observed chair"));
    assert!(prompt.contains("keep four separate thin legs or sled/loop frame when visible"));
    assert!(prompt.contains("do not invent a central pedestal"));
    assert!(prompt.contains("Do not generate multiple chairs"));
    assert!(!prompt.contains("preserve a flat rectangular tabletop"));
}

#[test]
fn generated_image_suitability_penalizes_low_contrast_background() {
    let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([222, 222, 222]));
    for y in 50..72 {
        for x in 20..108 {
            image.put_pixel(x, y, image::Rgb([234, 234, 232]));
        }
    }
    for x in [24, 104] {
        for y in 72..105 {
            image.put_pixel(x, y, image::Rgb([236, 236, 234]));
        }
    }
    let score = score_generated_object_rgb(&image);
    assert!(
        score.score < 0.35,
        "low-contrast white-on-gray image should be a poor TRELLIS/RMBG candidate: {score:?}"
    );
}

#[test]
fn generated_image_suitability_accepts_high_contrast_object() {
    let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
    for y in 32..96 {
        for x in 24..104 {
            image.put_pixel(x, y, image::Rgb([238, 238, 232]));
        }
    }
    let score = score_generated_object_rgb(&image);
    assert!(
        score.score > 0.90,
        "high-contrast object/matte image should rank well: {score:?}"
    );
}

#[test]
fn object_image_generation_policy_retries_until_candidate_passes_guardrail() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("scene.png");
    image::RgbImage::from_pixel(512, 256, image::Rgb([64, 64, 64]))
        .save(&source_path)
        .unwrap();
    let provider = RetryImageProvider::new(vec![
        low_contrast_candidate_png(),
        high_contrast_candidate_png(),
    ]);
    let config = SceneBuildConfig {
        source_scene_path: source_path.clone(),
        object_reference_image_path: source_path.clone(),
        output_dir: dir.path().join("run"),
        candidate_count: 1,
        quality_profile: SceneQualityProfile::Draft,
        reasoning_model: "test-reasoning".to_string(),
        image_model: "test-image".to_string(),
        allow_catalog_reuse: false,
    };
    let pipeline = ScenePipeline::new(config, provider);
    let request = ObjectImageRequest {
        object: SceneObjectSpec {
            id: "green_chair".to_string(),
            label: "green chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "dark green padded chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        },
        source_scene_path: source_path.display().to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: source_path.display().to_string(),
        prompt: "generate chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };

    let report = pipeline
        .generate_object_candidates_with_policy(
            &[request],
            ObjectImageGenerationPolicy {
                min_score: 0.80,
                max_attempts_per_object: 2,
                candidates_per_attempt: 1,
            },
        )
        .unwrap();

    assert_eq!(report.attempts.len(), 2);
    assert!(!report.attempts[0].accepted);
    assert!(report.attempts[1].accepted);
    assert_eq!(report.candidates.len(), 2);
    assert_eq!(report.selected_candidates.len(), 1);
    assert!(report.rejected_objects.is_empty());
    assert_eq!(report.selected_candidates[0].candidate_index, 1);
    assert!(
        dir.path()
            .join("run/objects/generated/green_chair_candidate_0.png")
            .exists()
    );
    assert!(
        dir.path()
            .join("run/objects/generated/green_chair_candidate_1.png")
            .exists()
    );
}

#[test]
fn object_image_generation_policy_stops_after_required_object_rejection() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("scene.png");
    image::RgbImage::from_pixel(512, 256, image::Rgb([64, 64, 64]))
        .save(&source_path)
        .unwrap();
    let provider = RetryImageProvider::new(vec![low_contrast_candidate_png()]);
    let config = SceneBuildConfig {
        source_scene_path: source_path.clone(),
        object_reference_image_path: source_path.clone(),
        output_dir: dir.path().join("run"),
        candidate_count: 1,
        quality_profile: SceneQualityProfile::Draft,
        reasoning_model: "test-reasoning".to_string(),
        image_model: "test-image".to_string(),
        allow_catalog_reuse: false,
    };
    let pipeline = ScenePipeline::new(config, provider);
    let request = |id: &str| ObjectImageRequest {
        object: SceneObjectSpec {
            id: id.to_string(),
            label: id.replace('_', " "),
            aliases: Vec::new(),
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "test object".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        },
        source_scene_path: source_path.display().to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: source_path.display().to_string(),
        prompt: "generate object".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };

    let report = pipeline
        .generate_object_candidates_with_policy(
            &[
                request("bad_first_object"),
                request("would_exhaust_provider"),
            ],
            ObjectImageGenerationPolicy {
                min_score: 0.80,
                max_attempts_per_object: 1,
                candidates_per_attempt: 1,
            },
        )
        .unwrap();

    assert_eq!(report.attempts.len(), 1);
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.rejected_objects.len(), 1);
    assert_eq!(report.rejected_objects[0].object_id, "bad_first_object");
    assert!(report.selected_candidates.is_empty());
}

#[test]
fn object_image_generation_policy_parallelizes_independent_requests() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("scene.png");
    image::RgbImage::from_pixel(512, 256, image::Rgb([64, 64, 64]))
        .save(&source_path)
        .unwrap();
    let provider = ParallelImageProvider::new(high_contrast_candidate_png());
    let config = SceneBuildConfig {
        source_scene_path: source_path.clone(),
        object_reference_image_path: source_path.clone(),
        output_dir: dir.path().join("run"),
        candidate_count: 1,
        quality_profile: SceneQualityProfile::Draft,
        reasoning_model: "test-reasoning".to_string(),
        image_model: "test-image".to_string(),
        allow_catalog_reuse: false,
    };
    let pipeline = ScenePipeline::new(config, provider.clone());
    let request = |id: &str| ObjectImageRequest {
        object: SceneObjectSpec {
            id: id.to_string(),
            label: id.replace('_', " "),
            aliases: Vec::new(),
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "test object".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        },
        source_scene_path: source_path.display().to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: source_path.display().to_string(),
        prompt: "generate object".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };

    let report = pipeline
        .generate_object_candidates_with_policy_parallel(
            &[request("first"), request("second"), request("third")],
            ObjectImageGenerationPolicy {
                min_score: 0.10,
                max_attempts_per_object: 1,
                candidates_per_attempt: 1,
            },
            3,
        )
        .unwrap();

    assert!(
        provider.max_active() > 1,
        "independent object image requests should overlap"
    );
    assert_eq!(report.attempts.len(), 3);
    assert_eq!(report.candidates.len(), 3);
    assert_eq!(report.selected_candidates.len(), 3);
    assert_eq!(report.attempts[0].object_id, "first");
    assert_eq!(report.attempts[1].object_id, "second");
    assert_eq!(report.attempts[2].object_id, "third");
    assert!(report.rejected_objects.is_empty());

    let metrics = fs::read_to_string(dir.path().join("run/metrics.jsonl")).unwrap();
    for line in metrics.lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("metric line should be valid json");
    }
}

#[test]
fn generated_candidate_matte_writes_transparent_background() {
    let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
    for y in 32..96 {
        for x in 24..104 {
            image.put_pixel(x, y, image::Rgb([238, 238, 232]));
        }
    }
    let suitability = score_generated_object_rgb(&image);
    let (matted, stats) = matte_generated_object_rgb(&image, suitability);
    assert_eq!(matted.get_pixel(0, 0).0[3], 0);
    assert_eq!(matted.get_pixel(64, 64).0[3], 255);
    assert!(
        (0.20..0.50).contains(&stats.alpha_coverage),
        "matte alpha should cover the object, not the whole frame: {stats:?}"
    );
    assert_eq!(stats.alpha_bbox, Some([24, 32, 104, 96]));
}

#[test]
fn schemas_are_strict_objects() {
    assert_eq!(
        object_manifest_schema()["additionalProperties"],
        json!(false)
    );
    assert_eq!(scene_bsn_schema()["additionalProperties"], json!(false));
    assert_eq!(
        scene_quality_rubric_schema()["additionalProperties"],
        json!(false)
    );
}

#[test]
fn image_data_url_uses_source_pixels_and_mime_type() {
    let dir = tempfile::tempdir().unwrap();
    let image_path = dir.path().join("input.jpg");
    fs::write(&image_path, [1u8, 2, 3, 4]).unwrap();
    let data_url = image_data_url(&image_path).unwrap();
    assert!(data_url.starts_with("data:image/jpeg;base64,"));
    assert!(data_url.ends_with("AQIDBA=="));
}

#[test]
fn resize_image_for_api_bounds_large_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let input_path = dir.path().join("large.png");
    let output_path = dir.path().join("large_1024.jpg");
    let image = image::RgbImage::from_pixel(2048, 1024, image::Rgb([128, 64, 32]));
    image.save(&input_path).unwrap();
    resize_image_for_api(&input_path, &output_path).unwrap();
    let resized = image::open(&output_path).unwrap();
    assert_eq!(resized.width(), 1024);
    assert_eq!(resized.height(), 512);
}
