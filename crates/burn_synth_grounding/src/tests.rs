use super::*;
use crate::depth::*;
use crate::locate::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn_depth::CameraIntrinsics;
use burn_synth_scene::{
    Detection, EstimatedCamera, EstimatedFloorPlane, ObjectDepthStats, ObjectGroundingEvidence,
    SceneGroundingEvidence, SceneInstanceSide, SceneObjectInstanceSpec, SceneObjectManifest,
    SceneObjectSpec,
};
use image::{Rgba, RgbaImage};
use serde_json::json;

#[test]
fn segmentation_grounding_attaches_bbox_masks_and_artifacts() {
    let run_id = format!(
        "burn_synth_grounding_segmentation_test_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    );
    let root = std::env::temp_dir().join(run_id);
    fs::create_dir_all(&root).unwrap();
    let image_path = root.join("source.png");
    RgbaImage::from_pixel(20, 10, Rgba([24, 48, 72, 255]))
        .save(&image_path)
        .unwrap();
    let detection = Detection {
        label: "chair".to_string(),
        bbox: [0.10, 0.20, 0.40, 0.80],
        point: Some([0.25, 0.80]),
        confidence: Some(0.9),
        source_query: "chair".to_string(),
    };
    let mut evidence = SceneGroundingEvidence {
        source_image_path: image_path.display().to_string(),
        depth: None,
        segmentation: None,
        detections: vec![detection.clone()],
        camera: EstimatedCamera::default(),
        floor: EstimatedFloorPlane::default(),
        objects: vec![ObjectGroundingEvidence {
            object_id: "chair".to_string(),
            instance_id: Some("chair_01".to_string()),
            reuse_group: Some("chair".to_string()),
            detection: Some(detection),
            mask: None,
            asset_id: None,
            contact_pixel: Some([0.25, 0.80]),
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: None,
            provenance: Vec::new(),
        }],
    };
    let mut runtime = SceneGroundingRuntime::default();
    let report = runtime
        .segmentation_grounding_evidence(
            &mut evidence,
            &image_path,
            &root,
            SegmentationGroundingConfig::default(),
        )
        .unwrap();

    assert_eq!(report.mask_count, 1);
    assert!(report.masks_path.exists());
    assert!(report.overlay_path.exists());
    assert_eq!(
        evidence
            .segmentation
            .as_ref()
            .and_then(|segmentation| segmentation.mask_count),
        Some(1)
    );
    let object_mask = evidence.objects[0].mask.as_ref().unwrap();
    assert_eq!(object_mask.image_size, [20, 10]);
    assert_eq!(object_mask.area_px, 6 * 6);
    assert!(
        object_mask
            .mask_png_path
            .as_ref()
            .is_some_and(|path| Path::new(path).exists())
    );

    fs::remove_dir_all(root).ok();
}

#[test]
fn depth_map_sidecar_persists_raw_f32_metric_depth() {
    let run_id = format!(
        "burn_synth_grounding_depth_sidecar_test_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    );
    let root = std::env::temp_dir().join(run_id);
    fs::create_dir_all(&root).unwrap();
    let depth_map = SceneDepthMapEvidence {
        depth_m: vec![1.0, 1.5, 2.0, f32::NAN],
        width: 2,
        height: 2,
        intrinsics: CameraIntrinsics {
            fx: 4.0,
            fy: 5.0,
            cx: 0.5,
            cy: 0.5,
            width: 2,
            height: 2,
        },
        focal_length_px: Some(4.0),
        vertical_fov_degrees: Some(53.0),
    };

    let artifacts = write_depth_map_sidecar(&root, &depth_map).expect("write sidecar");
    assert!(artifacts.raw_path.exists());
    assert!(artifacts.metadata_path.exists());
    assert_eq!(artifacts.metadata["encoding"], json!("f32le"));
    assert_eq!(artifacts.metadata["width"], json!(2));
    assert_eq!(artifacts.metadata["height"], json!(2));
    assert_eq!(artifacts.metadata["finite_positive_count"], json!(3));
    let bytes = fs::read(&artifacts.raw_path).expect("read raw sidecar");
    assert_eq!(bytes.len(), 4 * std::mem::size_of::<f32>());
    let decoded = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    assert_eq!(decoded[0], 1.0);
    assert_eq!(decoded[1], 1.5);
    assert_eq!(decoded[2], 2.0);
    assert!(decoded[3].is_nan());

    fs::remove_dir_all(root).ok();
}

#[test]
fn locate_anything_cache_key_ignores_non_execution_flags() {
    let base = LocateAnythingRuntimeConfig {
        model_root: PathBuf::from("assets/models/LocateAnything-3B"),
        backend: LocateAnythingRuntimeBackend::BurnNative,
        allow_experimental_native_detect: true,
        decode_mode: DecodeMode::Hybrid,
        max_new_tokens: 1024,
        in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
        ..LocateAnythingRuntimeConfig::default()
    };
    let mut same_runtime = base.clone();
    same_runtime.require_gpu = false;
    assert_eq!(
        LocateAnythingBurnNativeCacheKey::from_config(&base),
        LocateAnythingBurnNativeCacheKey::from_config(&same_runtime)
    );

    let mut different_tokens = base;
    different_tokens.in_token_limit += 1;
    assert_ne!(
        LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
        LocateAnythingBurnNativeCacheKey::from_config(&different_tokens)
    );

    let mut different_decode_filter = same_runtime.clone();
    different_decode_filter.top_p = None;
    assert_ne!(
        LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
        LocateAnythingBurnNativeCacheKey::from_config(&different_decode_filter)
    );
}

#[test]
fn depth_pro_cache_key_ignores_non_execution_policy_flags() {
    let base = DepthProGroundingConfig {
        cache_dir: Some(PathBuf::from("/tmp/depth-cache")),
        precision: GroundingDepthPrecision::F16,
        allow_download: true,
        require_gpu: true,
    };
    let mut same = base.clone();
    assert_eq!(
        DepthProRuntimeCacheKey::from_config(&base),
        DepthProRuntimeCacheKey::from_config(&same)
    );

    same.precision = GroundingDepthPrecision::F32;
    assert_ne!(
        DepthProRuntimeCacheKey::from_config(&base),
        DepthProRuntimeCacheKey::from_config(&same)
    );

    same = base.clone();
    same.allow_download = false;
    assert_eq!(
        DepthProRuntimeCacheKey::from_config(&base),
        DepthProRuntimeCacheKey::from_config(&same)
    );

    same = base.clone();
    same.require_gpu = false;
    assert_eq!(
        DepthProRuntimeCacheKey::from_config(&base),
        DepthProRuntimeCacheKey::from_config(&same)
    );

    same = base.clone();
    same.cache_dir = Some(PathBuf::from("/tmp/other-depth-cache"));
    assert_ne!(
        DepthProRuntimeCacheKey::from_config(&base),
        DepthProRuntimeCacheKey::from_config(&same)
    );
}

#[test]
fn locate_anything_evidence_maps_detections_to_instances() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chairs".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.10, 0.40, 0.80, 0.90],
            instances: vec![
                SceneObjectInstanceSpec {
                    id: Some("chair_left".to_string()),
                    bbox: [0.10, 0.40, 0.30, 0.90],
                    contact: None,
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Left),
                    slot_index: None,
                    target_footprint_m: None,
                },
                SceneObjectInstanceSpec {
                    id: Some("chair_right".to_string()),
                    bbox: [0.60, 0.40, 0.80, 0.90],
                    contact: None,
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: None,
                    target_footprint_m: None,
                },
            ],
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let detections = vec![
        Detection {
            label: "chair".to_string(),
            bbox: [0.61, 0.41, 0.79, 0.91],
            point: None,
            confidence: Some(0.8),
            source_query: "chair".to_string(),
        },
        Detection {
            label: "chair".to_string(),
            bbox: [0.11, 0.39, 0.29, 0.89],
            point: None,
            confidence: Some(0.9),
            source_query: "chair".to_string(),
        },
    ];
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        Path::new("/tmp/source.jpg"),
        detections,
        "locate_anything_test",
    )
    .unwrap();
    assert_eq!(evidence.detections.len(), 2);
    let left = evidence
        .objects
        .iter()
        .find(|object| object.instance_id.as_deref() == Some("chair_left"))
        .unwrap();
    let right = evidence
        .objects
        .iter()
        .find(|object| object.instance_id.as_deref() == Some("chair_right"))
        .unwrap();
    assert_eq!(
        left.detection.as_ref().unwrap().bbox,
        [0.11, 0.39, 0.29, 0.89]
    );
    assert_eq!(
        right.detection.as_ref().unwrap().bbox,
        [0.61, 0.41, 0.79, 0.91]
    );
    let object_union = evidence
        .objects
        .iter()
        .find(|object| object.object_id == "chairs" && object.instance_id.is_none())
        .unwrap();
    assert_eq!(
        object_union.detection.as_ref().unwrap().bbox,
        [0.11, 0.39, 0.79, 0.91]
    );
}

#[test]
fn locate_anything_queries_use_per_category_prompts() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.30, 0.40, 0.70, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("table".to_string()),
                instance_count: 1,
                object_prompt: "table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            SceneObjectSpec {
                id: "chair".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.10, 0.40, 0.80, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("chair".to_string()),
                instance_count: 1,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
        ],
    };
    assert_eq!(
        locate_anything_queries(&manifest),
        vec!["table".to_string(), "chair".to_string()]
    );
}

#[test]
fn locate_anything_allowed_categories_drop_bad_labels_and_dedupe_aliases() {
    let queries = locate_anything_queries_for_allowed_categories(
        &["table", "chair", "plant", "sofa", "couch"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        queries,
        vec![
            "table".to_string(),
            "chair".to_string(),
            "plant".to_string(),
            "sofa".to_string(),
            "couch".to_string()
        ]
    );

    let filter = filter_locate_anything_detections(
        vec![
            Detection {
                label: "ceiling light".to_string(),
                bbox: [0.10, 0.10, 0.20, 0.20],
                point: None,
                confidence: Some(0.9),
                source_query: "chair".to_string(),
            },
            Detection {
                label: "chair".to_string(),
                bbox: [0.30, 0.40, 0.45, 0.90],
                point: None,
                confidence: Some(0.8),
                source_query: "chair".to_string(),
            },
            Detection {
                label: "sofa".to_string(),
                bbox: [0.50, 0.40, 0.95, 0.90],
                point: None,
                confidence: Some(0.7),
                source_query: "sofa".to_string(),
            },
            Detection {
                label: "couch".to_string(),
                bbox: [0.51, 0.41, 0.94, 0.89],
                point: None,
                confidence: Some(0.6),
                source_query: "couch".to_string(),
            },
        ],
        &queries,
    );

    assert_eq!(filter.detections.len(), 2);
    assert_eq!(filter.dropped.len(), 1);
    assert_eq!(filter.deduped.len(), 1);
    assert!(filter.detections.iter().any(|d| d.label == "chair"));
    assert!(filter.detections.iter().any(|d| d.label == "sofa"));
}

#[test]
fn locate_anything_repeated_instances_ignore_manifest_bbox_matching() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "chairs".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.0, 0.0, 1.0, 1.0],
            instances: vec![
                SceneObjectInstanceSpec {
                    id: Some("chair_left".to_string()),
                    bbox: [0.70, 0.30, 0.90, 0.90],
                    contact: None,
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Left),
                    slot_index: Some(0),
                    target_footprint_m: None,
                },
                SceneObjectInstanceSpec {
                    id: Some("chair_right".to_string()),
                    bbox: [0.10, 0.30, 0.30, 0.90],
                    contact: None,
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: Some(1),
                    target_footprint_m: None,
                },
            ],
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let detections = vec![
        Detection {
            label: "chair".to_string(),
            bbox: [0.12, 0.32, 0.30, 0.90],
            point: None,
            confidence: Some(0.83),
            source_query: "chair".to_string(),
        },
        Detection {
            label: "chair".to_string(),
            bbox: [0.68, 0.34, 0.86, 0.91],
            point: None,
            confidence: Some(0.86),
            source_query: "chair".to_string(),
        },
    ];
    let root = std::env::temp_dir().join(format!(
        "locate_anything_detection_order_{}",
        std::process::id()
    ));
    let source_path = root.join("source.png");
    fs::create_dir_all(&root).unwrap();
    RgbaImage::from_pixel(20, 10, Rgba([0, 0, 0, 255]))
        .save(&source_path)
        .unwrap();
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        &source_path,
        detections,
        "locate_anything",
    )
    .unwrap();
    let left = evidence
        .objects
        .iter()
        .find(|object| object.instance_id.as_deref() == Some("chair_left"))
        .unwrap();
    let right = evidence
        .objects
        .iter()
        .find(|object| object.instance_id.as_deref() == Some("chair_right"))
        .unwrap();
    assert_eq!(
        left.detection.as_ref().unwrap().bbox,
        [0.12, 0.32, 0.30, 0.90]
    );
    assert_eq!(
        right.detection.as_ref().unwrap().bbox,
        [0.68, 0.34, 0.86, 0.91]
    );
}

#[test]
fn locate_anything_singleton_prefers_detector_confidence_over_manifest_iou() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "table".to_string(),
            label: "table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.45, 0.45, 0.55, 0.55],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("table".to_string()),
            instance_count: 1,
            object_prompt: "table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([2.0, 1.0]),
        }],
    };
    let detections = vec![
        Detection {
            label: "table".to_string(),
            bbox: [0.44, 0.44, 0.56, 0.56],
            point: None,
            confidence: Some(0.20),
            source_query: "table".to_string(),
        },
        Detection {
            label: "table".to_string(),
            bbox: [0.20, 0.35, 0.80, 0.80],
            point: None,
            confidence: Some(0.91),
            source_query: "table".to_string(),
        },
    ];
    let root = std::env::temp_dir().join(format!(
        "locate_anything_detector_confidence_{}",
        std::process::id()
    ));
    let source_path = root.join("source.png");
    fs::create_dir_all(&root).unwrap();
    RgbaImage::from_pixel(20, 10, Rgba([0, 0, 0, 255]))
        .save(&source_path)
        .unwrap();
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        &source_path,
        detections,
        "locate_anything",
    )
    .unwrap();
    let table = evidence
        .objects
        .iter()
        .find(|object| object.object_id == "table" && object.instance_id.is_none())
        .unwrap();
    assert_eq!(
        table.detection.as_ref().unwrap().bbox,
        [0.20, 0.35, 0.80, 0.80]
    );
}

#[test]
fn combined_locate_anything_labels_map_back_to_manifest_objects() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.30, 0.40, 0.70, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("table".to_string()),
                instance_count: 1,
                object_prompt: "table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.2, 1.2]),
            },
            SceneObjectSpec {
                id: "chair".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.10, 0.40, 0.80, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("chair".to_string()),
                instance_count: 1,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.65, 0.65]),
            },
        ],
    };
    let combined = "conference table</c>conference chair".to_string();
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        Path::new("/tmp/source.jpg"),
        vec![
            Detection {
                label: "conference chair".to_string(),
                bbox: [0.11, 0.39, 0.29, 0.89],
                point: None,
                confidence: None,
                source_query: combined.clone(),
            },
            Detection {
                label: "conference table".to_string(),
                bbox: [0.32, 0.42, 0.68, 0.88],
                point: None,
                confidence: None,
                source_query: combined,
            },
        ],
        "locate_anything_test",
    )
    .unwrap();

    let table = evidence
        .objects
        .iter()
        .find(|object| object.object_id == "table")
        .unwrap();
    let chair = evidence
        .objects
        .iter()
        .find(|object| object.object_id == "chair")
        .unwrap();
    assert_eq!(
        table.detection.as_ref().unwrap().bbox,
        [0.32, 0.42, 0.68, 0.88]
    );
    assert_eq!(
        chair.detection.as_ref().unwrap().bbox,
        [0.11, 0.39, 0.29, 0.89]
    );
    assert_eq!(table.target_footprint_m, Some([3.2, 1.2]));
    assert_eq!(chair.target_footprint_m, Some([0.65, 0.65]));
}

#[test]
fn locate_anything_evidence_does_not_duplicate_object_without_instances() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            bbox: [0.30, 0.40, 0.70, 0.90],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("table".to_string()),
            instance_count: 1,
            object_prompt: "conference table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([3.2, 1.2]),
        }],
    };
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        Path::new("/tmp/source.jpg"),
        vec![Detection {
            label: "conference table".to_string(),
            bbox: [0.32, 0.42, 0.68, 0.88],
            point: None,
            confidence: Some(0.8),
            source_query: "conference table".to_string(),
        }],
        "locate_anything_test",
    )
    .unwrap();
    assert_eq!(evidence.objects.len(), 1);
    assert_eq!(evidence.objects[0].object_id, "table");
    assert!(evidence.objects[0].instance_id.is_none());
    assert_eq!(evidence.objects[0].target_footprint_m, Some([3.2, 1.2]));
}

#[test]
fn singleton_object_uses_best_detection_instead_of_label_union() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![SceneObjectSpec {
            id: "table".to_string(),
            label: "conference table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.30, 0.40, 0.70, 0.90],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("table".to_string()),
            instance_count: 1,
            object_prompt: "conference table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([3.2, 1.2]),
        }],
    };
    let evidence = locate_anything_evidence_from_detections(
        &manifest,
        Path::new("/tmp/source.jpg"),
        vec![
            Detection {
                label: "table".to_string(),
                bbox: [0.386, 0.519, 0.659, 1.0],
                point: None,
                confidence: None,
                source_query: "table</c>chair".to_string(),
            },
            Detection {
                label: "table".to_string(),
                bbox: [0.778, 0.401, 0.832, 0.481],
                point: None,
                confidence: None,
                source_query: "table</c>chair".to_string(),
            },
        ],
        "locate_anything_test",
    )
    .unwrap();

    assert_eq!(evidence.objects.len(), 1);
    let table = evidence.objects.first().unwrap();
    assert_eq!(
        table.detection.as_ref().unwrap().bbox,
        [0.386, 0.519, 0.659, 1.0]
    );
}

#[test]
fn depth_annotation_adds_contact_geometry_and_footprint_hints() {
    let detection = Detection {
        label: "conference chair".to_string(),
        bbox: [0.25, 0.25, 0.75, 0.75],
        point: Some([0.5, 0.75]),
        confidence: Some(0.9),
        source_query: "conference chair".to_string(),
    };
    let mut evidence = SceneGroundingEvidence {
        source_image_path: "/tmp/source.jpg".to_string(),
        depth: None,
        segmentation: None,
        detections: vec![detection.clone()],
        camera: EstimatedCamera::default(),
        floor: EstimatedFloorPlane::default(),
        objects: vec![ObjectGroundingEvidence {
            object_id: "chair".to_string(),
            instance_id: None,
            reuse_group: Some("chair".to_string()),
            detection: Some(detection),
            mask: None,
            asset_id: None,
            contact_pixel: None,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: Some([0.7, 0.8]),
            provenance: Vec::new(),
        }],
    };
    let depth_map = SceneDepthMapEvidence {
        depth_m: vec![
            2.0, 2.0, 2.0, 2.0, //
            2.0, 2.2, 2.2, 2.0, //
            2.0, 2.4, 2.4, 2.0, //
            3.0, 3.0, 3.0, 3.0,
        ],
        width: 4,
        height: 4,
        intrinsics: CameraIntrinsics {
            fx: 4.0,
            fy: 4.0,
            cx: 1.5,
            cy: 1.5,
            width: 4,
            height: 4,
        },
        focal_length_px: Some(4.0),
        vertical_fov_degrees: Some(53.0),
    };

    let summary =
        annotate_grounding_evidence_with_depth_map(&mut evidence, &depth_map, "depth_pro");
    let object = evidence.objects.first().unwrap();

    assert_eq!(summary.annotated_objects, 1);
    assert_eq!(summary.depth_map_size, [4, 4]);
    assert!(summary.floor_candidate_sample_count > 0);
    assert!(object.depth_stats.as_ref().unwrap().contact_m.unwrap() > 0.0);
    assert_eq!(object.candidate_floor_contact_rays.len(), 1);
    assert!(object.metric_contact_point_m.unwrap()[2] > 0.0);
    assert_eq!(object.target_footprint_m, Some([0.7, 0.8]));
    assert!(object.provenance.contains(&"depth_pro".to_string()));
}

#[test]
fn depth_annotation_persists_object_excluded_floor_estimate() {
    let detection = Detection {
        label: "table".to_string(),
        bbox: [0.0, 0.58, 0.72, 1.0],
        point: Some([0.36, 0.98]),
        confidence: Some(0.9),
        source_query: "table".to_string(),
    };
    let mut evidence = SceneGroundingEvidence {
        source_image_path: "/tmp/source.jpg".to_string(),
        depth: None,
        segmentation: None,
        detections: vec![detection.clone()],
        camera: EstimatedCamera::default(),
        floor: EstimatedFloorPlane::default(),
        objects: vec![ObjectGroundingEvidence {
            object_id: "table".to_string(),
            instance_id: None,
            reuse_group: Some("table".to_string()),
            detection: Some(detection),
            mask: None,
            asset_id: None,
            contact_pixel: None,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: None,
            provenance: Vec::new(),
        }],
    };
    let width = 96u32;
    let height = 72u32;
    let intrinsics = CameraIntrinsics {
        fx: 90.0,
        fy: 90.0,
        cx: width as f32 * 0.5,
        cy: height as f32 * 0.5,
        width,
        height,
    };
    let mut depth_m = perspective_floor_depth_map(width, height, intrinsics, 1.35);
    let false_floor_depth_m = perspective_floor_depth_map(width, height, intrinsics, 3.0);
    let y_start = (height as f32 * 0.58).floor() as u32;
    for y in y_start..height {
        for x in 0..((width as f32 * 0.72).round() as u32) {
            let index = y as usize * width as usize + x as usize;
            depth_m[index] = false_floor_depth_m[index];
        }
    }
    let depth_map = SceneDepthMapEvidence {
        depth_m,
        width,
        height,
        intrinsics,
        focal_length_px: Some(90.0),
        vertical_fov_degrees: Some(45.0),
    };
    let exclusion_bboxes = floor_sample_exclusion_bboxes(&evidence);
    let (expected_floor, expected_count) =
        estimate_scene_floor_plane_with_exclusions(&depth_map, &exclusion_bboxes)
            .expect("excluded floor");
    let unexcluded_floor = estimate_scene_floor_plane(&depth_map).expect("unexcluded floor");

    let summary =
        annotate_grounding_evidence_with_depth_map(&mut evidence, &depth_map, "depth_pro");

    assert_eq!(summary.floor_sample_count, expected_count);
    assert_eq!(summary.floor_sample_count, summary.floor_inlier_count);
    assert!(summary.floor_candidate_sample_count >= summary.floor_inlier_count);
    assert_eq!(
        summary.floor_rejected_sample_count,
        summary
            .floor_candidate_sample_count
            .saturating_sub(summary.floor_inlier_count)
    );
    assert_eq!(evidence.floor, expected_floor);
    assert!(
        (evidence.floor.distance_m - unexcluded_floor.distance_m).abs() > 1.0e-3,
        "excluded floor should not be overwritten by unexcluded estimate"
    );
}

#[test]
fn depth_floor_estimator_prefers_upright_floor_over_lower_image_clutter() {
    let width = 96u32;
    let height = 72u32;
    let camera_height_m = 1.25f32;
    let intrinsics = CameraIntrinsics {
        fx: 88.0,
        fy: 88.0,
        cx: width as f32 * 0.5,
        cy: height as f32 * 0.5,
        width,
        height,
    };
    let mut depth_m = vec![8.0f32; width as usize * height as usize];
    depth_m.copy_from_slice(&perspective_floor_depth_map(
        width,
        height,
        intrinsics,
        camera_height_m,
    ));
    for y in 44..68 {
        for x in 30..66 {
            depth_m[y as usize * width as usize + x as usize] = 1.15;
        }
    }
    let depth_map = SceneDepthMapEvidence {
        depth_m,
        width,
        height,
        intrinsics,
        focal_length_px: Some(88.0),
        vertical_fov_degrees: Some(44.0),
    };

    let (floor, inliers) =
        estimate_scene_floor_plane_with_exclusions(&depth_map, &[]).expect("floor estimate");

    assert!(inliers > 64, "expected many floor inliers, got {inliers}");
    assert!(
        floor.normal[1] > 0.92,
        "floor normal should remain upright despite clutter: {:?}",
        floor
    );
    assert!(
        (floor.distance_m + camera_height_m).abs() < 0.08,
        "floor distance should recover camera height: {:?}",
        floor
    );
    assert!(
        floor.residual_m.unwrap_or(f32::INFINITY) < 0.06,
        "floor residual should stay bounded: {:?}",
        floor
    );
}

fn perspective_floor_depth_map(
    width: u32,
    height: u32,
    intrinsics: CameraIntrinsics,
    camera_height_m: f32,
) -> Vec<f32> {
    let mut depth_m = vec![8.0f32; width as usize * height as usize];
    for y in 0..height {
        let denom = y as f32 + 0.5 - intrinsics.cy;
        let floor_depth = if denom > 1.0 {
            camera_height_m * intrinsics.fy / denom
        } else {
            8.0
        };
        for x in 0..width {
            depth_m[y as usize * width as usize + x as usize] = floor_depth;
        }
    }
    depth_m
}

#[test]
fn far_field_filter_removes_small_background_detections() {
    let near_detection = Detection {
        label: "chair".to_string(),
        bbox: [0.10, 0.50, 0.30, 0.90],
        point: Some([0.20, 0.90]),
        confidence: None,
        source_query: "chair".to_string(),
    };
    let far_detection = Detection {
        label: "chair".to_string(),
        bbox: [0.80, 0.35, 0.84, 0.50],
        point: Some([0.82, 0.50]),
        confidence: None,
        source_query: "chair".to_string(),
    };
    let mut evidence = SceneGroundingEvidence {
        source_image_path: "/tmp/source.jpg".to_string(),
        depth: None,
        segmentation: None,
        detections: vec![near_detection.clone(), far_detection.clone()],
        camera: EstimatedCamera::default(),
        floor: EstimatedFloorPlane::default(),
        objects: vec![
            ObjectGroundingEvidence {
                object_id: "near_chair".to_string(),
                instance_id: None,
                reuse_group: Some("chair".to_string()),
                detection: Some(near_detection),
                mask: None,
                asset_id: None,
                contact_pixel: Some([0.20, 0.90]),
                depth_stats: Some(ObjectDepthStats {
                    median_m: 2.0,
                    min_m: 1.8,
                    max_m: 2.2,
                    contact_m: Some(2.0),
                    sample_count: Some(16),
                }),
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: Some([0.0, 0.0, 2.0]),
                target_footprint_m: None,
                provenance: vec!["depth_pro".to_string()],
            },
            ObjectGroundingEvidence {
                object_id: "far_chair".to_string(),
                instance_id: None,
                reuse_group: Some("chair".to_string()),
                detection: Some(far_detection),
                mask: None,
                asset_id: None,
                contact_pixel: Some([0.82, 0.50]),
                depth_stats: Some(ObjectDepthStats {
                    median_m: 12.0,
                    min_m: 10.0,
                    max_m: 12.5,
                    contact_m: Some(12.0),
                    sample_count: Some(16),
                }),
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: Some([2.0, 0.0, 12.0]),
                target_footprint_m: None,
                provenance: vec!["depth_pro".to_string()],
            },
        ],
    };
    let mut depth = vec![2.0; 100];
    for y in 3..5 {
        for x in 8..9 {
            depth[y * 10 + x] = 12.0;
        }
    }
    let depth_map = SceneDepthMapEvidence {
        depth_m: depth,
        width: 10,
        height: 10,
        intrinsics: CameraIntrinsics {
            fx: 10.0,
            fy: 10.0,
            cx: 4.5,
            cy: 4.5,
            width: 10,
            height: 10,
        },
        focal_length_px: Some(10.0),
        vertical_fov_degrees: Some(50.0),
    };

    let summary = filter_far_field_grounding_evidence(&mut evidence, &depth_map);
    assert!(summary.enabled);
    assert_eq!(summary.removed_detections, 1);
    assert_eq!(summary.removed_objects, 1);
    assert_eq!(evidence.detections.len(), 1);
    assert_eq!(evidence.objects.len(), 1);
    assert_eq!(evidence.objects[0].object_id, "near_chair");
}
