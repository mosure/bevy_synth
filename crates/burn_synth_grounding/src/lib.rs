mod depth;
mod image_util;
mod locate;
mod segmentation;
mod types;

pub use burn_locate_anything::import::LocateAnythingPrecision;
pub use burn_locate_anything::{
    DecodeMode, Detection as LocateAnythingDetection, DetectionQuery,
    LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT, LocateAnythingRuntime, LocateAnythingRuntimeBackend,
    LocateAnythingRuntimeConfig,
};
pub use burn_segmentation::{
    SegmentationModelKind, SegmentationPrecision, SegmentationQuantization,
    SegmentationRuntimeBackend,
};

pub use depth::{
    annotate_grounding_evidence_with_depth_map, estimate_scene_floor_plane,
    filter_far_field_grounding_evidence,
};
pub use image_util::{bbox_bottom_center, bbox_iou};
pub use locate::{
    default_locate_anything_allowed_categories, locate_anything_categories,
    locate_anything_evidence_from_detections, locate_anything_queries,
    locate_anything_queries_for_allowed_categories,
};
pub use types::*;

#[cfg(test)]
mod tests;
