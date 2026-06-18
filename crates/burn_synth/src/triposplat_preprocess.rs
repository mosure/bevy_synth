use burn_foreground::pipeline::PrepareImageConfig;

pub(crate) const TRIPOSPLAT_CANVAS_SIZE: usize = burn_triposplat::TRIPOSPLAT_CANONICAL_CANVAS_SIZE;

pub(crate) fn triposplat_prepare_image_config(erode_radius: usize) -> PrepareImageConfig {
    let default = PrepareImageConfig::default();
    PrepareImageConfig {
        bg_color: [0.0, 0.0, 0.0],
        padding_ratio: 0.1,
        max_dimension: usize::MAX,
        resize_shorter_to: Some(TRIPOSPLAT_CANVAS_SIZE),
        alpha_erode_radius: erode_radius,
        min_component_size: default.min_component_size,
    }
}
