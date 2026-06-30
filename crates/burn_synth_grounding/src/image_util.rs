use std::path::Path;

use burn_depth::ImageBoundingBox;
use burn_synth_scene::Detection;
use image::{Rgba, RgbaImage};

pub(crate) fn sanitize_artifact_stem(value: &str) -> String {
    let mut out = String::with_capacity(value.len().max(1));
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "_".to_string() } else { out }
}

pub(crate) fn bbox_area_normalized(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).abs().clamp(0.0, 1.0) * (bbox[3] - bbox[1]).abs().clamp(0.0, 1.0)
}

pub(crate) fn write_detection_overlay(
    source_scene_path: &Path,
    detections: &[Detection],
    output_path: &Path,
) -> Result<(), String> {
    let mut image = image::open(source_scene_path)
        .map_err(|err| {
            format!(
                "failed to load source image for detection overlay {}: {err}",
                source_scene_path.display()
            )
        })?
        .to_rgba8();
    for (index, detection) in detections.iter().enumerate() {
        let color = overlay_color(index);
        draw_normalized_bbox(&mut image, detection.bbox, color, 4);
        draw_normalized_cross(
            &mut image,
            detection
                .point
                .unwrap_or_else(|| bbox_bottom_center(detection.bbox)),
            color,
            8,
        );
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write detection overlay {}: {err}",
            output_path.display()
        )
    })
}

pub(crate) fn overlay_color(index: usize) -> Rgba<u8> {
    const COLORS: [[u8; 4]; 8] = [
        [230, 57, 70, 255],
        [42, 157, 143, 255],
        [69, 123, 157, 255],
        [233, 196, 106, 255],
        [131, 56, 236, 255],
        [255, 128, 0, 255],
        [0, 180, 216, 255],
        [255, 0, 110, 255],
    ];
    Rgba(COLORS[index % COLORS.len()])
}

pub(crate) fn draw_normalized_bbox(
    image: &mut RgbaImage,
    bbox: [f32; 4],
    color: Rgba<u8>,
    thickness: u32,
) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x0 = (bbox[0].min(bbox[2]).clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let x1 = (bbox[0].max(bbox[2]).clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y0 =
        (bbox[1].min(bbox[3]).clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    let y1 =
        (bbox[1].max(bbox[3]).clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    let thickness = thickness.max(1) as i32;
    for offset in 0..thickness {
        draw_line(image, x0, y0 + offset, x1, y0 + offset, color);
        draw_line(image, x0, y1 - offset, x1, y1 - offset, color);
        draw_line(image, x0 + offset, y0, x0 + offset, y1, color);
        draw_line(image, x1 - offset, y0, x1 - offset, y1, color);
    }
}

pub(crate) fn draw_normalized_cross(
    image: &mut RgbaImage,
    pixel: [f32; 2],
    color: Rgba<u8>,
    radius: i32,
) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x = (pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y = (pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    draw_line(image, x - radius, y, x + radius, y, color);
    draw_line(image, x, y - radius, x, y + radius, color);
}

pub(crate) fn draw_normalized_square(
    image: &mut RgbaImage,
    pixel: [f32; 2],
    color: Rgba<u8>,
    radius: i32,
) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x = (pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y = (pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    for yy in (y - radius)..=(y + radius) {
        for xx in (x - radius)..=(x + radius) {
            put_pixel_checked(image, xx, yy, color);
        }
    }
}

fn draw_line(image: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        put_pixel_checked(image, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn put_pixel_checked(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}

pub(crate) fn normalized_bbox_to_image_bbox(
    bbox: [f32; 4],
    image_width: u32,
    image_height: u32,
) -> ImageBoundingBox {
    let bbox = [
        bbox[0].clamp(0.0, 1.0),
        bbox[1].clamp(0.0, 1.0),
        bbox[2].clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ];
    let x0 = (bbox[0].min(bbox[2]) * image_width as f32).floor() as u32;
    let x1 = (bbox[0].max(bbox[2]) * image_width as f32).ceil() as u32;
    let y0 = (bbox[1].min(bbox[3]) * image_height as f32).floor() as u32;
    let y1 = (bbox[1].max(bbox[3]) * image_height as f32).ceil() as u32;
    let x0 = x0.min(image_width.saturating_sub(1));
    let y0 = y0.min(image_height.saturating_sub(1));
    let x1 = x1.min(image_width).max(x0 + 1);
    let y1 = y1.min(image_height).max(y0 + 1);
    ImageBoundingBox {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

pub fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
}

pub(crate) fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

pub(crate) fn union_bbox(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

pub fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let ix0 = left[0].max(right[0]);
    let iy0 = left[1].max(right[1]);
    let ix1 = left[2].min(right[2]);
    let iy1 = left[3].min(right[3]);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let intersection = iw * ih;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}
