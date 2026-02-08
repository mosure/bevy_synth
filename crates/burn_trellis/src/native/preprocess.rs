use std::path::Path;

use image::imageops::{FilterType, crop_imm, resize};
use image::{DynamicImage, RgbImage, RgbaImage};

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub max_side: u32,
    pub alpha_bbox_threshold: u8,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            max_side: 1024,
            alpha_bbox_threshold: (0.8 * 255.0) as u8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl PreprocessOutput {
    pub fn into_image(self) -> Result<RgbImage, String> {
        RgbImage::from_vec(self.width, self.height, self.rgb).ok_or_else(|| {
            "failed to construct image from preprocessed bytes (shape/length mismatch)".to_string()
        })
    }
}

pub fn preprocess_image_path(
    path: &Path,
    config: PreprocessConfig,
) -> Result<PreprocessOutput, String> {
    let image =
        image::open(path).map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
    preprocess_image(image, config)
}

pub fn preprocess_image(
    image: DynamicImage,
    config: PreprocessConfig,
) -> Result<PreprocessOutput, String> {
    let rgba = image.to_rgba8();
    let resized = resize_to_max_side(rgba, config.max_side);
    let bbox = alpha_bbox(&resized, config.alpha_bbox_threshold).unwrap_or((
        0,
        0,
        resized.width().saturating_sub(1),
        resized.height().saturating_sub(1),
    ));
    let (crop_x, crop_y, crop_size) = square_crop_bounds(resized.width(), resized.height(), bbox);
    let mut cropped = crop_imm(&resized, crop_x, crop_y, crop_size, crop_size).to_image();
    premultiply_alpha(&mut cropped);
    let rgb = rgba_to_rgb(&cropped);
    let (width, height) = cropped.dimensions();
    Ok(PreprocessOutput { width, height, rgb })
}

fn resize_to_max_side(image: RgbaImage, max_side: u32) -> RgbaImage {
    let (width, height) = image.dimensions();
    if width <= max_side && height <= max_side {
        return image;
    }
    let long_side = width.max(height) as f32;
    let scale = max_side as f32 / long_side;
    let out_w = ((width as f32) * scale).round().max(1.0) as u32;
    let out_h = ((height as f32) * scale).round().max(1.0) as u32;
    resize(&image, out_w, out_h, FilterType::CatmullRom)
}

fn alpha_bbox(image: &RgbaImage, threshold: u8) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = image.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            let alpha = image.get_pixel(x, y).0[3];
            if alpha > threshold {
                found = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

fn square_crop_bounds(
    image_width: u32,
    image_height: u32,
    bbox: (u32, u32, u32, u32),
) -> (u32, u32, u32) {
    let (min_x, min_y, max_x, max_y) = bbox;
    let bbox_w = max_x.saturating_sub(min_x);
    let bbox_h = max_y.saturating_sub(min_y);
    let mut size = bbox_w.max(bbox_h);
    size = size.min(image_width.max(1)).min(image_height.max(1));
    size = size.max(1);

    let center_x = (min_x + max_x) as f32 * 0.5;
    let center_y = (min_y + max_y) as f32 * 0.5;

    let mut x = (center_x - (size as f32 * 0.5)).floor() as i64;
    let mut y = (center_y - (size as f32 * 0.5)).floor() as i64;
    let max_x_start = image_width.saturating_sub(size) as i64;
    let max_y_start = image_height.saturating_sub(size) as i64;
    x = x.clamp(0, max_x_start);
    y = y.clamp(0, max_y_start);

    (x as u32, y as u32, size)
}

fn premultiply_alpha(image: &mut RgbaImage) {
    for pixel in image.pixels_mut() {
        let alpha = pixel.0[3] as f32 / 255.0;
        pixel.0[0] = (pixel.0[0] as f32 * alpha).clamp(0.0, 255.0) as u8;
        pixel.0[1] = (pixel.0[1] as f32 * alpha).clamp(0.0, 255.0) as u8;
        pixel.0[2] = (pixel.0[2] as f32 * alpha).clamp(0.0, 255.0) as u8;
    }
}

fn rgba_to_rgb(image: &RgbaImage) -> Vec<u8> {
    let mut out = Vec::with_capacity((image.width() * image.height() * 3) as usize);
    for pixel in image.pixels() {
        out.push(pixel.0[0]);
        out.push(pixel.0[1]);
        out.push(pixel.0[2]);
    }
    out
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};

    use super::{PreprocessConfig, preprocess_image};

    #[test]
    fn premultiplies_alpha() {
        let mut image = RgbaImage::new(2, 2);
        image.put_pixel(0, 0, Rgba([100, 50, 200, 128]));
        image.put_pixel(1, 0, Rgba([80, 40, 120, 64]));
        image.put_pixel(0, 1, Rgba([10, 20, 30, 0]));
        image.put_pixel(1, 1, Rgba([255, 255, 255, 255]));

        let output =
            preprocess_image(DynamicImage::ImageRgba8(image), PreprocessConfig::default()).unwrap();
        let out = output.into_image().unwrap();
        assert_eq!(out.width(), out.height());
        let p = out.get_pixel(0, 0).0;
        assert!(p[0] <= 100);
        assert!(p[1] <= 50);
        assert!(p[2] <= 200);
    }

    #[test]
    fn crops_square_around_alpha_bbox() {
        let mut image = RgbaImage::from_pixel(8, 6, Rgba([0, 0, 0, 0]));
        for y in 1..=2 {
            for x in 5..=6 {
                image.put_pixel(x, y, Rgba([200, 100, 50, 255]));
            }
        }
        let output =
            preprocess_image(DynamicImage::ImageRgba8(image), PreprocessConfig::default()).unwrap();
        assert_eq!(output.width, output.height);
        // Trellis bbox size uses max - min (inclusive-high semantics), so this crop is 1x1.
        assert_eq!(output.width, 1);
    }

    #[test]
    fn downsizes_large_inputs() {
        let image = RgbaImage::from_pixel(4000, 2000, Rgba([255, 255, 255, 255]));
        let output = preprocess_image(
            DynamicImage::ImageRgba8(image),
            PreprocessConfig {
                max_side: 1024,
                ..PreprocessConfig::default()
            },
        )
        .unwrap();
        assert!(output.width <= 1024);
        assert!(output.height <= 1024);
    }
}
