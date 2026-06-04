use std::collections::VecDeque;
use std::path::Path;

use burn::prelude::*;
use burn::tensor::ops::InterpolateMode;

use crate::preprocess::RmbgImageProcessor;
use crate::resize::resize_chw_align_corners_false;
use crate::rmbg14::BriaRmbg;

#[derive(Debug, Clone)]
pub struct PrepareImageConfig {
    pub bg_color: [f32; 3],
    pub padding_ratio: f32,
    pub max_dimension: usize,
    pub min_component_size: usize,
}

impl Default for PrepareImageConfig {
    fn default() -> Self {
        Self {
            bg_color: [1.0, 1.0, 1.0],
            padding_ratio: 0.1,
            max_dimension: 2000,
            min_component_size: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedImageData {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub alpha_mask: Option<Vec<f32>>,
    pub alpha_probs: Option<Vec<f32>>,
    pub bbox: Option<[usize; 4]>,
}

impl PreparedImageData {
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 4> {
        let flat = Tensor::<B, 1>::from_floats(self.data.as_slice(), device);
        flat.reshape([1, 3, self.height as i32, self.width as i32])
    }
}

#[derive(Debug)]
pub struct PrepareImageError(pub String);

impl std::fmt::Display for PrepareImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}", self.0)
    }
}

impl std::error::Error for PrepareImageError {}

pub(crate) struct LoadedImage {
    pub(crate) rgb: Vec<u8>,
    pub(crate) alpha: Option<Vec<u8>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) has_alpha: bool,
}

#[derive(Debug)]
pub struct RmbgPipeline<B: Backend> {
    pub model: BriaRmbg<B>,
    pub processor: RmbgImageProcessor,
}

impl<B: Backend> RmbgPipeline<B> {
    pub fn new(model: BriaRmbg<B>, processor: RmbgImageProcessor) -> Self {
        Self { model, processor }
    }

    pub fn infer_mask(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = image.device();
        let processed = self.processor.preprocess(image);
        let output = self.model.forward(processed);
        output
            .masks
            .first()
            .cloned()
            .unwrap_or_else(|| Tensor::<B, 4>::zeros([1, 1, 1, 1], &device))
    }

    #[cfg(feature = "import")]
    pub fn from_pretrained(
        weights_root: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::rmbg14::import::{load_rmbg, load_rmbg_config, load_rmbg_processor_config};

        let root = weights_root.as_ref();
        let config = load_rmbg_config(root)?;
        let weights_path = root.join("model.safetensors");
        let model = load_rmbg(device, weights_path, &config)?;
        let processor = load_rmbg_processor_config(root)?;

        Ok(Self::new(model, processor))
    }
}

pub fn prepare_image_data<B: Backend>(
    path: &Path,
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let loaded = load_image_rgb(path, config.max_dimension)?;
    prepare_loaded_image(loaded, pipeline, config)
}

pub fn prepare_image_data_from_bytes<B: Backend>(
    bytes: &[u8],
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let loaded = load_image_rgb_from_bytes(bytes, config.max_dimension)?;
    prepare_loaded_image(loaded, pipeline, config)
}

#[cfg(target_arch = "wasm32")]
pub async fn prepare_image_data_async<B: Backend>(
    path: &Path,
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let loaded = load_image_rgb(path, config.max_dimension)?;
    prepare_loaded_image_async(loaded, pipeline, config).await
}

#[cfg(target_arch = "wasm32")]
pub async fn prepare_image_data_from_bytes_async<B: Backend>(
    bytes: &[u8],
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let loaded = load_image_rgb_from_bytes(bytes, config.max_dimension)?;
    prepare_loaded_image_async(loaded, pipeline, config).await
}

fn prepare_loaded_image<B: Backend>(
    loaded: LoadedImage,
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let rgb = loaded.rgb;
    let alpha = loaded.alpha;
    let width = loaded.width;
    let height = loaded.height;
    let has_alpha = loaded.has_alpha;

    let alpha = if has_alpha {
        alpha.filter(|alpha| is_valid_alpha(alpha, width, height, 0.01))
    } else {
        None
    };

    let (alpha_mask, alpha_probs, bbox) = if let Some(alpha) = alpha {
        let alpha_mask = alpha
            .iter()
            .map(|value| *value as f32 / 255.0)
            .collect::<Vec<f32>>();
        let bbox = bbox_from_mask_all(&alpha, width, height)
            .ok_or_else(|| PrepareImageError("input image too small".to_string()))?;
        (alpha_mask, None, Some(bbox))
    } else {
        let pipeline = pipeline.ok_or_else(|| {
            PrepareImageError("RMBG pipeline required for images without alpha".to_string())
        })?;
        let alpha = infer_alpha_mask(pipeline, &rgb, width, height, config.min_component_size)?;
        (alpha.alpha_mask, Some(alpha.alpha_probs), Some(alpha.bbox))
    };

    let bbox = bbox.ok_or_else(|| PrepareImageError("missing bounding box".to_string()))?;
    build_prepared_image_from_alpha(&rgb, width, height, alpha_mask, alpha_probs, bbox, config)
}

#[cfg(target_arch = "wasm32")]
async fn prepare_loaded_image_async<B: Backend>(
    loaded: LoadedImage,
    pipeline: Option<&RmbgPipeline<B>>,
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let rgb = loaded.rgb;
    let alpha = loaded.alpha;
    let width = loaded.width;
    let height = loaded.height;
    let has_alpha = loaded.has_alpha;

    let alpha = if has_alpha {
        alpha.and_then(|alpha| {
            if is_valid_alpha(&alpha, width, height, 0.01) {
                Some(alpha)
            } else {
                None
            }
        })
    } else {
        None
    };

    let (alpha_mask, alpha_probs, bbox) = if let Some(alpha) = alpha {
        let alpha_mask = alpha
            .iter()
            .map(|value| *value as f32 / 255.0)
            .collect::<Vec<f32>>();
        let bbox = bbox_from_mask_all(&alpha, width, height)
            .ok_or_else(|| PrepareImageError("input image too small".to_string()))?;
        (alpha_mask, None, Some(bbox))
    } else {
        let pipeline = pipeline.ok_or_else(|| {
            PrepareImageError("RMBG pipeline required for images without alpha".to_string())
        })?;
        let alpha =
            infer_alpha_mask_async(pipeline, &rgb, width, height, config.min_component_size)
                .await?;
        (alpha.alpha_mask, Some(alpha.alpha_probs), Some(alpha.bbox))
    };

    let bbox = bbox.ok_or_else(|| PrepareImageError("missing bounding box".to_string()))?;
    build_prepared_image_from_alpha(&rgb, width, height, alpha_mask, alpha_probs, bbox, config)
}

pub(crate) fn build_prepared_image_from_alpha(
    rgb: &[u8],
    width: usize,
    height: usize,
    alpha_mask: Vec<f32>,
    alpha_probs: Option<Vec<f32>>,
    bbox: [usize; 4],
    config: &PrepareImageConfig,
) -> Result<PreparedImageData, PrepareImageError> {
    let mut rgb_f32 = rgb_to_chw(rgb, width, height);
    for value in &mut rgb_f32 {
        *value /= 255.0;
    }

    apply_alpha(&mut rgb_f32, &alpha_mask, config.bg_color, width, height);
    let (x, y, w, h) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let cropped = crop_rgb(&rgb_f32, width, height, x, y, w, h)?;
    let padded = pad_to_square(&cropped, w, h, config.padding_ratio, config.bg_color[0]);

    let mut output = padded.0;
    for value in &mut output {
        *value *= 255.0;
    }

    Ok(PreparedImageData {
        data: output,
        width: padded.1,
        height: padded.2,
        alpha_mask: Some(alpha_mask),
        alpha_probs,
        bbox: Some(bbox),
    })
}

pub fn prepare_image_tensor<B: Backend>(
    path: &Path,
    pipeline: Option<&RmbgPipeline<B>>,
    device: &B::Device,
    config: &PrepareImageConfig,
) -> Result<Tensor<B, 4>, PrepareImageError> {
    let prepared = prepare_image_data(path, pipeline, config)?;
    let flat = Tensor::<B, 1>::from_floats(prepared.data.as_slice(), device);
    Ok(flat.reshape([1, 3, prepared.height as i32, prepared.width as i32]))
}

pub fn prepare_image_tensor_from_bytes<B: Backend>(
    bytes: &[u8],
    pipeline: Option<&RmbgPipeline<B>>,
    device: &B::Device,
    config: &PrepareImageConfig,
) -> Result<Tensor<B, 4>, PrepareImageError> {
    let prepared = prepare_image_data_from_bytes(bytes, pipeline, config)?;
    let flat = Tensor::<B, 1>::from_floats(prepared.data.as_slice(), device);
    Ok(flat.reshape([1, 3, prepared.height as i32, prepared.width as i32]))
}

#[cfg(target_arch = "wasm32")]
pub async fn prepare_image_tensor_async<B: Backend>(
    path: &Path,
    pipeline: Option<&RmbgPipeline<B>>,
    device: &B::Device,
    config: &PrepareImageConfig,
) -> Result<Tensor<B, 4>, PrepareImageError> {
    let prepared = prepare_image_data_async(path, pipeline, config).await?;
    let flat = Tensor::<B, 1>::from_floats(prepared.data.as_slice(), device);
    Ok(flat.reshape([1, 3, prepared.height as i32, prepared.width as i32]))
}

#[cfg(target_arch = "wasm32")]
pub async fn prepare_image_tensor_from_bytes_async<B: Backend>(
    bytes: &[u8],
    pipeline: Option<&RmbgPipeline<B>>,
    device: &B::Device,
    config: &PrepareImageConfig,
) -> Result<Tensor<B, 4>, PrepareImageError> {
    let prepared = prepare_image_data_from_bytes_async(bytes, pipeline, config).await?;
    let flat = Tensor::<B, 1>::from_floats(prepared.data.as_slice(), device);
    Ok(flat.reshape([1, 3, prepared.height as i32, prepared.width as i32]))
}

pub(crate) fn load_image_rgb(
    path: &Path,
    max_dimension: usize,
) -> Result<LoadedImage, PrepareImageError> {
    let image = image::open(path)
        .map_err(|err| PrepareImageError(format!("invalid image path {path:?}: {err}")))?;
    load_image_rgb_from_dynamic(image, max_dimension)
}

pub(crate) fn load_image_rgb_from_bytes(
    bytes: &[u8],
    max_dimension: usize,
) -> Result<LoadedImage, PrepareImageError> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| PrepareImageError(format!("invalid image bytes: {err}")))?;
    load_image_rgb_from_dynamic(image, max_dimension)
}

fn load_image_rgb_from_dynamic(
    image: image::DynamicImage,
    max_dimension: usize,
) -> Result<LoadedImage, PrepareImageError> {
    let has_alpha = image.color().has_alpha();

    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();

    let (width, height, rgba) = if max_dimension > 0
        && (width as usize > max_dimension || height as usize > max_dimension)
    {
        let scale = if height > width {
            max_dimension as f32 / height as f32
        } else {
            max_dimension as f32 / width as f32
        };
        let new_width = (width as f32 * scale).floor().max(1.0) as u32;
        let new_height = (height as f32 * scale).floor().max(1.0) as u32;
        let resized = resize_rgba_inter_area(&rgba, new_width, new_height);
        (new_width, new_height, resized)
    } else {
        (width, height, rgba)
    };

    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    let mut alpha = if has_alpha {
        Some(Vec::with_capacity((width * height) as usize))
    } else {
        None
    };

    for pixel in rgba.pixels() {
        let channels = pixel.0;
        rgb.push(channels[0]);
        rgb.push(channels[1]);
        rgb.push(channels[2]);
        if let Some(alpha_vec) = alpha.as_mut() {
            alpha_vec.push(channels[3]);
        }
    }

    Ok(LoadedImage {
        rgb,
        alpha,
        width: width as usize,
        height: height as usize,
        has_alpha,
    })
}

pub(crate) fn is_valid_alpha(alpha: &[u8], width: usize, height: usize, min_ratio: f32) -> bool {
    let bins = 20usize;
    let mut hist = vec![0usize; bins];
    for value in alpha {
        let bin = (*value as usize * bins) / 256;
        hist[bin.min(bins - 1)] += 1;
    }
    let min_hist_val = (width * height) as f32 * min_ratio;
    (hist[0] as f32) >= min_hist_val && (hist[bins - 1] as f32) >= min_hist_val
}

struct AlphaMaskResult {
    alpha_mask: Vec<f32>,
    alpha_probs: Vec<f32>,
    bbox: [usize; 4],
}

fn infer_alpha_mask<B: Backend>(
    pipeline: &RmbgPipeline<B>,
    rgb: &[u8],
    width: usize,
    height: usize,
    min_component_size: usize,
) -> Result<AlphaMaskResult, PrepareImageError> {
    let device = &pipeline.model.conv_in.weight.val().device();
    let pixels = width * height;
    let mut rgb_chw = Vec::with_capacity(pixels * 3);
    for c in 0..3 {
        for idx in 0..pixels {
            rgb_chw.push(rgb[idx * 3 + c] as f32);
        }
    }

    let processor = &pipeline.processor;
    let target_size = if processor.do_resize {
        processor.size.unwrap_or([height, width])
    } else {
        [height, width]
    };
    let [target_height, target_width] = target_size;
    let mut resized = resize_chw_align_corners_false(
        &rgb_chw,
        3,
        height,
        width,
        target_height,
        target_width,
        processor.resize_mode.clone(),
    );

    if processor.do_rescale {
        for value in &mut resized {
            *value *= processor.rescale_factor;
        }
    }

    let max_value = resized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_value < 1e-3 {
        return Err(PrepareImageError(
            "invalid image: pure black image".to_string(),
        ));
    }

    if processor.do_normalize {
        let pixels = target_height * target_width;
        for c in 0..3 {
            let mean = processor.mean[c];
            let std = processor.std[c];
            let offset = c * pixels;
            for idx in 0..pixels {
                let value = resized[offset + idx];
                resized[offset + idx] = (value - mean) / std;
            }
        }
    }

    let input = Tensor::<B, 1>::from_floats(resized.as_slice(), device).reshape([
        1,
        3,
        target_height as i32,
        target_width as i32,
    ]);

    let output = pipeline.model.forward(input);
    let mask = output
        .masks
        .first()
        .cloned()
        .unwrap_or_else(|| Tensor::<B, 4>::zeros([1, 1, 1, 1], device));
    let mask_data = mask
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| PrepareImageError(format!("failed to read RMBG mask: {err:?}")))?;

    postprocess_alpha_mask(
        mask_data,
        target_height,
        target_width,
        width,
        height,
        min_component_size,
    )
}

#[cfg(target_arch = "wasm32")]
async fn infer_alpha_mask_async<B: Backend>(
    pipeline: &RmbgPipeline<B>,
    rgb: &[u8],
    width: usize,
    height: usize,
    min_component_size: usize,
) -> Result<AlphaMaskResult, PrepareImageError> {
    let device = &pipeline.model.conv_in.weight.val().device();
    let pixels = width * height;
    let mut rgb_chw = Vec::with_capacity(pixels * 3);
    for c in 0..3 {
        for idx in 0..pixels {
            rgb_chw.push(rgb[idx * 3 + c] as f32);
        }
    }

    let processor = &pipeline.processor;
    let target_size = if processor.do_resize {
        processor.size.unwrap_or([height, width])
    } else {
        [height, width]
    };
    let [target_height, target_width] = target_size;
    let mut resized = resize_chw_align_corners_false(
        &rgb_chw,
        3,
        height,
        width,
        target_height,
        target_width,
        processor.resize_mode.clone(),
    );

    if processor.do_rescale {
        for value in &mut resized {
            *value *= processor.rescale_factor;
        }
    }

    let max_value = resized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_value < 1e-3 {
        return Err(PrepareImageError(
            "invalid image: pure black image".to_string(),
        ));
    }

    if processor.do_normalize {
        let pixels = target_height * target_width;
        for c in 0..3 {
            let mean = processor.mean[c];
            let std = processor.std[c];
            let offset = c * pixels;
            for idx in 0..pixels {
                let value = resized[offset + idx];
                resized[offset + idx] = (value - mean) / std;
            }
        }
    }

    let input = Tensor::<B, 1>::from_floats(resized.as_slice(), device).reshape([
        1,
        3,
        target_height as i32,
        target_width as i32,
    ]);

    let output = pipeline.model.forward(input);
    let mask = output
        .masks
        .first()
        .cloned()
        .unwrap_or_else(|| Tensor::<B, 4>::zeros([1, 1, 1, 1], device));
    let mask_data = mask
        .into_data_async()
        .await
        .map_err(|err| PrepareImageError(format!("failed to materialize RMBG mask: {err:?}")))?
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| PrepareImageError(format!("failed to read RMBG mask: {err:?}")))?;

    postprocess_alpha_mask(
        mask_data,
        target_height,
        target_width,
        width,
        height,
        min_component_size,
    )
}

fn postprocess_alpha_mask(
    mask_data: Vec<f32>,
    target_height: usize,
    target_width: usize,
    width: usize,
    height: usize,
    min_component_size: usize,
) -> Result<AlphaMaskResult, PrepareImageError> {
    let data = if target_height != height || target_width != width {
        resize_chw_align_corners_false(
            &mask_data,
            1,
            target_height,
            target_width,
            height,
            width,
            InterpolateMode::Bilinear,
        )
    } else {
        mask_data
    };

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &data {
        min = min.min(*value);
        max = max.max(*value);
    }
    let denom = (max - min).max(1e-6);
    let normalized = data
        .iter()
        .map(|value| (value - min) / denom)
        .collect::<Vec<f32>>();

    let mut alpha_u8 = normalized
        .iter()
        .map(|value| (value * 255.0) as u8)
        .collect::<Vec<u8>>();

    let thresh = otsu_threshold(&alpha_u8);
    for value in &mut alpha_u8 {
        *value = if *value > thresh { 255 } else { 0 };
    }

    let cleaned = remove_small_objects(&alpha_u8, width, height, min_component_size);
    let bbox = find_bounding_box(&cleaned, width, height)
        .ok_or_else(|| PrepareImageError("input image too small".to_string()))?;

    let alpha_mask = cleaned
        .iter()
        .map(|value| if *value > 0 { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();

    Ok(AlphaMaskResult {
        alpha_mask,
        alpha_probs: normalized,
        bbox,
    })
}

pub(crate) fn apply_alpha(
    rgb: &mut [f32],
    alpha: &[f32],
    bg_color: [f32; 3],
    width: usize,
    height: usize,
) {
    let pixels = width * height;
    for (idx, &a) in alpha.iter().enumerate().take(pixels) {
        for (c, &bg) in bg_color.iter().enumerate() {
            let offset = c * pixels + idx;
            rgb[offset] = rgb[offset] * a + bg * (1.0 - a);
        }
    }
}

pub(crate) fn crop_rgb(
    rgb: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Result<Vec<f32>, PrepareImageError> {
    if x + w > width || y + h > height {
        return Err(PrepareImageError("crop outside image bounds".to_string()));
    }
    let mut cropped = vec![0.0f32; 3 * w * h];
    for c in 0..3 {
        for yy in 0..h {
            let src_row = (y + yy) * width;
            let dst_row = yy * w;
            for xx in 0..w {
                let src_idx = c * width * height + src_row + (x + xx);
                let dst_idx = c * w * h + dst_row + xx;
                cropped[dst_idx] = rgb[src_idx];
            }
        }
    }
    Ok(cropped)
}

pub(crate) fn pad_to_square(
    rgb: &[f32],
    width: usize,
    height: usize,
    padding_ratio: f32,
    pad_value: f32,
) -> (Vec<f32>, usize, usize) {
    let (pad_left, pad_right, pad_top, pad_bottom) = if width > height {
        let pad_lr = (width as f32 * padding_ratio) as usize;
        let extra = ((width - height) as f32 / 2.0).floor() as usize;
        let pad_tb = pad_lr + extra;
        (pad_lr, pad_lr, pad_tb, pad_tb)
    } else {
        let pad_tb = (height as f32 * padding_ratio) as usize;
        let extra = ((height - width) as f32 / 2.0).floor() as usize;
        let pad_lr = pad_tb + extra;
        (pad_lr, pad_lr, pad_tb, pad_tb)
    };

    let padded_width = width + pad_left + pad_right;
    let padded_height = height + pad_top + pad_bottom;
    let mut padded = vec![pad_value; 3 * padded_width * padded_height];

    for c in 0..3 {
        for yy in 0..height {
            let dst_row = (yy + pad_top) * padded_width;
            let src_row = yy * width;
            for xx in 0..width {
                let src_idx = c * width * height + src_row + xx;
                let dst_idx = c * padded_width * padded_height + dst_row + (xx + pad_left);
                padded[dst_idx] = rgb[src_idx];
            }
        }
    }

    (padded, padded_width, padded_height)
}

pub(crate) fn otsu_threshold(values: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &value in values {
        hist[value as usize] += 1;
    }
    let total = values.len() as f32;
    let mut sum = 0.0f32;
    for (i, count) in hist.iter().enumerate() {
        sum += i as f32 * *count as f32;
    }

    let mut sum_b = 0.0f32;
    let mut w_b = 0.0f32;
    let mut var_max = -1.0f32;
    let mut threshold = 0usize;

    for (i, count) in hist.iter().enumerate() {
        w_b += *count as f32;
        if w_b <= 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f <= 0.0 {
            break;
        }
        sum_b += i as f32 * *count as f32;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > var_max {
            var_max = var_between;
            threshold = i;
        }
    }

    threshold as u8
}

pub(crate) fn remove_small_objects(
    mask: &[u8],
    width: usize,
    height: usize,
    min_size: usize,
) -> Vec<u8> {
    let mut output = mask.to_vec();
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if visited[idx] || output[idx] == 0 {
                continue;
            }
            let mut component = Vec::new();
            visited[idx] = true;
            queue.push_back((x, y));

            while let Some((cx, cy)) = queue.pop_front() {
                let cidx = cy * width + cx;
                component.push(cidx);
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx as isize + dx;
                        let ny = cy as isize + dy;
                        if nx < 0 || ny < 0 {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if nx >= width || ny >= height {
                            continue;
                        }
                        let nidx = ny * width + nx;
                        if visited[nidx] || output[nidx] == 0 {
                            continue;
                        }
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }

            if component.len() < min_size {
                for idx in component {
                    output[idx] = 0;
                }
            }
        }
    }

    output
}

pub(crate) fn find_bounding_box(mask: &[u8], width: usize, height: usize) -> Option<[usize; 4]> {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    let mut best_bbox = None;
    let mut best_size = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if visited[idx] || mask[idx] == 0 {
                continue;
            }
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;
            let mut count = 0usize;
            visited[idx] = true;
            queue.push_back((x, y));

            while let Some((cx, cy)) = queue.pop_front() {
                count += 1;
                min_x = min_x.min(cx);
                max_x = max_x.max(cx);
                min_y = min_y.min(cy);
                max_y = max_y.max(cy);

                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx as isize + dx;
                        let ny = cy as isize + dy;
                        if nx < 0 || ny < 0 {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if nx >= width || ny >= height {
                            continue;
                        }
                        let nidx = ny * width + nx;
                        if visited[nidx] || mask[nidx] == 0 {
                            continue;
                        }
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }

            if count > best_size {
                best_size = count;
                best_bbox = Some([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]);
            }
        }
    }

    best_bbox
}

pub(crate) fn bbox_from_mask_all(mask: &[u8], width: usize, height: usize) -> Option<[usize; 4]> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask[idx] == 0 {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    if !found {
        None
    } else {
        Some([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1])
    }
}

pub(crate) fn rgb_to_chw(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut out = vec![0.0f32; pixels * 3];
    for idx in 0..pixels {
        let base = idx * 3;
        out[idx] = rgb[base] as f32;
        out[pixels + idx] = rgb[base + 1] as f32;
        out[pixels * 2 + idx] = rgb[base + 2] as f32;
    }
    out
}

fn resize_rgba_inter_area(
    input: &image::RgbaImage,
    out_width: u32,
    out_height: u32,
) -> image::RgbaImage {
    let in_width = input.width() as usize;
    let in_height = input.height() as usize;
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;

    if in_width == out_width_usize && in_height == out_height_usize {
        return input.clone();
    }

    let scale_x = in_width as f32 / out_width_usize as f32;
    let scale_y = in_height as f32 / out_height_usize as f32;
    let mut output = image::RgbaImage::new(out_width, out_height);

    for oy in 0..out_height_usize {
        let y0 = oy as f32 * scale_y;
        let y1 = (oy + 1) as f32 * scale_y;
        let y_start = y0.floor() as isize;
        let y_end = y1.ceil() as isize;
        for ox in 0..out_width_usize {
            let x0 = ox as f32 * scale_x;
            let x1 = (ox + 1) as f32 * scale_x;
            let x_start = x0.floor() as isize;
            let x_end = x1.ceil() as isize;

            let mut acc = [0.0f32; 4];
            let mut total = 0.0f32;

            for iy in y_start..y_end {
                if iy < 0 || iy >= in_height as isize {
                    continue;
                }
                let iy0 = iy as f32;
                let iy1 = iy0 + 1.0;
                let overlap_y = (y1.min(iy1) - y0.max(iy0)).max(0.0);
                if overlap_y <= 0.0 {
                    continue;
                }
                for ix in x_start..x_end {
                    if ix < 0 || ix >= in_width as isize {
                        continue;
                    }
                    let ix0 = ix as f32;
                    let ix1 = ix0 + 1.0;
                    let overlap_x = (x1.min(ix1) - x0.max(ix0)).max(0.0);
                    if overlap_x <= 0.0 {
                        continue;
                    }
                    let weight = overlap_x * overlap_y;
                    let pixel = input.get_pixel(ix as u32, iy as u32).0;
                    for c in 0..4 {
                        acc[c] += pixel[c] as f32 * weight;
                    }
                    total += weight;
                }
            }

            if total > 0.0 {
                for value in &mut acc {
                    *value = (*value / total).clamp(0.0, 255.0);
                }
            }

            output.put_pixel(
                ox as u32,
                oy as u32,
                image::Rgba([
                    acc[0].round() as u8,
                    acc[1].round() as u8,
                    acc[2].round() as u8,
                    acc[3].round() as u8,
                ]),
            );
        }
    }

    output
}
