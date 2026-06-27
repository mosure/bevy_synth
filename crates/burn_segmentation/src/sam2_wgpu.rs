use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use image::DynamicImage;
use image::imageops::FilterType;

use crate::image_encoder::burn_image_encoder::BurnSamImageEncoder;
use crate::mask_decoder::burn_mask_decoder::BurnSamMaskDecoder;
use crate::prompt_encoder::burn_prompt::BurnSamPromptEncoder;
use crate::{
    BinaryMask, SamImageEncoderVariant, SamImageEncoderWeights, SamMaskDecoderWeights,
    SamPromptEncoderWeights, SamPromptInput, SegmentationError, SegmentationMask,
    SegmentationModelKind, SegmentationPrompt, SegmentationResult, SegmentationRuntimeConfig,
    SegmentationStageTimings, validate_bbox,
};

type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const SAM_INPUT_SIZE: usize = 1024;
const SAM_MASK_THRESHOLD: f32 = 0.0;
const NORMALIZE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const NORMALIZE_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
pub struct Sam2WgpuRuntime {
    device: burn_wgpu::WgpuDevice,
    image_encoder: BurnSamImageEncoder<WgpuBackend>,
    prompt_encoder: BurnSamPromptEncoder<WgpuBackend>,
    mask_decoder: BurnSamMaskDecoder<WgpuBackend>,
    profile_stages: bool,
    last_timings: Option<SegmentationStageTimings>,
}

impl Sam2WgpuRuntime {
    pub fn new(config: &SegmentationRuntimeConfig) -> SegmentationResult<Self> {
        if config.model != SegmentationModelKind::Sam2 {
            return Err(SegmentationError::Unsupported(format!(
                "SAM2 WGPU runtime cannot run {}",
                config.model.label()
            )));
        }
        if config.allow_download && config.cdn_base_url.is_some() {
            return Err(SegmentationError::Unsupported(
                "SAM2 WGPU runtime does not yet implement CDN download/bootstrap".to_string(),
            ));
        }
        let model_root = config.model_root.as_ref().ok_or_else(|| {
            SegmentationError::Unsupported(
                "SAM2 WGPU runtime requires model_root containing component safetensors"
                    .to_string(),
            )
        })?;
        let image_encoder_path = component_path(model_root, "image_encoder.safetensors")?;
        let prompt_encoder_path = component_path(model_root, "prompt_encoder.safetensors")?;
        let mask_decoder_path = component_path(model_root, "mask_decoder.safetensors")?;

        let device = burn_wgpu::WgpuDevice::default();
        let image_encoder = BurnSamImageEncoder::<WgpuBackend>::from_weights(
            SamImageEncoderWeights::from_safetensors_file(&image_encoder_path)?,
            &device,
        );
        let prompt_encoder = BurnSamPromptEncoder::<WgpuBackend>::from_weights(
            SamPromptEncoderWeights::from_safetensors_file(&prompt_encoder_path)?,
            &device,
        );
        let mask_decoder = BurnSamMaskDecoder::<WgpuBackend>::from_weights(
            SamMaskDecoderWeights::from_safetensors_file(&mask_decoder_path)?,
            &device,
        );
        Ok(Self {
            device,
            image_encoder,
            prompt_encoder,
            mask_decoder,
            profile_stages: config.profile_stages,
            last_timings: None,
        })
    }

    pub fn last_stage_timings(&self) -> Option<SegmentationStageTimings> {
        self.last_timings
    }

    pub fn variant(&self) -> SamImageEncoderVariant {
        self.image_encoder.weights.config.variant
    }

    pub fn segment(
        &mut self,
        image: &DynamicImage,
        prompts: &[SegmentationPrompt],
    ) -> SegmentationResult<Vec<SegmentationMask>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        for prompt in prompts {
            validate_bbox(prompt.bbox)?;
        }
        if image.width() == 0 || image.height() == 0 {
            return Err(SegmentationError::Image(
                "cannot segment zero-sized image".to_string(),
            ));
        }

        let total_start = Instant::now();
        let preprocess_start = Instant::now();
        let input = preprocess_image_to_sam_input(image);
        let preprocess_ms = elapsed_ms(preprocess_start);
        let input = Tensor::<WgpuBackend, 1>::from_floats(input.as_slice(), &self.device)
            .reshape([1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE]);

        let encode_start = Instant::now();
        let image_features = self.image_encoder.forward(input);
        let high_res_s0_base = self
            .mask_decoder
            .project_high_res_s0(image_features.high_res_features_raw[0].clone());
        let high_res_s1_base = self
            .mask_decoder
            .project_high_res_s1(image_features.high_res_features_raw[1].clone());
        self.sync_if_profile()?;
        let encode_ms = elapsed_ms(encode_start);

        let prompt_start = Instant::now();
        let batch = prompts.len();
        let prompt_input = prompt_input_from_prompts(prompts)?;
        let sparse_prompt_embeddings = self
            .prompt_encoder
            .embed_points(&prompt_input, &self.device);
        let dense_prompt_embeddings = self.prompt_encoder.dense_no_mask(batch);
        let dense_pe = self.prompt_encoder.dense_pe(&self.device);
        self.sync_if_profile()?;
        let prompt_ms = elapsed_ms(prompt_start);

        let decode_start = Instant::now();
        let high_res_s0 = high_res_s0_base.repeat_dim(0, batch);
        let high_res_s1 = high_res_s1_base.repeat_dim(0, batch);
        let decoded = self.mask_decoder.forward(
            image_features.image_embed,
            dense_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            [high_res_s0, high_res_s1],
            false,
            true,
        );
        self.sync_if_profile()?;
        let decode_ms = elapsed_ms(decode_start);

        let postprocess_start = Instant::now();
        let masks = postprocess_masks(decoded.low_res_masks, image.width(), image.height())?;
        let scores = decoded
            .iou_predictions
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| SegmentationError::Image(format!("read SAM2 scores: {err}")))?;
        let postprocess_ms = elapsed_ms(postprocess_start);
        self.last_timings = Some(SegmentationStageTimings {
            preprocess_ms,
            encode_ms,
            prompt_ms,
            decode_ms,
            postprocess_ms,
            total_ms: elapsed_ms(total_start),
        });
        masks
            .into_iter()
            .enumerate()
            .map(|(index, binary)| {
                let prompt = prompts[index].clone();
                let bbox = binary.bbox_normalized().unwrap_or(prompt.bbox);
                Ok(SegmentationMask {
                    object_id: prompt.object_id.clone(),
                    label: prompt.label.clone(),
                    bbox,
                    score: scores.get(index).copied().unwrap_or_default(),
                    width: binary.width(),
                    height: binary.height(),
                    area_px: binary.area_px(),
                    mask_rle: binary.encode_rle(),
                    mask_png_path: None,
                    source_prompt: prompt,
                    provider: "burn-native-wgpu".to_string(),
                    model: SegmentationModelKind::Sam2.label().to_string(),
                })
            })
            .collect()
    }

    fn sync_if_profile(&self) -> SegmentationResult<()> {
        if self.profile_stages {
            WgpuBackend::sync(&self.device)
                .map_err(|err| SegmentationError::Image(format!("sync SAM2 WGPU device: {err}")))?;
        }
        Ok(())
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn component_path(model_root: &Path, file_name: &str) -> SegmentationResult<PathBuf> {
    for candidate in [
        model_root.join(file_name),
        model_root.join("components").join(file_name),
    ] {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(SegmentationError::Unsupported(format!(
        "missing SAM2 component safetensors `{file_name}` under {}",
        model_root.display()
    )))
}

pub fn preprocess_image_to_sam_input(image: &DynamicImage) -> Vec<f32> {
    let rgb = image.to_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        SAM_INPUT_SIZE as u32,
        SAM_INPUT_SIZE as u32,
        FilterType::Triangle,
    );
    let mut data = vec![0.0_f32; 3 * SAM_INPUT_SIZE * SAM_INPUT_SIZE];
    for y in 0..SAM_INPUT_SIZE {
        for x in 0..SAM_INPUT_SIZE {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                data[channel * SAM_INPUT_SIZE * SAM_INPUT_SIZE + y * SAM_INPUT_SIZE + x] =
                    (value - NORMALIZE_MEAN[channel]) / NORMALIZE_STD[channel];
            }
        }
    }
    data
}

fn prompt_input_from_prompts(prompts: &[SegmentationPrompt]) -> SegmentationResult<SamPromptInput> {
    let mut coords = Vec::with_capacity(prompts.len() * 3 * 2);
    let mut labels = Vec::with_capacity(prompts.len() * 3);
    for prompt in prompts {
        validate_bbox(prompt.bbox)?;
        coords.extend_from_slice(&[
            prompt.bbox[0] * SAM_INPUT_SIZE as f32,
            prompt.bbox[1] * SAM_INPUT_SIZE as f32,
            prompt.bbox[2] * SAM_INPUT_SIZE as f32,
            prompt.bbox[3] * SAM_INPUT_SIZE as f32,
            0.0,
            0.0,
        ]);
        labels.extend_from_slice(&[2, 3, -1]);
    }
    let input = SamPromptInput {
        coords,
        labels,
        batch: prompts.len(),
        points: 3,
    };
    input.validate()?;
    Ok(input)
}

fn postprocess_masks(
    low_res_masks: Tensor<WgpuBackend, 4>,
    width: u32,
    height: u32,
) -> SegmentationResult<Vec<BinaryMask>> {
    let [batch, mask_count, _low_h, _low_w] = low_res_masks.dims();
    if mask_count != 1 {
        return Err(SegmentationError::Image(format!(
            "expected one mask per prompt, got mask_count={mask_count}"
        )));
    }
    let resized = interpolate(
        low_res_masks,
        [height as usize, width as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    let data = resized
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| SegmentationError::Image(format!("read SAM2 masks: {err}")))?;
    let pixels_per_mask = width as usize * height as usize;
    (0..batch)
        .map(|index| {
            let start = index * pixels_per_mask;
            let end = start + pixels_per_mask;
            let binary = data[start..end]
                .iter()
                .map(|value| u8::from(*value > SAM_MASK_THRESHOLD))
                .collect::<Vec<_>>();
            BinaryMask::new(width, height, binary)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_io::{find_tensor, load_required_tensors_from_safetensors_file};

    #[test]
    fn sam2_prompt_batch_transforms_normalized_boxes() {
        let prompts = vec![SegmentationPrompt {
            object_id: "chair_1".to_string(),
            label: "chair".to_string(),
            bbox: [0.25, 0.5, 0.75, 1.0],
            point: None,
            source_query: Some("chair".to_string()),
        }];
        let input = prompt_input_from_prompts(&prompts).unwrap();
        assert_eq!(input.batch, 1);
        assert_eq!(input.points, 3);
        assert_eq!(input.coords, vec![256.0, 512.0, 768.0, 1024.0, 0.0, 0.0]);
        assert_eq!(input.labels, vec![2, 3, -1]);
    }

    #[test]
    fn sam2_preprocess_matches_reference_fixture_reasonably() {
        let image_path = std::env::var("SAM2_REFERENCE_IMAGE").unwrap_or_default();
        let reference_path = std::env::var("SAM2_REFERENCE_HOOK").unwrap_or_default();
        if image_path.is_empty() || reference_path.is_empty() {
            eprintln!(
                "skipping: set SAM2_REFERENCE_IMAGE and SAM2_REFERENCE_HOOK to run SAM2 preprocess parity"
            );
            return;
        }
        let image = image::open(&image_path).unwrap();
        let actual = preprocess_image_to_sam_input(&image);
        let reference = load_required_tensors_from_safetensors_file(
            Path::new(&reference_path),
            &["image_encoder_input"],
        )
        .unwrap();
        let expected = find_tensor(&reference, "image_encoder_input").unwrap();
        assert_eq!(expected.shape, vec![1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE]);
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (left, right) in actual.iter().zip(expected.data.iter()) {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            sum_sq += (delta as f64) * (delta as f64);
        }
        let rms = (sum_sq / actual.len() as f64).sqrt();
        eprintln!("sam2_preprocess max_abs={max_abs:.6e} rms={rms:.6e}");
        assert!(max_abs < 5.0e-2, "max_abs={max_abs}");
        assert!(rms < 8.0e-3, "rms={rms}");
    }

    #[test]
    fn sam2_wgpu_runtime_matches_reference_mask() {
        let component_root = std::env::var("SAM2_COMPONENT_ROOT").unwrap_or_default();
        let image_path = std::env::var("SAM2_REFERENCE_IMAGE").unwrap_or_default();
        let reference_path = std::env::var("SAM2_REFERENCE_HOOK").unwrap_or_default();
        if component_root.is_empty() || image_path.is_empty() || reference_path.is_empty() {
            eprintln!(
                "skipping: set SAM2_COMPONENT_ROOT, SAM2_REFERENCE_IMAGE, and SAM2_REFERENCE_HOOK to run SAM2 WGPU e2e parity"
            );
            return;
        }
        let reference = load_required_tensors_from_safetensors_file(
            Path::new(&reference_path),
            &["box_normalized", "postprocessed_masks", "iou_predictions"],
        )
        .unwrap();
        let bbox = find_tensor(&reference, "box_normalized").unwrap();
        let prompt = SegmentationPrompt {
            object_id: "chair_1".to_string(),
            label: "chair".to_string(),
            bbox: [bbox.data[0], bbox.data[1], bbox.data[2], bbox.data[3]],
            point: None,
            source_query: Some("chair".to_string()),
        };
        let image = image::open(&image_path).unwrap();
        let mut runtime = Sam2WgpuRuntime::new(&SegmentationRuntimeConfig {
            model: SegmentationModelKind::Sam2,
            backend: crate::SegmentationRuntimeBackend::BurnNative,
            model_root: Some(PathBuf::from(component_root)),
            ..SegmentationRuntimeConfig::default()
        })
        .unwrap();
        let observed = runtime.segment(&image, &[prompt]).unwrap();
        assert_eq!(observed.len(), 1);

        let expected_logits = find_tensor(&reference, "postprocessed_masks").unwrap();
        let expected_mask = BinaryMask::new(
            image.width(),
            image.height(),
            expected_logits
                .data
                .iter()
                .map(|value| u8::from(*value > SAM_MASK_THRESHOLD))
                .collect(),
        )
        .unwrap();
        let observed_mask =
            BinaryMask::decode_rle(observed[0].width, observed[0].height, &observed[0].mask_rle)
                .unwrap();
        let iou = observed_mask.iou(&expected_mask).unwrap();
        let expected_score = find_tensor(&reference, "iou_predictions").unwrap().data[0];
        let score_delta = (observed[0].score - expected_score).abs();
        eprintln!("sam2_wgpu_runtime mask_iou={iou:.6} score_delta={score_delta:.6e}");
        assert!(iou >= 0.98, "mask_iou={iou}");
        assert!(score_delta < 1.0e-3, "score_delta={score_delta}");
    }
}
