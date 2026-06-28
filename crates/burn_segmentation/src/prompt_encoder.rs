use std::path::Path;

#[cfg(feature = "bpk")]
use crate::tensor_io::load_all_tensors_from_burnpack_file;
use crate::tensor_io::{
    LoadedTensorF32, expect_2d, find_tensor, load_required_tensors_from_safetensors_file,
};
use crate::{SegmentationError, SegmentationResult};

#[derive(Clone, Debug, PartialEq)]
pub struct SamPromptEncoderConfig {
    pub embed_dim: usize,
    pub input_image_size: [usize; 2],
    pub image_embedding_size: [usize; 2],
}

impl SamPromptEncoderConfig {
    pub fn sam2_1024() -> Self {
        Self {
            embed_dim: 256,
            input_image_size: [1024, 1024],
            image_embedding_size: [64, 64],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamPromptEncoderWeights {
    pub config: SamPromptEncoderConfig,
    pub gaussian_matrix: Vec<f32>,
    pub gaussian_shape: [usize; 2],
    pub point_embeddings: Vec<Vec<f32>>,
    pub not_a_point_embed: Vec<f32>,
    pub no_mask_embed: Vec<f32>,
}

impl SamPromptEncoderWeights {
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> SegmentationResult<Self> {
        let path = path.as_ref();
        let tensors = load_required_tensors_from_safetensors_file(
            path,
            &[
                "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
                "sam_prompt_encoder.point_embeddings.0.weight",
                "sam_prompt_encoder.point_embeddings.1.weight",
                "sam_prompt_encoder.point_embeddings.2.weight",
                "sam_prompt_encoder.point_embeddings.3.weight",
                "sam_prompt_encoder.not_a_point_embed.weight",
                "sam_prompt_encoder.no_mask_embed.weight",
            ],
        )?;
        Self::from_loaded_tensors(&tensors)
    }

    #[cfg(feature = "bpk")]
    pub fn from_burnpack_file(path: impl AsRef<Path>) -> SegmentationResult<Self> {
        let tensors = load_all_tensors_from_burnpack_file(path.as_ref())?;
        Self::from_loaded_tensors(&tensors)
    }

    pub fn from_loaded_tensors(tensors: &[(String, LoadedTensorF32)]) -> SegmentationResult<Self> {
        let gaussian = find_tensor(
            tensors,
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        )?;
        let gaussian_shape = expect_2d(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            gaussian,
        )?;
        if gaussian_shape[0] != 2 {
            return Err(SegmentationError::Image(format!(
                "SAM positional Gaussian matrix expected [2, C], got {gaussian_shape:?}"
            )));
        }
        let embed_dim = gaussian_shape[1] * 2;
        let point_embeddings = (0..4)
            .map(|index| {
                let key = format!("sam_prompt_encoder.point_embeddings.{index}.weight");
                let tensor = find_tensor(tensors, &key)?;
                let shape = expect_2d(&key, tensor)?;
                if shape != [1, embed_dim] {
                    return Err(SegmentationError::Image(format!(
                        "{key} expected [1, {embed_dim}], got {shape:?}"
                    )));
                }
                Ok(tensor.data.clone())
            })
            .collect::<SegmentationResult<Vec<_>>>()?;
        let not_a_point = find_tensor(tensors, "sam_prompt_encoder.not_a_point_embed.weight")?;
        let no_mask = find_tensor(tensors, "sam_prompt_encoder.no_mask_embed.weight")?;
        for (key, tensor) in [
            ("sam_prompt_encoder.not_a_point_embed.weight", not_a_point),
            ("sam_prompt_encoder.no_mask_embed.weight", no_mask),
        ] {
            let shape = expect_2d(key, tensor)?;
            if shape != [1, embed_dim] {
                return Err(SegmentationError::Image(format!(
                    "{key} expected [1, {embed_dim}], got {shape:?}"
                )));
            }
        }
        Ok(Self {
            config: SamPromptEncoderConfig {
                embed_dim,
                ..SamPromptEncoderConfig::sam2_1024()
            },
            gaussian_matrix: gaussian.data.clone(),
            gaussian_shape,
            point_embeddings,
            not_a_point_embed: not_a_point.data.clone(),
            no_mask_embed: no_mask.data.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamPromptInput {
    pub coords: Vec<f32>,
    pub labels: Vec<i32>,
    pub batch: usize,
    pub points: usize,
}

impl SamPromptInput {
    pub fn from_box_transformed(coords_xyxy: [f32; 4]) -> Self {
        Self {
            coords: vec![
                coords_xyxy[0],
                coords_xyxy[1],
                coords_xyxy[2],
                coords_xyxy[3],
                0.0,
                0.0,
            ],
            labels: vec![2, 3, -1],
            batch: 1,
            points: 3,
        }
    }

    pub fn validate(&self) -> SegmentationResult<()> {
        if self.coords.len() != self.batch * self.points * 2 {
            return Err(SegmentationError::InvalidPrompt(format!(
                "prompt coords length {} does not match batch={} points={}",
                self.coords.len(),
                self.batch,
                self.points
            )));
        }
        if self.labels.len() != self.batch * self.points {
            return Err(SegmentationError::InvalidPrompt(format!(
                "prompt labels length {} does not match batch={} points={}",
                self.labels.len(),
                self.batch,
                self.points
            )));
        }
        Ok(())
    }
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
pub mod burn_prompt {
    use burn::prelude::*;

    use super::*;

    #[derive(Debug)]
    pub struct BurnSamPromptEncoder<B: Backend> {
        weights: SamPromptEncoderWeights,
        gaussian: Tensor<B, 2>,
        no_mask_embed: Tensor<B, 1>,
    }

    impl<B: Backend> BurnSamPromptEncoder<B> {
        pub fn from_weights(weights: SamPromptEncoderWeights, device: &B::Device) -> Self {
            let gaussian = Tensor::<B, 1>::from_floats(weights.gaussian_matrix.as_slice(), device)
                .reshape(weights.gaussian_shape);
            let no_mask_embed =
                Tensor::<B, 1>::from_floats(weights.no_mask_embed.as_slice(), device);
            Self {
                weights,
                gaussian,
                no_mask_embed,
            }
        }

        pub fn embed_points(&self, input: &SamPromptInput, device: &B::Device) -> Tensor<B, 3> {
            input.validate().expect("validated SAM prompt input");
            let [input_h, input_w] = self.weights.config.input_image_size;
            let embed_dim = self.weights.config.embed_dim;
            let coords = Tensor::<B, 1>::from_floats(input.coords.as_slice(), device)
                .reshape([input.batch * input.points, 2])
                .add_scalar(0.5);
            let x = coords
                .clone()
                .slice([0..input.batch * input.points, 0..1])
                .div_scalar(input_w as f64)
                .mul_scalar(2.0)
                .sub_scalar(1.0);
            let y = coords
                .slice([0..input.batch * input.points, 1..2])
                .div_scalar(input_h as f64)
                .mul_scalar(2.0)
                .sub_scalar(1.0);
            let coords = Tensor::cat(vec![x, y], 1);
            let args = coords
                .matmul(self.gaussian.clone())
                .mul_scalar(2.0 * std::f64::consts::PI);
            let embedding = Tensor::cat(vec![args.clone().sin(), args.cos()], 1).reshape([
                input.batch,
                input.points,
                embed_dim,
            ]);

            let mut fourier_mask = vec![1.0; input.batch * input.points * embed_dim];
            let mut label_add = vec![0.0; input.batch * input.points * embed_dim];
            for batch in 0..input.batch {
                for point in 0..input.points {
                    let label = input.labels[batch * input.points + point];
                    let add = match label {
                        -1 => Some(self.weights.not_a_point_embed.as_slice()),
                        0..=3 => Some(self.weights.point_embeddings[label as usize].as_slice()),
                        _ => None,
                    };
                    if let Some(add) = add {
                        let offset = (batch * input.points + point) * embed_dim;
                        label_add[offset..offset + embed_dim].copy_from_slice(add);
                        if label == -1 {
                            fourier_mask[offset..offset + embed_dim].fill(0.0);
                        }
                    }
                }
            }
            let mask = Tensor::<B, 1>::from_floats(fourier_mask.as_slice(), device).reshape([
                input.batch,
                input.points,
                embed_dim,
            ]);
            let label_add = Tensor::<B, 1>::from_floats(label_add.as_slice(), device).reshape([
                input.batch,
                input.points,
                embed_dim,
            ]);
            embedding * mask + label_add
        }

        pub fn dense_no_mask(&self, batch: usize) -> Tensor<B, 4> {
            let [height, width] = self.weights.config.image_embedding_size;
            let embed_dim = self.weights.config.embed_dim;
            self.no_mask_embed
                .clone()
                .reshape([1, embed_dim, 1, 1])
                .repeat_dim(0, batch)
                .repeat_dim(2, height)
                .repeat_dim(3, width)
        }

        pub fn dense_pe(&self, device: &B::Device) -> Tensor<B, 4> {
            let [height, width] = self.weights.config.image_embedding_size;
            let embed_dim = self.weights.config.embed_dim;
            let mut coords = Vec::with_capacity(height * width * 2);
            for y in 0..height {
                for x in 0..width {
                    coords.push((x as f32 + 0.5) / width as f32);
                    coords.push((y as f32 + 0.5) / height as f32);
                }
            }
            let coords = Tensor::<B, 1>::from_floats(coords.as_slice(), device)
                .reshape([height * width, 2])
                .mul_scalar(2.0)
                .sub_scalar(1.0);
            let args = coords
                .matmul(self.gaussian.clone())
                .mul_scalar(2.0 * std::f64::consts::PI);
            Tensor::cat(vec![args.clone().sin(), args.cos()], 1)
                .reshape([height, width, embed_dim])
                .permute([2, 0, 1])
                .reshape([1, embed_dim, height, width])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_input_validates_shape() {
        let input = SamPromptInput::from_box_transformed([1.0, 2.0, 3.0, 4.0]);
        input.validate().unwrap();
        assert_eq!(input.labels, vec![2, 3, -1]);
    }

    #[test]
    fn prompt_encoder_rejects_bad_embedding_shape() {
        let tensors = vec![(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix".to_string(),
            LoadedTensorF32 {
                shape: vec![3, 128],
                data: vec![0.0; 384],
            },
        )];
        assert!(SamPromptEncoderWeights::from_loaded_tensors(&tensors).is_err());
    }

    #[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
    #[test]
    fn sam2_prompt_encoder_wgpu_matches_reference_hook() {
        let weights_path = std::env::var("SAM2_PROMPT_WEIGHTS").unwrap_or_default();
        let reference_path = std::env::var("SAM2_REFERENCE_HOOK").unwrap_or_default();
        if weights_path.is_empty() || reference_path.is_empty() {
            eprintln!(
                "skipping: set SAM2_PROMPT_WEIGHTS and SAM2_REFERENCE_HOOK to run WGPU SAM2 prompt parity"
            );
            return;
        }
        type B = burn_wgpu::Wgpu<f32, i32, u32>;
        let device = burn_wgpu::WgpuDevice::default();
        let weights = SamPromptEncoderWeights::from_safetensors_file(&weights_path).unwrap();
        let embed_dim = weights.config.embed_dim;
        let reference = crate::tensor_io::load_required_tensors_from_safetensors_file(
            Path::new(&reference_path),
            &[
                "box_coords_transformed",
                "sparse_prompt_embeddings",
                "dense_pe",
            ],
        )
        .unwrap();
        let coords = crate::tensor_io::find_tensor(&reference, "box_coords_transformed").unwrap();
        let expected =
            crate::tensor_io::find_tensor(&reference, "sparse_prompt_embeddings").unwrap();
        assert_eq!(coords.shape, vec![1, 2, 2]);
        assert_eq!(expected.shape, vec![1, 3, embed_dim]);
        let encoder = burn_prompt::BurnSamPromptEncoder::<B>::from_weights(weights, &device);
        let prompt = SamPromptInput::from_box_transformed([
            coords.data[0],
            coords.data[1],
            coords.data[2],
            coords.data[3],
        ]);
        let actual = encoder
            .embed_points(&prompt, &device)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (left, right) in actual.iter().zip(expected.data.iter()) {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            sum_sq += (delta as f64) * (delta as f64);
        }
        let rms = (sum_sq / actual.len() as f64).sqrt();
        eprintln!(
            "sam2_prompt_encoder_wgpu_matches_reference_hook max_abs={max_abs:.6e} rms={rms:.6e}"
        );
        assert!(max_abs < 2.0e-4, "max_abs={max_abs}");
        assert!(rms < 2.0e-5, "rms={rms}");

        let expected = crate::tensor_io::find_tensor(&reference, "dense_pe").unwrap();
        assert_eq!(expected.shape, vec![1, embed_dim, 64, 64]);
        let actual = encoder
            .dense_pe(&device)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (left, right) in actual.iter().zip(expected.data.iter()) {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            sum_sq += (delta as f64) * (delta as f64);
        }
        let rms = (sum_sq / actual.len() as f64).sqrt();
        eprintln!(
            "sam2_prompt_dense_pe_wgpu_matches_reference_hook max_abs={max_abs:.6e} rms={rms:.6e}"
        );
        assert!(max_abs < 2.0e-5, "dense_pe max_abs={max_abs}");
        assert!(rms < 2.0e-6, "dense_pe rms={rms}");
    }
}
