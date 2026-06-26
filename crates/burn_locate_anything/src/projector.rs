use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::tensor_io::{LoadedTensorF32, load_required_tensors_from_safetensors_file};
use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProjectorConfig {
    pub kind: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            kind: "mlp".to_string(),
            input_dim: 4608,
            hidden_dim: 2048,
            output_dim: 2048,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectorWeights {
    pub config: ProjectorConfig,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
}

impl ProjectorWeights {
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        let path = path.as_ref();
        let tensors = load_required_tensors_from_safetensors_file(
            path,
            &[
                "mlp1.0.weight",
                "mlp1.0.bias",
                "mlp1.1.weight",
                "mlp1.1.bias",
                "mlp1.3.weight",
                "mlp1.3.bias",
            ],
        )?;
        Self::from_loaded_tensors(&tensors)
    }

    pub fn from_loaded_tensors(
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let norm_weight = find_tensor(tensors, "mlp1.0.weight")?;
        let norm_bias = find_tensor(tensors, "mlp1.0.bias")?;
        let fc1_weight = find_tensor(tensors, "mlp1.1.weight")?;
        let fc1_bias = find_tensor(tensors, "mlp1.1.bias")?;
        let fc2_weight = find_tensor(tensors, "mlp1.3.weight")?;
        let fc2_bias = find_tensor(tensors, "mlp1.3.bias")?;

        let input_dim = expect_1d("mlp1.0.weight", norm_weight)?;
        expect_1d_len("mlp1.0.bias", norm_bias, input_dim)?;
        let [hidden_dim, fc1_input_dim] = expect_2d("mlp1.1.weight", fc1_weight)?;
        if fc1_input_dim != input_dim {
            return Err(LocateAnythingError::Config(format!(
                "projector fc1 input dim mismatch: layer norm dim={input_dim}, fc1 input dim={fc1_input_dim}"
            )));
        }
        expect_1d_len("mlp1.1.bias", fc1_bias, hidden_dim)?;
        let [output_dim, fc2_input_dim] = expect_2d("mlp1.3.weight", fc2_weight)?;
        if fc2_input_dim != hidden_dim {
            return Err(LocateAnythingError::Config(format!(
                "projector fc2 input dim mismatch: hidden dim={hidden_dim}, fc2 input dim={fc2_input_dim}"
            )));
        }
        expect_1d_len("mlp1.3.bias", fc2_bias, output_dim)?;

        Ok(Self {
            config: ProjectorConfig {
                kind: "mlp".to_string(),
                input_dim,
                hidden_dim,
                output_dim,
            },
            norm_weight: norm_weight.data.clone(),
            norm_bias: norm_bias.data.clone(),
            fc1_weight: fc1_weight.data.clone(),
            fc1_bias: fc1_bias.data.clone(),
            fc2_weight: fc2_weight.data.clone(),
            fc2_bias: fc2_bias.data.clone(),
        })
    }
}

fn find_tensor<'a>(
    tensors: &'a [(String, LoadedTensorF32)],
    key: &str,
) -> LocateAnythingResult<&'a LoadedTensorF32> {
    tensors
        .iter()
        .find_map(|(name, tensor)| (name == key).then_some(tensor))
        .ok_or_else(|| LocateAnythingError::Runtime(format!("missing tensor `{key}`")))
}

fn expect_1d(key: &str, tensor: &LoadedTensorF32) -> LocateAnythingResult<usize> {
    if tensor.shape.len() != 1 {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected rank 1, got {:?}",
            tensor.shape
        )));
    }
    Ok(tensor.shape[0])
}

fn expect_1d_len(key: &str, tensor: &LoadedTensorF32, expected: usize) -> LocateAnythingResult<()> {
    let actual = expect_1d(key, tensor)?;
    if actual != expected {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected len {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn expect_2d(key: &str, tensor: &LoadedTensorF32) -> LocateAnythingResult<[usize; 2]> {
    if tensor.shape.len() != 2 {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected rank 2, got {:?}",
            tensor.shape
        )));
    }
    Ok([tensor.shape[0], tensor.shape[1]])
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
pub mod burn_projector {
    use burn::prelude::*;
    use burn::tensor::activation::gelu;

    use super::ProjectorWeights;

    #[derive(Debug)]
    pub struct BurnProjector<B: Backend> {
        pub weights: ProjectorWeights,
        norm_weight: Tensor<B, 1>,
        norm_bias: Tensor<B, 1>,
        fc1_weight: Tensor<B, 2>,
        fc1_bias: Tensor<B, 1>,
        fc2_weight: Tensor<B, 2>,
        fc2_bias: Tensor<B, 1>,
    }

    impl<B: Backend> BurnProjector<B> {
        pub fn from_weights(weights: ProjectorWeights, device: &B::Device) -> Self {
            let cfg = &weights.config;
            Self {
                norm_weight: tensor1(&weights.norm_weight, device),
                norm_bias: tensor1(&weights.norm_bias, device),
                fc1_weight: tensor2(&weights.fc1_weight, [cfg.hidden_dim, cfg.input_dim], device),
                fc1_bias: tensor1(&weights.fc1_bias, device),
                fc2_weight: tensor2(
                    &weights.fc2_weight,
                    [cfg.output_dim, cfg.hidden_dim],
                    device,
                ),
                fc2_bias: tensor1(&weights.fc2_bias, device),
                weights,
            }
        }

        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let [_tokens, channels] = input.dims();
            let hidden = self.weights.config.hidden_dim;
            let output = self.weights.config.output_dim;
            let (var, mean) = input.clone().var_mean_bias(1);
            let normalized = (input - mean) / var.add_scalar(1.0e-5).sqrt();
            let normalized = normalized * self.norm_weight.clone().reshape([1, channels])
                + self.norm_bias.clone().reshape([1, channels]);
            let projected = normalized.matmul(self.fc1_weight.clone().swap_dims(0, 1))
                + self.fc1_bias.clone().reshape([1, hidden]);
            gelu(projected).matmul(self.fc2_weight.clone().swap_dims(0, 1))
                + self.fc2_bias.clone().reshape([1, output])
        }
    }

    fn tensor1<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(values, device)
    }

    fn tensor2<B: Backend>(values: &[f32], shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(values, device).reshape(shape)
    }
}

#[cfg(all(
    test,
    any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    )
))]
mod tests {
    use std::time::Instant;

    use super::burn_projector::BurnProjector;
    use super::*;
    use crate::tensor_io::load_tensor_from_safetensors_file;

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn projector_matches_reference_hook_when_enabled() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        run_projector_parity::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn projector_matches_reference_hook_wgpu_when_enabled() {
        if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
            eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run WGPU projector parity");
            return;
        }
        let device = burn_wgpu::WgpuDevice::default();
        run_projector_parity::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn projector_matches_reference_hook_wgpu_f16_when_enabled() {
        if std::env::var("BURN_WGPU_F16_CORRECTNESS").is_err() {
            eprintln!("skipping: set BURN_WGPU_F16_CORRECTNESS=1 to run WGPU f16 projector parity");
            return;
        }
        let device = burn_wgpu::WgpuDevice::default();
        run_projector_parity::<burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>>(&device);
    }

    fn run_projector_parity<B: burn::prelude::Backend>(device: &B::Device) {
        if std::env::var("LOCATE_ANYTHING_PROJECTOR_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_PROJECTOR_PARITY=1 with LOCATE_ANYTHING_PROJECTOR_WEIGHTS and LOCATE_ANYTHING_PROJECTOR_HOOKS"
            );
            return;
        }
        let weights_path = std::env::var("LOCATE_ANYTHING_PROJECTOR_WEIGHTS")
            .expect("LOCATE_ANYTHING_PROJECTOR_WEIGHTS");
        let hooks_path = std::env::var("LOCATE_ANYTHING_PROJECTOR_HOOKS")
            .expect("LOCATE_ANYTHING_PROJECTOR_HOOKS");
        let token_limit = std::env::var("LOCATE_ANYTHING_PROJECTOR_TOKEN_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16);
        let mean_tolerance = env_f32("LOCATE_ANYTHING_PROJECTOR_MEAN_TOLERANCE", 2.0e-2);
        let rms_tolerance = env_f32("LOCATE_ANYTHING_PROJECTOR_RMS_TOLERANCE", 3.0e-2);
        let max_tolerance = env_f32("LOCATE_ANYTHING_PROJECTOR_MAX_TOLERANCE", 2.0e-1);
        let warmup_iters = env_usize("LOCATE_ANYTHING_PROJECTOR_WARMUP_ITERS", 0);

        let weights = ProjectorWeights::from_safetensors_file(weights_path).unwrap();
        let input = load_tensor_from_safetensors_file(&hooks_path, "vision.merged_tokens").unwrap();
        let reference = load_tensor_from_safetensors_file(&hooks_path, "projector.mlp1").unwrap();
        let input_shape: [usize; 2] = input.shape.clone().try_into().unwrap();
        let reference_shape: [usize; 2] = reference.shape.clone().try_into().unwrap();
        assert_eq!(input_shape[0], reference_shape[0]);
        assert_eq!(input_shape[1], weights.config.input_dim);
        assert_eq!(reference_shape[1], weights.config.output_dim);

        let rows = input_shape[0].min(token_limit.max(1));
        let input_data = input.data[..rows * input_shape[1]].to_vec();
        let reference_data = reference.data[..rows * reference_shape[1]].to_vec();
        let projector = BurnProjector::<B>::from_weights(weights, device);
        let input_tensor =
            burn::prelude::Tensor::<B, 1>::from_floats(input_data.as_slice(), device)
                .reshape([rows, input_shape[1]]);
        for _ in 0..warmup_iters {
            let _ = projector.forward(input_tensor.clone()).into_data();
        }
        let forward_started = Instant::now();
        let output = projector.forward(input_tensor);
        let output_data = output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data");
        let forward_readback_ms = forward_started.elapsed().as_secs_f64() * 1000.0;
        let stats = stats(&output_data, &reference_data);
        eprintln!(
            "projector parity rows={rows} forward_readback_ms={forward_readback_ms:.3} mean_abs={:.6e} rms={:.6e} max_abs={:.6e}",
            stats.mean_abs, stats.rms, stats.max_abs
        );
        assert!(
            stats.mean_abs < mean_tolerance,
            "projector mean_abs {:.6e} exceeded tolerance {:.6e}",
            stats.mean_abs,
            mean_tolerance
        );
        assert!(
            stats.rms < rms_tolerance,
            "projector rms {:.6e} exceeded tolerance {:.6e}",
            stats.rms,
            rms_tolerance
        );
        assert!(
            stats.max_abs < max_tolerance,
            "projector max_abs {:.6e} exceeded tolerance {:.6e}",
            stats.max_abs,
            max_tolerance
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct Stats {
        mean_abs: f32,
        rms: f32,
        max_abs: f32,
    }

    fn stats(lhs: &[f32], rhs: &[f32]) -> Stats {
        assert_eq!(lhs.len(), rhs.len());
        let mut sum_abs = 0.0;
        let mut sum_sq = 0.0;
        let mut max_abs = 0.0f32;
        for (&left, &right) in lhs.iter().zip(rhs.iter()) {
            let diff = left - right;
            let abs = diff.abs();
            sum_abs += abs;
            sum_sq += diff * diff;
            max_abs = max_abs.max(abs);
        }
        let len = lhs.len().max(1) as f32;
        Stats {
            mean_abs: sum_abs / len,
            rms: (sum_sq / len).sqrt(),
            max_abs,
        }
    }

    fn env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(default)
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    }
}
