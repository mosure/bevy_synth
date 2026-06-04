use burn::{module::Param, nn::Initializer, prelude::*, tensor::FloatDType};

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    pub dim: usize,
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm::new(device, self)
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
    epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(device: &B::Device, config: &LayerNormConfig) -> Self {
        let gamma = Initializer::Ones.init([config.dim], device);
        let beta = Initializer::Zeros.init([config.dim], device);

        Self {
            gamma,
            beta,
            epsilon: config.epsilon,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype: FloatDType = x.dtype().into();
        let acc_dtype = accumulation_dtype(dtype);
        let x_acc = cast_low_precision_to_f32(x, dtype);
        let (var, mean) = x_acc.clone().var_mean_bias(D - 1);
        let input_normalized = x_acc.sub(mean).div(var.add_scalar(self.epsilon).sqrt());

        let out = input_normalized
            .mul(self.gamma.val().unsqueeze().cast(acc_dtype))
            .add(self.beta.val().unsqueeze().cast(acc_dtype));
        cast_from_f32_accum(out, dtype)
    }
}

fn cast_low_precision_to_f32<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        tensor.cast(FloatDType::F32)
    } else {
        tensor
    }
}

fn accumulation_dtype(dtype: FloatDType) -> FloatDType {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        FloatDType::F32
    } else {
        dtype
    }
}

fn cast_from_f32_accum<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        tensor.cast(dtype)
    } else {
        tensor
    }
}
