use burn::{
    prelude::*,
    serde::{Deserialize, Serialize},
};

#[derive(Clone, Copy, Debug, Module, Serialize, Deserialize)]
pub struct RopeConfig {
    pub base_frequency: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            base_frequency: 100.0,
        }
    }
}

pub struct RotaryEmbedding;

impl RotaryEmbedding {
    pub fn apply<B: Backend>(
        tokens: Tensor<B, 4>,
        positions: &Tensor<B, 3>,
        config: RopeConfig,
    ) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<4>();
        assert!(
            dims[3].is_multiple_of(4),
            "RoPE expects the token dimension to be divisible by 4"
        );
        let batch = dims[0];
        let heads = dims[1];
        let token_count = dims[2];
        let head_dim = dims[3];
        let half_dim = head_dim / 2;
        let device = tokens.device();
        let steps = half_dim / 2;
        let inv_freq = Self::inv_frequencies(half_dim, config.base_frequency);
        let inv_freq =
            Tensor::<B, 1>::from_floats(inv_freq.as_slice(), &device).reshape([1, steps as i32]);

        let y_coords = positions
            .clone()
            .slice([0..batch as i32, 0..token_count as i32, 0..1])
            .reshape([(batch * token_count) as i32, 1]);
        let x_coords = positions
            .clone()
            .slice([0..batch as i32, 0..token_count as i32, 1..2])
            .reshape([(batch * token_count) as i32, 1]);

        let y_angles = y_coords.matmul(inv_freq.clone()).reshape([
            batch as i32,
            token_count as i32,
            steps as i32,
        ]);
        let x_angles =
            x_coords
                .matmul(inv_freq)
                .reshape([batch as i32, token_count as i32, steps as i32]);

        // HF DINOv3 layout:
        // angles_half = [y_freqs..., x_freqs...]
        // angles = [angles_half, angles_half]
        let half_angles: Tensor<B, 3> = Tensor::cat(vec![y_angles, x_angles], 2);
        let angles: Tensor<B, 3> = Tensor::cat(vec![half_angles.clone(), half_angles], 2);
        let angles: Tensor<B, 4> = angles.unsqueeze_dim(1).repeat_dim(1, heads);
        let cos = angles.clone().cos();
        let sin = angles.sin();
        let rotated = Self::rotate(tokens.clone());
        tokens * cos + rotated * sin
    }

    fn rotate<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<4>();
        let half = dims[3] / 2;
        let first = tokens.clone().slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            0..half as i32,
        ]);
        let second = tokens.slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            half as i32..dims[3] as i32,
        ]);
        let neg_second = second.mul_scalar(-1.0);
        Tensor::cat(vec![neg_second, first], 3)
    }

    fn inv_frequencies(feature_dim: usize, base: f32) -> Vec<f32> {
        let mut inv = Vec::with_capacity(feature_dim / 2);
        let denom = feature_dim as f32;
        for idx in (0..feature_dim).step_by(2) {
            let exponent = idx as f32 / denom;
            let freq = base.powf(exponent);
            inv.push(1.0 / freq);
        }
        inv
    }
}
