use burn::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_unit_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        (value as f32) / ((1u32 << 24) as f32)
    }
}

pub(crate) fn push_standard_normals(rng: &mut SplitMix64, values: &mut Vec<f32>, total: usize) {
    while values.len() < total {
        let u1 = rng.next_unit_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_unit_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        values.push(radius * theta.cos());
        if values.len() < total {
            values.push(radius * theta.sin());
        }
    }
}

pub(crate) fn deterministic_standard_normal_3d<B: Backend>(
    rng: &mut SplitMix64,
    shape: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 3> {
    let total = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    if total == 0 {
        return Tensor::zeros(shape, device);
    }
    let mut values = Vec::with_capacity(total);
    push_standard_normals(rng, &mut values, total);
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape(shape)
}
