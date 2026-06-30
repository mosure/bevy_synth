use burn_flex_gmm::{
    SparsePatchify3dConfig, SparsePatchify3dWeights, sparse_patchify3d_forward_flex,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.state >> 40) as u32) | 1;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn bench_sparse_patchify3d(c: &mut Criterion) {
    let cfg = SparsePatchify3dConfig {
        in_channels: 3,
        out_channels: 768,
        frames: 16,
        height: 224,
        width: 224,
        tubelet_size: 2,
        patch_h: 16,
        patch_w: 16,
    };
    let mut rng = Lcg::new(20260509);
    let input: Vec<f32> = (0..cfg.in_channels * cfg.frames * cfg.height * cfg.width)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.in_channels * cfg.tubelet_size * cfg.patch_h * cfg.patch_w;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();
    let weights = SparsePatchify3dWeights {
        weight: &weight,
        bias: &bias,
    };
    let mut group = c.benchmark_group("sparse_patchify3d_cpu");
    for keep_tokens in [16usize, 64, 256, 784] {
        let coords = evenly_spaced_coords(&cfg, keep_tokens);
        group.throughput(Throughput::Elements(coords.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_tokens", coords.len())),
            &coords,
            |b, coords| {
                b.iter(|| {
                    sparse_patchify3d_forward_flex(&cfg, weights, coords, &input)
                        .expect("sparse patchify")
                })
            },
        );
    }
    group.finish();
}

fn evenly_spaced_coords(cfg: &SparsePatchify3dConfig, keep_tokens: usize) -> Vec<[u32; 4]> {
    let grid_t = cfg.frames / cfg.tubelet_size;
    let grid_h = cfg.height / cfg.patch_h;
    let grid_w = cfg.width / cfg.patch_w;
    let dense_len = grid_t * grid_h * grid_w;
    let keep_tokens = keep_tokens.max(1).min(dense_len);
    let last = dense_len.saturating_sub(1);
    (0..keep_tokens)
        .map(|i| ((i * last) + (keep_tokens / 2)) / keep_tokens.max(1))
        .map(|index| {
            let frame = index / (grid_h * grid_w);
            let rem = index - frame * grid_h * grid_w;
            let row = rem / grid_w;
            let col = rem - row * grid_w;
            [0, frame as u32, row as u32, col as u32]
        })
        .collect()
}

criterion_group!(benches, bench_sparse_patchify3d);
criterion_main!(benches);
