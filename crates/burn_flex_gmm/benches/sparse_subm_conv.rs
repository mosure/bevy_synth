use burn_flex_gmm::{
    SparseSubmConvConfig, SparseSubmConvWeights, build_neighbor_rows, pack_flex_weight,
    sparse_subm_conv_forward_flex, sparse_subm_conv_forward_flex_precomputed,
    sparse_subm_conv_forward_legacy,
};
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "wgpu-kernel")]
use burn::tensor::Tensor;
#[cfg(feature = "wgpu-kernel")]
use burn_flex_gmm::wgpu::{
    DefaultWgpuBackend, clear_neighbor_rows_tensor_cache, neighbor_rows_tensor_from_coords,
    sparse_subm_conv_forward_wgpu,
};

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

fn line_coords(count: usize) -> Vec<[u32; 4]> {
    (0..count as u32).map(|x| [0, x, 0, 0]).collect()
}

fn bench_sparse_subm_conv(c: &mut Criterion) {
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(4096);
    let mut rng = Lcg::new(7);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();
    let weights = SparseSubmConvWeights {
        weight: &weight,
        bias: &bias,
    };

    let mut group = c.benchmark_group("sparse_subm_conv");
    group.bench_function("legacy", |b| {
        b.iter(|| {
            let _ = sparse_subm_conv_forward_legacy(&cfg, weights, &coords, &input).unwrap();
        })
    });
    group.bench_function("flex_gemm", |b| {
        b.iter(|| {
            let _ = sparse_subm_conv_forward_flex(&cfg, weights, &coords, &input).unwrap();
        })
    });
    let neighbor_rows = build_neighbor_rows(&cfg, coords.as_slice()).expect("neighbor rows");
    let packed_weight = pack_flex_weight(&cfg, weight.as_slice()).expect("packed weight");
    group.bench_function("flex_gemm_precomputed", |b| {
        b.iter(|| {
            let _ = sparse_subm_conv_forward_flex_precomputed(
                &cfg,
                weights,
                input.as_slice(),
                neighbor_rows.as_slice(),
                Some(packed_weight.as_slice()),
            )
            .unwrap();
        })
    });
    group.bench_function("flex_gemm_uncached_x24", |b| {
        b.iter(|| {
            for _ in 0..24 {
                let _ = sparse_subm_conv_forward_flex(&cfg, weights, &coords, &input).unwrap();
            }
        })
    });
    group.bench_function("flex_gemm_precomputed_x24", |b| {
        b.iter(|| {
            for _ in 0..24 {
                let _ = sparse_subm_conv_forward_flex_precomputed(
                    &cfg,
                    weights,
                    input.as_slice(),
                    neighbor_rows.as_slice(),
                    Some(packed_weight.as_slice()),
                )
                .unwrap();
            }
        })
    });
    #[cfg(feature = "wgpu-kernel")]
    {
        let device = burn_wgpu::WgpuDevice::default();
        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input.as_slice(), &device)
            .reshape([coords.len(), cfg.in_channels]);
        let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(weight.as_slice(), &device)
            .reshape([
                cfg.out_channels,
                cfg.kernel_d,
                cfg.kernel_h,
                cfg.kernel_w,
                cfg.in_channels_per_group,
            ]);
        let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(bias.as_slice(), &device);
        let neighbor_t = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("neighbor row tensor");

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "host");
        }
        group.bench_function("wgpu_neighbor_host_uncached", |b| {
            b.iter(|| {
                clear_neighbor_rows_tensor_cache();
                let tensor = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
                    .expect("host neighbor tensor");
                let _ = tensor.to_data();
            })
        });
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "device");
        }
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO", "scan");
        }
        group.bench_function("wgpu_neighbor_device_scan_uncached", |b| {
            b.iter(|| {
                clear_neighbor_rows_tensor_cache();
                let tensor = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
                    .expect("device scan neighbor tensor");
                let _ = tensor.to_data();
            })
        });
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO", "hash");
        }
        group.bench_function("wgpu_neighbor_device_hash_uncached", |b| {
            b.iter(|| {
                clear_neighbor_rows_tensor_cache();
                let tensor = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
                    .expect("device hash neighbor tensor");
                let _ = tensor.to_data();
            })
        });
        clear_neighbor_rows_tensor_cache();
        let _ = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("cached warmup");
        group.bench_function("wgpu_neighbor_device_cached_hit", |b| {
            b.iter(|| {
                let tensor = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
                    .expect("device cached tensor");
                let _ = tensor.to_data();
            })
        });

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "baseline");
        }
        group.bench_function("wgpu_kernel_baseline", |b| {
            b.iter(|| {
                let out = sparse_subm_conv_forward_wgpu(
                    &cfg,
                    input_t.clone(),
                    neighbor_t.clone(),
                    weight_t.clone(),
                    bias_t.clone(),
                )
                .expect("wgpu kernel");
                let _ = out.to_data();
            })
        });
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "fused_oc4");
        }
        group.bench_function("wgpu_kernel_fused_oc4", |b| {
            b.iter(|| {
                let out = sparse_subm_conv_forward_wgpu(
                    &cfg,
                    input_t.clone(),
                    neighbor_t.clone(),
                    weight_t.clone(),
                    bias_t.clone(),
                )
                .expect("wgpu fused-oc4 kernel");
                let _ = out.to_data();
            })
        });
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "baseline");
            std::env::set_var("BURN_FLEX_GMM_WGPU_SPLIT_K", "4");
        }
        group.bench_function("wgpu_kernel_splitk", |b| {
            b.iter(|| {
                let out = sparse_subm_conv_forward_wgpu(
                    &cfg,
                    input_t.clone(),
                    neighbor_t.clone(),
                    weight_t.clone(),
                    bias_t.clone(),
                )
                .expect("wgpu split-k kernel");
                let _ = out.to_data();
            })
        });
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_SPLIT_K");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_KERNEL");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO");
        }
        clear_neighbor_rows_tensor_cache();
    }
    group.finish();
}

criterion_group!(benches, bench_sparse_subm_conv);
criterion_main!(benches);
