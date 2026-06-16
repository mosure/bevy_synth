use burn::{
    backend::NdArray,
    tensor::{Distribution, Tensor},
};
use burn_triposplat::{
    ElasticGaussianFixedlenDecoderConfig, FlowState, LatentSeqMmFlowModelConfig,
    OctreeGaussianDecoder, OctreeProbabilityFixedlenDecoderConfig, OctreeSample,
    TripoSplatCondition, TripoSplatProfile,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

type BenchBackend = NdArray<f32>;

fn bench_tiny_flow_forward(c: &mut Criterion) {
    let device = Default::default();
    let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
    let model = config.clone().init::<BenchBackend>(&device);
    let state = FlowState::random(
        &device,
        1,
        config.q_token_length,
        config.in_channels,
        config.cam_channels,
    );
    let cond = TripoSplatCondition {
        feature1: Tensor::random(
            [1, 6, config.cond_channels],
            Distribution::Normal(0.0, 1.0),
            &device,
        ),
        feature2: config.cond2_channels.map(|channels| {
            Tensor::random([1, 5, channels], Distribution::Normal(0.0, 1.0), &device)
        }),
        rng_normals_consumed: 0,
    };
    let timestep = Tensor::<BenchBackend, 1>::from_floats([0.5], &device);

    c.bench_function("triposplat/tiny_flow_forward", |b| {
        b.iter(|| {
            let out = model.forward(
                black_box(state.clone()),
                black_box(timestep.clone()),
                black_box(cond.clone()),
            );
            black_box(out.latent.dims())
        })
    });
}

fn bench_tiny_profile_flow_sample(c: &mut Criterion) {
    let device = Default::default();
    let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
    let model = config.clone().init::<BenchBackend>(&device);
    let state = FlowState::random(
        &device,
        1,
        config.q_token_length,
        config.in_channels,
        config.cam_channels,
    );
    let cond = TripoSplatCondition {
        feature1: Tensor::random(
            [1, 6, config.cond_channels],
            Distribution::Normal(0.0, 1.0),
            &device,
        ),
        feature2: config.cond2_channels.map(|channels| {
            Tensor::random([1, 5, channels], Distribution::Normal(0.0, 1.0), &device)
        }),
        rng_normals_consumed: 0,
    };

    for profile in [
        TripoSplatProfile::Low,
        TripoSplatProfile::Balanced,
        TripoSplatProfile::High,
    ] {
        let settings = profile.settings();
        c.bench_function(
            &format!("triposplat/profile_{profile:?}/tiny_flow_sample"),
            |b| {
                b.iter(|| {
                    let out = model.sample_euler_cfg_prefix(
                        black_box(state.clone()),
                        black_box(cond.clone()),
                        black_box(settings.steps),
                        black_box(settings.steps),
                        black_box(settings.guidance_scale),
                        black_box(3.0),
                    );
                    black_box(out.latent.dims())
                })
            },
        );
    }
}

fn bench_tiny_octree_forward(c: &mut Criterion) {
    let device = Default::default();
    let config = OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests();
    let model = config.clone().init::<BenchBackend>(&device);
    let coords = Tensor::random([1, 16, 3], Distribution::Normal(0.0, 1.0), &device);
    let level = Tensor::<BenchBackend, 1>::from_floats([5.0], &device);
    let cond = Tensor::random(
        [1, 8, config.cond_channels],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    c.bench_function("triposplat/tiny_octree_forward", |b| {
        b.iter(|| {
            let out = model.forward(
                black_box(coords.clone()),
                black_box(level.clone()),
                black_box(cond.clone()),
                None,
            );
            black_box(out.logits.dims())
        })
    });
}

fn bench_tiny_gaussian_forward(c: &mut Criterion) {
    let device = Default::default();
    let config = ElasticGaussianFixedlenDecoderConfig::tiny_for_tests();
    let model = config.clone().init::<BenchBackend>(&device);
    let sample = OctreeSample {
        points: Tensor::random([1, 8, 3], Distribution::Normal(0.0, 1.0), &device),
        log_probs: Tensor::zeros([1, 8], &device),
    };
    let cond = Tensor::random(
        [1, 8, config.cond_channels],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    c.bench_function("triposplat/tiny_gaussian_forward", |b| {
        b.iter(|| {
            let out = model.forward(black_box(&sample), black_box(cond.clone()));
            black_box(out.dims())
        })
    });
}

fn bench_tiny_profile_decode_cloud(c: &mut Criterion) {
    let device = <BenchBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let decoder = OctreeGaussianDecoder::<BenchBackend>::new(
        &device,
        OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
        ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
    );
    let latent = Tensor::random([1, 8, 4], Distribution::Normal(0.0, 1.0), &device);

    for profile in [
        TripoSplatProfile::Low,
        TripoSplatProfile::Balanced,
        TripoSplatProfile::High,
    ] {
        let settings = profile.settings();
        c.bench_function(
            &format!("triposplat/profile_{profile:?}/tiny_decode_cloud"),
            |b| {
                b.iter(|| {
                    let cloud = decoder
                        .decode_to_cloud_with_seed(
                            black_box(latent.clone()),
                            black_box(settings.num_gaussians),
                            black_box(42),
                        )
                        .expect("profile decode benchmark should produce a splat cloud");
                    black_box(cloud.len())
                })
            },
        );
    }
}

criterion_group! {
    name = triposplat_stage_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_tiny_flow_forward, bench_tiny_profile_flow_sample, bench_tiny_octree_forward, bench_tiny_gaussian_forward, bench_tiny_profile_decode_cloud
}
criterion_main!(triposplat_stage_benches);
