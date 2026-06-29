use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::BackendTypes,
};
use burn_synth_render::{
    CameraIntrinsics, ObjectTransformTensors, ObjectTransformValues, SoftRenderConfig,
    cpu_render_soft_surface, scalar_tensor, soft_silhouette_depth_loss, tensor_from_image,
    tensor_from_points,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

type BenchBackend = NdArray<f32>;
type BenchAutodiffBackend = Autodiff<BenchBackend>;

fn fixture_points(count: usize) -> Vec<[f32; 3]> {
    let side = (count as f32).sqrt().ceil() as usize;
    let mut points = Vec::with_capacity(count);
    for y in 0..side {
        for x in 0..side {
            if points.len() == count {
                break;
            }
            let xf = x as f32 / side.max(1) as f32 - 0.5;
            let yf = y as f32 / side.max(1) as f32 - 0.5;
            points.push([xf * 0.7, yf * 0.45, (xf * yf) * 0.05]);
        }
    }
    points
}

fn camera(width: usize, height: usize) -> CameraIntrinsics {
    CameraIntrinsics {
        fx: width as f32 * 1.1,
        fy: height as f32 * 1.45,
        cx: width as f32 * 0.5,
        cy: height as f32 * 0.5,
        width,
        height,
    }
}

fn config(width: usize, height: usize) -> SoftRenderConfig {
    SoftRenderConfig {
        width,
        height,
        sigma_px: 2.0,
        depth_sigma_m: 0.08,
        mask_weight: 1.0,
        depth_weight: 0.2,
    }
}

fn bench_soft_renderer(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_point_renderer");
    for (points_count, width, height) in [(64, 32, 24), (256, 48, 36)] {
        let points = fixture_points(points_count);
        let transform = ObjectTransformValues {
            tx: 0.04,
            ty: -0.01,
            tz: 2.4,
            yaw: 0.3,
            scale: 1.05,
        };
        let cfg = config(width, height);
        let cam = camera(width, height);
        let (target_mask, target_depth) = cpu_render_soft_surface(&points, transform, cam, cfg);

        group.bench_function(
            format!("forward_loss_{points_count}_{width}x{height}"),
            |b| {
                let device = <BenchBackend as BackendTypes>::Device::default();
                let point_tensor = tensor_from_points::<BenchBackend>(&points, &device);
                let mask = tensor_from_image::<BenchBackend>(&target_mask, width, height, &device);
                let depth =
                    tensor_from_image::<BenchBackend>(&target_depth, width, height, &device);
                let tensors =
                    ObjectTransformTensors::<BenchBackend>::from_values(transform, &device);
                b.iter(|| {
                    let loss = soft_silhouette_depth_loss(
                        black_box(point_tensor.clone()),
                        black_box(&tensors),
                        black_box(cam),
                        black_box(cfg),
                        black_box(mask.clone()),
                        black_box(depth.clone()),
                    );
                    black_box(loss.into_scalar());
                });
            },
        );

        group.bench_function(
            format!("backward_transform_{points_count}_{width}x{height}"),
            |b| {
                let device = <BenchAutodiffBackend as BackendTypes>::Device::default();
                let point_tensor = tensor_from_points::<BenchAutodiffBackend>(&points, &device);
                let mask =
                    tensor_from_image::<BenchAutodiffBackend>(&target_mask, width, height, &device);
                let depth = tensor_from_image::<BenchAutodiffBackend>(
                    &target_depth,
                    width,
                    height,
                    &device,
                );
                b.iter(|| {
                    let tensors = ObjectTransformTensors::<BenchAutodiffBackend> {
                        tx: scalar_tensor(transform.tx + 0.02, &device).require_grad(),
                        ty: scalar_tensor(transform.ty - 0.01, &device).require_grad(),
                        tz: scalar_tensor(transform.tz + 0.05, &device).require_grad(),
                        yaw: scalar_tensor(transform.yaw - 0.1, &device).require_grad(),
                        scale: scalar_tensor(transform.scale * 0.98, &device).require_grad(),
                    };
                    let loss = soft_silhouette_depth_loss(
                        black_box(point_tensor.clone()),
                        black_box(&tensors),
                        black_box(cam),
                        black_box(cfg),
                        black_box(mask.clone()),
                        black_box(depth.clone()),
                    );
                    black_box(loss.backward());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_soft_renderer);
criterion_main!(benches);
