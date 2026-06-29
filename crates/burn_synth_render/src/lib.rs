use burn::tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

pub mod normal;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftRenderConfig {
    pub width: usize,
    pub height: usize,
    pub sigma_px: f32,
    pub depth_sigma_m: f32,
    pub mask_weight: f32,
    pub depth_weight: f32,
}

impl Default for SoftRenderConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 24,
            sigma_px: 2.0,
            depth_sigma_m: 0.08,
            mask_weight: 1.0,
            depth_weight: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectTransformValues {
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub yaw: f32,
    pub scale: f32,
}

impl Default for ObjectTransformValues {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 2.5,
            yaw: 0.0,
            scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObjectTransformTensors<B: Backend> {
    pub tx: Tensor<B, 1>,
    pub ty: Tensor<B, 1>,
    pub tz: Tensor<B, 1>,
    pub yaw: Tensor<B, 1>,
    pub scale: Tensor<B, 1>,
}

impl<B: Backend> ObjectTransformTensors<B> {
    pub fn from_values(values: ObjectTransformValues, device: &B::Device) -> Self {
        Self {
            tx: scalar_tensor(values.tx, device),
            ty: scalar_tensor(values.ty, device),
            tz: scalar_tensor(values.tz, device),
            yaw: scalar_tensor(values.yaw, device),
            scale: scalar_tensor(values.scale, device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SoftRenderOutput<B: Backend> {
    pub mask: Tensor<B, 2>,
    pub depth: Tensor<B, 2>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderMetrics {
    pub mask_mse: f32,
    pub mask_psnr_db: f32,
    pub threshold_iou: f32,
    pub depth_masked_mae: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftPoseOptimizationConfig {
    pub iterations: usize,
    pub learning_rate_translation: f32,
    pub learning_rate_yaw: f32,
    pub learning_rate_scale: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub max_translation_step: f32,
    pub optimize_ty: bool,
}

impl Default for SoftPoseOptimizationConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            learning_rate_translation: 0.006,
            learning_rate_yaw: 0.003,
            learning_rate_scale: 0.003,
            min_scale: 0.05,
            max_scale: 20.0,
            max_translation_step: 0.05,
            optimize_ty: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftPoseOptimizationStep {
    pub iteration: usize,
    pub loss: f32,
    pub transform: ObjectTransformValues,
    pub gradient: ObjectTransformValues,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftPoseOptimizationResult {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub transform: ObjectTransformValues,
    pub steps: Vec<SoftPoseOptimizationStep>,
}

pub fn scalar_tensor<B: Backend>(value: f32, device: &B::Device) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([value], device)
}

pub fn tensor_from_points<B: Backend>(points: &[[f32; 3]], device: &B::Device) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(points.len() * 3);
    for point in points {
        values.extend_from_slice(point);
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([points.len(), 3])
}

pub fn tensor_from_image<B: Backend>(
    image: &[f32],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert_eq!(image.len(), width * height);
    Tensor::<B, 1>::from_floats(image, device).reshape([height, width])
}

pub fn image_from_tensor<B: Backend>(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("tensor data should be readable as f32")
}

pub fn transform_points<B: Backend>(
    local_points: Tensor<B, 2>,
    transform: &ObjectTransformTensors<B>,
) -> Tensor<B, 2> {
    let [points, _] = local_points.shape().dims();
    let x = local_points.clone().slice([0..points, 0..1]);
    let y = local_points.clone().slice([0..points, 1..2]);
    let z = local_points.slice([0..points, 2..3]);

    let shape = [points, 1];
    let scale = transform.scale.clone().reshape([1, 1]).expand(shape);
    let tx = transform.tx.clone().reshape([1, 1]).expand(shape);
    let ty = transform.ty.clone().reshape([1, 1]).expand(shape);
    let tz = transform.tz.clone().reshape([1, 1]).expand(shape);
    let cos = transform.yaw.clone().cos().reshape([1, 1]).expand(shape);
    let sin = transform.yaw.clone().sin().reshape([1, 1]).expand(shape);

    let x_scaled = x.mul(scale.clone());
    let y_scaled = y.mul(scale.clone());
    let z_scaled = z.mul(scale);

    let world_x = x_scaled.clone().mul(cos.clone()) + z_scaled.clone().mul(sin.clone()) + tx;
    let world_y = y_scaled + ty;
    let world_z = z_scaled.mul(cos) - x_scaled.mul(sin) + tz;

    Tensor::cat(vec![world_x, world_y, world_z], 1)
}

pub fn project_points<B: Backend>(
    camera_points: Tensor<B, 2>,
    intrinsics: CameraIntrinsics,
) -> Tensor<B, 2> {
    let [points, _] = camera_points.shape().dims();
    let x = camera_points.clone().slice([0..points, 0..1]);
    let y = camera_points.clone().slice([0..points, 1..2]);
    let z = camera_points.slice([0..points, 2..3]).add_scalar(1.0e-6);

    let px = x.div(z.clone()).mul_scalar(intrinsics.fx as f64) + intrinsics.cx as f64;
    let py = y.div(z).mul_scalar(intrinsics.fy as f64) + intrinsics.cy as f64;
    Tensor::cat(vec![px, py], 1)
}

pub fn render_soft_surface<B: Backend>(
    local_points: Tensor<B, 2>,
    transform: &ObjectTransformTensors<B>,
    intrinsics: CameraIntrinsics,
    config: SoftRenderConfig,
) -> SoftRenderOutput<B> {
    let camera_points = transform_points(local_points, transform);
    let [points, _] = camera_points.shape().dims();
    let projected = project_points(camera_points.clone(), intrinsics);
    let depth = camera_points.slice([0..points, 2..3]);

    let grid = pixel_grid::<B>(config.width, config.height, &projected.device());
    let pixels = config.width * config.height;
    let projected_expanded = projected.unsqueeze_dim::<3>(1).expand([points, pixels, 2]);
    let grid_expanded = grid.unsqueeze_dim::<3>(0).expand([points, pixels, 2]);
    let diff = projected_expanded - grid_expanded;
    let dist2 = diff
        .powi_scalar(2)
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        .div_scalar(2.0 * (config.sigma_px as f64) * (config.sigma_px as f64));
    let density = dist2.mul_scalar(-1.0).exp();
    let density_sum = density.clone().sum_dim(0).squeeze_dim::<1>(0);
    let mask = density_sum
        .clone()
        .mul_scalar(-1.0)
        .exp()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .reshape([config.height, config.width]);

    let depth_weights = density * depth.reshape([points, 1]).expand([points, pixels]);
    let weighted_depth = depth_weights.sum_dim(0).squeeze_dim::<1>(0);
    let depth_image = weighted_depth
        .div(density_sum.add_scalar(1.0e-6))
        .reshape([config.height, config.width]);

    SoftRenderOutput {
        mask,
        depth: depth_image,
    }
}

pub fn soft_silhouette_depth_loss<B: Backend>(
    local_points: Tensor<B, 2>,
    transform: &ObjectTransformTensors<B>,
    intrinsics: CameraIntrinsics,
    config: SoftRenderConfig,
    target_mask: Tensor<B, 2>,
    target_depth: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let rendered = render_soft_surface(local_points, transform, intrinsics, config);
    let mask_loss = (rendered.mask.clone() - target_mask.clone())
        .powi_scalar(2)
        .mean();
    let depth_delta = (rendered.depth - target_depth)
        .div_scalar(config.depth_sigma_m.max(1.0e-5) as f64)
        .powi_scalar(2)
        .mul(target_mask.clone());
    let depth_loss = depth_delta.sum().div(target_mask.sum().add_scalar(1.0e-6));
    mask_loss.mul_scalar(config.mask_weight as f64)
        + depth_loss.mul_scalar(config.depth_weight as f64)
}

pub fn compute_render_metrics(
    predicted_mask: &[f32],
    predicted_depth: &[f32],
    target_mask: &[f32],
    target_depth: &[f32],
) -> RenderMetrics {
    assert_eq!(predicted_mask.len(), target_mask.len());
    assert_eq!(predicted_depth.len(), target_depth.len());
    assert_eq!(predicted_mask.len(), predicted_depth.len());

    let mut mask_mse = 0.0;
    let mut intersection = 0usize;
    let mut union = 0usize;
    let mut depth_abs = 0.0;
    let mut depth_count = 0usize;
    for i in 0..predicted_mask.len() {
        let delta = predicted_mask[i] - target_mask[i];
        mask_mse += delta * delta;
        let pred_on = predicted_mask[i] >= 0.5;
        let target_on = target_mask[i] >= 0.5;
        if pred_on && target_on {
            intersection += 1;
        }
        if pred_on || target_on {
            union += 1;
        }
        if target_on {
            depth_abs += (predicted_depth[i] - target_depth[i]).abs();
            depth_count += 1;
        }
    }
    mask_mse /= predicted_mask.len().max(1) as f32;
    let mask_psnr_db = if mask_mse <= 1.0e-12 {
        120.0
    } else {
        10.0 * (1.0 / mask_mse).log10()
    };
    let threshold_iou = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };
    let depth_masked_mae = if depth_count == 0 {
        0.0
    } else {
        depth_abs / depth_count as f32
    };

    RenderMetrics {
        mask_mse,
        mask_psnr_db,
        threshold_iou,
        depth_masked_mae,
    }
}

pub fn cpu_render_soft_surface(
    local_points: &[[f32; 3]],
    transform: ObjectTransformValues,
    intrinsics: CameraIntrinsics,
    config: SoftRenderConfig,
) -> (Vec<f32>, Vec<f32>) {
    let mut projected = Vec::with_capacity(local_points.len());
    let cos = transform.yaw.cos();
    let sin = transform.yaw.sin();
    for point in local_points {
        let x = point[0] * transform.scale;
        let y = point[1] * transform.scale;
        let z = point[2] * transform.scale;
        let world_x = x * cos + z * sin + transform.tx;
        let world_y = y + transform.ty;
        let world_z = z * cos - x * sin + transform.tz;
        let safe_z = world_z + 1.0e-6;
        projected.push((
            world_x / safe_z * intrinsics.fx + intrinsics.cx,
            world_y / safe_z * intrinsics.fy + intrinsics.cy,
            world_z,
        ));
    }

    let pixels = config.width * config.height;
    let mut mask = vec![0.0; pixels];
    let mut depth_weighted = vec![0.0; pixels];
    let mut density_sum = vec![0.0; pixels];
    let denom = 2.0 * config.sigma_px * config.sigma_px;
    for y in 0..config.height {
        for x in 0..config.width {
            let idx = y * config.width + x;
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            for &(point_x, point_y, point_z) in &projected {
                let dx = point_x - px;
                let dy = point_y - py;
                let density = (-(dx * dx + dy * dy) / denom).exp();
                density_sum[idx] += density;
                depth_weighted[idx] += density * point_z;
            }
            mask[idx] = 1.0 - (-density_sum[idx]).exp();
        }
    }

    let mut depth = vec![0.0; pixels];
    for i in 0..pixels {
        depth[i] = depth_weighted[i] / (density_sum[i] + 1.0e-6);
    }

    (mask, depth)
}

pub fn cpu_soft_silhouette_depth_loss(
    local_points: &[[f32; 3]],
    transform: ObjectTransformValues,
    intrinsics: CameraIntrinsics,
    config: SoftRenderConfig,
    target_mask: &[f32],
    target_depth: &[f32],
) -> f32 {
    let (mask, depth) = cpu_render_soft_surface(local_points, transform, intrinsics, config);
    let mask_loss = mask
        .iter()
        .zip(target_mask.iter())
        .map(|(actual, target)| {
            let delta = actual - target;
            delta * delta
        })
        .sum::<f32>()
        / mask.len().max(1) as f32;
    let mut depth_loss = 0.0;
    let mut target_sum = 0.0;
    for i in 0..mask.len() {
        let delta = (depth[i] - target_depth[i]) / config.depth_sigma_m.max(1.0e-5);
        depth_loss += delta * delta * target_mask[i];
        target_sum += target_mask[i];
    }
    depth_loss /= target_sum + 1.0e-6;
    mask_loss * config.mask_weight + depth_loss * config.depth_weight
}

#[derive(Clone, Copy, Debug)]
pub struct CpuLossReference<'a> {
    pub local_points: &'a [[f32; 3]],
    pub intrinsics: CameraIntrinsics,
    pub config: SoftRenderConfig,
    pub target_mask: &'a [f32],
    pub target_depth: &'a [f32],
}

impl CpuLossReference<'_> {
    pub fn loss(self, transform: ObjectTransformValues) -> f32 {
        cpu_soft_silhouette_depth_loss(
            self.local_points,
            transform,
            self.intrinsics,
            self.config,
            self.target_mask,
            self.target_depth,
        )
    }
}

pub fn finite_difference_gradient(
    reference: CpuLossReference<'_>,
    transform: ObjectTransformValues,
    field: TransformField,
    epsilon: f32,
) -> f32 {
    let mut plus = transform;
    let mut minus = transform;
    match field {
        TransformField::Tx => {
            plus.tx += epsilon;
            minus.tx -= epsilon;
        }
        TransformField::Ty => {
            plus.ty += epsilon;
            minus.ty -= epsilon;
        }
        TransformField::Tz => {
            plus.tz += epsilon;
            minus.tz -= epsilon;
        }
        TransformField::Yaw => {
            plus.yaw += epsilon;
            minus.yaw -= epsilon;
        }
        TransformField::Scale => {
            plus.scale += epsilon;
            minus.scale -= epsilon;
        }
    }
    let plus_loss = reference.loss(plus);
    let minus_loss = reference.loss(minus);
    (plus_loss - minus_loss) / (2.0 * epsilon)
}

pub fn optimize_soft_pose_ndarray(
    local_points: &[[f32; 3]],
    initial: ObjectTransformValues,
    intrinsics: CameraIntrinsics,
    render_config: SoftRenderConfig,
    target_mask: &[f32],
    target_depth: &[f32],
    optimize_config: SoftPoseOptimizationConfig,
) -> SoftPoseOptimizationResult {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::BackendTypes;

    type Backend = Autodiff<NdArray<f32>>;

    let reference = CpuLossReference {
        local_points,
        intrinsics,
        config: render_config,
        target_mask,
        target_depth,
    };
    let initial_loss = reference.loss(initial);
    let device = <Backend as BackendTypes>::Device::default();
    let point_tensor = tensor_from_points::<Backend>(local_points, &device);
    let target_mask_tensor = tensor_from_image::<Backend>(
        target_mask,
        render_config.width,
        render_config.height,
        &device,
    );
    let target_depth_tensor = tensor_from_image::<Backend>(
        target_depth,
        render_config.width,
        render_config.height,
        &device,
    );
    let mut current = initial;
    let mut steps = Vec::with_capacity(optimize_config.iterations);

    for iteration in 0..optimize_config.iterations {
        let transform = ObjectTransformTensors::<Backend> {
            tx: scalar_tensor(current.tx, &device).require_grad(),
            ty: scalar_tensor(current.ty, &device).require_grad(),
            tz: scalar_tensor(current.tz, &device).require_grad(),
            yaw: scalar_tensor(current.yaw, &device).require_grad(),
            scale: scalar_tensor(current.scale, &device).require_grad(),
        };
        let loss = soft_silhouette_depth_loss(
            point_tensor.clone(),
            &transform,
            intrinsics,
            render_config,
            target_mask_tensor.clone(),
            target_depth_tensor.clone(),
        );
        let loss_value = loss.clone().inner().into_scalar();
        let grads = loss.backward();
        let gradient = ObjectTransformValues {
            tx: transform.tx.grad(&grads).unwrap().into_scalar(),
            ty: transform.ty.grad(&grads).unwrap().into_scalar(),
            tz: transform.tz.grad(&grads).unwrap().into_scalar(),
            yaw: transform.yaw.grad(&grads).unwrap().into_scalar(),
            scale: transform.scale.grad(&grads).unwrap().into_scalar(),
        };
        steps.push(SoftPoseOptimizationStep {
            iteration,
            loss: loss_value,
            transform: current,
            gradient,
        });

        current.tx -= clipped_step(
            gradient.tx,
            optimize_config.learning_rate_translation,
            optimize_config.max_translation_step,
        );
        if optimize_config.optimize_ty {
            current.ty -= clipped_step(
                gradient.ty,
                optimize_config.learning_rate_translation,
                optimize_config.max_translation_step,
            );
        }
        current.tz -= clipped_step(
            gradient.tz,
            optimize_config.learning_rate_translation,
            optimize_config.max_translation_step,
        );
        current.yaw -= clipped_step(
            gradient.yaw,
            optimize_config.learning_rate_yaw,
            10.0_f32.to_radians(),
        );
        current.scale = (current.scale
            - clipped_step(
                gradient.scale,
                optimize_config.learning_rate_scale,
                optimize_config.max_translation_step,
            ))
        .clamp(optimize_config.min_scale, optimize_config.max_scale);
        current = sanitize_transform(current);
    }

    let final_loss = reference.loss(current);
    SoftPoseOptimizationResult {
        initial_loss,
        final_loss,
        transform: current,
        steps,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformField {
    Tx,
    Ty,
    Tz,
    Yaw,
    Scale,
}

fn pixel_grid<B: Backend>(width: usize, height: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(width * height * 2);
    for y in 0..height {
        for x in 0..width {
            values.push(x as f32 + 0.5);
            values.push(y as f32 + 0.5);
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([width * height, 2])
}

fn clipped_step(gradient: f32, learning_rate: f32, max_abs_step: f32) -> f32 {
    if !gradient.is_finite() {
        return 0.0;
    }
    (gradient * learning_rate).clamp(-max_abs_step, max_abs_step)
}

fn sanitize_transform(mut transform: ObjectTransformValues) -> ObjectTransformValues {
    if !transform.tx.is_finite() {
        transform.tx = 0.0;
    }
    if !transform.ty.is_finite() {
        transform.ty = 0.0;
    }
    if !transform.tz.is_finite() || transform.tz <= 1.0e-4 {
        transform.tz = 1.0e-4;
    }
    if !transform.yaw.is_finite() {
        transform.yaw = 0.0;
    }
    if !transform.scale.is_finite() {
        transform.scale = 1.0;
    }
    transform
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::BackendTypes;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    fn fixture_points() -> Vec<[f32; 3]> {
        let mut points = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                points.push([x as f32 * 0.06, y as f32 * 0.05, 0.0]);
            }
        }
        points.extend([
            [-0.24, -0.14, 0.05],
            [0.24, -0.14, 0.05],
            [-0.24, 0.14, -0.05],
            [0.24, 0.14, -0.05],
        ]);
        points
    }

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 34.0,
            fy: 34.0,
            cx: 16.0,
            cy: 12.0,
            width: 32,
            height: 24,
        }
    }

    fn config() -> SoftRenderConfig {
        SoftRenderConfig {
            width: 32,
            height: 24,
            sigma_px: 1.8,
            depth_sigma_m: 0.06,
            mask_weight: 1.0,
            depth_weight: 0.2,
        }
    }

    fn tensor_loss<B: Backend>(
        points: &[[f32; 3]],
        transform: &ObjectTransformTensors<B>,
        target_mask: &[f32],
        target_depth: &[f32],
        device: &B::Device,
    ) -> Tensor<B, 1> {
        soft_silhouette_depth_loss(
            tensor_from_points(points, device),
            transform,
            camera(),
            config(),
            tensor_from_image(target_mask, config().width, config().height, device),
            tensor_from_image(target_depth, config().width, config().height, device),
        )
    }

    fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn render_matches_cpu_reference_small_scene() {
        let points = fixture_points();
        let transform = ObjectTransformValues {
            tx: 0.04,
            ty: -0.02,
            tz: 2.4,
            yaw: 0.25,
            scale: 1.2,
        };
        let device = <TestBackend as BackendTypes>::Device::default();
        let output = render_soft_surface(
            tensor_from_points::<TestBackend>(&points, &device),
            &ObjectTransformTensors::<TestBackend>::from_values(transform, &device),
            camera(),
            config(),
        );
        let actual_mask = image_from_tensor(output.mask);
        let actual_depth = image_from_tensor(output.depth);
        let (expected_mask, expected_depth) =
            cpu_render_soft_surface(&points, transform, camera(), config());

        let mask_max_abs = max_abs_delta(&actual_mask, &expected_mask);
        let depth_max_abs = max_abs_delta(&actual_depth, &expected_depth);
        assert!(
            mask_max_abs < 1.0e-5,
            "mask max abs too high: {mask_max_abs}"
        );
        assert!(
            depth_max_abs < 1.0e-4,
            "depth max abs too high: {depth_max_abs}"
        );
    }

    #[test]
    fn visual_metrics_prefer_matching_transform() {
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: -0.08,
            ty: 0.02,
            tz: 2.35,
            yaw: -0.55,
            scale: 1.15,
        };
        let shifted = ObjectTransformValues {
            tx: 0.2,
            ty: target.ty,
            tz: 2.7,
            yaw: 0.45,
            scale: 0.82,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let (same_mask, same_depth) = cpu_render_soft_surface(&points, target, camera(), config());
        let (shifted_mask, shifted_depth) =
            cpu_render_soft_surface(&points, shifted, camera(), config());
        let same = compute_render_metrics(&same_mask, &same_depth, &target_mask, &target_depth);
        let mismatched =
            compute_render_metrics(&shifted_mask, &shifted_depth, &target_mask, &target_depth);

        assert!(same.mask_psnr_db > 100.0);
        assert_eq!(same.threshold_iou, 1.0);
        assert!(
            mismatched.mask_psnr_db < 30.0,
            "mismatched transform should be visually worse: {mismatched:?}"
        );
        assert!(
            mismatched.threshold_iou < 0.75,
            "mismatched transform IoU should expose visual error: {mismatched:?}"
        );
    }

    #[test]
    fn loss_decreases_when_transform_matches_target() {
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: 0.12,
            ty: -0.02,
            tz: 2.25,
            yaw: 0.7,
            scale: 1.05,
        };
        let wrong = ObjectTransformValues {
            tx: -0.18,
            ty: 0.04,
            tz: 2.75,
            yaw: -0.15,
            scale: 0.8,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let correct_loss = cpu_soft_silhouette_depth_loss(
            &points,
            target,
            camera(),
            config(),
            &target_mask,
            &target_depth,
        );
        let wrong_loss = cpu_soft_silhouette_depth_loss(
            &points,
            wrong,
            camera(),
            config(),
            &target_mask,
            &target_depth,
        );

        assert!(
            correct_loss < 1.0e-8,
            "target self-loss should be near zero: {correct_loss}"
        );
        assert!(
            wrong_loss > correct_loss + 0.05,
            "wrong transform should have higher loss: correct={correct_loss} wrong={wrong_loss}"
        );
    }

    #[test]
    fn autodiff_gradients_match_finite_difference_for_transform() {
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: 0.07,
            ty: 0.03,
            tz: 2.42,
            yaw: 0.38,
            scale: 1.12,
        };
        let initial = ObjectTransformValues {
            tx: -0.02,
            ty: -0.01,
            tz: 2.55,
            yaw: -0.18,
            scale: 0.94,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let device = <TestAutodiffBackend as BackendTypes>::Device::default();
        let transform = ObjectTransformTensors::<TestAutodiffBackend> {
            tx: scalar_tensor(initial.tx, &device).require_grad(),
            ty: scalar_tensor(initial.ty, &device).require_grad(),
            tz: scalar_tensor(initial.tz, &device).require_grad(),
            yaw: scalar_tensor(initial.yaw, &device).require_grad(),
            scale: scalar_tensor(initial.scale, &device).require_grad(),
        };
        let loss = tensor_loss(&points, &transform, &target_mask, &target_depth, &device);
        let grads = loss.backward();
        let reference = CpuLossReference {
            local_points: &points,
            intrinsics: camera(),
            config: config(),
            target_mask: &target_mask,
            target_depth: &target_depth,
        };

        let checks = [
            (
                "tx",
                transform.tx.grad(&grads).unwrap().into_scalar(),
                finite_difference_gradient(reference, initial, TransformField::Tx, 1.0e-3),
            ),
            (
                "ty",
                transform.ty.grad(&grads).unwrap().into_scalar(),
                finite_difference_gradient(reference, initial, TransformField::Ty, 1.0e-3),
            ),
            (
                "tz",
                transform.tz.grad(&grads).unwrap().into_scalar(),
                finite_difference_gradient(reference, initial, TransformField::Tz, 1.0e-3),
            ),
            (
                "yaw",
                transform.yaw.grad(&grads).unwrap().into_scalar(),
                finite_difference_gradient(reference, initial, TransformField::Yaw, 1.0e-3),
            ),
            (
                "scale",
                transform.scale.grad(&grads).unwrap().into_scalar(),
                finite_difference_gradient(reference, initial, TransformField::Scale, 1.0e-3),
            ),
        ];

        for (name, autodiff, finite_diff) in checks {
            let abs = (autodiff - finite_diff).abs();
            let rel = abs / finite_diff.abs().max(1.0e-3);
            assert!(
                abs < 5.0e-2 || rel < 0.15,
                "{name} gradient mismatch: autodiff={autodiff} finite_diff={finite_diff} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    fn gradient_descent_step_reduces_loss() {
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: 0.08,
            ty: 0.03,
            tz: 2.45,
            yaw: 0.2,
            scale: 1.08,
        };
        let mut current = ObjectTransformValues {
            tx: -0.06,
            ty: -0.02,
            tz: 2.6,
            yaw: -0.1,
            scale: 0.98,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let initial_loss = cpu_soft_silhouette_depth_loss(
            &points,
            current,
            camera(),
            config(),
            &target_mask,
            &target_depth,
        );

        let device = <TestAutodiffBackend as BackendTypes>::Device::default();
        for _ in 0..5 {
            let transform = ObjectTransformTensors::<TestAutodiffBackend> {
                tx: scalar_tensor(current.tx, &device).require_grad(),
                ty: scalar_tensor(current.ty, &device).require_grad(),
                tz: scalar_tensor(current.tz, &device).require_grad(),
                yaw: scalar_tensor(current.yaw, &device).require_grad(),
                scale: scalar_tensor(current.scale, &device).require_grad(),
            };
            let loss = tensor_loss(&points, &transform, &target_mask, &target_depth, &device);
            let grads = loss.backward();
            current.tx -= 0.01 * transform.tx.grad(&grads).unwrap().into_scalar();
            current.ty -= 0.01 * transform.ty.grad(&grads).unwrap().into_scalar();
            current.tz -= 0.01 * transform.tz.grad(&grads).unwrap().into_scalar();
            current.yaw -= 0.005 * transform.yaw.grad(&grads).unwrap().into_scalar();
            current.scale = (current.scale
                - 0.005 * transform.scale.grad(&grads).unwrap().into_scalar())
            .clamp(0.5, 1.8);
        }

        let final_loss = cpu_soft_silhouette_depth_loss(
            &points,
            current,
            camera(),
            config(),
            &target_mask,
            &target_depth,
        );
        assert!(
            final_loss < initial_loss,
            "gradient descent should reduce loss: initial={initial_loss} final={final_loss} current={current:?}"
        );
    }

    #[test]
    fn optimize_soft_pose_ndarray_reduces_crop_alignment_loss() {
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: 0.05,
            ty: 0.02,
            tz: 2.35,
            yaw: 0.22,
            scale: 1.08,
        };
        let initial = ObjectTransformValues {
            tx: -0.03,
            ty: -0.02,
            tz: 2.50,
            yaw: -0.08,
            scale: 0.96,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let result = optimize_soft_pose_ndarray(
            &points,
            initial,
            camera(),
            config(),
            &target_mask,
            &target_depth,
            SoftPoseOptimizationConfig {
                iterations: 14,
                learning_rate_translation: 0.008,
                learning_rate_yaw: 0.004,
                learning_rate_scale: 0.004,
                min_scale: 0.5,
                max_scale: 1.6,
                max_translation_step: 0.04,
                optimize_ty: true,
            },
        );
        assert!(
            result.final_loss < result.initial_loss * 0.85,
            "optimizer should reduce loss: {result:?}"
        );
        assert_eq!(result.steps.len(), 14);
        assert!((result.transform.tx - target.tx).abs() < (initial.tx - target.tx).abs());
        assert!((result.transform.tz - target.tz).abs() < (initial.tz - target.tz).abs());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_soft_render_smoke() {
        if std::env::var("BURN_SYNTH_RENDER_WGPU_SMOKE")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!("skipping WGPU smoke; set BURN_SYNTH_RENDER_WGPU_SMOKE=1 to enable");
            return;
        }
        type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32, u32>>;
        let points = fixture_points();
        let target = ObjectTransformValues {
            tx: 0.02,
            ty: 0.0,
            tz: 2.5,
            yaw: 0.15,
            scale: 1.0,
        };
        let (target_mask, target_depth) =
            cpu_render_soft_surface(&points, target, camera(), config());
        let device = <WgpuBackend as BackendTypes>::Device::default();
        let transform = ObjectTransformTensors::<WgpuBackend> {
            tx: scalar_tensor(target.tx, &device).require_grad(),
            ty: scalar_tensor(target.ty, &device),
            tz: scalar_tensor(target.tz, &device).require_grad(),
            yaw: scalar_tensor(target.yaw, &device).require_grad(),
            scale: scalar_tensor(target.scale, &device).require_grad(),
        };
        let loss = tensor_loss(&points, &transform, &target_mask, &target_depth, &device);
        let grads = loss.backward();
        let tx_grad = transform.tx.grad(&grads).unwrap().into_scalar();
        assert!(tx_grad.is_finite());
    }
}
