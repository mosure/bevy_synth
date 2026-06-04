use burn::{
    nn,
    prelude::*,
    tensor::{FloatDType, activation::softmax},
};

use crate::{
    components::{CrossOnlyBlock, FeedForwardNet, PcdAbsolutePositionEmbedder, SinusoidalEmbedder},
    gaussian::{GaussianSplat, GaussianSplatCloud},
    rng::SplitMix64,
};

pub const OCTREE_MAX_VOXEL_LEVEL: usize = 8;

#[derive(Clone, Debug)]
pub struct OctreeSample<B: Backend> {
    pub points: Tensor<B, 3>,
    pub log_probs: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct OctreePrediction<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub probs: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
struct OctreeHostNode {
    coords: [usize; 3],
    count: usize,
    log_prob: f32,
}

#[derive(Config, Debug)]
pub struct OctreeProbabilityFixedlenDecoderConfig {
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub share_mod: bool,
    pub additional_level_embed: bool,
    pub qk_rms_norm_cross: bool,
    pub no_norm: bool,
}

impl OctreeProbabilityFixedlenDecoderConfig {
    pub fn triposplat() -> Self {
        Self {
            model_channels: 1024,
            cond_channels: 16,
            num_blocks: 4,
            num_heads: 16,
            mlp_ratio: 4.0,
            share_mod: true,
            additional_level_embed: false,
            qk_rms_norm_cross: true,
            no_norm: false,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            model_channels: 32,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: 4,
            mlp_ratio: 2.0,
            share_mod: true,
            additional_level_embed: false,
            qk_rms_norm_cross: true,
            no_norm: false,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> OctreeProbabilityFixedlenDecoder<B> {
        OctreeProbabilityFixedlenDecoder::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct OctreeProbabilityFixedlenDecoder<B: Backend> {
    pub input_layer: nn::Linear<B>,
    pub l_embedder: SinusoidalEmbedder<B>,
    pub l_embedder2: Option<SinusoidalEmbedder<B>>,
    pub ada_ln_modulation: Option<nn::Linear<B>>,
    pub blocks: Vec<CrossOnlyBlock<B>>,
    pub out_proj: nn::Linear<B>,
    pub in_proj: nn::Linear<B>,
    pos_embedder: PcdAbsolutePositionEmbedder,
    config: OctreeProbabilityFixedlenDecoderConfig,
}

impl<B: Backend> OctreeProbabilityFixedlenDecoder<B> {
    pub fn new(device: &B::Device, config: OctreeProbabilityFixedlenDecoderConfig) -> Self {
        Self {
            input_layer: nn::LinearConfig::new(config.model_channels, config.model_channels)
                .with_bias(true)
                .init(device),
            l_embedder: SinusoidalEmbedder::new(device, config.model_channels, 256, 1024.0, true),
            l_embedder2: config
                .additional_level_embed
                .then(|| SinusoidalEmbedder::new(device, config.model_channels, 256, 100.0, true)),
            ada_ln_modulation: config.share_mod.then(|| {
                nn::LinearConfig::new(config.model_channels, 6 * config.model_channels)
                    .with_bias(true)
                    .init(device)
            }),
            blocks: (0..config.num_blocks)
                .map(|_| {
                    CrossOnlyBlock::new(
                        device,
                        config.model_channels,
                        config.cond_channels,
                        config.num_heads,
                        config.mlp_ratio,
                        config.qk_rms_norm_cross,
                        config.share_mod,
                    )
                })
                .collect(),
            out_proj: nn::LinearConfig::new(config.model_channels, 8)
                .with_bias(true)
                .init(device),
            in_proj: nn::LinearConfig::new(3, config.model_channels)
                .with_bias(true)
                .init(device),
            pos_embedder: PcdAbsolutePositionEmbedder::v2(config.model_channels),
            config,
        }
    }

    pub fn forward(
        &self,
        coords_norm: Tensor<B, 3>,
        level: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        total_points: Option<Tensor<B, 1>>,
    ) -> OctreePrediction<B> {
        let dtype = self.float_dtype();
        let coords_norm = cast_float_tensor(coords_norm, dtype);
        let level = cast_float_tensor(level, dtype);
        let cond = cast_float_tensor(cond, dtype);
        let total_points = total_points.map(|tensor| cast_float_tensor(tensor, dtype));
        let [batch, tokens, _] = coords_norm.dims();
        let h =
            self.in_proj.forward(coords_norm.clone()) + self.pos_embedder.forward_3d(coords_norm);
        let mut h = self.input_layer.forward(h);
        let mut mod_signal = self.l_embedder.forward(level);
        if let (Some(embedder2), Some(total_points)) = (&self.l_embedder2, total_points) {
            mod_signal =
                mod_signal + embedder2.forward(total_points.log() / core::f32::consts::LN_2);
        }
        if self.config.share_mod {
            mod_signal = self
                .ada_ln_modulation
                .as_ref()
                .expect("shared octree adaLN modulation missing")
                .forward(crate::components::silu(mod_signal));
        }
        for block in &self.blocks {
            h = block.forward(h, mod_signal.clone(), cond.clone());
        }
        if !self.config.no_norm {
            h = layer_norm_last(h, 1.0e-6);
        } else {
            h = h / (1.0 + 2.0 * self.config.num_blocks as f32).sqrt();
        }
        let logits = self.out_proj.forward(h).reshape([batch, tokens, 8]);
        let probs = softmax_probabilities(logits.clone(), 2);
        OctreePrediction { logits, probs }
    }

    pub fn sample_regular(
        device: &B::Device,
        batch: usize,
        num_points: usize,
        level: usize,
    ) -> OctreeSample<B> {
        let resolution = (1usize << level.max(1)) as f32;
        let mut values = Vec::with_capacity(batch * num_points * 3);
        for _ in 0..batch {
            for index in 0..num_points {
                let x = (index as f32 / num_points.max(1) as f32 * resolution).fract();
                values.push(x);
                values.push(radical_inverse(2, index as u32));
                values.push(radical_inverse(3, index as u32));
            }
        }
        OctreeSample {
            points: Tensor::<B, 1>::from_floats(values.as_slice(), device)
                .reshape([batch, num_points, 3]),
            log_probs: Tensor::<B, 2>::zeros([batch, num_points], device),
        }
    }

    pub fn sample_systematic_host(
        &self,
        cond: Tensor<B, 3>,
        num_points: usize,
        level: usize,
        seed: u64,
    ) -> Result<OctreeSample<B>, String> {
        let [batch, _tokens, _channels] = cond.dims();
        let dtype: FloatDType = cond.dtype().into();
        if batch == 0 {
            return Err("octree systematic sampler requires a non-empty batch".to_string());
        }
        if num_points == 0 {
            return Err("octree systematic sampler requires at least one point".to_string());
        }
        if level == 0 || level > OCTREE_MAX_VOXEL_LEVEL {
            return Err(format!(
                "octree systematic sampler level must be in [1, {OCTREE_MAX_VOXEL_LEVEL}], got {level}"
            ));
        }

        let device = cond.device();
        let child_offsets = [
            [0usize, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        let mut rng = SplitMix64::new(seed);
        let mut prev = vec![
            vec![OctreeHostNode {
                coords: [0, 0, 0],
                count: num_points,
                log_prob: 0.0,
            }];
            batch
        ];

        for lv in 1..=level {
            let max_parent_count = prev.iter().map(Vec::len).max().unwrap_or(0);
            if max_parent_count == 0 {
                return Err(format!(
                    "octree systematic sampler exhausted all nodes at level {}",
                    lv - 1
                ));
            }

            let parent_resolution = 1usize << (lv - 1);
            let resolution = 1usize << lv;
            let mut coords = vec![0.0f32; batch * max_parent_count * 3];
            for (batch_index, nodes) in prev.iter().enumerate() {
                for (node_index, node) in nodes.iter().enumerate() {
                    let base = (batch_index * max_parent_count + node_index) * 3;
                    coords[base] = (node.coords[0] as f32 + 0.5) / parent_resolution as f32;
                    coords[base + 1] = (node.coords[1] as f32 + 0.5) / parent_resolution as f32;
                    coords[base + 2] = (node.coords[2] as f32 + 0.5) / parent_resolution as f32;
                }
            }

            let coords = Tensor::<B, 1>::from_floats(coords.as_slice(), &device).reshape([
                batch,
                max_parent_count,
                3,
            ]);
            let level_tensor =
                Tensor::<B, 1>::from_floats(vec![resolution as f32; batch].as_slice(), &device);
            let total_points =
                Tensor::<B, 1>::from_floats(vec![num_points as f32; batch].as_slice(), &device);
            let probs = self
                .forward(coords, level_tensor, cond.clone(), Some(total_points))
                .probs
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "failed to read octree probabilities for host sampler".to_string())?;

            let mut next = Vec::with_capacity(batch);
            for (batch_index, nodes) in prev.iter().enumerate() {
                let mut next_nodes = Vec::new();
                for (node_index, node) in nodes.iter().enumerate() {
                    if node.count == 0 {
                        continue;
                    }
                    let row_start = (batch_index * max_parent_count + node_index) * 8;
                    let row = &probs[row_start..row_start + 8];
                    let child_counts =
                        systematic_child_counts(row, node.count, rng.next_unit_f32());
                    for (child_index, count) in child_counts.into_iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let child = child_offsets[child_index];
                        let prob = sanitized_prob(row, child_index);
                        next_nodes.push(OctreeHostNode {
                            coords: [
                                node.coords[0] * 2 + child[0],
                                node.coords[1] * 2 + child[1],
                                node.coords[2] * 2 + child[2],
                            ],
                            count,
                            log_prob: node.log_prob + prob.ln(),
                        });
                    }
                }
                next.push(next_nodes);
            }
            prev = next;
        }

        let resolution = 1usize << level;
        let mut points = Vec::with_capacity(batch * num_points * 3);
        let mut log_probs = Vec::with_capacity(batch * num_points);
        for (batch_index, nodes) in prev.iter().enumerate() {
            let produced = nodes.iter().map(|node| node.count).sum::<usize>();
            if produced != num_points {
                return Err(format!(
                    "octree systematic sampler produced {produced} point(s) for batch {batch_index}, expected {num_points}"
                ));
            }
            for node in nodes {
                for _ in 0..node.count {
                    points.push((node.coords[0] as f32 + rng.next_unit_f32()) / resolution as f32);
                    points.push((node.coords[1] as f32 + rng.next_unit_f32()) / resolution as f32);
                    points.push((node.coords[2] as f32 + rng.next_unit_f32()) / resolution as f32);
                    log_probs.push(node.log_prob);
                }
            }
        }

        Ok(OctreeSample {
            points: Tensor::<B, 1>::from_floats(points.as_slice(), &device)
                .cast(dtype)
                .reshape([batch, num_points, 3]),
            log_probs: Tensor::<B, 1>::from_floats(log_probs.as_slice(), &device)
                .cast(dtype)
                .reshape([batch, num_points]),
        })
    }

    pub async fn sample_systematic_host_async(
        &self,
        cond: Tensor<B, 3>,
        num_points: usize,
        level: usize,
        seed: u64,
    ) -> Result<OctreeSample<B>, String> {
        let [batch, _tokens, _channels] = cond.dims();
        let dtype: FloatDType = cond.dtype().into();
        if batch == 0 {
            return Err("octree systematic sampler requires a non-empty batch".to_string());
        }
        if num_points == 0 {
            return Err("octree systematic sampler requires at least one point".to_string());
        }
        if level == 0 || level > OCTREE_MAX_VOXEL_LEVEL {
            return Err(format!(
                "octree systematic sampler level must be in [1, {OCTREE_MAX_VOXEL_LEVEL}], got {level}"
            ));
        }

        let device = cond.device();
        let child_offsets = [
            [0usize, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        let mut rng = SplitMix64::new(seed);
        let mut prev = vec![
            vec![OctreeHostNode {
                coords: [0, 0, 0],
                count: num_points,
                log_prob: 0.0,
            }];
            batch
        ];

        for lv in 1..=level {
            let max_parent_count = prev.iter().map(Vec::len).max().unwrap_or(0);
            if max_parent_count == 0 {
                return Err(format!(
                    "octree systematic sampler exhausted all nodes at level {}",
                    lv - 1
                ));
            }

            let parent_resolution = 1usize << (lv - 1);
            let resolution = 1usize << lv;
            let mut coords = vec![0.0f32; batch * max_parent_count * 3];
            for (batch_index, nodes) in prev.iter().enumerate() {
                for (node_index, node) in nodes.iter().enumerate() {
                    let base = (batch_index * max_parent_count + node_index) * 3;
                    coords[base] = (node.coords[0] as f32 + 0.5) / parent_resolution as f32;
                    coords[base + 1] = (node.coords[1] as f32 + 0.5) / parent_resolution as f32;
                    coords[base + 2] = (node.coords[2] as f32 + 0.5) / parent_resolution as f32;
                }
            }

            let coords = Tensor::<B, 1>::from_floats(coords.as_slice(), &device).reshape([
                batch,
                max_parent_count,
                3,
            ]);
            let level_tensor =
                Tensor::<B, 1>::from_floats(vec![resolution as f32; batch].as_slice(), &device);
            let total_points =
                Tensor::<B, 1>::from_floats(vec![num_points as f32; batch].as_slice(), &device);
            let probs = self
                .forward(coords, level_tensor, cond.clone(), Some(total_points))
                .probs
                .to_data_async()
                .await
                .map_err(|_| "failed to read octree probabilities for host sampler".to_string())?
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "failed to read octree probabilities for host sampler".to_string())?;

            let mut next = Vec::with_capacity(batch);
            for (batch_index, nodes) in prev.iter().enumerate() {
                let mut next_nodes = Vec::new();
                for (node_index, node) in nodes.iter().enumerate() {
                    if node.count == 0 {
                        continue;
                    }
                    let row_start = (batch_index * max_parent_count + node_index) * 8;
                    let row = &probs[row_start..row_start + 8];
                    let child_counts =
                        systematic_child_counts(row, node.count, rng.next_unit_f32());
                    for (child_index, count) in child_counts.into_iter().enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let child = child_offsets[child_index];
                        let prob = sanitized_prob(row, child_index);
                        next_nodes.push(OctreeHostNode {
                            coords: [
                                node.coords[0] * 2 + child[0],
                                node.coords[1] * 2 + child[1],
                                node.coords[2] * 2 + child[2],
                            ],
                            count,
                            log_prob: node.log_prob + prob.ln(),
                        });
                    }
                }
                next.push(next_nodes);
            }
            prev = next;
        }

        let resolution = 1usize << level;
        let mut points = Vec::with_capacity(batch * num_points * 3);
        let mut log_probs = Vec::with_capacity(batch * num_points);
        for (batch_index, nodes) in prev.iter().enumerate() {
            let produced = nodes.iter().map(|node| node.count).sum::<usize>();
            if produced != num_points {
                return Err(format!(
                    "octree systematic sampler produced {produced} point(s) for batch {batch_index}, expected {num_points}"
                ));
            }
            for node in nodes {
                for _ in 0..node.count {
                    points.push((node.coords[0] as f32 + rng.next_unit_f32()) / resolution as f32);
                    points.push((node.coords[1] as f32 + rng.next_unit_f32()) / resolution as f32);
                    points.push((node.coords[2] as f32 + rng.next_unit_f32()) / resolution as f32);
                    log_probs.push(node.log_prob);
                }
            }
        }

        Ok(OctreeSample {
            points: Tensor::<B, 1>::from_floats(points.as_slice(), &device)
                .cast(dtype)
                .reshape([batch, num_points, 3]),
            log_probs: Tensor::<B, 1>::from_floats(log_probs.as_slice(), &device)
                .cast(dtype)
                .reshape([batch, num_points]),
        })
    }

    fn float_dtype(&self) -> FloatDType {
        self.in_proj.weight.val().dtype().into()
    }
}

#[derive(Clone, Debug)]
pub struct GaussianRepresentationConfig {
    pub num_gaussians: usize,
    pub perturb_offset: bool,
    pub perturbe_size: f32,
    pub offset_scale: f32,
    pub filter_kernel_size_3d: f32,
    pub scaling_bias: f32,
    pub opacity_bias: f32,
    pub lr_xyz: f32,
    pub lr_features_dc: f32,
    pub lr_opacity: f32,
    pub lr_scaling: f32,
    pub lr_rotation: f32,
}

impl GaussianRepresentationConfig {
    pub fn triposplat() -> Self {
        Self {
            num_gaussians: 32,
            perturb_offset: true,
            perturbe_size: 1.5,
            offset_scale: 0.05,
            filter_kernel_size_3d: 0.0009,
            scaling_bias: 0.004,
            opacity_bias: 0.1,
            lr_xyz: 1.0,
            lr_features_dc: 1.0,
            lr_opacity: 1.0,
            lr_scaling: 1.0,
            lr_rotation: 0.1,
        }
    }
}

#[derive(Config, Debug)]
pub struct ElasticGaussianFixedlenDecoderConfig {
    pub in_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub no_norm: bool,
    pub use_learned_offset_scale: bool,
    pub use_per_offset: bool,
}

impl ElasticGaussianFixedlenDecoderConfig {
    pub fn triposplat() -> Self {
        Self {
            in_channels: 3,
            model_channels: 1024,
            cond_channels: 16,
            num_blocks: 16,
            num_heads: 16,
            mlp_ratio: 4.0,
            no_norm: false,
            use_learned_offset_scale: true,
            use_per_offset: true,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            in_channels: 3,
            model_channels: 32,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: 4,
            mlp_ratio: 2.0,
            no_norm: false,
            use_learned_offset_scale: true,
            use_per_offset: true,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ElasticGaussianFixedlenDecoder<B> {
        ElasticGaussianFixedlenDecoder::new(
            device,
            self.clone(),
            GaussianRepresentationConfig::triposplat(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianFeatureLayout {
    pub xyz: (usize, usize),
    pub features_dc: (usize, usize),
    pub scaling: (usize, usize),
    pub rotation: (usize, usize),
    pub opacity: (usize, usize),
    pub offset_scale: Option<(usize, usize)>,
    pub out_channels: usize,
}

impl GaussianFeatureLayout {
    pub fn new(config: &GaussianRepresentationConfig, include_offset_scale: bool) -> Self {
        let ng = config.num_gaussians;
        let mut cursor = 0;
        let xyz = take(&mut cursor, ng * 3);
        let features_dc = take(&mut cursor, ng * 3);
        let scaling = take(&mut cursor, ng * 3);
        let rotation = take(&mut cursor, ng * 4);
        let opacity = take(&mut cursor, ng);
        let offset_scale = include_offset_scale.then(|| take(&mut cursor, ng));
        Self {
            xyz,
            features_dc,
            scaling,
            rotation,
            opacity,
            offset_scale,
            out_channels: cursor,
        }
    }
}

fn take(cursor: &mut usize, size: usize) -> (usize, usize) {
    let start = *cursor;
    *cursor += size;
    (start, *cursor)
}

#[derive(Module, Debug)]
pub struct ElasticGaussianFixedlenDecoder<B: Backend> {
    pub in_proj: nn::Linear<B>,
    pub input_layer: nn::Linear<B>,
    pub self_attns: Vec<crate::components::MultiHeadAttention<B>>,
    pub cross_attns: Vec<crate::components::MultiHeadAttention<B>>,
    pub mlps: Vec<FeedForwardNet<B>>,
    pub norms: Vec<nn::LayerNorm<B>>,
    pub out_proj: nn::Linear<B>,
    pos_embedder: PcdAbsolutePositionEmbedder,
    pub layout: GaussianFeatureLayout,
    rep_config: GaussianRepresentationConfig,
    config: ElasticGaussianFixedlenDecoderConfig,
}

impl<B: Backend> ElasticGaussianFixedlenDecoder<B> {
    pub fn new(
        device: &B::Device,
        config: ElasticGaussianFixedlenDecoderConfig,
        rep_config: GaussianRepresentationConfig,
    ) -> Self {
        let layout = GaussianFeatureLayout::new(
            &rep_config,
            config.use_learned_offset_scale && config.use_per_offset,
        );
        Self {
            in_proj: nn::LinearConfig::new(config.in_channels, config.model_channels)
                .with_bias(true)
                .init(device),
            input_layer: nn::LinearConfig::new(config.model_channels, config.model_channels)
                .with_bias(true)
                .init(device),
            self_attns: (0..config.num_blocks)
                .map(|_| {
                    crate::components::MultiHeadAttention::new(
                        device,
                        config.model_channels,
                        config.num_heads,
                        None,
                        crate::components::AttentionKind::SelfAttention,
                        true,
                        true,
                        false,
                    )
                })
                .collect(),
            cross_attns: (0..config.num_blocks)
                .map(|_| {
                    crate::components::MultiHeadAttention::new(
                        device,
                        config.model_channels,
                        config.num_heads,
                        Some(config.cond_channels),
                        crate::components::AttentionKind::CrossAttention,
                        true,
                        true,
                        false,
                    )
                })
                .collect(),
            mlps: (0..config.num_blocks)
                .map(|_| FeedForwardNet::new(device, config.model_channels, config.mlp_ratio))
                .collect(),
            norms: (0..(config.num_blocks * 3))
                .map(|_| {
                    nn::LayerNormConfig::new(config.model_channels)
                        .with_epsilon(1.0e-6)
                        .init(device)
                })
                .collect(),
            out_proj: nn::LinearConfig::new(config.model_channels, layout.out_channels)
                .with_bias(true)
                .init(device),
            pos_embedder: PcdAbsolutePositionEmbedder::v2(config.model_channels),
            layout,
            rep_config,
            config,
        }
    }

    pub fn forward(&self, x: &OctreeSample<B>, cond: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_inner(x, cond, false)
            .expect("Gaussian decoder unchecked forward should not run finite diagnostics")
    }

    pub fn forward_checked(
        &self,
        x: &OctreeSample<B>,
        cond: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, String> {
        self.forward_inner(x, cond, true)
    }

    fn forward_inner(
        &self,
        x: &OctreeSample<B>,
        cond: Tensor<B, 3>,
        check_finite: bool,
    ) -> Result<Tensor<B, 3>, String> {
        let dtype = self.float_dtype();
        let points = cast_float_tensor(x.points.clone(), dtype);
        let cond = cast_float_tensor(cond, dtype);
        if check_finite {
            validate_tensor_finite("gaussian_decoder.points", &points)?;
            validate_tensor_finite("gaussian_decoder.cond", &cond)?;
        }
        let mut h = self.in_proj.forward(points.clone()) + self.pos_embedder.forward_3d(points);
        if check_finite {
            validate_tensor_finite("gaussian_decoder.in_proj_plus_pos", &h)?;
        }
        h = self.input_layer.forward(h);
        if check_finite {
            validate_tensor_finite("gaussian_decoder.input_layer", &h)?;
        }
        for index in 0..self.config.num_blocks {
            let norm0 = &self.norms[index * 3];
            let norm1 = &self.norms[index * 3 + 1];
            let norm2 = &self.norms[index * 3 + 2];
            let norm0_out = norm0.forward(h.clone());
            if check_finite {
                validate_tensor_finite(
                    &format!("gaussian_decoder.block{index}.norm0"),
                    &norm0_out,
                )?;
            }
            let self_delta = self.self_attns[index].forward(norm0_out, None, None);
            if check_finite {
                validate_tensor_finite(
                    &format!("gaussian_decoder.block{index}.self_attn"),
                    &self_delta,
                )?;
            }
            h = h + self_delta;
            if check_finite {
                validate_tensor_finite(&format!("gaussian_decoder.block{index}.post_self"), &h)?;
            }
            let norm1_out = norm1.forward(h.clone());
            if check_finite {
                validate_tensor_finite(
                    &format!("gaussian_decoder.block{index}.norm1"),
                    &norm1_out,
                )?;
            }
            let cross_delta = self.cross_attns[index].forward(norm1_out, Some(cond.clone()), None);
            if check_finite {
                validate_tensor_finite(
                    &format!("gaussian_decoder.block{index}.cross_attn"),
                    &cross_delta,
                )?;
            }
            h = h + cross_delta;
            if check_finite {
                validate_tensor_finite(&format!("gaussian_decoder.block{index}.post_cross"), &h)?;
            }
            let norm2_out = norm2.forward(h.clone());
            if check_finite {
                validate_tensor_finite(
                    &format!("gaussian_decoder.block{index}.norm2"),
                    &norm2_out,
                )?;
            }
            let mlp_delta = self.mlps[index].forward(norm2_out);
            if check_finite {
                validate_tensor_finite(&format!("gaussian_decoder.block{index}.mlp"), &mlp_delta)?;
            }
            h = h + mlp_delta;
            if check_finite {
                validate_tensor_finite(&format!("gaussian_decoder.block{index}.post_mlp"), &h)?;
            }
        }
        if !self.config.no_norm {
            h = layer_norm_last(h, 1.0e-6);
            if check_finite {
                validate_tensor_finite("gaussian_decoder.final_norm", &h)?;
            }
        }
        let out = self.out_proj.forward(h);
        if check_finite {
            validate_tensor_finite("gaussian_decoder.out_proj", &out)?;
        }
        Ok(out)
    }

    pub fn build_cloud(
        &self,
        sample: &OctreeSample<B>,
        features: Tensor<B, 3>,
    ) -> Result<GaussianSplatCloud, String> {
        let [batch, tokens, channels] = features.dims();
        if batch == 0 {
            return Err("Gaussian decoder output has empty batch".to_string());
        }
        if channels != self.layout.out_channels {
            return Err(format!(
                "Gaussian decoder feature width mismatch: got {channels}, expected {}",
                self.layout.out_channels
            ));
        }
        let points = sample
            .points
            .clone()
            .slice([0..1, 0..tokens, 0..3])
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read TripoSplat decoder points".to_string())?;
        let features = features
            .slice([0..1, 0..tokens, 0..channels])
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read TripoSplat decoder features".to_string())?;
        Ok(GaussianSplatCloud::new(build_splats_host(
            &points,
            &features,
            tokens,
            self.layout,
            &self.rep_config,
        )?))
    }

    pub async fn build_cloud_async(
        &self,
        sample: &OctreeSample<B>,
        features: Tensor<B, 3>,
    ) -> Result<GaussianSplatCloud, String> {
        let [batch, tokens, channels] = features.dims();
        if batch == 0 {
            return Err("Gaussian decoder output has empty batch".to_string());
        }
        if channels != self.layout.out_channels {
            return Err(format!(
                "Gaussian decoder feature width mismatch: got {channels}, expected {}",
                self.layout.out_channels
            ));
        }
        let points = sample
            .points
            .clone()
            .slice([0..1, 0..tokens, 0..3])
            .to_data_async()
            .await
            .map_err(|_| "failed to read TripoSplat decoder points".to_string())?
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read TripoSplat decoder points".to_string())?;
        let features = features
            .slice([0..1, 0..tokens, 0..channels])
            .to_data_async()
            .await
            .map_err(|_| "failed to read TripoSplat decoder features".to_string())?
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read TripoSplat decoder features".to_string())?;
        Ok(GaussianSplatCloud::new(build_splats_host(
            &points,
            &features,
            tokens,
            self.layout,
            &self.rep_config,
        )?))
    }

    fn float_dtype(&self) -> FloatDType {
        self.in_proj.weight.val().dtype().into()
    }
}

fn cast_float_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    let current: FloatDType = tensor.dtype().into();
    if current == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn layer_norm_last<B: Backend>(x: Tensor<B, 3>, epsilon: f64) -> Tensor<B, 3> {
    let dtype: FloatDType = x.dtype().into();
    let x_acc = cast_low_precision_to_f32(x, dtype);
    let (var, mean) = x_acc.clone().var_mean_bias(2);
    cast_from_f32_accum((x_acc - mean) / var.add_scalar(epsilon).sqrt(), dtype)
}

fn softmax_probabilities<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let dtype: FloatDType = x.dtype().into();
    softmax(cast_low_precision_to_f32(x, dtype), dim)
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

#[derive(Module, Debug)]
pub struct OctreeGaussianDecoder<B: Backend> {
    pub octree: OctreeProbabilityFixedlenDecoder<B>,
    pub gs: ElasticGaussianFixedlenDecoder<B>,
}

impl<B: Backend> OctreeGaussianDecoder<B> {
    pub fn new(
        device: &B::Device,
        octree_config: OctreeProbabilityFixedlenDecoderConfig,
        gs_config: ElasticGaussianFixedlenDecoderConfig,
    ) -> Self {
        Self {
            octree: octree_config.init(device),
            gs: gs_config.init(device),
        }
    }

    pub fn triposplat(device: &B::Device) -> Self {
        Self::new(
            device,
            OctreeProbabilityFixedlenDecoderConfig::triposplat(),
            ElasticGaussianFixedlenDecoderConfig::triposplat(),
        )
    }

    pub fn gaussians_per_point(&self) -> usize {
        self.gs.rep_config.num_gaussians
    }

    pub fn decode_to_cloud(
        &self,
        latent: Tensor<B, 3>,
        num_gaussians: usize,
    ) -> Result<GaussianSplatCloud, String> {
        self.decode_to_cloud_with_seed(latent, num_gaussians, 0)
    }

    pub fn decode_to_cloud_with_seed(
        &self,
        latent: Tensor<B, 3>,
        num_gaussians: usize,
        seed: u64,
    ) -> Result<GaussianSplatCloud, String> {
        let [batch, _tokens, _channels] = latent.dims();
        let num_points = (num_gaussians / self.gaussians_per_point()).max(1);
        let sample = self.octree.sample_systematic_host(
            latent.clone(),
            num_points,
            OCTREE_MAX_VOXEL_LEVEL,
            seed,
        )?;
        debug_assert_eq!(sample.points.dims()[0], batch);
        let features = self.gs.forward(&sample, latent);
        self.gs.build_cloud(&sample, features)
    }

    pub async fn decode_to_cloud_with_seed_async(
        &self,
        latent: Tensor<B, 3>,
        num_gaussians: usize,
        seed: u64,
    ) -> Result<GaussianSplatCloud, String> {
        let [batch, _tokens, _channels] = latent.dims();
        let num_points = (num_gaussians / self.gaussians_per_point()).max(1);
        let sample = self
            .octree
            .sample_systematic_host_async(latent.clone(), num_points, OCTREE_MAX_VOXEL_LEVEL, seed)
            .await?;
        debug_assert_eq!(sample.points.dims()[0], batch);
        let features = self.gs.forward(&sample, latent);
        self.gs.build_cloud_async(&sample, features).await
    }

    pub fn decode_to_cloud_with_seed_checked(
        &self,
        latent: Tensor<B, 3>,
        num_gaussians: usize,
        seed: u64,
    ) -> Result<GaussianSplatCloud, String> {
        validate_tensor_finite("octree_gaussian_decoder.latent", &latent)?;
        let [batch, _tokens, _channels] = latent.dims();
        let num_points = (num_gaussians / self.gaussians_per_point()).max(1);
        let sample = self.octree.sample_systematic_host(
            latent.clone(),
            num_points,
            OCTREE_MAX_VOXEL_LEVEL,
            seed,
        )?;
        debug_assert_eq!(sample.points.dims()[0], batch);
        validate_tensor_finite("octree_gaussian_decoder.sample_points", &sample.points)?;
        validate_tensor_finite(
            "octree_gaussian_decoder.sample_log_probs",
            &sample.log_probs,
        )?;
        let features = self.gs.forward_checked(&sample, latent)?;
        self.gs.build_cloud(&sample, features)
    }
}

fn sanitized_prob(row: &[f32], index: usize) -> f32 {
    let mut sum = 0.0f32;
    for value in row {
        if value.is_finite() && *value > 0.0 {
            sum += *value;
        }
    }
    if sum <= 0.0 {
        return 1.0 / row.len().max(1) as f32;
    }
    row.get(index)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.0)
        .max(1.0e-12)
        / sum
}

fn systematic_child_counts(row: &[f32], count: usize, u0_unit: f32) -> [usize; 8] {
    let mut probs = [0.0f32; 8];
    let mut sum = 0.0f32;
    for (index, value) in row.iter().take(8).enumerate() {
        let value = if value.is_finite() && *value > 0.0 {
            *value
        } else {
            0.0
        };
        probs[index] = value;
        sum += value;
    }
    if sum <= 0.0 {
        probs = [1.0 / 8.0; 8];
    } else {
        for prob in &mut probs {
            *prob /= sum;
        }
    }

    let mut cdf = [0.0f32; 8];
    let mut running = 0.0f32;
    for (index, prob) in probs.iter().enumerate() {
        running = (running + *prob).min(1.0 - 1.0e-12);
        cdf[index] = running;
    }

    let mut out = [0usize; 8];
    if count == 0 {
        return out;
    }
    let start = u0_unit.clamp(0.0, 1.0 - 1.0e-12) / count as f32;
    for index in 0..count {
        let u = (start + index as f32 / count as f32).min(1.0 - 1.0e-12);
        let child = cdf
            .iter()
            .position(|threshold| u <= *threshold)
            .unwrap_or(7);
        out[child] += 1;
    }
    out
}

fn build_splats_host(
    points: &[f32],
    features: &[f32],
    tokens: usize,
    layout: GaussianFeatureLayout,
    config: &GaussianRepresentationConfig,
) -> Result<Vec<GaussianSplat>, String> {
    let ng = config.num_gaussians;
    let mut splats = Vec::with_capacity(tokens * ng);
    let perturb = hammersley_perturbation(ng, config.perturbe_size);
    let base_offset_scale = inverse_softplus(config.offset_scale);
    let scale_bias = inverse_softplus(config.scaling_bias);
    let opacity_bias = logit(config.opacity_bias);
    for token in 0..tokens {
        let point = [
            points[token * 3],
            points[token * 3 + 1],
            points[token * 3 + 2],
        ];
        for (axis, value) in point.iter().enumerate() {
            validate_decoder_scalar("octree_point", token, 0, axis, *value)?;
        }
        let row = &features[token * layout.out_channels..(token + 1) * layout.out_channels];
        for (index, perturbation) in perturb.iter().enumerate().take(ng) {
            let raw_xyz = [
                decoder_feature(row, layout.xyz.0, index, 3, 0, "xyz", token)?,
                decoder_feature(row, layout.xyz.0, index, 3, 1, "xyz", token)?,
                decoder_feature(row, layout.xyz.0, index, 3, 2, "xyz", token)?,
            ];
            let xyz_offset = [
                raw_xyz[0] * config.lr_xyz + perturbation[0],
                raw_xyz[1] * config.lr_xyz + perturbation[1],
                raw_xyz[2] * config.lr_xyz + perturbation[2],
            ];
            let learned_scale = layout
                .offset_scale
                .map(|range| {
                    decoder_feature(row, range.0, index, 1, 0, "offset_scale", token)
                        .map(|value| softplus_scalar(value + base_offset_scale))
                })
                .transpose()?
                .unwrap_or(config.offset_scale);
            let offset = [
                xyz_offset[0].tanh() * 0.5 * config.perturbe_size * learned_scale,
                xyz_offset[1].tanh() * 0.5 * config.perturbe_size * learned_scale,
                xyz_offset[2].tanh() * 0.5 * config.perturbe_size * learned_scale,
            ];
            let raw_scale = [
                decoder_feature(row, layout.scaling.0, index, 3, 0, "scaling", token)?
                    * config.lr_scaling,
                decoder_feature(row, layout.scaling.0, index, 3, 1, "scaling", token)?
                    * config.lr_scaling,
                decoder_feature(row, layout.scaling.0, index, 3, 2, "scaling", token)?
                    * config.lr_scaling,
            ];
            let scale = raw_scale.map(|value| {
                let value = softplus_scalar(value + scale_bias);
                (value * value + config.filter_kernel_size_3d.powi(2)).sqrt()
            });
            splats.push(GaussianSplat {
                position: [
                    point[0] + offset[0] - 0.5,
                    point[1] + offset[1] - 0.5,
                    point[2] + offset[2] - 0.5,
                ],
                features_dc: [
                    decoder_feature(row, layout.features_dc.0, index, 3, 0, "features_dc", token)?
                        * config.lr_features_dc,
                    decoder_feature(row, layout.features_dc.0, index, 3, 1, "features_dc", token)?
                        * config.lr_features_dc,
                    decoder_feature(row, layout.features_dc.0, index, 3, 2, "features_dc", token)?
                        * config.lr_features_dc,
                ],
                opacity: sigmoid_scalar(
                    decoder_feature(row, layout.opacity.0, index, 1, 0, "opacity", token)?
                        * config.lr_opacity
                        + opacity_bias,
                ),
                scale,
                rotation: [
                    1.0 + decoder_feature(row, layout.rotation.0, index, 4, 0, "rotation", token)?
                        * config.lr_rotation,
                    decoder_feature(row, layout.rotation.0, index, 4, 1, "rotation", token)?
                        * config.lr_rotation,
                    decoder_feature(row, layout.rotation.0, index, 4, 2, "rotation", token)?
                        * config.lr_rotation,
                    decoder_feature(row, layout.rotation.0, index, 4, 3, "rotation", token)?
                        * config.lr_rotation,
                ],
            });
        }
    }
    Ok(splats)
}

fn decoder_feature(
    row: &[f32],
    start: usize,
    gaussian_index: usize,
    stride: usize,
    component: usize,
    field: &str,
    token: usize,
) -> Result<f32, String> {
    let value = row[start + gaussian_index * stride + component];
    validate_decoder_scalar(field, token, gaussian_index, component, value)?;
    Ok(value)
}

fn validate_decoder_scalar(
    field: &str,
    token: usize,
    gaussian_index: usize,
    component: usize,
    value: f32,
) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!(
            "Gaussian decoder produced non-finite value: token={token} gaussian_index={gaussian_index} field={field} component={component} value={value}"
        ));
    }
    Ok(())
}

fn validate_tensor_finite<B: Backend, const D: usize>(
    label: &str,
    tensor: &Tensor<B, D>,
) -> Result<(), String> {
    let dims = tensor.dims();
    let values = tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read {label} for finite diagnostics: {err:?}"))?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} produced non-finite value at linear_index={index} dims={dims:?} value={value}"
        ));
    }
    Ok(())
}

fn hammersley_perturbation(count: usize, perturbe_size: f32) -> Vec<[f32; 3]> {
    (0..count)
        .map(|index| {
            [
                atanh(((index as f32 / count.max(1) as f32) * 2.0 - 1.0) / perturbe_size),
                atanh((radical_inverse(2, index as u32) * 2.0 - 1.0) / perturbe_size),
                atanh((radical_inverse(3, index as u32) * 2.0 - 1.0) / perturbe_size),
            ]
        })
        .collect()
}

fn radical_inverse(base: u32, mut value: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut inv = inv_base;
    let mut out = 0.0;
    while value > 0 {
        let digit = value % base;
        out += digit as f32 * inv;
        value /= base;
        inv *= inv_base;
    }
    out
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus_scalar(value: f32) -> f32 {
    (1.0 + value.exp()).ln()
}

fn inverse_softplus(value: f32) -> f32 {
    (value.exp() - 1.0).ln()
}

fn logit(value: f32) -> f32 {
    let value = value.clamp(1.0e-6, 1.0 - 1.0e-6);
    (value / (1.0 - value)).ln()
}

fn atanh(value: f32) -> f32 {
    0.5 * ((1.0 + value) / (1.0 - value)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn gaussian_layout_matches_upstream_channel_count() {
        let layout = GaussianFeatureLayout::new(&GaussianRepresentationConfig::triposplat(), true);
        assert_eq!(layout.out_channels, 480);
    }

    #[test]
    fn tiny_octree_decoder_smoke_produces_requested_multiple() {
        let device = Default::default();
        let decoder = OctreeGaussianDecoder::<TestBackend>::new(
            &device,
            OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
            ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
        );
        let latent = Tensor::<TestBackend, 3>::zeros([1, 8, 4], &device);
        let cloud = decoder.decode_to_cloud(latent, 64).expect("decode cloud");
        assert_eq!(cloud.len(), 64);
    }

    #[test]
    fn tiny_octree_decoder_checked_smoke_reports_finite_path() {
        let device = Default::default();
        let decoder = OctreeGaussianDecoder::<TestBackend>::new(
            &device,
            OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
            ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
        );
        let latent = Tensor::<TestBackend, 3>::zeros([1, 8, 4], &device);
        let cloud = decoder
            .decode_to_cloud_with_seed_checked(latent, 64, 7)
            .expect("checked decode cloud");
        assert_eq!(cloud.len(), 64);
    }

    #[test]
    fn systematic_octree_sampler_preserves_requested_count_and_bounds() {
        let device = Default::default();
        let config = OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let cond = Tensor::<TestBackend, 3>::zeros([1, 8, config.cond_channels], &device);
        let sample = model
            .sample_systematic_host(cond, 17, 3, 123)
            .expect("systematic sample");

        assert_eq!(sample.points.dims(), [1, 17, 3]);
        assert_eq!(sample.log_probs.dims(), [1, 17]);
        let points = sample
            .points
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("points vec");
        assert!(points.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn systematic_octree_sampler_is_seed_deterministic() {
        let device = Default::default();
        let config = OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let cond = Tensor::<TestBackend, 3>::zeros([1, 8, config.cond_channels], &device);
        let first = model
            .sample_systematic_host(cond.clone(), 9, 2, 77)
            .expect("first sample")
            .points
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("first vec");
        let second = model
            .sample_systematic_host(cond, 9, 2, 77)
            .expect("second sample")
            .points
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("second vec");

        assert_eq!(first, second);
    }
}
