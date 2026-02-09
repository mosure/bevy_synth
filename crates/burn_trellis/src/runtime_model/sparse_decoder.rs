use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

const F16_SUFFIX: &str = "_f16";
const LAYER_NORM32_EPS: f32 = 1.0e-6;
const F_LAYER_NORM_EPS: f32 = 1.0e-5;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DecoderConfigFile {
    #[allow(dead_code)]
    pub name: String,
    pub args: DecoderArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DecoderArgs {
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub model_channels: Vec<usize>,
    pub latent_channels: usize,
    pub num_blocks: Vec<usize>,
    #[allow(dead_code)]
    pub block_type: Vec<String>,
    #[allow(dead_code)]
    pub up_block_type: Vec<String>,
    #[allow(dead_code)]
    pub block_args: Vec<serde_json::Value>,
    #[serde(default)]
    pub pred_subdiv: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub resolution: Option<usize>,
    #[serde(default)]
    pub voxel_margin: Option<f32>,
    #[serde(default)]
    pub use_fp16: Option<bool>,
}

#[derive(Debug, Clone)]
pub(crate) struct SparseSubdivisionLogits {
    pub coords: Vec<[u32; 4]>,
    pub logits: Vec<f32>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct SparseDecodeResult {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<f32>,
    pub out_channels: usize,
    pub subdivisions: Vec<SparseSubdivisionLogits>,
}

#[derive(Debug, Clone)]
pub(crate) struct SparseUnetDecoderRuntime {
    out_channels: usize,
    pred_subdiv: bool,
    voxel_margin: f32,
    compute_fp16: bool,
    model_channels: Vec<usize>,
    from_latent: LinearLayer,
    output_layer: LinearLayer,
    stages: Vec<DecoderStage>,
}

#[derive(Debug, Clone)]
struct DecoderStage {
    convnext_blocks: Vec<ConvNeXtBlock>,
    upsample_block: Option<C2SUpsampleBlock>,
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    conv: SparseConvLayer,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    mlp_0: LinearLayer,
    mlp_2: LinearLayer,
}

#[derive(Debug, Clone)]
struct C2SUpsampleBlock {
    in_channels: usize,
    out_channels: usize,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    to_subdiv: Option<LinearLayer>,
    conv1: SparseConvLayer,
    conv2: SparseConvLayer,
}

#[derive(Debug, Clone)]
struct LinearLayer {
    in_channels: usize,
    out_channels: usize,
    // Row-major [out, in] as stored by PyTorch linear layers.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SparseConvLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_channels_per_group: usize,
    out_channels_per_group: usize,
    groups: usize,
    // Row-major [out, kd, kh, kw, in_per_group]
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl SparseUnetDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = std::fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        let parsed: DecoderConfigFile = serde_json::from_slice(&config_bytes).map_err(|err| {
            format!(
                "failed to parse sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        if parsed.args.model_channels.is_empty() {
            return Err(format!(
                "sparse decoder config '{}' has empty model_channels",
                config_path.display()
            ));
        }
        if parsed.args.num_blocks.len() != parsed.args.model_channels.len() {
            return Err(format!(
                "sparse decoder config '{}' has mismatched num_blocks/model_channels lengths",
                config_path.display()
            ));
        }

        let weight_path = resolve_model_weight_candidates(
            model_stem,
            weights_root,
            image_large_root,
        )
        .into_iter()
        .find(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
        .ok_or_else(|| {
            format!("unable to resolve safetensors weights for sparse decoder stem '{model_stem}'")
        })?;

        let file = File::open(&weight_path).map_err(|err| {
            format!(
                "failed to open sparse decoder safetensors '{}': {err}",
                weight_path.display()
            )
        })?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
            format!(
                "failed to mmap sparse decoder safetensors '{}': {err}",
                weight_path.display()
            )
        })?;
        let safetensors = SafeTensors::deserialize(&mmap).map_err(|err| {
            format!(
                "failed to deserialize sparse decoder safetensors '{}': {err}",
                weight_path.display()
            )
        })?;

        let out_channels = parsed.args.out_channels.unwrap_or_else(|| {
            if parsed.name == "FlexiDualGridVaeDecoder" {
                7
            } else {
                6
            }
        });

        let from_latent = load_linear(
            &safetensors,
            "from_latent.weight",
            "from_latent.bias",
            parsed.args.latent_channels,
            parsed.args.model_channels[0],
        )?;
        let output_layer = load_linear(
            &safetensors,
            "output_layer.weight",
            "output_layer.bias",
            *parsed
                .args
                .model_channels
                .last()
                .expect("checked non-empty model_channels"),
            out_channels,
        )?;

        let mut stages = Vec::with_capacity(parsed.args.num_blocks.len());
        for stage_idx in 0..parsed.args.num_blocks.len() {
            let stage_channels = parsed.args.model_channels[stage_idx];
            let mut convnext_blocks = Vec::with_capacity(parsed.args.num_blocks[stage_idx]);
            for block_idx in 0..parsed.args.num_blocks[stage_idx] {
                let prefix = format!("blocks.{stage_idx}.{block_idx}");
                convnext_blocks.push(ConvNeXtBlock {
                    conv: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv.weight").as_str(),
                        format!("{prefix}.conv.bias").as_str(),
                        stage_channels,
                        stage_channels,
                    )?,
                    norm_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.weight").as_str(),
                        stage_channels,
                    )?,
                    norm_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_0: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.0.weight").as_str(),
                        format!("{prefix}.mlp.0.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_2: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.2.weight").as_str(),
                        format!("{prefix}.mlp.2.bias").as_str(),
                        0,
                    )?,
                });
            }

            let upsample_block = if stage_idx + 1 < parsed.args.model_channels.len() {
                let up_idx = parsed.args.num_blocks[stage_idx];
                let prefix = format!("blocks.{stage_idx}.{up_idx}");
                let in_channels = parsed.args.model_channels[stage_idx];
                let out_channels = parsed.args.model_channels[stage_idx + 1];
                let conv1_out = out_channels
                    .checked_mul(8)
                    .ok_or_else(|| "conv1_out channels overflow".to_string())?;
                let to_subdiv = match parsed.args.pred_subdiv.unwrap_or(true) {
                    true => Some(load_linear(
                        &safetensors,
                        format!("{prefix}.to_subdiv.weight").as_str(),
                        format!("{prefix}.to_subdiv.bias").as_str(),
                        in_channels,
                        8,
                    )?),
                    false => None,
                };

                Some(C2SUpsampleBlock {
                    in_channels,
                    out_channels,
                    norm1_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.weight").as_str(),
                        in_channels,
                    )?,
                    norm1_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.bias").as_str(),
                        in_channels,
                    )?,
                    to_subdiv,
                    conv1: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv1.weight").as_str(),
                        format!("{prefix}.conv1.bias").as_str(),
                        in_channels,
                        conv1_out,
                    )?,
                    conv2: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv2.weight").as_str(),
                        format!("{prefix}.conv2.bias").as_str(),
                        out_channels,
                        out_channels,
                    )?,
                })
            } else {
                None
            };

            stages.push(DecoderStage {
                convnext_blocks,
                upsample_block,
            });
        }

        Ok(Self {
            out_channels,
            pred_subdiv: parsed.args.pred_subdiv.unwrap_or(true),
            voxel_margin: parsed.args.voxel_margin.unwrap_or(0.5),
            compute_fp16: parsed.args.use_fp16.unwrap_or(false) && !decoder_force_fp32(),
            model_channels: parsed.args.model_channels,
            from_latent,
            output_layer,
            stages,
        })
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn pred_subdiv(&self) -> bool {
        self.pred_subdiv
    }

    pub fn voxel_margin(&self) -> f32 {
        self.voxel_margin
    }

    pub fn decode(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        let count = coords.len().min(rows.len());
        if count == 0 {
            return Ok(SparseDecodeResult {
                coords: Vec::new(),
                feats: Vec::new(),
                out_channels: self.out_channels,
                subdivisions: Vec::new(),
            });
        }

        let mut state_coords = coords[..count].to_vec();
        let mut state_feats = flatten_rows_32(&rows[..count]);
        state_feats = linear_forward(
            state_feats.as_slice(),
            count,
            &self.from_latent,
            "from_latent",
        )?;
        if self.compute_fp16 {
            quantize_f16_inplace(state_feats.as_mut_slice());
        }

        let mut subdivisions = Vec::new();
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let stage_channels = self.model_channels[stage_idx];
            for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
                let row_count = state_coords.len();
                if row_count == 0 {
                    break;
                }
                let residual = state_feats.clone();
                let mut h = sparse_subm_conv_forward(
                    state_coords.as_slice(),
                    state_feats.as_slice(),
                    &block.conv,
                    format!("stage {stage_idx} block {block_idx} conv").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                layer_norm_inplace(
                    h.as_mut_slice(),
                    row_count,
                    stage_channels,
                    Some(block.norm_weight.as_slice()),
                    Some(block.norm_bias.as_slice()),
                    LAYER_NORM32_EPS,
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                h = linear_forward(
                    h.as_slice(),
                    row_count,
                    &block.mlp_0,
                    format!("stage {stage_idx} block {block_idx} mlp_0").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                silu_inplace(h.as_mut_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                h = linear_forward(
                    h.as_slice(),
                    row_count,
                    &block.mlp_2,
                    format!("stage {stage_idx} block {block_idx} mlp_2").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                add_inplace(h.as_mut_slice(), residual.as_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                state_feats = h;
            }

            if let Some(up) = stage.upsample_block.as_ref() {
                let parent_coords = state_coords.clone();
                let parent_feats = state_feats.clone();
                let parent_rows = parent_coords.len();
                if parent_rows == 0 {
                    continue;
                }

                let subdiv_logits = if let Some(to_subdiv) = up.to_subdiv.as_ref() {
                    let mut logits = linear_forward(
                        parent_feats.as_slice(),
                        parent_rows,
                        to_subdiv,
                        format!("stage {stage_idx} to_subdiv").as_str(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(logits.as_mut_slice());
                    }
                    if should_center_subdivision_logits() {
                        row_center_logits(logits.as_mut_slice(), parent_rows);
                    }
                    logits
                } else {
                    let guide = guide_subdivisions
                        .and_then(|levels| levels.get(stage_idx))
                        .ok_or_else(|| {
                            format!(
                                "decoder stage {stage_idx} requires guide_subdivisions but none were provided"
                            )
                        })?;
                    map_guide_subdivision_logits(parent_coords.as_slice(), guide)?
                };

                let subdivision_mask =
                    logits_to_mask(subdiv_logits.as_slice(), parent_rows, false)?;
                if self.pred_subdiv {
                    subdivisions.push(SparseSubdivisionLogits {
                        spatial_shape: spatial_shape_from_coords(parent_coords.as_slice()),
                        coords: parent_coords.clone(),
                        logits: subdiv_logits.clone(),
                    });
                }

                let mut h_norm = parent_feats.clone();
                layer_norm_inplace(
                    h_norm.as_mut_slice(),
                    parent_rows,
                    up.in_channels,
                    Some(up.norm1_weight.as_slice()),
                    Some(up.norm1_bias.as_slice()),
                    LAYER_NORM32_EPS,
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h_norm.as_mut_slice());
                }
                silu_inplace(h_norm.as_mut_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h_norm.as_mut_slice());
                }
                let h_conv1 = sparse_subm_conv_forward(
                    parent_coords.as_slice(),
                    h_norm.as_slice(),
                    &up.conv1,
                    format!("stage {stage_idx} up conv1").as_str(),
                )?;
                let mut h_conv1 = h_conv1;
                if self.compute_fp16 {
                    quantize_f16_inplace(h_conv1.as_mut_slice());
                }
                let (child_coords, mut h_up) = channel2spatial(
                    parent_coords.as_slice(),
                    h_conv1.as_slice(),
                    up.out_channels
                        .checked_mul(8)
                        .ok_or_else(|| "up.out_channels * 8 overflow".to_string())?,
                    subdivision_mask.as_slice(),
                )?;
                let (child_coords_skip, x_up) = channel2spatial(
                    parent_coords.as_slice(),
                    parent_feats.as_slice(),
                    up.in_channels,
                    subdivision_mask.as_slice(),
                )?;
                if child_coords != child_coords_skip {
                    return Err(format!(
                        "decoder stage {stage_idx} channel2spatial coord mismatch between conv and skip branches"
                    ));
                }

                let skip_in_channels = up.in_channels / 8;
                if skip_in_channels == 0 || up.out_channels % skip_in_channels != 0 {
                    return Err(format!(
                        "decoder stage {stage_idx} invalid skip channel ratio (in={}, out={})",
                        up.in_channels, up.out_channels
                    ));
                }
                let repeat_factor = up.out_channels / skip_in_channels;
                let skip = repeat_interleave_channels(
                    x_up.as_slice(),
                    child_coords.len(),
                    skip_in_channels,
                    repeat_factor,
                );

                layer_norm_inplace(
                    h_up.as_mut_slice(),
                    child_coords.len(),
                    up.out_channels,
                    None,
                    None,
                    LAYER_NORM32_EPS,
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h_up.as_mut_slice());
                }
                silu_inplace(h_up.as_mut_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h_up.as_mut_slice());
                }
                let mut h = sparse_subm_conv_forward(
                    child_coords.as_slice(),
                    h_up.as_slice(),
                    &up.conv2,
                    format!("stage {stage_idx} up conv2").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                add_inplace(h.as_mut_slice(), skip.as_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                state_coords = child_coords;
                state_feats = h;
            }
        }

        let rows_final = state_coords.len();
        layer_norm_inplace(
            state_feats.as_mut_slice(),
            rows_final,
            *self
                .model_channels
                .last()
                .expect("checked non-empty model_channels"),
            None,
            None,
            F_LAYER_NORM_EPS,
        )?;
        let state_feats = linear_forward(
            state_feats.as_slice(),
            rows_final,
            &self.output_layer,
            "output_layer",
        )?;

        Ok(SparseDecodeResult {
            coords: state_coords,
            feats: state_feats,
            out_channels: self.out_channels,
            subdivisions,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stage0_subdivision_logits(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseSubdivisionLogits, String> {
        if self.stages.is_empty() {
            return Err("decoder has no stages".to_string());
        }
        let stage = &self.stages[0];
        let up = stage
            .upsample_block
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no upsample block".to_string())?;
        let to_subdiv = up
            .to_subdiv
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no to_subdiv head".to_string())?;

        let count = coords.len().min(rows.len());
        if count == 0 {
            return Ok(SparseSubdivisionLogits {
                coords: Vec::new(),
                logits: Vec::new(),
                spatial_shape: [1, 1, 1],
            });
        }

        let state_coords = coords[..count].to_vec();
        let mut state_feats = flatten_rows_32(&rows[..count]);
        state_feats = linear_forward(
            state_feats.as_slice(),
            count,
            &self.from_latent,
            "from_latent(stage0)",
        )?;
        if self.compute_fp16 {
            quantize_f16_inplace(state_feats.as_mut_slice());
        }

        let stage_channels = self.model_channels[0];
        for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
            let row_count = state_coords.len();
            if row_count == 0 {
                break;
            }
            let residual = state_feats.clone();
            let mut h = sparse_subm_conv_forward(
                state_coords.as_slice(),
                state_feats.as_slice(),
                &block.conv,
                format!("stage0 block {block_idx} conv(stage0)").as_str(),
            )?;
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            layer_norm_inplace(
                h.as_mut_slice(),
                row_count,
                stage_channels,
                Some(block.norm_weight.as_slice()),
                Some(block.norm_bias.as_slice()),
                LAYER_NORM32_EPS,
            )?;
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            h = linear_forward(
                h.as_slice(),
                row_count,
                &block.mlp_0,
                format!("stage0 block {block_idx} mlp_0(stage0)").as_str(),
            )?;
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            silu_inplace(h.as_mut_slice());
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            h = linear_forward(
                h.as_slice(),
                row_count,
                &block.mlp_2,
                format!("stage0 block {block_idx} mlp_2(stage0)").as_str(),
            )?;
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            add_inplace(h.as_mut_slice(), residual.as_slice());
            if self.compute_fp16 {
                quantize_f16_inplace(h.as_mut_slice());
            }
            state_feats = h;
        }

        let mut subdiv_logits = linear_forward(
            state_feats.as_slice(),
            state_coords.len(),
            to_subdiv,
            "stage0 to_subdiv",
        )?;
        if self.compute_fp16 {
            quantize_f16_inplace(subdiv_logits.as_mut_slice());
        }
        if should_center_subdivision_logits() {
            row_center_logits(subdiv_logits.as_mut_slice(), state_coords.len());
        }

        Ok(SparseSubdivisionLogits {
            spatial_shape: spatial_shape_from_coords(state_coords.as_slice()),
            coords: state_coords,
            logits: subdiv_logits,
        })
    }
}

fn load_linear(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
    })
}

fn load_linear_dynamic(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
    })
}

fn load_sparse_conv(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<SparseConvLayer, String> {
    let (w_shape, weight) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 5 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=5, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let kd = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let in_channels_per_group = w_shape[4];
    if kd == 0 || kh == 0 || kw == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid kernel dims ({kd},{kh},{kw})"
        ));
    }
    if in_channels_per_group == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid in_channels_per_group=0"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }
    let in_channels = if expected_in > 0 {
        expected_in
    } else {
        in_channels_per_group
    };
    if in_channels < in_channels_per_group || !in_channels.is_multiple_of(in_channels_per_group) {
        return Err(format!(
            "tensor '{weight_key}' expected_in={in_channels} is incompatible with in_per_group={in_channels_per_group}"
        ));
    }
    let groups = in_channels / in_channels_per_group;
    if groups == 0 || !out_channels.is_multiple_of(groups) {
        return Err(format!(
            "tensor '{weight_key}' has incompatible grouped channels (groups={groups}, out_channels={out_channels})"
        ));
    }
    let out_channels_per_group = out_channels / groups;

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let expected_weight_len = out_channels
        .checked_mul(kd)
        .and_then(|value| value.checked_mul(kh))
        .and_then(|value| value.checked_mul(kw))
        .and_then(|value| value.checked_mul(in_channels_per_group))
        .ok_or_else(|| format!("tensor '{weight_key}' weight shape product overflow"))?;
    if weight.len() != expected_weight_len {
        return Err(format!(
            "tensor '{weight_key}' element count mismatch: expected {expected_weight_len}, got {}",
            weight.len()
        ));
    }

    Ok(SparseConvLayer {
        in_channels,
        out_channels,
        kernel_d: kd,
        kernel_h: kh,
        kernel_w: kw,
        in_channels_per_group,
        out_channels_per_group,
        groups,
        weight,
        bias,
    })
}

fn load_vector(
    safetensors: &SafeTensors<'_>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let (shape, data) = load_tensor_f32(safetensors, key)?;
    if shape.len() != 1 {
        return Err(format!(
            "tensor '{key}' expected rank=1, got rank={}",
            shape.len()
        ));
    }
    if expected_len > 0 && shape[0] != expected_len {
        return Err(format!(
            "tensor '{key}' expected len={expected_len}, got len={}",
            shape[0]
        ));
    }
    Ok(data)
}

fn load_tensor_f32(
    safetensors: &SafeTensors<'_>,
    key: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let view = safetensors
        .tensor(key)
        .map_err(|err| format!("missing tensor '{key}' in safetensors: {err}"))?;
    let shape = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F32 => bytes_to_f32(view.data())?,
        Dtype::F16 => bytes_to_f16(view.data())?,
        Dtype::BF16 => bytes_to_bf16(view.data())?,
        other => {
            return Err(format!(
                "tensor '{key}' has unsupported dtype {other:?}; expected f32/f16/bf16"
            ));
        }
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| format!("tensor '{key}' shape product overflow: {:?}", shape))?;
    if data.len() != expected {
        return Err(format!(
            "tensor '{key}' element count mismatch: expected {expected}, got {}",
            data.len()
        ));
    }
    Ok((shape, data))
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid f32 tensor payload byte length {}; must be divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bytes_to_f16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid f16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn bytes_to_bf16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid bf16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn flatten_rows_32(rows: &[[f32; 32]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * 32);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn linear_forward(
    input: &[f32],
    rows: usize,
    layer: &LinearLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows
        .checked_mul(layer.in_channels)
        .ok_or_else(|| format!("{context}: input size overflow"))?;
    if input.len() != expected {
        return Err(format!(
            "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
            input.len(),
            expected,
            rows,
            layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }
    let mut output = vec![0.0f32; rows * layer.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * layer.out_channels;
        output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
    }
    unsafe {
        matrixmultiply::sgemm(
            rows,
            layer.in_channels,
            layer.out_channels,
            1.0,
            input.as_ptr(),
            layer.in_channels as isize,
            1,
            layer.weight.as_ptr(),
            1,
            layer.in_channels as isize,
            1.0,
            output.as_mut_ptr(),
            layer.out_channels as isize,
            1,
        );
    }
    Ok(output)
}

fn sparse_subm_conv_forward(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows
        .checked_mul(layer.in_channels)
        .ok_or_else(|| format!("{context}: input size overflow"))?;
    if input.len() != expected {
        return Err(format!(
            "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
            input.len(),
            expected,
            rows,
            layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }

    let mut output = vec![0.0f32; rows * layer.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * layer.out_channels;
        output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
    }

    let mut coord_to_row = HashMap::with_capacity(rows.saturating_mul(2));
    for (row_idx, coord) in coords.iter().copied().enumerate() {
        coord_to_row.insert(coord, row_idx);
    }

    let center_d = (layer.kernel_d / 2) as i32;
    let center_h = (layer.kernel_h / 2) as i32;
    let center_w = (layer.kernel_w / 2) as i32;
    let axis_order = conv_kernel_axis_order();
    let axis_sign = conv_kernel_axis_signs();
    for (out_row_idx, out_coord) in coords.iter().copied().enumerate().take(rows) {
        let batch = out_coord[0];
        let ox = out_coord[1] as i32;
        let oy = out_coord[2] as i32;
        let oz = out_coord[3] as i32;
        let out_base = out_row_idx * layer.out_channels;

        for kd_idx in 0..layer.kernel_d {
            for kh_idx in 0..layer.kernel_h {
                for kw_idx in 0..layer.kernel_w {
                    let deltas = [
                        axis_sign[0] * (kd_idx as i32 - center_d),
                        axis_sign[1] * (kh_idx as i32 - center_h),
                        axis_sign[2] * (kw_idx as i32 - center_w),
                    ];
                    let mut spatial = [ox, oy, oz];
                    spatial[axis_order[0]] += deltas[0];
                    spatial[axis_order[1]] += deltas[1];
                    spatial[axis_order[2]] += deltas[2];
                    if spatial[0] < 0 || spatial[1] < 0 || spatial[2] < 0 {
                        continue;
                    }
                    let neighbor = [
                        batch,
                        spatial[0] as u32,
                        spatial[1] as u32,
                        spatial[2] as u32,
                    ];
                    let Some(in_row_idx) = coord_to_row.get(&neighbor).copied() else {
                        continue;
                    };
                    let in_row = &input
                        [in_row_idx * layer.in_channels..(in_row_idx + 1) * layer.in_channels];
                    for group_idx in 0..layer.groups {
                        let in_group_base = group_idx * layer.in_channels_per_group;
                        let out_group_base = group_idx * layer.out_channels_per_group;
                        for out_local in 0..layer.out_channels_per_group {
                            let out_idx = out_group_base + out_local;
                            let weight_base =
                                (((out_idx * layer.kernel_d + kd_idx) * layer.kernel_h + kh_idx)
                                    * layer.kernel_w
                                    + kw_idx)
                                    * layer.in_channels_per_group;
                            let mut accum = 0.0f32;
                            for in_local in 0..layer.in_channels_per_group {
                                accum += in_row[in_group_base + in_local]
                                    * layer.weight[weight_base + in_local];
                            }
                            output[out_base + out_idx] += accum;
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

fn conv_kernel_axis_order() -> [usize; 3] {
    match std::env::var("TRELLIS2_CONV_AXIS_ORDER")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("xzy") => [0, 2, 1],
        Some("yxz") => [1, 0, 2],
        Some("yzx") => [1, 2, 0],
        Some("zxy") => [2, 0, 1],
        Some("zyx") => [2, 1, 0],
        _ => [0, 1, 2],
    }
}

fn conv_kernel_axis_signs() -> [i32; 3] {
    let raw = std::env::var("TRELLIS2_CONV_AXIS_SIGN")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "+++".to_string());
    let mut signs = [1i32, 1, 1];
    for (idx, ch) in raw.chars().take(3).enumerate() {
        signs[idx] = if ch == '-' { -1 } else { 1 };
    }
    signs
}

fn layer_norm_inplace(
    data: &mut [f32],
    rows: usize,
    channels: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), String> {
    if rows == 0 || channels == 0 {
        return Ok(());
    }
    if data.len() != rows * channels {
        return Err(format!(
            "layer_norm_inplace: invalid data len {}, expected {}",
            data.len(),
            rows * channels
        ));
    }
    if let Some(weight) = weight
        && weight.len() != channels
    {
        return Err(format!(
            "layer_norm_inplace: invalid weight len {}, expected {}",
            weight.len(),
            channels
        ));
    }
    if let Some(bias) = bias
        && bias.len() != channels
    {
        return Err(format!(
            "layer_norm_inplace: invalid bias len {}, expected {}",
            bias.len(),
            channels
        ));
    }

    for row_idx in 0..rows {
        let base = row_idx * channels;
        let row = &mut data[base..base + channels];
        let mean = row.iter().copied().sum::<f32>() / channels as f32;
        let var = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ch in 0..channels {
            let mut value = (row[ch] - mean) * inv_std;
            if let Some(weight) = weight {
                value *= weight[ch];
            }
            if let Some(bias) = bias {
                value += bias[ch];
            }
            row[ch] = value;
        }
    }
    Ok(())
}

fn silu_inplace(data: &mut [f32]) {
    for value in data {
        *value = *value / (1.0 + (-*value).exp());
    }
}

fn quantize_f16_inplace(data: &mut [f32]) {
    for value in data {
        *value = f16::from_f32(*value).to_f32();
    }
}

fn row_center_logits(data: &mut [f32], rows: usize) {
    if rows == 0 {
        return;
    }
    if data.len() != rows * 8 {
        return;
    }
    for row_idx in 0..rows {
        let row = &mut data[row_idx * 8..(row_idx + 1) * 8];
        let mean = row.iter().copied().sum::<f32>() / 8.0;
        for value in row {
            *value -= mean;
        }
    }
}

fn should_center_subdivision_logits() -> bool {
    std::env::var("TRELLIS2_DECODER_CENTER_SUBDIV_LOGITS")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn decoder_force_fp32() -> bool {
    std::env::var("TRELLIS2_DECODER_FORCE_FP32")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn add_inplace(lhs: &mut [f32], rhs: &[f32]) {
    if lhs.len() != rhs.len() {
        return;
    }
    for (left, right) in lhs.iter_mut().zip(rhs.iter()) {
        *left += *right;
    }
}

fn logits_to_mask(
    logits: &[f32],
    rows: usize,
    enforce_non_empty: bool,
) -> Result<Vec<[bool; 8]>, String> {
    if logits.len() != rows * 8 {
        return Err(format!(
            "subdivision logits len {} does not match rows*8={}",
            logits.len(),
            rows * 8
        ));
    }
    let mut out = Vec::with_capacity(rows);
    let max_children = decoder_max_children_per_parent();
    for row_idx in 0..rows {
        let mut mask = [false; 8];
        let row = &logits[row_idx * 8..(row_idx + 1) * 8];
        for child in 0..8 {
            mask[child] = row[child] > 0.0;
        }
        if let Some(max_children) = max_children {
            let selected = mask.iter().filter(|flag| **flag).count();
            if selected > max_children {
                let mut order = (0..8usize).collect::<Vec<_>>();
                order.sort_by(|a, b| {
                    row[*b]
                        .partial_cmp(&row[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut limited = [false; 8];
                for idx in order.into_iter().take(max_children) {
                    limited[idx] = true;
                }
                mask = limited;
            }
        }
        if enforce_non_empty && !mask.iter().any(|flag| *flag) {
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (idx, value) in row.iter().enumerate().skip(1) {
                if *value > best_val {
                    best_val = *value;
                    best_idx = idx;
                }
            }
            mask[best_idx] = true;
        }
        out.push(mask);
    }
    Ok(out)
}

fn decoder_max_children_per_parent() -> Option<usize> {
    if let Ok(value) = std::env::var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT")
        && let Ok(parsed) = value.trim().parse::<usize>()
    {
        if parsed == 0 {
            return None;
        }
        return Some(parsed.min(8));
    }

    if env_flag("TRELLIS2_PARITY_STRICT") || env_flag("TRELLIS2_DECODER_UNCAPPED") {
        return None;
    }
    // Default to uncapped subdivision for decoder parity.
    None
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn channel2spatial(
    coords: &[[u32; 4]],
    feats: &[f32],
    in_channels: usize,
    subdivision_mask: &[[bool; 8]],
) -> Result<(Vec<[u32; 4]>, Vec<f32>), String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if feats.len() != rows * in_channels {
        return Err(format!(
            "channel2spatial: invalid feats len {}, expected {}",
            feats.len(),
            rows * in_channels
        ));
    }
    if !in_channels.is_multiple_of(8) {
        return Err(format!(
            "channel2spatial: in_channels={} is not divisible by 8",
            in_channels
        ));
    }
    if subdivision_mask.len() != rows {
        return Err(format!(
            "channel2spatial: subdivision rows {} do not match coords rows {}",
            subdivision_mask.len(),
            rows
        ));
    }

    let out_channels = in_channels / 8;
    let mut out_coords = Vec::new();
    let mut out_feats = Vec::new();
    for row_idx in 0..rows {
        let coord = coords[row_idx];
        let row_feats = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for (child, selected) in subdivision_mask[row_idx].iter().enumerate().take(8usize) {
            if !*selected {
                continue;
            }
            let cx = (child & 1) as u32;
            let cy = ((child >> 1) & 1) as u32;
            let cz = ((child >> 2) & 1) as u32;
            out_coords.push([
                coord[0],
                coord[1].saturating_mul(2).saturating_add(cx),
                coord[2].saturating_mul(2).saturating_add(cy),
                coord[3].saturating_mul(2).saturating_add(cz),
            ]);
            let child_base = child * out_channels;
            out_feats.extend_from_slice(&row_feats[child_base..child_base + out_channels]);
        }
    }
    Ok((out_coords, out_feats))
}

fn repeat_interleave_channels(
    feats: &[f32],
    rows: usize,
    in_channels: usize,
    repeat_factor: usize,
) -> Vec<f32> {
    if rows == 0 || in_channels == 0 || repeat_factor == 0 {
        return Vec::new();
    }
    let out_channels = in_channels * repeat_factor;
    let mut out = Vec::with_capacity(rows * out_channels);
    for row_idx in 0..rows {
        let row = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for value in row {
            for _ in 0..repeat_factor {
                out.push(*value);
            }
        }
    }
    out
}

fn map_guide_subdivision_logits(
    coords: &[[u32; 4]],
    guide: &SparseSubdivisionLogits,
) -> Result<Vec<f32>, String> {
    if guide.logits.len() != guide.coords.len() * 8 {
        return Err(format!(
            "guide subdivision logits invalid length: logits={} coords={}",
            guide.logits.len(),
            guide.coords.len()
        ));
    }
    let mut map = HashMap::with_capacity(guide.coords.len() * 2);
    for (idx, coord) in guide.coords.iter().enumerate() {
        let row = &guide.logits[idx * 8..(idx + 1) * 8];
        map.insert(*coord, row.to_vec());
    }

    let mut out = Vec::with_capacity(coords.len() * 8);
    let strict = env_flag("TRELLIS2_PARITY_STRICT");
    for coord in coords {
        if let Some(row) = map.get(coord) {
            out.extend_from_slice(row);
        } else if strict {
            return Err(format!(
                "guide subdivision logits missing coord {:?} in parity strict mode",
                coord
            ));
        } else {
            // If the guide row is missing, keep all children disabled.
            out.extend_from_slice(&[-1.0; 8]);
        }
    }
    Ok(out)
}

fn spatial_shape_from_coords(coords: &[[u32; 4]]) -> [u32; 3] {
    if coords.is_empty() {
        return [1, 1, 1];
    }
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut max_z = 0u32;
    for coord in coords {
        max_x = max_x.max(coord[1]);
        max_y = max_y.max(coord[2]);
        max_z = max_z.max(coord[3]);
    }
    [
        max_x.saturating_add(1),
        max_y.saturating_add(1),
        max_z.saturating_add(1),
    ]
}

fn resolve_model_weight_candidates(
    model_stem: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> Vec<PathBuf> {
    let source =
        resolve_model_source_path(model_stem, "safetensors", weights_root, image_large_root);
    let burnpack = source.with_extension("bpk");
    let burnpack_f16 = with_file_stem_suffix(&burnpack, F16_SUFFIX);
    let source_f16 = with_file_stem_suffix(&source, F16_SUFFIX);
    let prefer_f16 = prefer_f16_burnpack();
    let candidates = if prefer_f16 {
        vec![source_f16, source, burnpack_f16, burnpack]
    } else {
        vec![source, source_f16, burnpack, burnpack_f16]
    };
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    let precision = std::env::var("TRELLIS2_BPK_PRECISION")
        .ok()
        .or_else(|| std::env::var("BURN_SYNTH_BPK_PRECISION").ok());
    match precision
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("f32" | "fp32" | "float32" | "32") => false,
        Some("f16" | "fp16" | "float16" | "half" | "16") => true,
        Some(_) | None => true,
    }
}

fn resolve_model_source_path(
    stem: &str,
    ext: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        let image_large_root = image_large_root.unwrap_or(weights_root);
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{
        LinearLayer, SparseConvLayer, linear_forward, logits_to_mask, sparse_subm_conv_forward,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn make_unit_conv_3x1x1(weight: [f32; 3]) -> SparseConvLayer {
        SparseConvLayer {
            in_channels: 1,
            out_channels: 1,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 1,
            out_channels_per_group: 1,
            groups: 1,
            weight: weight.to_vec(),
            bias: vec![0.0],
        }
    }

    #[test]
    fn sparse_conv_uses_neighbor_voxels() {
        let coords = vec![[0, 0, 0, 0], [0, 1, 0, 0]];
        let input = vec![1.0f32, 2.0f32];
        // kernel offsets: [-1, 0, +1]
        let layer = make_unit_conv_3x1x1([10.0, 1.0, 100.0]);

        let output =
            sparse_subm_conv_forward(coords.as_slice(), input.as_slice(), &layer, "test conv")
                .expect("sparse conv should succeed");
        assert_eq!(output.len(), 2);
        // x=0: center(1*1) + right-neighbor(2*100)
        assert!((output[0] - 201.0).abs() < 1.0e-5);
        // x=1: left-neighbor(1*10) + center(2*1)
        assert!((output[1] - 12.0).abs() < 1.0e-5);
    }

    #[test]
    fn decoder_default_child_cap_is_uncapped_without_strict_mode() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
    }

    #[test]
    fn parity_strict_defaults_to_uncapped_children() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::set_var("TRELLIS2_PARITY_STRICT", "1");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
        }
    }

    #[test]
    fn explicit_zero_child_cap_env_means_uncapped() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::set_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT", "0");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
    }

    #[test]
    fn linear_forward_matches_naive_matmul() {
        let layer = LinearLayer {
            in_channels: 3,
            out_channels: 2,
            // [out, in]
            weight: vec![
                1.0, 2.0, 3.0, // out0
                -1.0, 0.5, 4.0, // out1
            ],
            bias: vec![0.25, -0.5],
        };
        let input = vec![
            2.0, -1.0, 0.5, // row0
            -3.0, 4.0, 1.0, // row1
        ];
        let output = linear_forward(input.as_slice(), 2, &layer, "test linear")
            .expect("linear forward should succeed");
        assert_eq!(output.len(), 4);

        let mut expected = Vec::new();
        for row in 0..2 {
            let x = &input[row * 3..(row + 1) * 3];
            // out0
            expected.push(layer.bias[0] + x[0] * 1.0 + x[1] * 2.0 + x[2] * 3.0);
            // out1
            expected.push(layer.bias[1] - x[0] + x[1] * 0.5 + x[2] * 4.0);
        }
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-5, "got={got} want={want}");
        }
    }
}
