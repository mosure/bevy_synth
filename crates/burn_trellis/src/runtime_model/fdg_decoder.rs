use std::path::Path;

use super::sparse_decoder::{
    SparseDecodeResult, SparseSubdivisionLogits, SparseUnetDecoderRuntime, SparseUpsampledCoords,
};
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::DefaultWgpuBackend;

#[derive(Debug, Clone)]
pub(crate) struct FdgDecodedOutput {
    pub coords: Vec<[u32; 4]>,
    pub vertices: Vec<[f32; 3]>,
    pub intersected: Vec<[bool; 3]>,
    pub intersection_logits: Vec<[f32; 3]>,
    pub quad_lerp: Vec<f32>,
    pub subdivisions: Vec<SparseSubdivisionLogits>,
}

#[derive(Debug)]
pub(crate) struct FdgDecoderRuntime {
    inner: SparseUnetDecoderRuntime,
}

impl FdgDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        _prefer_wgpu: bool,
    ) -> Result<Self, String> {
        let inner =
            SparseUnetDecoderRuntime::load_from_stem(weights_root, image_large_root, model_stem)?;
        if inner.out_channels() < 7 {
            return Err(format!(
                "fdg decoder runtime out_channels={} is below required 7",
                inner.out_channels()
            ));
        }
        if !inner.pred_subdiv() {
            return Err("fdg decoder runtime expects pred_subdiv=true".to_string());
        }
        Ok(Self { inner })
    }

    pub fn out_channels(&self) -> usize {
        self.inner.out_channels()
    }

    pub fn voxel_margin(&self) -> f32 {
        self.inner.voxel_margin()
    }

    #[allow(dead_code)]
    pub fn decode_sparse(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<FdgDecodedOutput, String> {
        let decoded = self.decode_sparse_result(coords, rows)?;
        decode_fdg_outputs(&decoded, self.voxel_margin())
    }

    #[allow(dead_code)]
    pub fn decode_with_guidance(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: &[SparseSubdivisionLogits],
    ) -> Result<FdgDecodedOutput, String> {
        let decoded = self.decode_with_guidance_result(coords, rows, guide_subdivisions)?;
        decode_fdg_outputs(&decoded, self.voxel_margin())
    }

    pub fn decode_sparse_result(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseDecodeResult, String> {
        self.inner.decode(coords, rows, None)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn decode_sparse_result_with_tensors(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
    ) -> Result<SparseDecodeResult, String> {
        self.inner.decode_with_tensors(coords_wgpu, rows_wgpu, None)
    }

    pub fn decode_with_guidance_result(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: &[SparseSubdivisionLogits],
    ) -> Result<SparseDecodeResult, String> {
        self.inner.decode(coords, rows, Some(guide_subdivisions))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn decode_with_guidance_result_with_tensors(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        guide_subdivisions: &[SparseSubdivisionLogits],
    ) -> Result<SparseDecodeResult, String> {
        self.inner
            .decode_with_tensors(coords_wgpu, rows_wgpu, Some(guide_subdivisions))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stage0_subdivision_logits(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseSubdivisionLogits, String> {
        self.inner.stage0_subdivision_logits(coords, rows)
    }

    pub fn upsample_coords_result(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        self.inner
            .upsample_coords_result(coords, rows, upsample_times)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn upsample_coords_result_with_tensors(
        &self,
        coords_wgpu: Tensor<DefaultWgpuBackend, 2, Int>,
        rows_wgpu: Tensor<DefaultWgpuBackend, 2>,
        upsample_times: usize,
    ) -> Result<SparseUpsampledCoords, String> {
        self.inner
            .upsample_coords_result_with_tensors(coords_wgpu, rows_wgpu, upsample_times)
    }
}

pub(crate) fn decode_fdg_outputs(
    decoded: &SparseDecodeResult,
    voxel_margin: f32,
) -> Result<FdgDecodedOutput, String> {
    let coords = decoded.coords_host("fdg decoder coords materialization")?;
    let feats = decoded.feats_host("fdg decoder feats materialization")?;
    decode_fdg_outputs_from_host(
        coords,
        feats.as_slice(),
        decoded.out_channels,
        decoded.subdivisions.as_slice(),
        voxel_margin,
    )
}

pub(crate) fn decode_fdg_outputs_from_host(
    coords: Vec<[u32; 4]>,
    feats: &[f32],
    out_channels: usize,
    subdivisions: &[SparseSubdivisionLogits],
    voxel_margin: f32,
) -> Result<FdgDecodedOutput, String> {
    if out_channels < 7 {
        return Err(format!(
            "fdg decoder expected at least 7 channels, got {}",
            out_channels
        ));
    }
    let row_count = coords.len();
    if feats.len() != row_count * out_channels {
        return Err(format!(
            "fdg decoder output feats len {} does not match rows*out_channels = {}",
            feats.len(),
            row_count * out_channels
        ));
    }

    let mut vertices = Vec::with_capacity(row_count);
    let mut intersected = Vec::with_capacity(row_count);
    let mut intersection_logits = Vec::with_capacity(row_count);
    let mut quad_lerp = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let row = &feats[row_idx * out_channels..(row_idx + 1) * out_channels];
        let vx = (1.0 + 2.0 * voxel_margin) * sigmoid(row[0]) - voxel_margin;
        let vy = (1.0 + 2.0 * voxel_margin) * sigmoid(row[1]) - voxel_margin;
        let vz = (1.0 + 2.0 * voxel_margin) * sigmoid(row[2]) - voxel_margin;
        vertices.push([vx, vy, vz]);
        let logits = [row[3], row[4], row[5]];
        intersected.push([logits[0] > 0.0, logits[1] > 0.0, logits[2] > 0.0]);
        intersection_logits.push(logits);
        quad_lerp.push(softplus(row[6]));
    }

    Ok(FdgDecodedOutput {
        coords,
        vertices,
        intersected,
        intersection_logits,
        quad_lerp,
        subdivisions: subdivisions.to_vec(),
    })
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}
