use std::path::Path;

use super::sparse_decoder::{
    SparseDecodeResult, SparseSubdivisionLogits, SparseUnetDecoderRuntime,
};
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::DefaultWgpuBackend;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TexDecodedOutput {
    pub coords: Vec<[u32; 4]>,
    pub attrs: Vec<[f32; 6]>,
}

#[derive(Debug)]
pub(crate) struct SparseUnetVaeDecoderRuntime {
    inner: SparseUnetDecoderRuntime,
}

impl SparseUnetVaeDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        _prefer_wgpu: bool,
    ) -> Result<Self, String> {
        let inner =
            SparseUnetDecoderRuntime::load_from_stem(weights_root, image_large_root, model_stem)?;
        if inner.out_channels() < 6 {
            return Err(format!(
                "sparse unet vae decoder runtime out_channels={} is below required 6",
                inner.out_channels()
            ));
        }
        if inner.pred_subdiv() {
            return Err("sparse unet vae decoder runtime expects pred_subdiv=false".to_string());
        }
        Ok(Self { inner })
    }

    pub fn out_channels(&self) -> usize {
        self.inner.out_channels()
    }

    #[allow(dead_code)]
    pub fn decode_with_guidance(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: &[SparseSubdivisionLogits],
    ) -> Result<TexDecodedOutput, String> {
        let decoded = self.decode_with_guidance_result(coords, rows, guide_subdivisions)?;
        decode_tex_outputs(&decoded)
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
}

pub(crate) fn decode_tex_outputs(decoded: &SparseDecodeResult) -> Result<TexDecodedOutput, String> {
    let decoded_coords = decoded.coords_host("tex decoder coords materialization")?;
    let decoded_feats = decoded.feats_host("tex decoder feats materialization")?;
    decode_tex_outputs_from_host(
        decoded_coords,
        decoded_feats.as_slice(),
        decoded.out_channels,
    )
}

pub(crate) fn decode_tex_outputs_from_host(
    decoded_coords: Vec<[u32; 4]>,
    decoded_feats: &[f32],
    out_channels: usize,
) -> Result<TexDecodedOutput, String> {
    if out_channels < 6 {
        return Err(format!(
            "sparse unet vae decoder expected at least 6 channels, got {}",
            out_channels
        ));
    }
    if decoded_feats.len() != decoded_coords.len() * out_channels {
        return Err(format!(
            "tex decoder output feats len {} does not match rows*out_channels = {}",
            decoded_feats.len(),
            decoded_coords.len() * out_channels
        ));
    }

    let mut attrs = Vec::with_capacity(decoded_coords.len());
    for row_idx in 0..decoded_coords.len() {
        let row = &decoded_feats[row_idx * out_channels..(row_idx + 1) * out_channels];
        let mut attr = [0.0f32; 6];
        for ch in 0..6 {
            // Python decode_tex_slat postprocessing: ret * 0.5 + 0.5
            attr[ch] = row[ch] * 0.5 + 0.5;
        }
        attrs.push(attr);
    }

    Ok(TexDecodedOutput {
        coords: decoded_coords,
        attrs,
    })
}

pub(crate) fn decode_tex_attrs_from_host(
    decoded_feats: &[f32],
    out_channels: usize,
    expected_rows: Option<usize>,
) -> Result<Vec<[f32; 6]>, String> {
    if out_channels < 6 {
        return Err(format!(
            "sparse unet vae decoder expected at least 6 channels, got {}",
            out_channels
        ));
    }
    if decoded_feats.len() % out_channels != 0 {
        return Err(format!(
            "tex decoder feats len {} is not divisible by out_channels={}",
            decoded_feats.len(),
            out_channels
        ));
    }
    let row_count = decoded_feats.len() / out_channels;
    if let Some(expected) = expected_rows
        && row_count != expected
    {
        return Err(format!(
            "tex decoder output row mismatch: expected_rows={} actual_rows={}",
            expected, row_count
        ));
    }

    let mut attrs = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let row = &decoded_feats[row_idx * out_channels..(row_idx + 1) * out_channels];
        let mut attr = [0.0f32; 6];
        for ch in 0..6 {
            // Python decode_tex_slat postprocessing: ret * 0.5 + 0.5
            attr[ch] = row[ch] * 0.5 + 0.5;
        }
        attrs.push(attr);
    }
    Ok(attrs)
}
