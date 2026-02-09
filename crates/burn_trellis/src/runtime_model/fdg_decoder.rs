use std::path::Path;

use super::sparse_decoder::{
    SparseDecodeResult, SparseSubdivisionLogits, SparseUnetDecoderRuntime,
};

#[derive(Debug, Clone)]
pub(crate) struct FdgDecodedOutput {
    pub coords: Vec<[u32; 4]>,
    pub vertices: Vec<[f32; 3]>,
    pub intersected: Vec<[bool; 3]>,
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

    pub fn decode_sparse(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<FdgDecodedOutput, String> {
        let decoded = self.inner.decode(coords, rows, None)?;
        decode_fdg_outputs(&decoded, self.voxel_margin())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stage0_subdivision_logits(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseSubdivisionLogits, String> {
        self.inner.stage0_subdivision_logits(coords, rows)
    }
}

fn decode_fdg_outputs(
    decoded: &SparseDecodeResult,
    voxel_margin: f32,
) -> Result<FdgDecodedOutput, String> {
    if decoded.out_channels < 7 {
        return Err(format!(
            "fdg decoder expected at least 7 channels, got {}",
            decoded.out_channels
        ));
    }
    let row_count = decoded.coords.len();
    if decoded.feats.len() != row_count * decoded.out_channels {
        return Err(format!(
            "fdg decoder output feats len {} does not match rows*out_channels = {}",
            decoded.feats.len(),
            row_count * decoded.out_channels
        ));
    }

    let mut vertices = Vec::with_capacity(row_count);
    let mut intersected = Vec::with_capacity(row_count);
    let mut quad_lerp = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let row =
            &decoded.feats[row_idx * decoded.out_channels..(row_idx + 1) * decoded.out_channels];
        let vx = (1.0 + 2.0 * voxel_margin) * sigmoid(row[0]) - voxel_margin;
        let vy = (1.0 + 2.0 * voxel_margin) * sigmoid(row[1]) - voxel_margin;
        let vz = (1.0 + 2.0 * voxel_margin) * sigmoid(row[2]) - voxel_margin;
        vertices.push([vx, vy, vz]);
        intersected.push([row[3] > 0.0, row[4] > 0.0, row[5] > 0.0]);
        quad_lerp.push(softplus(row[6]));
    }

    Ok(FdgDecodedOutput {
        coords: decoded.coords.clone(),
        vertices,
        intersected,
        quad_lerp,
        subdivisions: decoded.subdivisions.clone(),
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
