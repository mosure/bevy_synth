use std::path::Path;

use super::sparse_decoder::{SparseSubdivisionLogits, SparseUnetDecoderRuntime};

#[derive(Debug, Clone)]
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

    pub fn decode_with_guidance(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: &[SparseSubdivisionLogits],
    ) -> Result<TexDecodedOutput, String> {
        let decoded = self.inner.decode(coords, rows, Some(guide_subdivisions))?;
        if decoded.out_channels < 6 {
            return Err(format!(
                "sparse unet vae decoder expected at least 6 channels, got {}",
                decoded.out_channels
            ));
        }
        if decoded.feats.len() != decoded.coords.len() * decoded.out_channels {
            return Err(format!(
                "tex decoder output feats len {} does not match rows*out_channels = {}",
                decoded.feats.len(),
                decoded.coords.len() * decoded.out_channels
            ));
        }

        let mut attrs = Vec::with_capacity(decoded.coords.len());
        for row_idx in 0..decoded.coords.len() {
            let row = &decoded.feats
                [row_idx * decoded.out_channels..(row_idx + 1) * decoded.out_channels];
            let mut attr = [0.0f32; 6];
            for ch in 0..6 {
                // Python decode_tex_slat postprocessing: ret * 0.5 + 0.5
                attr[ch] = row[ch] * 0.5 + 0.5;
            }
            attrs.push(attr);
        }

        Ok(TexDecodedOutput {
            coords: decoded.coords,
            attrs,
        })
    }
}
