use burn::{prelude::*, tensor::Distribution};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(feature = "import")]
use crate::model::triposg::load_policy::BurnpackLoadPolicy;
#[cfg(feature = "import")]
use crate::model::triposg::scheduler::RectifiedFlowSchedulerConfig;
use crate::model::triposg::{
    dit::TripoSGDiT,
    image_encoder::{DinoImageProcessor, TripoSGImageEncoder},
    scheduler::RectifiedFlowScheduler,
    vae::TripoSGVae,
};
use crate::pipeline::geometry::{
    FlashExtractConfig, HierarchicalExtractConfig, flash_extract_geometry,
    hierarchical_extract_geometry,
};
#[cfg(target_arch = "wasm32")]
use crate::pipeline::geometry::flash_extract_geometry_async_wasm;
use crate::pipeline::mesh::{DenseGrid, Mesh, grid_to_mesh, sdf_to_mesh_diff_dmc};
use crate::readback::tensor_to_vec_f32;

#[derive(Debug)]
pub struct TripoSGPipeline<B: Backend> {
    pub vae: TripoSGVae<B>,
    pub transformer: TripoSGDiT<B>,
    pub scheduler: RectifiedFlowScheduler,
    pub image_encoder: Option<TripoSGImageEncoder<B>>,
    pub image_processor: DinoImageProcessor,
}

#[derive(Debug)]
pub struct TripoSGPipelineOutput<B: Backend> {
    pub latents: Tensor<B, 3>,
    pub decoded: Option<Tensor<B, 3>>,
}

#[derive(Debug)]
pub struct TripoSGMeshOutput<B: Backend> {
    pub latents: Tensor<B, 3>,
    pub grid: DenseGrid,
    pub mesh: Option<Mesh>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripoSGSamplerProgress {
    pub step_index: usize,
    pub total_steps: usize,
    pub timestep: f32,
    pub step_ms: f64,
}

#[cfg(feature = "import")]
#[derive(Debug, Clone, Copy)]
/// Loader options for `TripoSGPipeline::from_pretrained_with_options`.
pub struct TripoSGLoadOptions {
    /// Load source `.safetensors` directly instead of `.bpk` burnpacks.
    pub use_safetensors: bool,
    /// Burnpack file selection policy (`_f16.bpk` vs `.bpk` preference).
    pub burnpack_policy: BurnpackLoadPolicy,
    /// Whether to load the backend-resident DINO image encoder.
    ///
    /// Set this to `false` when a separate CPU DINO path is used to avoid
    /// loading duplicate image-encoder weights.
    pub load_image_encoder: bool,
    /// Explicit strict-DINO preprocessing mode.
    ///
    /// `Some(true)` is the canonical behavior to match upstream TripoSG.
    /// `Some(false)` keeps the fast backend-native interpolation path.
    /// `None` preserves the processor-config default.
    pub strict_dino_preprocess: Option<bool>,
}

#[cfg(feature = "import")]
impl Default for TripoSGLoadOptions {
    fn default() -> Self {
        Self {
            use_safetensors: false,
            burnpack_policy: BurnpackLoadPolicy::default(),
            load_image_encoder: true,
            strict_dino_preprocess: Some(true),
        }
    }
}

impl<B: Backend> TripoSGPipeline<B> {
    pub fn new(
        vae: TripoSGVae<B>,
        transformer: TripoSGDiT<B>,
        scheduler: RectifiedFlowScheduler,
        image_encoder: TripoSGImageEncoder<B>,
        image_processor: DinoImageProcessor,
    ) -> Self {
        Self::new_with_optional_image_encoder(
            vae,
            transformer,
            scheduler,
            Some(image_encoder),
            image_processor,
        )
    }

    pub fn new_with_optional_image_encoder(
        vae: TripoSGVae<B>,
        transformer: TripoSGDiT<B>,
        scheduler: RectifiedFlowScheduler,
        image_encoder: Option<TripoSGImageEncoder<B>>,
        image_processor: DinoImageProcessor,
    ) -> Self {
        Self {
            vae,
            transformer,
            scheduler,
            image_encoder,
            image_processor,
        }
    }

    pub fn encode_image(&self, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let image = self.image_processor.preprocess(image);
        self.image_encoder
            .as_ref()
            .expect("TripoSG image encoder unavailable")
            .forward(image)
    }

    pub fn prepare_latents(
        &self,
        batch_size: usize,
        num_tokens: usize,
        num_channels: usize,
        device: &B::Device,
        latents: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        if let Some(latents) = latents {
            return latents;
        }
        Tensor::<B, 3>::random(
            [batch_size as i32, num_tokens as i32, num_channels as i32],
            Distribution::Normal(0.0, 1.0),
            device,
        )
    }

    pub fn sample(
        &mut self,
        image: Tensor<B, 4>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        query_coords: Option<Tensor<B, 3>>,
        latents: Option<Tensor<B, 3>>,
    ) -> TripoSGPipelineOutput<B> {
        let batch_size = image.shape().dims::<4>()[0];

        let image_embeds = self.encode_image(image);
        self.sample_from_embeds(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            query_coords,
            latents,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_from_embeds(
        &mut self,
        image_embeds: Tensor<B, 3>,
        batch_size: usize,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        query_coords: Option<Tensor<B, 3>>,
        latents: Option<Tensor<B, 3>>,
    ) -> TripoSGPipelineOutput<B> {
        self.sample_from_embeds_with_progress(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            query_coords,
            latents,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_from_embeds_with_progress<F: FnMut(TripoSGSamplerProgress)>(
        &mut self,
        image_embeds: Tensor<B, 3>,
        batch_size: usize,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        query_coords: Option<Tensor<B, 3>>,
        latents: Option<Tensor<B, 3>>,
        mut on_step: F,
    ) -> TripoSGPipelineOutput<B> {
        let device = image_embeds.device();
        let do_guidance = guidance_scale > 1.0;
        let image_embeds = if do_guidance {
            let zeros = Tensor::<B, 3>::zeros(image_embeds.shape(), &device);
            Tensor::cat(vec![zeros, image_embeds], 0)
        } else {
            image_embeds
        };

        self.scheduler
            .set_timesteps(num_inference_steps, None, None, None)
            .expect("failed to set timesteps");

        let num_channels = self.transformer.config().in_channels;
        let mut latents =
            self.prepare_latents(batch_size, num_tokens, num_channels, &device, latents);

        let timesteps = self.scheduler.timesteps().to_vec();
        let total_steps = timesteps.len();
        let model_batch = if do_guidance {
            batch_size * 2
        } else {
            batch_size
        };
        for (step_index, &t) in timesteps.iter().enumerate() {
            let step_start = Instant::now();
            let latent_model_input = if do_guidance {
                Tensor::cat(vec![latents.clone(), latents.clone()], 0)
            } else {
                latents.clone()
            };
            let timestep = Tensor::<B, 1>::full([model_batch], t, &device);

            let mut noise_pred = self.transformer.forward(
                latent_model_input,
                timestep,
                image_embeds.clone(),
                None,
                None,
            );

            if do_guidance {
                let half = batch_size;
                let noise_uncond =
                    noise_pred
                        .clone()
                        .slice([0..half, 0..num_tokens, 0..num_channels]);
                let noise_cond =
                    noise_pred.slice([half..(half * 2), 0..num_tokens, 0..num_channels]);
                noise_pred =
                    noise_uncond.clone() + (noise_cond - noise_uncond).mul_scalar(guidance_scale);
            }

            latents = self.scheduler.step(noise_pred, t, latents);
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            on_step(TripoSGSamplerProgress {
                step_index: step_index + 1,
                total_steps,
                timestep: t,
                step_ms,
            });
        }

        let decoded = query_coords.map(|coords| self.vae.decode(coords, latents.clone(), None));
        TripoSGPipelineOutput { latents, decoded }
    }

    pub fn decode_grid(
        &self,
        latents: Tensor<B, 3>,
        bounds: [f32; 6],
        resolution: usize,
        chunk_size: usize,
    ) -> Result<DenseGrid, Box<dyn std::error::Error>> {
        let resolution = resolution.max(2);
        let chunk_size = chunk_size.max(1);
        let values = decode_grid_values(&latents, &self.vae, bounds, resolution, chunk_size)?;

        Ok(DenseGrid {
            values,
            size: [resolution, resolution, resolution],
            bounds,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh(
        &mut self,
        image: Tensor<B, 4>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        bounds: [f32; 6],
        resolution: usize,
        chunk_size: usize,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let output = self.sample(
            image,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = self.decode_grid(output.latents.clone(), bounds, resolution, chunk_size)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Generate a mesh from precomputed image embeddings, bypassing the image encoder.
    pub fn sample_mesh_from_embeds(
        &mut self,
        image_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        bounds: [f32; 6],
        resolution: usize,
        chunk_size: usize,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let batch_size = image_embeds.shape().dims::<3>()[0];
        let output = self.sample_from_embeds(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = self.decode_grid(output.latents.clone(), bounds, resolution, chunk_size)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh_hierarchical(
        &mut self,
        image: Tensor<B, 4>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &HierarchicalExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let output = self.sample(
            image,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = hierarchical_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Generate a mesh with hierarchical extraction from precomputed image embeddings.
    pub fn sample_mesh_hierarchical_from_embeds(
        &mut self,
        image_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &HierarchicalExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let batch_size = image_embeds.shape().dims::<3>()[0];
        let output = self.sample_from_embeds(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = hierarchical_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh_flash(
        &mut self,
        image: Tensor<B, 4>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &FlashExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        self.sample_mesh_flash_with_progress(
            image,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            config,
            latents,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh_flash_with_progress<F: FnMut(TripoSGSamplerProgress)>(
        &mut self,
        image: Tensor<B, 4>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &FlashExtractConfig,
        latents: Option<Tensor<B, 3>>,
        mut on_step: F,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let batch_size = image.shape().dims::<4>()[0];
        let image_embeds = self.encode_image(image);
        let output = self.sample_from_embeds_with_progress(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
            &mut on_step,
        );
        self.extract_flash_mesh_from_latents(output.latents, config)
    }

    #[allow(clippy::too_many_arguments)]
    /// Generate a mesh with flash extraction from precomputed image embeddings.
    pub fn sample_mesh_flash_from_embeds(
        &mut self,
        image_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &FlashExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let batch_size = image_embeds.shape().dims::<3>()[0];
        let output = self.sample_from_embeds(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        self.extract_flash_mesh_from_latents(output.latents, config)
    }

    #[allow(clippy::too_many_arguments)]
    /// Generate a mesh with flash extraction from precomputed image embeddings (wasm async).
    #[cfg(target_arch = "wasm32")]
    pub async fn sample_mesh_flash_from_embeds_async_wasm(
        &mut self,
        image_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &FlashExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGMeshOutput<B>, String> {
        let batch_size = image_embeds.shape().dims::<3>()[0];
        let output = self.sample_from_embeds(
            image_embeds,
            batch_size,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        self.extract_flash_mesh_from_latents_async_wasm(output.latents, config)
            .await
    }

    /// Extract flash geometry/mesh from sampled latents using the canonical native path.
    pub fn extract_flash_grid_from_latents(
        &self,
        latents: Tensor<B, 3>,
        config: &FlashExtractConfig,
    ) -> Result<DenseGrid, Box<dyn std::error::Error>> {
        flash_extract_geometry(latents, &self.vae, config)
    }

    /// Extract flash geometry/mesh from sampled latents using the canonical native path.
    pub fn extract_flash_mesh_from_latents(
        &self,
        latents: Tensor<B, 3>,
        config: &FlashExtractConfig,
    ) -> Result<TripoSGMeshOutput<B>, Box<dyn std::error::Error>> {
        let grid = self.extract_flash_grid_from_latents(latents.clone(), config)?;
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        Ok(TripoSGMeshOutput {
            latents,
            grid,
            mesh,
        })
    }

    /// Extract flash geometry/mesh from sampled latents using async tensor readback on wasm.
    #[cfg(target_arch = "wasm32")]
    pub async fn extract_flash_grid_from_latents_async_wasm(
        &self,
        latents: Tensor<B, 3>,
        config: &FlashExtractConfig,
    ) -> Result<DenseGrid, String> {
        flash_extract_geometry_async_wasm(latents, &self.vae, config)
            .await
            .map_err(|err| format!("flash extraction failed: {err}"))
    }

    /// Extract flash geometry/mesh from sampled latents using async tensor readback on wasm.
    #[cfg(target_arch = "wasm32")]
    pub async fn extract_flash_mesh_from_latents_async_wasm(
        &self,
        latents: Tensor<B, 3>,
        config: &FlashExtractConfig,
    ) -> Result<TripoSGMeshOutput<B>, String> {
        let grid = self
            .extract_flash_grid_from_latents_async_wasm(latents.clone(), config)
            .await?;
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        Ok(TripoSGMeshOutput {
            latents,
            grid,
            mesh,
        })
    }
}

fn dense_grid_step(start: f32, end: f32, steps: usize) -> f32 {
    if steps <= 1 {
        0.0
    } else {
        (end - start) / (steps as f32 - 1.0)
    }
}

fn dense_grid_index_to_xyz(index: usize, resolution: usize) -> (usize, usize, usize) {
    let plane = resolution * resolution;
    let z = index / plane;
    let rem = index - z * plane;
    let y = rem / resolution;
    let x = rem - y * resolution;
    (x, y, z)
}

pub(crate) fn decode_grid_values<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let total = resolution * resolution * resolution;
    let device = latents.device();
    if should_device_decode_accumulate::<B>(total) {
        decode_grid_values_device_accumulate(latents, vae, bounds, resolution, chunk_size, &device)
    } else {
        decode_grid_values_chunked_host(latents, vae, bounds, resolution, chunk_size, &device)
    }
}

pub(crate) fn should_device_decode_accumulate<B: Backend>(total_points: usize) -> bool {
    let backend = std::any::type_name::<B>().to_ascii_lowercase();
    let is_gpu_backend =
        backend.contains("wgpu") || backend.contains("cuda") || backend.contains("cube");
    if !is_gpu_backend {
        return false;
    }
    let max_points = 16_777_216usize;
    total_points <= max_points
}

pub(crate) fn decode_grid_values_device_accumulate<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    device: &B::Device,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let total = resolution * resolution * resolution;
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunks = Vec::<Tensor<B, 3>>::new();
    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);
        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        chunks.push(decoded_chunk_tensor(latents, vae, &coords, device)?);
        coords.clear();
    }

    if !coords.is_empty() {
        chunks.push(decoded_chunk_tensor(latents, vae, &coords, device)?);
    }

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let decoded = if chunks.len() == 1 {
        chunks.pop().expect("single chunk exists")
    } else {
        Tensor::cat(chunks, 1)
    };

    let mut values = tensor_to_vec_f32(decoded)
        .map_err(|err| format!("failed to convert decoded grid: {err}"))?;
    values.truncate(total);
    Ok(values)
}

fn decode_grid_values_chunked_host<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    device: &B::Device,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let total = resolution * resolution * resolution;
    let mut values = vec![0.0f32; total];
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunk_start = 0usize;

    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);
        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        let end = chunk_start + count;
        write_decoded_chunk_contiguous(
            latents,
            vae,
            &coords,
            device,
            &mut values[chunk_start..end],
        )?;
        coords.clear();
        chunk_start = end;
    }

    if !coords.is_empty() {
        let count = coords.len() / 3;
        let end = chunk_start + count;
        write_decoded_chunk_contiguous(
            latents,
            vae,
            &coords,
            device,
            &mut values[chunk_start..end],
        )?;
    }

    Ok(values)
}

fn decoded_chunk_tensor<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[f32],
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let count = coords.len() / 3;
    if count == 0 {
        return Ok(Tensor::<B, 3>::zeros([1, 0, 1], device));
    }
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([count as i32, 3])
        .unsqueeze_dim(0);
    Ok(vae.decode(coords_tensor, latents.clone(), None))
}

fn write_decoded_chunk_contiguous<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[f32],
    device: &B::Device,
    output_slice: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let count = coords.len() / 3;
    if count == 0 {
        return Ok(());
    }
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([count as i32, 3])
        .unsqueeze_dim(0);
    let decoded = vae.decode(coords_tensor, latents.clone(), None);
    let data = tensor_to_vec_f32(decoded)
        .map_err(|err| format!("failed to convert decoded grid: {err}"))?;
    output_slice.copy_from_slice(&data[..output_slice.len()]);
    Ok(())
}

#[cfg(feature = "import")]
impl<B: Backend> TripoSGPipeline<B> {
    pub fn from_pretrained(
        weights_root: impl AsRef<std::path::Path>,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_pretrained_with_options(weights_root, device, TripoSGLoadOptions::default())
    }

    pub fn from_pretrained_with_options(
        weights_root: impl AsRef<std::path::Path>,
        device: &B::Device,
        options: TripoSGLoadOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::model::triposg::dit::import::{
            load_triposg_dit_from_safetensors, load_triposg_dit_with_policy,
        };
        use crate::model::triposg::image_encoder::import::{
            load_dinov2_processor, load_triposg_dinov2_from_safetensors,
            load_triposg_dinov2_with_policy,
        };
        use crate::model::triposg::vae::import::{
            load_triposg_vae_from_safetensors, load_triposg_vae_with_policy,
        };

        let root = weights_root.as_ref();
        let vae_path = root.join("vae/diffusion_pytorch_model.safetensors");
        let dit_path = root.join("transformer/diffusion_pytorch_model.safetensors");
        let scheduler_path = root.join("scheduler/scheduler_config.json");
        let dino_path = root.join("image_encoder_dinov2/model.safetensors");
        let use_safetensors = options.use_safetensors;

        let vae_config_path = root.join("vae/config.json");
        let vae_config =
            crate::model::triposg::vae::TripoSGVaeConfig::from_config_file(vae_config_path)
                .unwrap_or_else(|_| crate::model::triposg::vae::TripoSGVaeConfig::midi_3d());
        let vae = if use_safetensors {
            load_triposg_vae_from_safetensors(&vae_config, device, &vae_path)?
        } else {
            load_triposg_vae_with_policy(&vae_config, device, &vae_path, options.burnpack_policy)?
        };

        let dit_config_path = root.join("transformer/config.json");
        let mut dit_configs = Vec::new();
        if let Ok(config) =
            crate::model::triposg::dit::TripoSGDiTConfig::from_config_file(&dit_config_path)
        {
            dit_configs.push(config);
        } else if dit_path.exists() {
            dit_configs.push(crate::model::triposg::dit::TripoSGDiTConfig::triposg_pretrained());
        } else {
            dit_configs.push(crate::model::triposg::dit::TripoSGDiTConfig::midi_3d());
        }

        // Support both historical DiT variants by retrying with the alternate config when
        // config.json and burnpack tensor shapes diverge.
        push_unique_dit_config(
            &mut dit_configs,
            crate::model::triposg::dit::TripoSGDiTConfig::triposg_pretrained(),
        );
        push_unique_dit_config(
            &mut dit_configs,
            crate::model::triposg::dit::TripoSGDiTConfig::midi_3d(),
        );

        let mut load_errors = Vec::new();
        let mut dit = None;
        for config in dit_configs {
            let loaded = if use_safetensors {
                load_triposg_dit_from_safetensors(&config, device, &dit_path)
            } else {
                load_triposg_dit_with_policy(&config, device, &dit_path, options.burnpack_policy)
            };
            match loaded {
                Ok(model) => {
                    dit = Some(model);
                    break;
                }
                Err(err) => {
                    load_errors.push(format!(
                        "cross_attention_dim={} cross_attention_2_dim={:?}: {err}",
                        config.cross_attention_dim, config.cross_attention_2_dim
                    ));
                }
            }
        }
        let Some(dit) = dit else {
            return Err(format!(
                "failed to load TripoSG DiT with all known configs:\n{}",
                load_errors.join("\n")
            )
            .into());
        };

        let scheduler_config = RectifiedFlowSchedulerConfig::from_config_file(scheduler_path)
            .unwrap_or_else(|_| RectifiedFlowSchedulerConfig::midi_3d());
        let scheduler = RectifiedFlowScheduler::new(scheduler_config);

        let image_encoder = if options.load_image_encoder {
            Some(if use_safetensors {
                load_triposg_dinov2_from_safetensors(device, &dino_path)?
            } else {
                load_triposg_dinov2_with_policy(device, &dino_path, options.burnpack_policy)?
            })
        } else {
            None
        };
        let mut image_processor = load_dinov2_processor(root)?;
        if let Some(strict) = options.strict_dino_preprocess {
            image_processor.set_strict_preprocess(strict);
        }

        Ok(Self::new_with_optional_image_encoder(
            vae,
            dit,
            scheduler,
            image_encoder,
            image_processor,
        ))
    }
}

#[cfg(feature = "import")]
fn push_unique_dit_config(
    configs: &mut Vec<crate::model::triposg::dit::TripoSGDiTConfig>,
    candidate: crate::model::triposg::dit::TripoSGDiTConfig,
) {
    if !configs.iter().any(|existing| {
        existing.in_channels == candidate.in_channels
            && existing.width == candidate.width
            && existing.num_layers == candidate.num_layers
            && existing.num_attention_heads == candidate.num_attention_heads
            && existing.cross_attention_dim == candidate.cross_attention_dim
            && existing.cross_attention_2_dim == candidate.cross_attention_2_dim
    }) {
        configs.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::triposg::vae::{TripoSGVae, TripoSGVaeConfig};

    #[test]
    fn dense_grid_index_round_trip() {
        let resolution = 8usize;
        let total = resolution * resolution * resolution;
        for idx in 0..total {
            let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
            let round_trip = x + y * resolution + z * resolution * resolution;
            assert_eq!(round_trip, idx, "index mapping mismatch at idx={idx}");
        }
    }

    #[test]
    fn decode_grid_device_accumulate_matches_chunked_host() {
        type B = burn::backend::NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let vae = TripoSGVae::new(
            &device,
            TripoSGVaeConfig {
                embed_frequency: 2,
                embed_include_pi: false,
                embedding_type: "frequency".to_string(),
                in_channels: 3,
                latent_channels: 4,
                num_attention_heads: 1,
                num_layers_decoder: 1,
                num_layers_encoder: 1,
                width_decoder: 8,
                width_encoder: 8,
            },
        );
        let latents = Tensor::<B, 3>::random([1, 32, 4], Distribution::Normal(0.0, 1.0), &device);
        let bounds = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let resolution = 10usize;
        let chunk_size = 37usize;

        let device_values = decode_grid_values_device_accumulate(
            &latents, &vae, bounds, resolution, chunk_size, &device,
        )
        .expect("device accumulate path");
        let host_values = decode_grid_values_chunked_host(
            &latents, &vae, bounds, resolution, chunk_size, &device,
        )
        .expect("host chunked path");

        assert_eq!(device_values.len(), host_values.len());
        let mut max_abs = 0.0f32;
        for (lhs, rhs) in device_values.iter().zip(host_values.iter()) {
            max_abs = max_abs.max((lhs - rhs).abs());
        }
        assert!(
            max_abs <= 1.0e-6,
            "device/host grid decode mismatch max_abs={max_abs}"
        );
    }
}
