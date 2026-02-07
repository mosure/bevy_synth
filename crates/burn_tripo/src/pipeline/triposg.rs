use burn::{prelude::*, tensor::Distribution};

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
use crate::pipeline::mesh::{DenseGrid, Mesh, grid_to_mesh, sdf_to_mesh_diff_dmc};
use crate::readback::tensor_to_vec_f32;

#[derive(Debug)]
pub struct TripoSGPipeline<B: Backend> {
    pub vae: TripoSGVae<B>,
    pub transformer: TripoSGDiT<B>,
    pub scheduler: RectifiedFlowScheduler,
    pub image_encoder: TripoSGImageEncoder<B>,
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

impl<B: Backend> TripoSGPipeline<B> {
    pub fn new(
        vae: TripoSGVae<B>,
        transformer: TripoSGDiT<B>,
        scheduler: RectifiedFlowScheduler,
        image_encoder: TripoSGImageEncoder<B>,
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
        self.image_encoder.forward(image)
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
        let model_batch = if do_guidance {
            batch_size * 2
        } else {
            batch_size
        };
        let timestep_template = Tensor::<B, 1>::zeros([model_batch as i32], &device);
        for &t in timesteps.iter() {
            let latent_model_input = if do_guidance {
                Tensor::cat(vec![latents.clone(), latents.clone()], 0)
            } else {
                latents.clone()
            };
            let timestep = timestep_template.clone().add_scalar(t);

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
        let total = resolution * resolution * resolution;
        let device = latents.device();
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
            if count >= chunk_size {
                let end = chunk_start + count;
                write_decoded_chunk_contiguous(
                    &latents,
                    &self.vae,
                    &coords,
                    &device,
                    &mut values[chunk_start..end],
                )?;
                coords.clear();
                chunk_start = end;
            }
        }

        if !coords.is_empty() {
            let count = coords.len() / 3;
            let end = chunk_start + count;
            write_decoded_chunk_contiguous(
                &latents,
                &self.vae,
                &coords,
                &device,
                &mut values[chunk_start..end],
            )?;
        }

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
        let output = self.sample(
            image,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = flash_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
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
        let grid = flash_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        Ok(TripoSGMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }
}

pub(crate) fn generate_dense_grid_coords(bounds: [f32; 6], resolution: usize) -> Vec<f32> {
    let xs = linspace(bounds[0], bounds[3], resolution);
    let ys = linspace(bounds[1], bounds[4], resolution);
    let zs = linspace(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(resolution * resolution * resolution * 3);
    for &z in &zs {
        for &y in &ys {
            for &x in &xs {
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }
    coords
}

pub(crate) fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let step = (end - start) / (steps as f32 - 1.0);
    (0..steps).map(|i| start + step * i as f32).collect()
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
        use crate::model::triposg::dit::import::load_triposg_dit;
        use crate::model::triposg::image_encoder::import::{
            load_dinov2_processor, load_triposg_dinov2,
        };
        use crate::model::triposg::vae::import::load_triposg_vae;

        let root = weights_root.as_ref();
        let vae_path = root.join("vae/diffusion_pytorch_model.safetensors");
        let dit_path = root.join("transformer/diffusion_pytorch_model.safetensors");
        let scheduler_path = root.join("scheduler/scheduler_config.json");
        let dino_path = root.join("image_encoder_dinov2/model.safetensors");

        let vae_config_path = root.join("vae/config.json");
        let vae_config =
            crate::model::triposg::vae::TripoSGVaeConfig::from_config_file(vae_config_path)
                .unwrap_or_else(|_| crate::model::triposg::vae::TripoSGVaeConfig::midi_3d());
        let vae = load_triposg_vae(&vae_config, device, vae_path)?;

        let dit_config_path = root.join("transformer/config.json");
        let dit_config =
            crate::model::triposg::dit::TripoSGDiTConfig::from_config_file(dit_config_path)
                .unwrap_or_else(|_| {
                    if dit_path.exists() {
                        crate::model::triposg::dit::TripoSGDiTConfig::triposg_pretrained()
                    } else {
                        crate::model::triposg::dit::TripoSGDiTConfig::midi_3d()
                    }
                });
        let dit = load_triposg_dit(&dit_config, device, dit_path)?;

        let scheduler_config = RectifiedFlowSchedulerConfig::from_config_file(scheduler_path)
            .unwrap_or_else(|_| RectifiedFlowSchedulerConfig::midi_3d());
        let scheduler = RectifiedFlowScheduler::new(scheduler_config);

        let image_encoder = load_triposg_dinov2(device, dino_path)?;
        let image_processor = load_dinov2_processor(root)?;

        Ok(Self::new(
            vae,
            dit,
            scheduler,
            image_encoder,
            image_processor,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_grid_coords_match_linspace_order() {
        let bounds = [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0];
        let resolution = 4;
        let coords = generate_dense_grid_coords(bounds, resolution);
        let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
        let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
        let step_z = dense_grid_step(bounds[2], bounds[5], resolution);
        let total = resolution * resolution * resolution;
        for idx in 0..total {
            let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
            let base = idx * 3;
            let expected = [
                bounds[0] + step_x * x as f32,
                bounds[1] + step_y * y as f32,
                bounds[2] + step_z * z as f32,
            ];
            let actual = [coords[base], coords[base + 1], coords[base + 2]];
            for i in 0..3 {
                assert!(
                    (expected[i] - actual[i]).abs() < 1e-6,
                    "coord mismatch at idx {idx} axis {i}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }
}
