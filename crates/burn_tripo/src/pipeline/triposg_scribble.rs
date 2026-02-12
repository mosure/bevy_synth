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
use crate::pipeline::triposg::TripoSGPipelineOutput;

#[derive(Debug)]
pub struct TripoSGScribblePipeline<B: Backend> {
    pub vae: TripoSGVae<B>,
    pub transformer: TripoSGDiT<B>,
    pub scheduler: RectifiedFlowScheduler,
    pub image_encoder: TripoSGImageEncoder<B>,
    pub image_processor: DinoImageProcessor,
}

#[derive(Debug)]
pub struct TripoSGScribbleMeshOutput<B: Backend> {
    pub latents: Tensor<B, 3>,
    pub grid: DenseGrid,
    pub mesh: Option<Mesh>,
}

impl<B: Backend> TripoSGScribblePipeline<B> {
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

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        image: Tensor<B, 4>,
        text_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        query_coords: Option<Tensor<B, 3>>,
        latents: Option<Tensor<B, 3>>,
    ) -> TripoSGPipelineOutput<B> {
        let image_embeds = self.encode_image(image);
        self.sample_with_embeddings(
            text_embeds,
            image_embeds,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            query_coords,
            latents,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_with_embeddings(
        &mut self,
        text_embeds: Tensor<B, 3>,
        image_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        query_coords: Option<Tensor<B, 3>>,
        latents: Option<Tensor<B, 3>>,
    ) -> TripoSGPipelineOutput<B> {
        let device = text_embeds.device();
        let batch_size = text_embeds.shape().dims::<3>()[0];
        let do_guidance = guidance_scale > 1.0;

        let text_embeds = if do_guidance {
            let zeros = Tensor::<B, 3>::zeros(text_embeds.shape(), &device);
            Tensor::cat(vec![zeros, text_embeds], 0)
        } else {
            text_embeds
        };

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
        for &t in timesteps.iter() {
            let latent_model_input = if do_guidance {
                Tensor::cat(vec![latents.clone(), latents.clone()], 0)
            } else {
                latents.clone()
            };
            let model_batch = latent_model_input.shape().dims::<3>()[0];
            let timestep_values = vec![t; model_batch];
            let timestep = Tensor::<B, 1>::from_floats(timestep_values.as_slice(), &device);

            let mut noise_pred = self.transformer.forward(
                latent_model_input,
                timestep,
                text_embeds.clone(),
                Some(image_embeds.clone()),
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
        let values = super::triposg::decode_grid_values(
            &latents, &self.vae, bounds, resolution, chunk_size,
        )?;

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
        text_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        bounds: [f32; 6],
        resolution: usize,
        chunk_size: usize,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGScribbleMeshOutput<B>, Box<dyn std::error::Error>> {
        let output = self.sample(
            image,
            text_embeds,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = self.decode_grid(output.latents.clone(), bounds, resolution, chunk_size)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGScribbleMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh_hierarchical(
        &mut self,
        image: Tensor<B, 4>,
        text_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &HierarchicalExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGScribbleMeshOutput<B>, Box<dyn std::error::Error>> {
        let output = self.sample(
            image,
            text_embeds,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = hierarchical_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = grid_to_mesh(&grid, 0.0);
        Ok(TripoSGScribbleMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_mesh_flash(
        &mut self,
        image: Tensor<B, 4>,
        text_embeds: Tensor<B, 3>,
        num_inference_steps: usize,
        num_tokens: usize,
        guidance_scale: f32,
        config: &FlashExtractConfig,
        latents: Option<Tensor<B, 3>>,
    ) -> Result<TripoSGScribbleMeshOutput<B>, Box<dyn std::error::Error>> {
        let output = self.sample(
            image,
            text_embeds,
            num_inference_steps,
            num_tokens,
            guidance_scale,
            None,
            latents,
        );
        let grid = flash_extract_geometry(output.latents.clone(), &self.vae, config)?;
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        Ok(TripoSGScribbleMeshOutput {
            latents: output.latents,
            grid,
            mesh,
        })
    }
}

#[cfg(feature = "import")]
impl<B: Backend> TripoSGScribblePipeline<B> {
    pub fn from_pretrained(
        weights_root: impl AsRef<std::path::Path>,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::model::triposg::dit::import::{
            load_triposg_dit, load_triposg_dit_from_safetensors,
        };
        use crate::model::triposg::image_encoder::import::{
            load_dinov2_processor, load_triposg_dinov2, load_triposg_dinov2_from_safetensors,
        };
        use crate::model::triposg::vae::import::{
            load_triposg_vae, load_triposg_vae_from_safetensors,
        };

        let root = weights_root.as_ref();
        let vae_path = root.join("vae/diffusion_pytorch_model.safetensors");
        let dit_path = root.join("transformer/diffusion_pytorch_model.safetensors");
        let scheduler_path = root.join("scheduler/scheduler_config.json");
        let dino_path = root.join("image_encoder_dinov2/model.safetensors");
        let use_safetensors = std::env::var("TRIPOSG_LOAD_SAFETENSORS").is_ok();

        let vae_config_path = root.join("vae/config.json");
        let vae_config =
            crate::model::triposg::vae::TripoSGVaeConfig::from_config_file(vae_config_path)
                .unwrap_or_else(|_| crate::model::triposg::vae::TripoSGVaeConfig::midi_3d());
        let vae = if use_safetensors {
            load_triposg_vae_from_safetensors(&vae_config, device, &vae_path)?
        } else {
            load_triposg_vae(&vae_config, device, &vae_path)?
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
                load_triposg_dit(&config, device, &dit_path)
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

        let image_encoder = if use_safetensors {
            load_triposg_dinov2_from_safetensors(device, &dino_path)?
        } else {
            load_triposg_dinov2(device, &dino_path)?
        };
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
