use burn::prelude::*;
use burn::tensor::FloatDType;

#[derive(Config, Debug)]
pub struct RectifiedFlowSchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub use_dynamic_shifting: bool,
}

impl RectifiedFlowSchedulerConfig {
    pub fn midi_3d() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 2.0,
            use_dynamic_shifting: false,
        }
    }

    pub fn init(&self) -> RectifiedFlowScheduler {
        RectifiedFlowScheduler::new(self.clone())
    }

    #[cfg(feature = "import")]
    pub fn from_config_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let config: RectifiedFlowSchedulerConfigFile = serde_json::from_slice(&bytes)?;
        Ok(Self {
            num_train_timesteps: config.num_train_timesteps.unwrap_or(1000),
            shift: config.shift.unwrap_or(1.0),
            use_dynamic_shifting: config.use_dynamic_shifting.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
pub struct RectifiedFlowScheduler {
    pub config: RectifiedFlowSchedulerConfig,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    step_index: Option<usize>,
    begin_index: Option<usize>,
}

impl RectifiedFlowScheduler {
    pub fn new(config: RectifiedFlowSchedulerConfig) -> Self {
        let (timesteps, sigmas) = build_training_schedule(&config);
        Self {
            config,
            timesteps,
            sigmas,
            step_index: None,
            begin_index: None,
        }
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    pub fn begin_index(&self) -> Option<usize> {
        self.begin_index
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
    }

    pub fn set_timesteps(
        &mut self,
        num_inference_steps: usize,
        timesteps: Option<Vec<f32>>,
        sigmas: Option<Vec<f32>>,
        mu: Option<f32>,
    ) -> Result<(), String> {
        if timesteps.is_some() && sigmas.is_some() {
            return Err("Only one of timesteps or sigmas can be passed".to_string());
        }

        let mut sigmas = if let Some(custom_sigmas) = sigmas {
            custom_sigmas
        } else if let Some(custom_timesteps) = timesteps {
            custom_timesteps
                .iter()
                .map(|t| self.t_to_sigma(*t))
                .collect()
        } else {
            build_inference_sigmas(self.config.num_train_timesteps, num_inference_steps)
        };

        if self.config.use_dynamic_shifting {
            let mu = mu.ok_or_else(|| {
                "mu must be provided when use_dynamic_shifting is enabled".to_string()
            })?;
            sigmas = sigmas
                .into_iter()
                .map(|sigma| self.time_shift_dynamic(mu, 1.0, sigma))
                .collect();
        } else {
            sigmas = sigmas
                .into_iter()
                .map(|sigma| self.time_shift(sigma))
                .collect();
        }

        self.timesteps = sigmas
            .iter()
            .map(|sigma| sigma * self.config.num_train_timesteps as f32)
            .collect();

        self.sigmas = sigmas;
        self.sigmas.push(0.0);

        self.step_index = None;
        self.begin_index = None;
        Ok(())
    }

    pub fn step<B: Backend>(
        &mut self,
        model_output: Tensor<B, 3>,
        timestep: f32,
        sample: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        if self.step_index.is_none() {
            self.init_step_index(timestep);
        }
        let step_index = self.step_index.unwrap_or(0);
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        let output_dtype: FloatDType = model_output.dtype().into();
        let sample_f32 = sample.cast(FloatDType::F32);
        let model_f32 = model_output.cast(FloatDType::F32);
        let prev_sample = sample_f32.add(model_f32.mul_scalar(sigma - sigma_next));
        let prev_sample = prev_sample.cast(output_dtype);
        self.step_index = Some(step_index + 1);
        prev_sample
    }

    pub fn scale_noise<B: Backend>(
        &self,
        original_samples: Tensor<B, 3>,
        noise: Tensor<B, 3>,
        timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let sigmas = timesteps
            .div_scalar(self.config.num_train_timesteps as f32)
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2);
        let scaled = original_samples.clone().sub(sigmas.clone() * original_samples);
        scaled.add(sigmas * noise)
    }

    pub fn scale_model_input<B: Backend>(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        latents
    }

    fn init_step_index(&mut self, timestep: f32) {
        if let Some(begin_index) = self.begin_index {
            self.step_index = Some(begin_index);
            return;
        }
        self.step_index = self.index_for_timestep(timestep);
    }

    fn index_for_timestep(&self, timestep: f32) -> Option<usize> {
        let indices: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| if *value == timestep { Some(idx) } else { None })
            .collect();
        if indices.is_empty() {
            return None;
        }
        if indices.len() > 1 {
            indices.get(1).copied()
        } else {
            indices.first().copied()
        }
    }

    fn time_shift(&self, t: f32) -> f32 {
        self.config.shift * t / (1.0 + (self.config.shift - 1.0) * t)
    }

    fn time_shift_dynamic(&self, mu: f32, sigma: f32, t: f32) -> f32 {
        let exp_mu = mu.exp();
        exp_mu / (exp_mu + (1.0 / t - 1.0).powf(sigma))
    }

    fn t_to_sigma(&self, timestep: f32) -> f32 {
        timestep / self.config.num_train_timesteps as f32
    }
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize)]
struct RectifiedFlowSchedulerConfigFile {
    num_train_timesteps: Option<usize>,
    shift: Option<f32>,
    use_dynamic_shifting: Option<bool>,
}

fn build_training_schedule(config: &RectifiedFlowSchedulerConfig) -> (Vec<f32>, Vec<f32>) {
    let num_steps = config.num_train_timesteps.max(1);
    let mut sigmas = Vec::with_capacity(num_steps);
    let mut timesteps = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        let t = (1.0 - i as f32 / num_steps as f32) * num_steps as f32;
        let sigma = t / num_steps as f32;
        let sigma = if config.use_dynamic_shifting {
            sigma
        } else {
            config.shift * sigma / (1.0 + (config.shift - 1.0) * sigma)
        };
        timesteps.push(sigma * num_steps as f32);
        sigmas.push(sigma);
    }
    (timesteps, sigmas)
}

fn build_inference_sigmas(num_train_timesteps: usize, num_inference_steps: usize) -> Vec<f32> {
    let steps = num_inference_steps.max(1);
    let mut sigmas = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = (1.0 - i as f32 / steps as f32) * num_train_timesteps as f32;
        sigmas.push(t / num_train_timesteps as f32);
    }
    sigmas
}
