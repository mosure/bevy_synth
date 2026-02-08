use crate::trellis_config::TrellisSamplerParams;

#[derive(Debug, Clone)]
pub struct FlowEulerGuidanceIntervalSampler {
    sigma_min: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FlowEulerSampleConfig {
    pub steps: usize,
    pub rescale_t: f32,
    pub guidance_strength: f32,
    pub guidance_rescale: f32,
    pub guidance_interval: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct FlowEulerSampleTrace {
    pub samples: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
}

impl FlowEulerGuidanceIntervalSampler {
    pub fn new(sigma_min: f32) -> Self {
        Self { sigma_min }
    }

    pub fn from_params(
        sigma_min: f32,
        params: &TrellisSamplerParams,
    ) -> (Self, FlowEulerSampleConfig) {
        (
            Self::new(sigma_min),
            FlowEulerSampleConfig {
                steps: params.steps.max(1),
                rescale_t: params.rescale_t.max(f32::EPSILON),
                guidance_strength: params.guidance_strength,
                guidance_rescale: params.guidance_rescale.max(0.0),
                guidance_interval: params.guidance_interval,
            },
        )
    }

    pub fn sample<F>(&self, noise: &[f32], config: FlowEulerSampleConfig, predict_v: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32, bool) -> Vec<f32>,
    {
        self.sample_with_trace(noise, config, predict_v).samples
    }

    pub fn sample_with_trace<F>(
        &self,
        noise: &[f32],
        config: FlowEulerSampleConfig,
        mut predict_v: F,
    ) -> FlowEulerSampleTrace
    where
        F: FnMut(&[f32], f32, bool) -> Vec<f32>,
    {
        let mut sample = noise.to_vec();
        let mut step_0_x_t: Option<Vec<f32>> = None;
        let mut step_last_x_t: Option<Vec<f32>> = None;
        let t_pairs = timestep_pairs(config.steps, config.rescale_t);
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let pred_v = self.predict_with_cfg(&sample, t, &config, &mut predict_v);
            let dt = t - t_prev;
            for (idx, value) in sample.iter_mut().enumerate() {
                *value -= dt * pred_v[idx];
            }
            if step_idx == 0 {
                step_0_x_t = Some(sample.clone());
            }
            if step_idx + 1 == config.steps {
                step_last_x_t = Some(sample.clone());
            }
        }
        let step_0_x_t = step_0_x_t.unwrap_or_else(|| sample.clone());
        let step_last_x_t = step_last_x_t.unwrap_or_else(|| sample.clone());
        FlowEulerSampleTrace {
            samples: sample,
            step_0_x_t,
            step_last_x_t,
        }
    }

    fn predict_with_cfg<F>(
        &self,
        x_t: &[f32],
        t: f32,
        config: &FlowEulerSampleConfig,
        predict_v: &mut F,
    ) -> Vec<f32>
    where
        F: FnMut(&[f32], f32, bool) -> Vec<f32>,
    {
        let in_guidance_interval =
            config.guidance_interval[0] <= t && t <= config.guidance_interval[1];
        if !in_guidance_interval {
            return predict_v(x_t, t, true);
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            return predict_v(x_t, t, true);
        }
        if w.abs() < f32::EPSILON {
            return predict_v(x_t, t, false);
        }

        let pos = predict_v(x_t, t, true);
        let neg = predict_v(x_t, t, false);
        let mut pred = vec![0.0f32; pos.len()];
        for idx in 0..pred.len() {
            pred[idx] = w * pos[idx] + (1.0 - w) * neg[idx];
        }

        if config.guidance_rescale <= 0.0 {
            return pred;
        }

        let x0_pos = pred_to_xstart(x_t, t, &pos, self.sigma_min);
        let x0_cfg = pred_to_xstart(x_t, t, &pred, self.sigma_min);
        let std_pos = stddev(&x0_pos);
        let std_cfg = stddev(&x0_cfg).max(1.0e-12);
        let scale = std_pos / std_cfg;
        let mut x0 = vec![0.0f32; x0_cfg.len()];
        for idx in 0..x0.len() {
            let x0_rescaled = x0_cfg[idx] * scale;
            x0[idx] = config.guidance_rescale * x0_rescaled
                + (1.0 - config.guidance_rescale) * x0_cfg[idx];
        }
        xstart_to_pred(x_t, t, &x0, self.sigma_min)
    }
}

fn timestep_pairs(steps: usize, rescale_t: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let a = 1.0 - (i as f32 / steps as f32);
        let b = 1.0 - ((i + 1) as f32 / steps as f32);
        let t = rescaled_t(a, rescale_t);
        let t_prev = rescaled_t(b, rescale_t);
        out.push((t, t_prev));
    }
    out
}

fn rescaled_t(t: f32, rescale_t: f32) -> f32 {
    rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
}

fn pred_to_xstart(x_t: &[f32], t: f32, pred: &[f32], sigma_min: f32) -> Vec<f32> {
    let factor = sigma_min + (1.0 - sigma_min) * t;
    let keep = 1.0 - sigma_min;
    let mut out = vec![0.0f32; x_t.len()];
    for idx in 0..out.len() {
        out[idx] = keep * x_t[idx] - factor * pred[idx];
    }
    out
}

fn xstart_to_pred(x_t: &[f32], t: f32, x0: &[f32], sigma_min: f32) -> Vec<f32> {
    let factor = sigma_min + (1.0 - sigma_min) * t;
    let keep = 1.0 - sigma_min;
    let mut out = vec![0.0f32; x_t.len()];
    for idx in 0..out.len() {
        out[idx] = (keep * x_t[idx] - x0[idx]) / factor;
    }
    out
}

fn stddev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|value| {
            let d = *value - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{FlowEulerGuidanceIntervalSampler, FlowEulerSampleConfig};

    #[test]
    fn converges_to_target_for_identity_velocity() {
        let sampler = FlowEulerGuidanceIntervalSampler::new(1.0e-5);
        let noise = vec![1.0f32; 8];
        let cfg = FlowEulerSampleConfig {
            steps: 64,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };
        let out = sampler.sample(&noise, cfg, |x_t, _t, _cond| x_t.to_vec());
        let avg_abs = out.iter().map(|v| v.abs()).sum::<f32>() / out.len() as f32;
        assert!(avg_abs < 0.4, "expected decay toward zero, got {avg_abs}");
    }

    #[test]
    fn ignores_cfg_outside_interval() {
        let sampler = FlowEulerGuidanceIntervalSampler::new(1.0e-5);
        let noise = vec![0.5f32; 4];
        let cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 9.0,
            guidance_rescale: 0.7,
            guidance_interval: [0.0, 0.1],
        };
        let out = sampler.sample(&noise, cfg, |x_t, _t, cond| {
            if cond {
                x_t.iter().map(|v| v * 0.5).collect()
            } else {
                x_t.iter().map(|v| v * 100.0).collect()
            }
        });
        assert!(out.iter().all(|v| v.abs() < 0.5));
    }
}
