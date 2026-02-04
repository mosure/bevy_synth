use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

#[derive(Debug, Clone)]
pub struct RmbgImageProcessor {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale_factor: f32,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub do_resize: bool,
    pub size: Option<[usize; 2]>,
    pub resize_mode: InterpolateMode,
}

impl Default for RmbgImageProcessor {
    fn default() -> Self {
        Self {
            mean: [0.5, 0.5, 0.5],
            std: [1.0, 1.0, 1.0],
            rescale_factor: 1.0 / 255.0,
            do_rescale: true,
            do_normalize: true,
            do_resize: true,
            size: Some([1024, 1024]),
            resize_mode: InterpolateMode::Bilinear,
        }
    }
}

impl RmbgImageProcessor {
    #[cfg(feature = "import")]
    pub fn from_config(config: RmbgProcessorConfig) -> Self {
        let resize_mode = match config.resample.unwrap_or(2) {
            3 => InterpolateMode::Bicubic,
            2 => InterpolateMode::Bilinear,
            _ => InterpolateMode::Nearest,
        };
        Self {
            mean: config.image_mean.unwrap_or([0.5, 0.5, 0.5]),
            std: config.image_std.unwrap_or([1.0, 1.0, 1.0]),
            rescale_factor: config.rescale_factor.unwrap_or(1.0 / 255.0),
            do_rescale: config.do_rescale.unwrap_or(true),
            do_normalize: config.do_normalize.unwrap_or(true),
            do_resize: config.do_resize.unwrap_or(true),
            size: config.size.map(|s| [s.height, s.width]).or(Some([1024, 1024])),
            resize_mode,
        }
    }

    pub fn preprocess<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut image = image;

        if self.do_resize && let Some([height, width]) = self.size {
            let options = InterpolateOptions {
                mode: self.resize_mode.clone(),
            };
            image = interpolate(image, [height, width], options);
        }

        if self.do_rescale {
            image = image.mul_scalar(self.rescale_factor);
        }

        if self.do_normalize {
            let device = image.device();
            let mean = Tensor::<B, 1>::from_floats(self.mean, &device).reshape([1, 3, 1, 1]);
            let std = Tensor::<B, 1>::from_floats(self.std, &device).reshape([1, 3, 1, 1]);
            image = image.sub(mean).div(std);
        }

        image
    }
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize, Debug, Clone)]
pub struct RmbgProcessorConfig {
    pub do_normalize: Option<bool>,
    pub do_pad: Option<bool>,
    pub do_rescale: Option<bool>,
    pub do_resize: Option<bool>,
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub resample: Option<i64>,
    pub rescale_factor: Option<f32>,
    pub size: Option<RmbgProcessorSize>,
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize, Debug, Clone)]
pub struct RmbgProcessorSize {
    pub width: usize,
    pub height: usize,
}
