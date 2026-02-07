use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{self, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::module::{interpolate, max_pool2d};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use crate::resize::resize_chw_align_corners_false;

#[derive(Config, Debug)]
pub struct RmbgConfig {
    pub in_ch: usize,
    pub out_ch: usize,
}

impl RmbgConfig {
    pub fn rmbg_1_4() -> Self {
        Self {
            in_ch: 3,
            out_ch: 1,
        }
    }

    #[cfg(feature = "import")]
    pub fn from_config_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let config: RmbgConfigFile = serde_json::from_slice(bytes)?;
        Ok(Self {
            in_ch: config.in_ch.unwrap_or(3),
            out_ch: config.out_ch.unwrap_or(1),
        })
    }

    #[cfg(feature = "import")]
    pub fn from_config_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Self::from_config_bytes(&bytes)
    }
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize)]
struct RmbgConfigFile {
    in_ch: Option<usize>,
    out_ch: Option<usize>,
}

#[derive(Module, Debug)]
pub struct RebnConv<B: Backend> {
    pub conv_s1: Conv2d<B>,
    pub bn_s1: nn::BatchNorm<B>,
    pub relu_s1: nn::Relu,
}

impl<B: Backend> RebnConv<B> {
    pub fn new(
        device: &B::Device,
        in_ch: usize,
        out_ch: usize,
        dirate: usize,
        stride: usize,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_stride([stride, stride])
            .with_dilation([dirate, dirate])
            .with_padding(PaddingConfig2d::Explicit(dirate, dirate))
            .init(device);
        let bn = nn::BatchNormConfig::new(out_ch).init(device);
        let relu = nn::Relu::new();
        Self {
            conv_s1: conv,
            bn_s1: bn,
            relu_s1: relu,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_s1.forward(x);
        let x = self.bn_s1.forward(x);
        self.relu_s1.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Rsu7<B: Backend> {
    pub rebnconvin: RebnConv<B>,
    pub rebnconv1: RebnConv<B>,
    pub rebnconv2: RebnConv<B>,
    pub rebnconv3: RebnConv<B>,
    pub rebnconv4: RebnConv<B>,
    pub rebnconv5: RebnConv<B>,
    pub rebnconv6: RebnConv<B>,
    pub rebnconv7: RebnConv<B>,
    pub rebnconv6d: RebnConv<B>,
    pub rebnconv5d: RebnConv<B>,
    pub rebnconv4d: RebnConv<B>,
    pub rebnconv3d: RebnConv<B>,
    pub rebnconv2d: RebnConv<B>,
    pub rebnconv1d: RebnConv<B>,
}

impl<B: Backend> Rsu7<B> {
    pub fn new(device: &B::Device, in_ch: usize, mid_ch: usize, out_ch: usize) -> Self {
        Self {
            rebnconvin: RebnConv::new(device, in_ch, out_ch, 1, 1),
            rebnconv1: RebnConv::new(device, out_ch, mid_ch, 1, 1),
            rebnconv2: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv3: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv4: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv5: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv6: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv7: RebnConv::new(device, mid_ch, mid_ch, 2, 1),
            rebnconv6d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv5d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv4d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv3d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv2d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv1d: RebnConv::new(device, mid_ch * 2, out_ch, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hxin = self.rebnconvin.forward(x);
        let hx1 = self.rebnconv1.forward(hxin.clone());
        let hx = max_pool2d_ceil(hx1.clone());
        let hx2 = self.rebnconv2.forward(hx);
        let hx = max_pool2d_ceil(hx2.clone());
        let hx3 = self.rebnconv3.forward(hx);
        let hx = max_pool2d_ceil(hx3.clone());
        let hx4 = self.rebnconv4.forward(hx);
        let hx = max_pool2d_ceil(hx4.clone());
        let hx5 = self.rebnconv5.forward(hx);
        let hx = max_pool2d_ceil(hx5.clone());
        let hx6 = self.rebnconv6.forward(hx);
        let hx7 = self.rebnconv7.forward(hx6.clone());

        let hx6d = self.rebnconv6d.forward(Tensor::cat(vec![hx7, hx6], 1));
        let hx6dup = upsample_like(hx6d, &hx5);
        let hx5d = self.rebnconv5d.forward(Tensor::cat(vec![hx6dup, hx5], 1));
        let hx5dup = upsample_like(hx5d, &hx4);
        let hx4d = self.rebnconv4d.forward(Tensor::cat(vec![hx5dup, hx4], 1));
        let hx4dup = upsample_like(hx4d, &hx3);
        let hx3d = self.rebnconv3d.forward(Tensor::cat(vec![hx4dup, hx3], 1));
        let hx3dup = upsample_like(hx3d, &hx2);
        let hx2d = self.rebnconv2d.forward(Tensor::cat(vec![hx3dup, hx2], 1));
        let hx2dup = upsample_like(hx2d, &hx1);
        let hx1d = self.rebnconv1d.forward(Tensor::cat(vec![hx2dup, hx1], 1));
        hx1d + hxin
    }
}

#[derive(Module, Debug)]
pub struct Rsu6<B: Backend> {
    pub rebnconvin: RebnConv<B>,
    pub rebnconv1: RebnConv<B>,
    pub rebnconv2: RebnConv<B>,
    pub rebnconv3: RebnConv<B>,
    pub rebnconv4: RebnConv<B>,
    pub rebnconv5: RebnConv<B>,
    pub rebnconv6: RebnConv<B>,
    pub rebnconv5d: RebnConv<B>,
    pub rebnconv4d: RebnConv<B>,
    pub rebnconv3d: RebnConv<B>,
    pub rebnconv2d: RebnConv<B>,
    pub rebnconv1d: RebnConv<B>,
}

impl<B: Backend> Rsu6<B> {
    pub fn new(device: &B::Device, in_ch: usize, mid_ch: usize, out_ch: usize) -> Self {
        Self {
            rebnconvin: RebnConv::new(device, in_ch, out_ch, 1, 1),
            rebnconv1: RebnConv::new(device, out_ch, mid_ch, 1, 1),
            rebnconv2: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv3: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv4: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv5: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv6: RebnConv::new(device, mid_ch, mid_ch, 2, 1),
            rebnconv5d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv4d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv3d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv2d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv1d: RebnConv::new(device, mid_ch * 2, out_ch, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hxin = self.rebnconvin.forward(x);
        let hx1 = self.rebnconv1.forward(hxin.clone());
        let hx = max_pool2d_ceil(hx1.clone());
        let hx2 = self.rebnconv2.forward(hx);
        let hx = max_pool2d_ceil(hx2.clone());
        let hx3 = self.rebnconv3.forward(hx);
        let hx = max_pool2d_ceil(hx3.clone());
        let hx4 = self.rebnconv4.forward(hx);
        let hx = max_pool2d_ceil(hx4.clone());
        let hx5 = self.rebnconv5.forward(hx);
        let hx6 = self.rebnconv6.forward(hx5.clone());

        let hx5d = self.rebnconv5d.forward(Tensor::cat(vec![hx6, hx5], 1));
        let hx5dup = upsample_like(hx5d, &hx4);
        let hx4d = self.rebnconv4d.forward(Tensor::cat(vec![hx5dup, hx4], 1));
        let hx4dup = upsample_like(hx4d, &hx3);
        let hx3d = self.rebnconv3d.forward(Tensor::cat(vec![hx4dup, hx3], 1));
        let hx3dup = upsample_like(hx3d, &hx2);
        let hx2d = self.rebnconv2d.forward(Tensor::cat(vec![hx3dup, hx2], 1));
        let hx2dup = upsample_like(hx2d, &hx1);
        let hx1d = self.rebnconv1d.forward(Tensor::cat(vec![hx2dup, hx1], 1));
        hx1d + hxin
    }
}

#[derive(Module, Debug)]
pub struct Rsu5<B: Backend> {
    pub rebnconvin: RebnConv<B>,
    pub rebnconv1: RebnConv<B>,
    pub rebnconv2: RebnConv<B>,
    pub rebnconv3: RebnConv<B>,
    pub rebnconv4: RebnConv<B>,
    pub rebnconv5: RebnConv<B>,
    pub rebnconv4d: RebnConv<B>,
    pub rebnconv3d: RebnConv<B>,
    pub rebnconv2d: RebnConv<B>,
    pub rebnconv1d: RebnConv<B>,
}

impl<B: Backend> Rsu5<B> {
    pub fn new(device: &B::Device, in_ch: usize, mid_ch: usize, out_ch: usize) -> Self {
        Self {
            rebnconvin: RebnConv::new(device, in_ch, out_ch, 1, 1),
            rebnconv1: RebnConv::new(device, out_ch, mid_ch, 1, 1),
            rebnconv2: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv3: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv4: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv5: RebnConv::new(device, mid_ch, mid_ch, 2, 1),
            rebnconv4d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv3d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv2d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv1d: RebnConv::new(device, mid_ch * 2, out_ch, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hxin = self.rebnconvin.forward(x);
        let hx1 = self.rebnconv1.forward(hxin.clone());
        let hx = max_pool2d_ceil(hx1.clone());
        let hx2 = self.rebnconv2.forward(hx);
        let hx = max_pool2d_ceil(hx2.clone());
        let hx3 = self.rebnconv3.forward(hx);
        let hx = max_pool2d_ceil(hx3.clone());
        let hx4 = self.rebnconv4.forward(hx);
        let hx5 = self.rebnconv5.forward(hx4.clone());

        let hx4d = self.rebnconv4d.forward(Tensor::cat(vec![hx5, hx4], 1));
        let hx4dup = upsample_like(hx4d, &hx3);
        let hx3d = self.rebnconv3d.forward(Tensor::cat(vec![hx4dup, hx3], 1));
        let hx3dup = upsample_like(hx3d, &hx2);
        let hx2d = self.rebnconv2d.forward(Tensor::cat(vec![hx3dup, hx2], 1));
        let hx2dup = upsample_like(hx2d, &hx1);
        let hx1d = self.rebnconv1d.forward(Tensor::cat(vec![hx2dup, hx1], 1));
        hx1d + hxin
    }
}

#[derive(Module, Debug)]
pub struct Rsu4<B: Backend> {
    pub rebnconvin: RebnConv<B>,
    pub rebnconv1: RebnConv<B>,
    pub rebnconv2: RebnConv<B>,
    pub rebnconv3: RebnConv<B>,
    pub rebnconv4: RebnConv<B>,
    pub rebnconv3d: RebnConv<B>,
    pub rebnconv2d: RebnConv<B>,
    pub rebnconv1d: RebnConv<B>,
}

impl<B: Backend> Rsu4<B> {
    pub fn new(device: &B::Device, in_ch: usize, mid_ch: usize, out_ch: usize) -> Self {
        Self {
            rebnconvin: RebnConv::new(device, in_ch, out_ch, 1, 1),
            rebnconv1: RebnConv::new(device, out_ch, mid_ch, 1, 1),
            rebnconv2: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv3: RebnConv::new(device, mid_ch, mid_ch, 1, 1),
            rebnconv4: RebnConv::new(device, mid_ch, mid_ch, 2, 1),
            rebnconv3d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv2d: RebnConv::new(device, mid_ch * 2, mid_ch, 1, 1),
            rebnconv1d: RebnConv::new(device, mid_ch * 2, out_ch, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hxin = self.rebnconvin.forward(x);
        let hx1 = self.rebnconv1.forward(hxin.clone());
        let hx = max_pool2d_ceil(hx1.clone());
        let hx2 = self.rebnconv2.forward(hx);
        let hx = max_pool2d_ceil(hx2.clone());
        let hx3 = self.rebnconv3.forward(hx);
        let hx4 = self.rebnconv4.forward(hx3.clone());

        let hx3d = self.rebnconv3d.forward(Tensor::cat(vec![hx4, hx3], 1));
        let hx3dup = upsample_like(hx3d, &hx2);
        let hx2d = self.rebnconv2d.forward(Tensor::cat(vec![hx3dup, hx2], 1));
        let hx2dup = upsample_like(hx2d, &hx1);
        let hx1d = self.rebnconv1d.forward(Tensor::cat(vec![hx2dup, hx1], 1));
        hx1d + hxin
    }
}

#[derive(Module, Debug)]
pub struct Rsu4f<B: Backend> {
    pub rebnconvin: RebnConv<B>,
    pub rebnconv1: RebnConv<B>,
    pub rebnconv2: RebnConv<B>,
    pub rebnconv3: RebnConv<B>,
    pub rebnconv4: RebnConv<B>,
    pub rebnconv3d: RebnConv<B>,
    pub rebnconv2d: RebnConv<B>,
    pub rebnconv1d: RebnConv<B>,
}

impl<B: Backend> Rsu4f<B> {
    pub fn new(device: &B::Device, in_ch: usize, mid_ch: usize, out_ch: usize) -> Self {
        Self {
            rebnconvin: RebnConv::new(device, in_ch, out_ch, 1, 1),
            rebnconv1: RebnConv::new(device, out_ch, mid_ch, 1, 1),
            rebnconv2: RebnConv::new(device, mid_ch, mid_ch, 2, 1),
            rebnconv3: RebnConv::new(device, mid_ch, mid_ch, 4, 1),
            rebnconv4: RebnConv::new(device, mid_ch, mid_ch, 8, 1),
            rebnconv3d: RebnConv::new(device, mid_ch * 2, mid_ch, 4, 1),
            rebnconv2d: RebnConv::new(device, mid_ch * 2, mid_ch, 2, 1),
            rebnconv1d: RebnConv::new(device, mid_ch * 2, out_ch, 1, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hxin = self.rebnconvin.forward(x);
        let hx1 = self.rebnconv1.forward(hxin.clone());
        let hx2 = self.rebnconv2.forward(hx1.clone());
        let hx3 = self.rebnconv3.forward(hx2.clone());
        let hx4 = self.rebnconv4.forward(hx3.clone());

        let hx3d = self.rebnconv3d.forward(Tensor::cat(vec![hx4, hx3], 1));
        let hx2d = self.rebnconv2d.forward(Tensor::cat(vec![hx3d, hx2], 1));
        let hx1d = self.rebnconv1d.forward(Tensor::cat(vec![hx2d, hx1], 1));
        hx1d + hxin
    }
}

#[derive(Debug)]
pub struct BriaRmbgOutput<B: Backend> {
    pub masks: Vec<Tensor<B, 4>>,
    pub features: Vec<Tensor<B, 4>>,
}

#[derive(Module, Debug)]
pub struct BriaRmbg<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub stage1: Rsu7<B>,
    pub stage2: Rsu6<B>,
    pub stage3: Rsu5<B>,
    pub stage4: Rsu4<B>,
    pub stage5: Rsu4f<B>,
    pub stage6: Rsu4f<B>,
    pub stage5d: Rsu4f<B>,
    pub stage4d: Rsu4<B>,
    pub stage3d: Rsu5<B>,
    pub stage2d: Rsu6<B>,
    pub stage1d: Rsu7<B>,
    pub side1: Conv2d<B>,
    pub side2: Conv2d<B>,
    pub side3: Conv2d<B>,
    pub side4: Conv2d<B>,
    pub side5: Conv2d<B>,
    pub side6: Conv2d<B>,
}

impl<B: Backend> BriaRmbg<B> {
    pub fn new(device: &B::Device, config: RmbgConfig) -> Self {
        let conv_in = Conv2dConfig::new([config.in_ch, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let stage1 = Rsu7::new(device, 64, 32, 64);
        let stage2 = Rsu6::new(device, 64, 32, 128);
        let stage3 = Rsu5::new(device, 128, 64, 256);
        let stage4 = Rsu4::new(device, 256, 128, 512);
        let stage5 = Rsu4f::new(device, 512, 256, 512);
        let stage6 = Rsu4f::new(device, 512, 256, 512);

        let stage5d = Rsu4f::new(device, 1024, 256, 512);
        let stage4d = Rsu4::new(device, 1024, 128, 256);
        let stage3d = Rsu5::new(device, 512, 64, 128);
        let stage2d = Rsu6::new(device, 256, 32, 64);
        let stage1d = Rsu7::new(device, 128, 16, 64);

        let side1 = Conv2dConfig::new([64, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let side2 = Conv2dConfig::new([64, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let side3 = Conv2dConfig::new([128, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let side4 = Conv2dConfig::new([256, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let side5 = Conv2dConfig::new([512, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let side6 = Conv2dConfig::new([512, config.out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv_in,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            stage6,
            stage5d,
            stage4d,
            stage3d,
            stage2d,
            stage1d,
            side1,
            side2,
            side3,
            side4,
            side5,
            side6,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> BriaRmbgOutput<B> {
        let hxin = self.conv_in.forward(x.clone());
        let hx1 = self.stage1.forward(hxin);
        let hx = max_pool2d_ceil(hx1.clone());
        let hx2 = self.stage2.forward(hx);
        let hx = max_pool2d_ceil(hx2.clone());
        let hx3 = self.stage3.forward(hx);
        let hx = max_pool2d_ceil(hx3.clone());
        let hx4 = self.stage4.forward(hx);
        let hx = max_pool2d_ceil(hx4.clone());
        let hx5 = self.stage5.forward(hx);
        let hx = max_pool2d_ceil(hx5.clone());
        let hx6 = self.stage6.forward(hx);
        let hx6up = upsample_like(hx6.clone(), &hx5);

        let hx5d = self
            .stage5d
            .forward(Tensor::cat(vec![hx6up, hx5.clone()], 1));
        let hx5dup = upsample_like(hx5d.clone(), &hx4);
        let hx4d = self
            .stage4d
            .forward(Tensor::cat(vec![hx5dup, hx4.clone()], 1));
        let hx4dup = upsample_like(hx4d.clone(), &hx3);
        let hx3d = self
            .stage3d
            .forward(Tensor::cat(vec![hx4dup, hx3.clone()], 1));
        let hx3dup = upsample_like(hx3d.clone(), &hx2);
        let hx2d = self
            .stage2d
            .forward(Tensor::cat(vec![hx3dup, hx2.clone()], 1));
        let hx2dup = upsample_like(hx2d.clone(), &hx1);
        let hx1d = self
            .stage1d
            .forward(Tensor::cat(vec![hx2dup, hx1.clone()], 1));

        let d1 = upsample_to(self.side1.forward(hx1d.clone()), &x);
        let d2 = upsample_to(self.side2.forward(hx2d.clone()), &x);
        let d3 = upsample_to(self.side3.forward(hx3d.clone()), &x);
        let d4 = upsample_to(self.side4.forward(hx4d.clone()), &x);
        let d5 = upsample_to(self.side5.forward(hx5d.clone()), &x);
        let d6 = upsample_to(self.side6.forward(hx6.clone()), &x);

        let masks = vec![
            sigmoid(d1),
            sigmoid(d2),
            sigmoid(d3),
            sigmoid(d4),
            sigmoid(d5),
            sigmoid(d6),
        ];
        let features = vec![hx1d, hx2d, hx3d, hx4d, hx5d, hx6];

        BriaRmbgOutput { masks, features }
    }
}

fn upsample_like<B: Backend>(src: Tensor<B, 4>, target: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [_b, _c, height, width] = target.dims();
    if std::env::var("RMBG_STRICT_INTERP").is_ok() {
        return interpolate_align_corners_false(src, height, width);
    }
    let options = InterpolateOptions {
        mode: InterpolateMode::Bilinear,
    };
    interpolate(src, [height, width], options)
}

fn upsample_to<B: Backend>(src: Tensor<B, 4>, target: &Tensor<B, 4>) -> Tensor<B, 4> {
    upsample_like(src, target)
}

fn interpolate_align_corners_false<B: Backend>(
    input: Tensor<B, 4>,
    out_height: usize,
    out_width: usize,
) -> Tensor<B, 4> {
    let device = input.device();
    let [batch, channels, in_height, in_width] = input.dims();
    let data = input
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("failed to read tensor data for strict RMBG interpolation");

    let batch_stride = channels * in_height * in_width;
    let mut resized = Vec::with_capacity(batch * channels * out_height * out_width);
    for b in 0..batch {
        let start = b * batch_stride;
        let end = start + batch_stride;
        let chunk = &data[start..end];
        let out = resize_chw_align_corners_false(
            chunk,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            InterpolateMode::Bilinear,
        );
        resized.extend(out);
    }

    let flat = Tensor::<B, 1>::from_floats(resized.as_slice(), &device);
    flat.reshape([
        batch as i32,
        channels as i32,
        out_height as i32,
        out_width as i32,
    ])
}

fn max_pool2d_ceil<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_b, _c, height, width] = x.dims();
    let kernel = 2usize;
    let stride = 2usize;

    let out_h = height.saturating_sub(kernel).div_ceil(stride) + 1;
    let out_w = width.saturating_sub(kernel).div_ceil(stride) + 1;

    let pad_h = (out_h - 1) * stride + kernel - height;
    let pad_w = (out_w - 1) * stride + kernel - width;

    let mut x = x;
    if pad_h > 0 || pad_w > 0 {
        x = x.pad((0, pad_w, 0, pad_h), f32::NEG_INFINITY);
    }

    max_pool2d(x, [kernel, kernel], [stride, stride], [0, 0], [1, 1])
}

#[cfg(feature = "import")]
pub mod import {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use burn::module::{Module, ModuleMapper, Param};
    use burn::prelude::*;
    use burn::tensor::Bytes;
    use burn::tensor::FloatDType;
    use burn_store::{
        BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };

    use super::{BriaRmbg, RmbgConfig};

    const F16_SUFFIX: &str = "_f16";

    pub fn load_rmbg<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
        config: &RmbgConfig,
    ) -> Result<BriaRmbg<B>, Box<dyn std::error::Error>> {
        let weights_path = weights_path.as_ref();
        let burnpack_candidates = candidate_burnpack_paths(weights_path);
        let burnpack_path = burnpack_candidates
            .iter()
            .find(|candidate| candidate.exists())
            .cloned();
        let Some(burnpack_path) = burnpack_path else {
            let checked = burnpack_candidates
                .iter()
                .map(|candidate| candidate.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Burnpack weights missing. Checked: {checked}. Run `triposg_import` to generate .bpk files."
            )
            .into());
        };

        let mut model = BriaRmbg::new(device, config.clone());
        let mut store = BurnpackStore::from_file(&burnpack_path).validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load RMBG burnpack: {err}"))?;
        Ok(model)
    }

    pub fn load_rmbg_from_burnpack_bytes<B: Backend>(
        device: &B::Device,
        burnpack_bytes: Vec<u8>,
        config: &RmbgConfig,
    ) -> Result<BriaRmbg<B>, Box<dyn std::error::Error>> {
        let mut model = BriaRmbg::new(device, config.clone());
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load RMBG burnpack bytes: {err}"))?;
        Ok(model)
    }

    pub fn load_rmbg_config_from_json_bytes(
        bytes: &[u8],
    ) -> Result<RmbgConfig, Box<dyn std::error::Error>> {
        RmbgConfig::from_config_bytes(bytes)
    }

    pub fn load_rmbg_config(
        root: impl AsRef<Path>,
    ) -> Result<RmbgConfig, Box<dyn std::error::Error>> {
        let path = root.as_ref().join("config.json");
        if path.exists() {
            return RmbgConfig::from_config_file(path);
        }
        Ok(RmbgConfig::rmbg_1_4())
    }

    pub fn resolve_rmbg_weights_root() -> PathBuf {
        if let Ok(root) = std::env::var("RMBG_WEIGHTS_ROOT") {
            let path = PathBuf::from(root);
            if path.exists() {
                return path;
            }
        }
        let tripo_assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../burn_3d_synth_tripo/assets/models/RMBG-1.4");
        if tripo_assets.exists() {
            return tripo_assets;
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/RMBG-1.4")
    }

    pub fn load_rmbg_processor_config(
        root: impl AsRef<Path>,
    ) -> Result<super::super::preprocess::RmbgImageProcessor, Box<dyn std::error::Error>> {
        let path = root.as_ref().join("preprocessor_config.json");
        if !path.exists() {
            return Ok(super::super::preprocess::RmbgImageProcessor::default());
        }
        let bytes = fs::read(path)?;
        load_rmbg_processor_from_json_bytes(&bytes)
    }

    pub fn load_rmbg_processor_from_json_bytes(
        bytes: &[u8],
    ) -> Result<super::super::preprocess::RmbgImageProcessor, Box<dyn std::error::Error>> {
        let config: super::super::preprocess::RmbgProcessorConfig = serde_json::from_slice(&bytes)?;
        Ok(super::super::preprocess::RmbgImageProcessor::from_config(
            config,
        ))
    }

    fn build_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in key_remap_rules() {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|err| format!("invalid remap rule {from}->{to}: {err}"))?;
        }

        let store = SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true)
            .remap(remapper)
            .validate(true);

        Ok(store)
    }

    fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
        &[
            (r"\.bn_s1\.weight$", ".bn_s1.gamma"),
            (r"\.bn_s1\.bias$", ".bn_s1.beta"),
        ]
    }

    fn candidate_burnpack_paths(path: &Path) -> Vec<PathBuf> {
        let default = burnpack_path(path, false);
        let f16 = burnpack_path(path, true);
        if f16 == default {
            vec![default]
        } else if prefer_f16_burnpack() {
            vec![f16, default]
        } else {
            vec![default, f16]
        }
    }

    fn prefer_f16_burnpack() -> bool {
        preferred_precision_from_env("RMBG_BPK_PRECISION", "BURN_3D_SYNTH_BPK_PRECISION")
    }

    fn preferred_precision_from_env(primary: &str, fallback: &str) -> bool {
        let value = std::env::var(primary)
            .ok()
            .or_else(|| std::env::var(fallback).ok());
        match value
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("f32" | "fp32" | "float32" | "32") => false,
            Some("f16" | "fp16" | "float16" | "half" | "16") => true,
            Some(_) | None => true,
        }
    }

    fn burnpack_path(path: &Path, use_f16: bool) -> PathBuf {
        let path = if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("bpk"))
            .unwrap_or(false)
        {
            path.to_path_buf()
        } else {
            path.with_extension("bpk")
        };

        if use_f16 {
            with_file_stem_suffix(&path, F16_SUFFIX)
        } else {
            path
        }
    }

    fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
        let Some(stem) = path.file_stem() else {
            return path.to_path_buf();
        };
        let stem = stem.to_string_lossy();
        if stem.ends_with(suffix) {
            return path.to_path_buf();
        }

        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        let mut file_name = format!("{stem}{suffix}");
        if !ext.is_empty() {
            file_name.push('.');
            file_name.push_str(ext);
        }
        path.with_file_name(file_name)
    }

    pub fn load_rmbg_from_safetensors<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
        config: &RmbgConfig,
    ) -> Result<BriaRmbg<B>, Box<dyn std::error::Error>> {
        let mut model = BriaRmbg::new(device, config.clone());
        let mut store = build_store(weights_path.as_ref())?;
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load RMBG weights: {err}"))?;
        Ok(model)
    }

    pub fn import_rmbg_burnpack<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
        config: &RmbgConfig,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let weights_path = weights_path.as_ref();
        let burnpack_path = burnpack_path(weights_path, use_f16);
        let model = load_rmbg_from_safetensors::<B>(device, weights_path, config)?;
        let model = if use_f16 {
            cast_module_float_dtype(model, FloatDType::F16)
        } else {
            model
        };
        save_burnpack(&model, &burnpack_path)?;
        Ok(burnpack_path)
    }

    struct FloatDTypeMapper {
        dtype: FloatDType,
    }

    impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            let tensor = tensor.cast(self.dtype);
            Param::from_mapped_value(id, tensor, mapper)
        }
    }

    fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
        let mut mapper = FloatDTypeMapper { dtype };
        module.map(&mut mapper)
    }

    fn save_burnpack<B: Backend>(
        model: &BriaRmbg<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save RMBG burnpack: {err}"))?;
        Ok(())
    }
}
