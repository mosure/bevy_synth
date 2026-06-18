#[cfg(feature = "legacy-wgpu-kernel")]
include!("wgpu_legacy.rs");

#[cfg(not(feature = "legacy-wgpu-kernel"))]
mod current {
    use burn::tensor::{Int, Tensor, activation::silu};

    use crate::{SparseSubmConvConfig, kernel_rows};

    /// Default WGPU backend type used by the tensor convenience wrappers.
    pub type DefaultWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct NeighborRowsBuildStats {
        pub cache_hits: u64,
        pub cache_misses: u64,
        pub host_builds: u64,
        pub device_builds: u64,
        pub device_scan_builds: u64,
        pub device_hash_builds: u64,
        pub device_scan_build_ns: u64,
        pub device_hash_build_ns: u64,
        pub device_hash_rows: u64,
        pub device_hash_probe_total: u64,
        pub device_hash_probe_max: u64,
        pub device_hash_insert_fail_rows: u64,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct SparseWgpuKernelStats {
        pub calls: u64,
        pub splitk_calls: u64,
        pub fused_variant_calls: u64,
        pub single_group_specialized_calls: u64,
        pub total_dispatches: u64,
        pub total_rows: u64,
        pub total_output_elements: u64,
        pub total_elapsed_ns: u64,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum SparseWgpuKernelVariant {
        Auto,
        Baseline,
        FusedOc4,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NeighborDeviceAlgoPreference {
        Auto,
        Scan,
        SortedHash,
        HashTableSerial,
        BucketHash,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SparseWgpuForwardConfig {
        pub kernel_variant: SparseWgpuKernelVariant,
        pub split_k: Option<usize>,
    }

    impl Default for SparseWgpuForwardConfig {
        fn default() -> Self {
            Self {
                kernel_variant: SparseWgpuKernelVariant::Auto,
                split_k: None,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SparseWgpuResolvedForwardConfig {
        pub kernel_variant: SparseWgpuKernelVariant,
        pub split_k: usize,
    }

    fn unavailable(name: &str) -> String {
        format!(
            "{name}: burn_flex_gmm WGPU CubeCL kernel is not available in the default Burn 0.21 path; port the legacy-wgpu-kernel implementation before enabling this runtime path"
        )
    }

    pub fn reset_neighbor_rows_build_stats() {}

    pub fn neighbor_rows_build_stats() -> NeighborRowsBuildStats {
        NeighborRowsBuildStats::default()
    }

    pub fn reset_sparse_wgpu_kernel_stats() {}

    pub fn sparse_wgpu_kernel_stats() -> SparseWgpuKernelStats {
        SparseWgpuKernelStats::default()
    }

    pub fn resolve_sparse_wgpu_forward_config(
        config: &SparseSubmConvConfig,
        _rows: usize,
        forward: SparseWgpuForwardConfig,
    ) -> Result<SparseWgpuResolvedForwardConfig, String> {
        let _ = kernel_rows(config)?;
        Ok(SparseWgpuResolvedForwardConfig {
            kernel_variant: match forward.kernel_variant {
                SparseWgpuKernelVariant::Auto => SparseWgpuKernelVariant::Baseline,
                variant => variant,
            },
            split_k: forward.split_k.unwrap_or(1).max(1),
        })
    }

    pub fn rope_rotate_pairs_wgpu(
        _x: Tensor<DefaultWgpuBackend, 4>,
        _cos: Tensor<DefaultWgpuBackend, 4>,
        _sin: Tensor<DefaultWgpuBackend, 4>,
    ) -> Result<Tensor<DefaultWgpuBackend, 4>, String> {
        Err(unavailable("rope_rotate_pairs_wgpu"))
    }

    pub fn rope_rotate_pairs_from_coords_wgpu(
        _x: Tensor<DefaultWgpuBackend, 4>,
        _coords: Tensor<DefaultWgpuBackend, 2, Int>,
        _rope_freq: [f32; 2],
    ) -> Result<Tensor<DefaultWgpuBackend, 4>, String> {
        Err(unavailable("rope_rotate_pairs_from_coords_wgpu"))
    }

    pub fn linear_skinny_forward_wgpu(
        input: Tensor<DefaultWgpuBackend, 2>,
        weight: Tensor<DefaultWgpuBackend, 2>,
        bias: Tensor<DefaultWgpuBackend, 1>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        let [rows, in_channels] = input.dims();
        let [weight_in, out_channels] = weight.dims();
        let [bias_channels] = bias.dims();
        if in_channels != weight_in {
            return Err(format!(
                "linear_skinny_forward_wgpu input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in},{out_channels}]"
            ));
        }
        if bias_channels != out_channels {
            return Err(format!(
                "linear_skinny_forward_wgpu bias mismatch: bias={bias_channels} out_channels={out_channels}"
            ));
        }
        Ok(input.matmul(weight).add(bias.unsqueeze_dim(0)))
    }

    pub fn layer_norm_affine_forward_wgpu(
        input: Tensor<DefaultWgpuBackend, 2>,
        weight: Tensor<DefaultWgpuBackend, 1>,
        bias: Tensor<DefaultWgpuBackend, 1>,
        eps: f32,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        let [_, channels] = input.dims();
        let [weight_channels] = weight.dims();
        let [bias_channels] = bias.dims();
        if weight_channels != channels || bias_channels != channels {
            return Err(format!(
                "layer_norm_affine_forward_wgpu parameter mismatch: channels={channels} weight={weight_channels} bias={bias_channels}"
            ));
        }
        let mean = input.clone().mean_dim(1);
        let centered = input - mean;
        let var = centered.clone().square().mean_dim(1);
        Ok(
            centered.div(var.add_scalar(eps).sqrt()) * weight.unsqueeze_dim(0)
                + bias.unsqueeze_dim(0),
        )
    }

    pub fn layer_norm_affine_silu_forward_wgpu(
        input: Tensor<DefaultWgpuBackend, 2>,
        weight: Tensor<DefaultWgpuBackend, 1>,
        bias: Tensor<DefaultWgpuBackend, 1>,
        eps: f32,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        Ok(silu(layer_norm_affine_forward_wgpu(
            input, weight, bias, eps,
        )?))
    }

    pub fn dense_trilinear_sample_attrs_wgpu(
        _positions: Tensor<DefaultWgpuBackend, 2>,
        _occupancy: Tensor<DefaultWgpuBackend, 1, Int>,
        _attrs: Tensor<DefaultWgpuBackend, 2>,
        _spatial: [usize; 3],
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        Err(unavailable("dense_trilinear_sample_attrs_wgpu"))
    }

    pub fn neighbor_rows_tensor_from_coords_tensor(
        _config: &SparseSubmConvConfig,
        _coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2, Int>, String> {
        Err(unavailable("neighbor_rows_tensor_from_coords_tensor"))
    }

    pub fn sparse_subm_conv_forward_wgpu(
        config: &SparseSubmConvConfig,
        input: Tensor<DefaultWgpuBackend, 2>,
        neighbor_rows: Tensor<DefaultWgpuBackend, 2, Int>,
        weight: Tensor<DefaultWgpuBackend, 5>,
        bias: Tensor<DefaultWgpuBackend, 1>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        sparse_subm_conv_forward_wgpu_with_config(
            config,
            input,
            neighbor_rows,
            weight,
            bias,
            SparseWgpuForwardConfig::default(),
        )
    }

    pub fn sparse_subm_conv_forward_wgpu_with_config(
        _config: &SparseSubmConvConfig,
        _input: Tensor<DefaultWgpuBackend, 2>,
        _neighbor_rows: Tensor<DefaultWgpuBackend, 2, Int>,
        _weight: Tensor<DefaultWgpuBackend, 5>,
        _bias: Tensor<DefaultWgpuBackend, 1>,
        _forward: SparseWgpuForwardConfig,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        Err(unavailable("sparse_subm_conv_forward_wgpu_with_config"))
    }
}

#[cfg(not(feature = "legacy-wgpu-kernel"))]
pub use current::*;
