#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RuntimeQualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeQualityDefaults {
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub resolution: usize,
    pub chunk_size: usize,
    pub dense_octree_depth: usize,
    pub hierarchical_octree_depth: usize,
    pub band_threshold: f32,
    pub flash_octree_depth: usize,
    pub flash_min_resolution: usize,
    pub flash_mini_grid_num: usize,
    pub flash_num_chunks: usize,
    pub flash_mc_level: f32,
}

pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_TRIPOSG_TARGET_FACES: usize = 10_000;
pub const DEFAULT_TRELLIS_TARGET_FACES: usize = 1_000_000;
pub const DEFAULT_CHUNK_SIZE: usize = 10_000;
pub const DEFAULT_TRIPOSG_GUIDANCE_SCALE: f32 = 7.0;

impl RuntimeQualityPreset {
    pub fn defaults(self) -> RuntimeQualityDefaults {
        match self {
            Self::Fast => RuntimeQualityDefaults {
                num_steps: 12,
                num_tokens: 512,
                guidance_scale: DEFAULT_TRIPOSG_GUIDANCE_SCALE,
                resolution: 128,
                chunk_size: DEFAULT_CHUNK_SIZE,
                dense_octree_depth: 6,
                hierarchical_octree_depth: 7,
                band_threshold: 1.0,
                flash_octree_depth: 7,
                flash_min_resolution: 31,
                flash_mini_grid_num: 2,
                flash_num_chunks: 4096,
                flash_mc_level: 0.0,
            },
            Self::Balanced => RuntimeQualityDefaults {
                num_steps: 20,
                num_tokens: 1024,
                guidance_scale: DEFAULT_TRIPOSG_GUIDANCE_SCALE,
                resolution: 192,
                chunk_size: DEFAULT_CHUNK_SIZE,
                dense_octree_depth: 7,
                hierarchical_octree_depth: 8,
                band_threshold: 1.0,
                flash_octree_depth: 8,
                flash_min_resolution: 31,
                flash_mini_grid_num: 4,
                flash_num_chunks: 8192,
                flash_mc_level: 0.0,
            },
            Self::Full => RuntimeQualityDefaults {
                num_steps: 50,
                num_tokens: 2048,
                guidance_scale: DEFAULT_TRIPOSG_GUIDANCE_SCALE,
                resolution: 256,
                chunk_size: DEFAULT_CHUNK_SIZE,
                dense_octree_depth: 8,
                hierarchical_octree_depth: 9,
                band_threshold: 1.0,
                flash_octree_depth: 9,
                flash_min_resolution: 63,
                flash_mini_grid_num: 4,
                flash_num_chunks: DEFAULT_CHUNK_SIZE,
                flash_mc_level: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_TRIPOSG_GUIDANCE_SCALE, RuntimeQualityPreset};

    #[test]
    fn balanced_defaults_are_the_canonical_triposg_runtime_values() {
        let defaults = RuntimeQualityPreset::Balanced.defaults();
        assert_eq!(defaults.num_steps, 20);
        assert_eq!(defaults.num_tokens, 1024);
        assert_eq!(defaults.guidance_scale, DEFAULT_TRIPOSG_GUIDANCE_SCALE);
        assert_eq!(defaults.resolution, 192);
        assert_eq!(defaults.flash_octree_depth, 8);
        assert_eq!(defaults.flash_num_chunks, 8192);
    }

    #[test]
    fn presets_are_monotonic_for_primary_quality_controls() {
        let fast = RuntimeQualityPreset::Fast.defaults();
        let balanced = RuntimeQualityPreset::Balanced.defaults();
        let full = RuntimeQualityPreset::Full.defaults();
        assert!(fast.num_steps < balanced.num_steps);
        assert!(balanced.num_steps < full.num_steps);
        assert!(fast.num_tokens < balanced.num_tokens);
        assert!(balanced.num_tokens < full.num_tokens);
        assert!(fast.resolution < balanced.resolution);
        assert!(balanced.resolution < full.resolution);
    }
}
