use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingModelConfig {
    pub model_type: String,
    pub image_token_index: u32,
    pub box_start_token_id: u32,
    pub box_end_token_id: u32,
    pub coord_start_token_id: u32,
    pub coord_end_token_id: u32,
    pub none_token_id: u32,
    pub ref_start_token_id: u32,
    pub ref_end_token_id: u32,
    pub text_config: LocateAnythingTextConfig,
    pub vision_config: LocateAnythingVisionBackboneConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingTextConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub switch_token_id: Option<u32>,
    #[serde(default)]
    pub null_token_id: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingVisionBackboneConfig {
    pub model_type: String,
    pub patch_size: usize,
    pub init_pos_emb_height: usize,
    pub init_pos_emb_width: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub merge_kernel_size: [usize; 2],
}

impl LocateAnythingModelConfig {
    pub fn from_model_root(model_root: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        Self::from_config_path(model_root.as_ref().join("config.json"))
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| {
            LocateAnythingError::Config(format!("failed to read {}: {err}", path.display()))
        })?;
        let config = serde_json::from_slice::<Self>(&bytes).map_err(|err| {
            LocateAnythingError::Config(format!("failed to parse {}: {err}", path.display()))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> LocateAnythingResult<()> {
        if self.model_type != "locateanything" {
            return Err(LocateAnythingError::Config(format!(
                "expected model_type=locateanything, got {}",
                self.model_type
            )));
        }
        if self.vision_config.model_type != "moonvit" {
            return Err(LocateAnythingError::Config(format!(
                "unsupported vision model type {}",
                self.vision_config.model_type
            )));
        }
        if !self
            .text_config
            .architectures
            .iter()
            .any(|arch| arch == "Qwen2ForCausalLM" || arch == "Qwen3ForCausalLM")
        {
            return Err(LocateAnythingError::Config(format!(
                "unsupported text architecture {:?}",
                self.text_config.architectures
            )));
        }
        if self.text_config.num_attention_heads == 0
            || !self
                .text_config
                .hidden_size
                .is_multiple_of(self.text_config.num_attention_heads)
        {
            return Err(LocateAnythingError::Config(format!(
                "hidden_size {} must divide num_attention_heads {}",
                self.text_config.hidden_size, self.text_config.num_attention_heads
            )));
        }
        if self.text_config.num_key_value_heads == 0
            || !self
                .text_config
                .num_attention_heads
                .is_multiple_of(self.text_config.num_key_value_heads)
        {
            return Err(LocateAnythingError::Config(format!(
                "num_attention_heads {} must divide num_key_value_heads {}",
                self.text_config.num_attention_heads, self.text_config.num_key_value_heads
            )));
        }
        if self.vision_config.num_attention_heads == 0
            || !self
                .vision_config
                .hidden_size
                .is_multiple_of(self.vision_config.num_attention_heads)
        {
            return Err(LocateAnythingError::Config(format!(
                "vision hidden_size {} must divide num_attention_heads {}",
                self.vision_config.hidden_size, self.vision_config.num_attention_heads
            )));
        }
        Ok(())
    }

    pub fn text_head_dim(&self) -> usize {
        self.text_config.hidden_size / self.text_config.num_attention_heads
    }

    pub fn vision_head_dim(&self) -> usize {
        self.vision_config.hidden_size / self.vision_config.num_attention_heads
    }

    pub fn coordinate_bins(&self) -> usize {
        (self.coord_end_token_id - self.coord_start_token_id + 1) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_public_locate_anything_config_when_present() {
        let path = Path::new("assets/models/LocateAnything-3B/config.json");
        if !path.exists() {
            eprintln!("skipping config parse test; {} is missing", path.display());
            return;
        }
        let config = LocateAnythingModelConfig::from_config_path(path).unwrap();
        assert_eq!(config.text_config.num_hidden_layers, 36);
        assert_eq!(config.text_config.hidden_size, 2048);
        assert_eq!(config.text_config.num_attention_heads, 16);
        assert_eq!(config.text_config.num_key_value_heads, 2);
        assert_eq!(config.text_head_dim(), 128);
        assert_eq!(config.vision_config.num_hidden_layers, 27);
        assert_eq!(config.vision_head_dim(), 72);
        assert_eq!(config.coordinate_bins(), 1001);
    }
}
