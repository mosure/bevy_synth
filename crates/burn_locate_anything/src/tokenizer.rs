use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::models::bpe::BPE;
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::normalizers::NFC;
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::pre_tokenizers::sequence::Sequence;
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
#[cfg(not(target_arch = "wasm32"))]
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct QwenTokenizerConfig {
    pub tokenizer_json: Option<String>,
    pub coordinate_bins: usize,
    pub box_start_token: String,
    pub box_end_token: String,
    pub point_start_token: String,
    pub point_end_token: String,
}

impl Default for QwenTokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_json: None,
            coordinate_bins: 1000,
            box_start_token: "<box>".to_string(),
            box_end_token: "</box>".to_string(),
            point_start_token: "<point>".to_string(),
            point_end_token: "</point>".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingMediaTokens {
    pub image_token: String,
    pub image_token_id: u32,
    pub image_start_token: String,
    pub image_end_token: String,
    pub image_placeholder: String,
    pub merge_kernel_size: [usize; 2],
}

impl Default for LocateAnythingMediaTokens {
    fn default() -> Self {
        Self {
            image_token: "<IMG_CONTEXT>".to_string(),
            image_token_id: 151_665,
            image_start_token: "<img>".to_string(),
            image_end_token: "</img>".to_string(),
            image_placeholder: "image".to_string(),
            merge_kernel_size: [2, 2],
        }
    }
}

impl LocateAnythingMediaTokens {
    pub fn from_model_root(model_root: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        let model_root = model_root.as_ref();
        let mut tokens = Self::default();
        let processor_config = model_root.join("processor_config.json");
        if processor_config.exists() {
            let bytes = std::fs::read(&processor_config).map_err(|err| {
                LocateAnythingError::Config(format!(
                    "failed to read {}: {err}",
                    processor_config.display()
                ))
            })?;
            let config = serde_json::from_slice::<ProcessorConfigJson>(&bytes).map_err(|err| {
                LocateAnythingError::Config(format!(
                    "failed to parse {}: {err}",
                    processor_config.display()
                ))
            })?;
            if let Some(image_token) = config.image_token {
                tokens.image_token = image_token;
            }
            if let Some(image_start_token) = config.image_start_token {
                tokens.image_start_token = image_start_token;
            }
            if let Some(image_end_token) = config.image_end_token {
                tokens.image_end_token = image_end_token;
            }
            if let Some(image_placeholder) = config.image_placeholder {
                tokens.image_placeholder = image_placeholder;
            }
            if let Some(merge_kernel_size) = config.merge_kernel_size {
                tokens.merge_kernel_size = merge_kernel_size;
            }
        }
        let added_tokens = model_root.join("added_tokens.json");
        if added_tokens.exists() {
            let map = read_added_tokens_map(&added_tokens)?;
            if let Some(id) = map.get(&tokens.image_token) {
                tokens.image_token_id = *id;
            }
        }
        Ok(tokens)
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ProcessorConfigJson {
    image_token: Option<String>,
    image_start_token: Option<String>,
    image_end_token: Option<String>,
    image_placeholder: Option<String>,
    merge_kernel_size: Option<[usize; 2]>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Deserialize)]
struct AddedTokenDecoderEntry {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    special: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: BTreeMap<String, AddedTokenDecoderEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingPromptInputs {
    pub prompt_text: String,
    pub expanded_text: String,
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub image_token_positions: Vec<usize>,
    pub image_context_tokens: usize,
}

impl LocateAnythingPromptInputs {
    pub fn validate_image_token_count(&self) -> LocateAnythingResult<()> {
        if self.image_token_positions.len() != self.image_context_tokens {
            return Err(LocateAnythingError::Config(format!(
                "prompt has {} image token ids but expected {} from image grid",
                self.image_token_positions.len(),
                self.image_context_tokens
            )));
        }
        Ok(())
    }
}

pub struct QwenTokenizer {
    #[cfg(not(target_arch = "wasm32"))]
    tokenizer: Tokenizer,
    media_tokens: LocateAnythingMediaTokens,
}

impl QwenTokenizer {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_model_root(model_root: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        let model_root = model_root.as_ref();
        let media_tokens = LocateAnythingMediaTokens::from_model_root(model_root)?;
        let tokenizer_json = model_root.join("tokenizer.json");
        let tokenizer = if tokenizer_json.exists() {
            Tokenizer::from_file(&tokenizer_json).map_err(|err| {
                LocateAnythingError::Config(format!(
                    "failed to load {}: {err}",
                    tokenizer_json.display()
                ))
            })?
        } else {
            build_qwen_bpe_tokenizer(model_root)?
        };
        Ok(Self {
            tokenizer,
            media_tokens,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_model_root(model_root: impl AsRef<Path>) -> LocateAnythingResult<Self> {
        let media_tokens = LocateAnythingMediaTokens::from_model_root(model_root)?;
        let _ = media_tokens;
        Err(LocateAnythingError::Unsupported(
            "Burn-native LocateAnything tokenizer is not available on wasm yet; use the explicit reference/host path or add a wasm-safe tokenizer implementation"
                .to_string(),
        ))
    }

    pub fn media_tokens(&self) -> &LocateAnythingMediaTokens {
        &self.media_tokens
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode(&self, text: &str) -> LocateAnythingResult<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(|err| {
            LocateAnythingError::Runtime(format!("LocateAnything tokenizer encode failed: {err}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn encode(&self, _text: &str) -> LocateAnythingResult<Vec<u32>> {
        Err(LocateAnythingError::Unsupported(
            "Burn-native LocateAnything tokenizer encode is not available on wasm yet".to_string(),
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> LocateAnythingResult<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| {
                LocateAnythingError::Runtime(format!(
                    "LocateAnything tokenizer decode failed: {err}"
                ))
            })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn decode(&self, _ids: &[u32], _skip_special_tokens: bool) -> LocateAnythingResult<String> {
        Err(LocateAnythingError::Unsupported(
            "Burn-native LocateAnything tokenizer decode is not available on wasm yet".to_string(),
        ))
    }

    pub fn build_prompt_inputs(
        &self,
        query: &str,
        image_grid_hws: &[[usize; 2]],
    ) -> LocateAnythingResult<LocateAnythingPromptInputs> {
        let image_context_tokens =
            merged_image_token_count(image_grid_hws, self.media_tokens.merge_kernel_size);
        let prompt_text = upstream_detection_prompt(query);
        let expanded_text = apply_single_image_chat_template(
            &prompt_text,
            image_context_tokens,
            &self.media_tokens,
        );
        let input_ids = self.encode(&expanded_text)?;
        let image_token_positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (*id == self.media_tokens.image_token_id).then_some(index))
            .collect::<Vec<_>>();
        let attention_mask = vec![1; input_ids.len()];
        let inputs = LocateAnythingPromptInputs {
            prompt_text,
            expanded_text,
            input_ids,
            attention_mask,
            image_token_positions,
            image_context_tokens,
        };
        inputs.validate_image_token_count()?;
        Ok(inputs)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_qwen_bpe_tokenizer(model_root: &Path) -> LocateAnythingResult<Tokenizer> {
    let vocab = model_root.join("vocab.json");
    let merges = model_root.join("merges.txt");
    let vocab_path = path_str(&vocab)?;
    let merges_path = path_str(&merges)?;
    let bpe = BPE::from_file(vocab_path, merges_path)
        .unk_token("<|endoftext|>".to_string())
        .build()
        .map_err(|err| {
            LocateAnythingError::Config(format!(
                "failed to build BPE tokenizer from {} and {}: {err}",
                vocab.display(),
                merges.display()
            ))
        })?;
    let mut tokenizer = Tokenizer::new(bpe);
    let qwen_split = Split::new(
        SplitPattern::Regex(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                .to_string(),
        ),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|err| LocateAnythingError::Config(format!("invalid Qwen tokenizer regex: {err}")))?;
    tokenizer.with_normalizer(Some(NFC));
    tokenizer
        .with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(qwen_split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ])))
        .with_decoder(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default().trim_offsets(false)));

    let added_tokens = load_added_tokens(model_root)?;
    if !added_tokens.is_empty() {
        tokenizer.add_tokens(&added_tokens);
    }
    Ok(tokenizer)
}

#[cfg(not(target_arch = "wasm32"))]
fn path_str(path: &Path) -> LocateAnythingResult<&str> {
    path.to_str().ok_or_else(|| {
        LocateAnythingError::Config(format!("path {} is not valid UTF-8", path.display()))
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_added_tokens(model_root: &Path) -> LocateAnythingResult<Vec<AddedToken>> {
    let mut by_id = BTreeMap::<u32, AddedToken>::new();
    let tokenizer_config = model_root.join("tokenizer_config.json");
    if tokenizer_config.exists() {
        let bytes = std::fs::read(&tokenizer_config).map_err(|err| {
            LocateAnythingError::Config(format!(
                "failed to read {}: {err}",
                tokenizer_config.display()
            ))
        })?;
        let config = serde_json::from_slice::<TokenizerConfigJson>(&bytes).map_err(|err| {
            LocateAnythingError::Config(format!(
                "failed to parse {}: {err}",
                tokenizer_config.display()
            ))
        })?;
        for (raw_id, entry) in config.added_tokens_decoder {
            let id = raw_id.parse::<u32>().map_err(|err| {
                LocateAnythingError::Config(format!(
                    "invalid added token id `{raw_id}` in {}: {err}",
                    tokenizer_config.display()
                ))
            })?;
            let token = AddedToken::from(entry.content.clone(), entry.special)
                .lstrip(entry.lstrip)
                .normalized(entry.normalized)
                .rstrip(entry.rstrip)
                .single_word(entry.single_word);
            by_id.insert(id, token);
        }
    }
    let added_tokens = model_root.join("added_tokens.json");
    if added_tokens.exists() {
        for (token, id) in read_added_tokens_map(&added_tokens)? {
            by_id
                .entry(id)
                .or_insert_with(|| AddedToken::from(token, true).normalized(false));
        }
    }
    Ok(by_id.into_values().collect())
}

fn read_added_tokens_map(path: &Path) -> LocateAnythingResult<BTreeMap<String, u32>> {
    let bytes = std::fs::read(path).map_err(|err| {
        LocateAnythingError::Config(format!("failed to read {}: {err}", path.display()))
    })?;
    serde_json::from_slice::<BTreeMap<String, u32>>(&bytes).map_err(|err| {
        LocateAnythingError::Config(format!("failed to parse {}: {err}", path.display()))
    })
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PromptTemplate {
    pub system: String,
    pub user_prefix: String,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self {
            system:
                "You are a visual grounding model. Return labels with boxes and optional points."
                    .to_string(),
            user_prefix: "Locate".to_string(),
        }
    }
}

pub fn grounding_prompt(query: &str) -> String {
    let template = PromptTemplate::default();
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}: {}<|im_end|>\n<|im_start|>assistant\n",
        template.system,
        template.user_prefix,
        query.trim()
    )
}

pub fn upstream_detection_prompt(query: &str) -> String {
    format!(
        "Locate all the instances that matches the following description: {}.",
        query.trim().trim_end_matches('.')
    )
}

pub fn merged_image_token_count(
    image_grid_hws: &[[usize; 2]],
    merge_kernel_size: [usize; 2],
) -> usize {
    let merge_h = merge_kernel_size[0].max(1);
    let merge_w = merge_kernel_size[1].max(1);
    image_grid_hws
        .iter()
        .map(|[h, w]| h.saturating_mul(*w) / merge_h.saturating_mul(merge_w))
        .sum()
}

pub fn apply_single_image_chat_template(
    prompt: &str,
    image_context_tokens: usize,
    media_tokens: &LocateAnythingMediaTokens,
) -> String {
    let image_context = media_tokens.image_token.repeat(image_context_tokens);
    format!(
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n<image 1>{}{}{}{}<|im_end|>\n<|im_start|>assistant\n",
        media_tokens.image_start_token, image_context, media_tokens.image_end_token, prompt
    )
}

pub fn coordinate_token(value: f32, bins: usize) -> String {
    let bins = bins.max(2);
    let value = if value.is_finite() { value } else { 0.0 };
    let index = (value.clamp(0.0, 1.0) * (bins as f32 - 1.0)).round() as usize;
    format!("<loc_{index:04}>")
}

pub fn coordinate_token_from_id(token_id: u32, coord_start_token_id: u32) -> String {
    let value = token_id.saturating_sub(coord_start_token_id);
    format!("<{value}>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_io::load_tensor_from_safetensors_file;
    use std::path::PathBuf;

    #[test]
    fn prompt_contains_query() {
        let prompt = grounding_prompt("chairs and tables");
        assert!(prompt.contains("chairs and tables"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn coordinate_tokens_are_bounded() {
        assert_eq!(coordinate_token(-1.0, 1000), "<loc_0000>");
        assert_eq!(coordinate_token(1.5, 1000), "<loc_0999>");
    }

    #[test]
    fn upstream_prompt_matches_reference_phrase() {
        assert_eq!(
            upstream_detection_prompt("conference table."),
            "Locate all the instances that matches the following description: conference table."
        );
    }

    #[test]
    fn media_expansion_uses_merged_image_token_count() {
        let media = LocateAnythingMediaTokens::default();
        let token_count = merged_image_token_count(&[[50, 86]], media.merge_kernel_size);
        assert_eq!(token_count, 1075);
        let prompt = apply_single_image_chat_template(
            &upstream_detection_prompt("conference table"),
            token_count,
            &media,
        );
        assert!(prompt.contains("<image 1><img><IMG_CONTEXT>"));
        assert_eq!(prompt.matches("<IMG_CONTEXT>").count(), 1075);
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen_tokenizer_matches_reference_preprocess_ids_when_fixture_present() {
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping tokenizer parity fixture; repo root not found");
            return;
        };
        let model_root = root.join("assets/models/LocateAnything-3B");
        let fixture = root
            .join("tmp/runs/20260626T020100Z_locateanything_patch_embed_parity_galaxy/preprocess.safetensors");
        if !model_root.join("vocab.json").exists() || !fixture.exists() {
            eprintln!(
                "skipping tokenizer parity fixture; missing {} or {}",
                model_root.display(),
                fixture.display()
            );
            return;
        }
        let tokenizer = QwenTokenizer::from_model_root(&model_root).unwrap();
        let inputs = tokenizer
            .build_prompt_inputs("conference table", &[[50, 86]])
            .unwrap();
        let reference = load_tensor_from_safetensors_file(&fixture, "input_ids").unwrap();
        let reference_ids = reference
            .data
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        assert_eq!(inputs.input_ids, reference_ids);
        assert_eq!(inputs.image_token_positions.len(), 1075);
    }

    #[test]
    fn tokenizer_decode_preserves_generated_special_text_when_model_present() {
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping tokenizer decode fixture; repo root not found");
            return;
        };
        let model_root = root.join("assets/models/LocateAnything-3B");
        if !model_root.join("vocab.json").exists() {
            eprintln!(
                "skipping tokenizer decode fixture; missing {}",
                model_root.display()
            );
            return;
        }
        let tokenizer = QwenTokenizer::from_model_root(&model_root).unwrap();
        let text = tokenizer
            .decode(
                &[
                    151_672, 3122, 151_673, 151_668, 151_777, 151_877, 151_977, 152_077, 151_669,
                ],
                false,
            )
            .unwrap();
        assert!(text.contains("<ref>"));
        assert!(text.contains("<box>"));
        assert!(text.contains("<100>"));
    }

    fn find_repo_root_for_test() -> Option<PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_locate_anything").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}
