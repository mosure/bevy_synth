use std::fs;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{Detection, ObjectGroundingEvidence, SceneGroundingEvidence, SceneObjectManifest};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneObjectCategory {
    Chair,
    Table,
    Sofa,
    Plant,
    TabletopAccessory,
    Display,
    Controller,
    Monitor,
    Light,
    Wall,
    Window,
    Floor,
    Rug,
    Other,
}

impl SceneObjectCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chair => "chair",
            Self::Table => "table",
            Self::Sofa => "sofa",
            Self::Plant => "plant",
            Self::TabletopAccessory => "tabletop_accessory",
            Self::Display => "display",
            Self::Controller => "controller",
            Self::Monitor => "monitor",
            Self::Light => "light",
            Self::Wall => "wall",
            Self::Window => "window",
            Self::Floor => "floor",
            Self::Rug => "rug",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCategoryFilterConfig {
    #[serde(default = "default_scene_category_allowlist")]
    pub allow: Vec<String>,
    #[serde(default = "default_scene_category_denylist")]
    pub deny: Vec<String>,
}

impl Default for SceneCategoryFilterConfig {
    fn default() -> Self {
        Self {
            allow: default_scene_category_allowlist(),
            deny: default_scene_category_denylist(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlConfig {
    #[serde(default)]
    pub scene: ScenePipelineTomlScene,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlScene {
    #[serde(default)]
    pub categories: Option<SceneCategoryFilterConfig>,
    #[serde(default)]
    pub instances: Option<ScenePipelineTomlInstances>,
    #[serde(default)]
    pub models: Option<ScenePipelineTomlModels>,
    #[serde(default)]
    pub grounding: Option<ScenePipelineTomlGrounding>,
    #[serde(default)]
    pub output: Option<ScenePipelineTomlOutput>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlInstances {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub type_aware_categories: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlModels {
    #[serde(default)]
    pub image_to_3d: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlGrounding {
    #[serde(default)]
    pub locate_anything: Option<bool>,
    #[serde(default)]
    pub depth: Option<bool>,
    #[serde(default)]
    pub segmentation: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ScenePipelineTomlOutput {
    #[serde(default)]
    pub pbr: Option<bool>,
    #[serde(default)]
    pub target_faces: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCategoryFilterReport {
    pub config: SceneCategoryFilterConfig,
    pub kept_objects: Vec<SceneCategoryFilterEntry>,
    pub dropped_objects: Vec<SceneCategoryFilterEntry>,
    pub kept_detections: Vec<SceneCategoryFilterEntry>,
    pub dropped_detections: Vec<SceneCategoryFilterEntry>,
    pub kept_evidence_objects: Vec<SceneCategoryFilterEntry>,
    pub dropped_evidence_objects: Vec<SceneCategoryFilterEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCategoryFilterEntry {
    pub id: String,
    pub label: String,
    pub category: String,
    pub reason: String,
}

pub fn default_scene_category_allowlist() -> Vec<String> {
    ["chair", "table", "sofa", "plant"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn default_scene_category_denylist() -> Vec<String> {
    [
        "tabletop_accessory",
        "display",
        "controller",
        "monitor",
        "light",
        "ceiling_light",
        "wall",
        "window",
        "floor",
        "rug",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub fn scene_category_filter_prompt(config: &SceneCategoryFilterConfig) -> String {
    format!(
        "Allowed reconstruction object categories: {}. Excluded categories: {}. Do not plan excluded categories as standalone 3D assets.",
        config.allow.join(", "),
        config.deny.join(", ")
    )
}

pub fn scene_pipeline_toml_template() -> &'static str {
    r#"[scene.categories]
allow = ["chair", "table", "sofa", "plant"]
deny = ["tabletop_accessory", "display", "controller", "monitor", "light", "ceiling_light", "wall", "window", "floor", "rug"]

[scene.instances]
mode = "type-aware-reuse"
type_aware_categories = ["chair"]

[scene.models]
image_to_3d = "trellis"

[scene.grounding]
locate_anything = true
depth = true
segmentation = true

[scene.output]
pbr = true
target_faces = 80000
"#
}

pub fn load_scene_pipeline_toml(source: &str) -> Result<ScenePipelineTomlConfig, String> {
    let contents = if source.starts_with("http://") || source.starts_with("https://") {
        fetch_toml_url(source)?
    } else {
        fs::read_to_string(source).map_err(|err| format!("read scene config {source}: {err}"))?
    };
    toml::from_str(&contents).map_err(|err| format!("parse scene config {source}: {err}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_toml_url(url: &str) -> Result<String, String> {
    let response = reqwest::blocking::get(url).map_err(|err| format!("fetch {url}: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("fetch {url}: HTTP {status}"));
    }
    response.text().map_err(|err| format!("read {url}: {err}"))
}

#[cfg(target_arch = "wasm32")]
fn fetch_toml_url(url: &str) -> Result<String, String> {
    Err(format!(
        "synchronous scene config URL fetch is unavailable on wasm for {url}; use the async Bevy/wasm loader and pass parsed config"
    ))
}

pub fn apply_scene_category_filter(
    manifest: &SceneObjectManifest,
    evidence: Option<&SceneGroundingEvidence>,
    config: &SceneCategoryFilterConfig,
) -> (
    SceneObjectManifest,
    Option<SceneGroundingEvidence>,
    SceneCategoryFilterReport,
) {
    let mut report = SceneCategoryFilterReport {
        config: config.clone(),
        kept_objects: Vec::new(),
        dropped_objects: Vec::new(),
        kept_detections: Vec::new(),
        dropped_detections: Vec::new(),
        kept_evidence_objects: Vec::new(),
        dropped_evidence_objects: Vec::new(),
    };

    let mut filtered_manifest = manifest.clone();
    filtered_manifest.objects.retain(|object| {
        let category = infer_object_category(object);
        let (allowed, reason) = category_allowed(category, config);
        let entry = SceneCategoryFilterEntry {
            id: object.id.clone(),
            label: object.label.clone(),
            category: category.as_str().to_string(),
            reason,
        };
        if allowed {
            report.kept_objects.push(entry);
        } else {
            report.dropped_objects.push(entry);
        }
        allowed
    });

    let filtered_evidence = evidence.map(|evidence| {
        let mut next = evidence.clone();
        next.detections.retain(|detection| {
            let category = infer_detection_category(detection);
            let (allowed, reason) = category_allowed(category, config);
            let entry = SceneCategoryFilterEntry {
                id: detection.source_query.clone(),
                label: detection.label.clone(),
                category: category.as_str().to_string(),
                reason,
            };
            if allowed {
                report.kept_detections.push(entry);
            } else {
                report.dropped_detections.push(entry);
            }
            allowed
        });
        next.objects.retain(|object| {
            let category = object
                .detection
                .as_ref()
                .map(infer_detection_category)
                .unwrap_or_else(|| infer_evidence_object_category(object, manifest));
            let (allowed, reason) = category_allowed(category, config);
            let entry = SceneCategoryFilterEntry {
                id: object
                    .instance_id
                    .as_ref()
                    .map(|instance| format!("{}:{instance}", object.object_id))
                    .unwrap_or_else(|| object.object_id.clone()),
                label: object
                    .detection
                    .as_ref()
                    .map(|detection| detection.label.clone())
                    .unwrap_or_else(|| object.object_id.clone()),
                category: category.as_str().to_string(),
                reason,
            };
            if allowed {
                report.kept_evidence_objects.push(entry);
            } else {
                report.dropped_evidence_objects.push(entry);
            }
            allowed
        });
        next
    });

    (filtered_manifest, filtered_evidence, report)
}

fn infer_evidence_object_category(
    object: &ObjectGroundingEvidence,
    manifest: &SceneObjectManifest,
) -> SceneObjectCategory {
    manifest
        .objects
        .iter()
        .find(|candidate| candidate.id == object.object_id)
        .map(infer_object_category)
        .unwrap_or_else(|| canonical_scene_category(&object.object_id))
}

fn category_allowed(
    category: SceneObjectCategory,
    config: &SceneCategoryFilterConfig,
) -> (bool, String) {
    if category_in_list(category, &config.deny) {
        return (false, "category denied".to_string());
    }
    if config.allow.is_empty() || category_in_list(category, &config.allow) {
        return (true, "category allowed".to_string());
    }
    (false, "category not allowlisted".to_string())
}

fn category_in_list(category: SceneObjectCategory, values: &[String]) -> bool {
    values
        .iter()
        .filter_map(|value| parse_scene_category(value))
        .any(|candidate| candidate == category)
}

pub fn infer_object_category(object: &crate::SceneObjectSpec) -> SceneObjectCategory {
    let label_category = canonical_scene_category(&object.label);
    if label_category != SceneObjectCategory::Other {
        return label_category;
    }

    let mut primary_text = String::new();
    primary_text.push_str(&object.aliases.join(" "));
    if let Some(reuse_group) = object.reuse_group.as_ref() {
        primary_text.push(' ');
        primary_text.push_str(reuse_group);
    }
    let primary_category = canonical_scene_category(&primary_text);
    if primary_category != SceneObjectCategory::Other {
        return primary_category;
    }
    canonical_scene_category(&object.object_prompt)
}

pub fn infer_detection_category(detection: &Detection) -> SceneObjectCategory {
    let label_category = canonical_scene_category(&detection.label);
    if label_category != SceneObjectCategory::Other {
        return label_category;
    }
    canonical_scene_category(&detection.source_query)
}

pub fn canonical_scene_category(value: &str) -> SceneObjectCategory {
    parse_scene_category(value).unwrap_or(SceneObjectCategory::Other)
}

pub fn parse_scene_category(value: &str) -> Option<SceneObjectCategory> {
    let key = normalized_category_text(value);
    if key.is_empty() {
        return None;
    }
    let words = key.split_whitespace().collect::<Vec<_>>();
    let has = |needle: &str| words.contains(&needle);
    let has_phrase = |phrase: &str| key.contains(phrase);

    if has_phrase("tabletop")
        || has_phrase("table top")
        || has_phrase("table accessory")
        || has_phrase("table display")
        || has_phrase("conference display")
    {
        return Some(SceneObjectCategory::TabletopAccessory);
    }
    if has("controller") || has_phrase("control panel") {
        return Some(SceneObjectCategory::Controller);
    }
    if has("display") || has("screen") {
        return Some(SceneObjectCategory::Display);
    }
    if has("monitor") {
        return Some(SceneObjectCategory::Monitor);
    }
    if has("light") || has("lamp") || has_phrase("ceiling light") {
        return Some(SceneObjectCategory::Light);
    }
    if has("wall") {
        return Some(SceneObjectCategory::Wall);
    }
    if has("window") {
        return Some(SceneObjectCategory::Window);
    }
    if has("floor") {
        return Some(SceneObjectCategory::Floor);
    }
    if has("rug") || has("carpet") {
        return Some(SceneObjectCategory::Rug);
    }
    if has("sofa") || has("couch") || has("sectional") || has("settee") || has("banquette") {
        return Some(SceneObjectCategory::Sofa);
    }
    if has("chair") || has("seat") || has("stool") || has("armchair") {
        return Some(SceneObjectCategory::Chair);
    }
    if has("table") || has("desk") {
        return Some(SceneObjectCategory::Table);
    }
    if has("plant") || has("potted") || has("tree") {
        return Some(SceneObjectCategory::Plant);
    }
    None
}

fn normalized_category_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
