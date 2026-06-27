#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

#[cfg(target_arch = "wasm32")]
use base64::Engine;
#[cfg(target_arch = "wasm32")]
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};

use crate::{
    GaussianSplatCloud, SynthAsset, SynthMesh, SynthMeshMaterial, SynthMeshPbrTextures,
    SynthMeshTexture,
};

const CACHE_VERSION: u32 = 6;
const TRIPOSPLAT_SPLAT_CACHE_NAMESPACE: &str = "triposplat-v2";
const SCENE_CACHE_NAMESPACE: &str = "scene-v1";
const INDEX_FILE_NAME: &str = "index.json";
#[cfg(not(target_arch = "wasm32"))]
const MESH_DIR_NAME: &str = "meshes";
#[cfg(not(target_arch = "wasm32"))]
const SCENE_DIR_NAME: &str = "scenes";
#[cfg(not(target_arch = "wasm32"))]
const SOURCE_IMAGE_DIR_NAME: &str = "source_images";

pub type CacheResult<T> = Result<T, CacheError>;

#[derive(Debug)]
pub enum CacheError {
    Io(String),
    Serialization(String),
    Storage(String),
    InvalidData(String),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Serialization(err) => write!(f, "serialization error: {err}"),
            Self::Storage(err) => write!(f, "storage error: {err}"),
            Self::InvalidData(err) => write!(f, "invalid data: {err}"),
        }
    }
}

impl std::error::Error for CacheError {}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CachedAssetKind {
    #[default]
    Mesh,
    GaussianSplat,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedMeshMetadata {
    pub cache_key: String,
    pub source_image_path: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_payload_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_mime: Option<String>,
    #[serde(default)]
    pub asset_kind: CachedAssetKind,
    pub mesh_payload_id: String,
    #[serde(default)]
    pub gltf_output_id: Option<String>,
    pub glb_output_id: String,
    #[serde(default)]
    pub splat_payload_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_aabb: Option<CachedAssetAabb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_frame: Option<CachedAssetFrame>,
    pub updated_at_unix_ms: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedAssetAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedAssetFrame {
    pub yaw_offset_degrees: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedWorldItem {
    pub cache_key: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CachedSourceImage {
    pub file_name: String,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedCameraState {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub focus: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CachedSceneMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_accepted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback_iteration: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_stage: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedSceneMetadata {
    pub scene_key: String,
    pub label: String,
    pub source_scene_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_payload_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_mime: Option<String>,
    pub scene_payload_id: String,
    pub pipeline: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<CachedSceneMetrics>,
    pub updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CachedScenePayload {
    #[serde(default)]
    pub world_items: Vec<CachedWorldItem>,
    #[serde(default)]
    pub camera: Option<CachedCameraState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bsn: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_bindings: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_summary: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheIndex {
    version: u32,
    #[serde(default)]
    meshes: Vec<CachedMeshMetadata>,
    #[serde(default)]
    scenes: Vec<CachedSceneMetadata>,
    #[serde(default)]
    world_items: Vec<CachedWorldItem>,
    #[serde(default)]
    camera: Option<CachedCameraState>,
}

impl Default for CacheIndex {
    fn default() -> Self {
        Self {
            version: CACHE_VERSION,
            meshes: Vec::new(),
            scenes: Vec::new(),
            world_items: Vec::new(),
            camera: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeshPayloadMaterial {
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    alpha: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeshPayloadTexture {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeshPayloadPbrTextures {
    base_color: MeshPayloadTexture,
    metallic_roughness: MeshPayloadTexture,
    #[serde(default)]
    normal: Option<MeshPayloadTexture>,
    #[serde(default)]
    emissive: Option<MeshPayloadTexture>,
    #[serde(default)]
    occlusion: Option<MeshPayloadTexture>,
}

impl From<SynthMeshMaterial> for MeshPayloadMaterial {
    fn from(material: SynthMeshMaterial) -> Self {
        Self {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            alpha: material.alpha,
        }
    }
}

impl From<MeshPayloadMaterial> for SynthMeshMaterial {
    fn from(material: MeshPayloadMaterial) -> Self {
        Self {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            alpha: material.alpha,
        }
    }
}

impl From<SynthMeshTexture> for MeshPayloadTexture {
    fn from(texture: SynthMeshTexture) -> Self {
        Self {
            width: texture.width,
            height: texture.height,
            rgba8: texture.rgba8,
        }
    }
}

impl From<MeshPayloadTexture> for SynthMeshTexture {
    fn from(texture: MeshPayloadTexture) -> Self {
        Self {
            width: texture.width,
            height: texture.height,
            rgba8: texture.rgba8,
        }
    }
}

impl From<SynthMeshPbrTextures> for MeshPayloadPbrTextures {
    fn from(textures: SynthMeshPbrTextures) -> Self {
        Self {
            base_color: textures.base_color.into(),
            metallic_roughness: textures.metallic_roughness.into(),
            normal: textures.normal.map(Into::into),
            emissive: textures.emissive.map(Into::into),
            occlusion: textures.occlusion.map(Into::into),
        }
    }
}

impl From<MeshPayloadPbrTextures> for SynthMeshPbrTextures {
    fn from(textures: MeshPayloadPbrTextures) -> Self {
        Self {
            base_color: textures.base_color.into(),
            metallic_roughness: textures.metallic_roughness.into(),
            normal: textures.normal.map(Into::into),
            emissive: textures.emissive.map(Into::into),
            occlusion: textures.occlusion.map(Into::into),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeshPayload {
    vertices: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
    #[serde(default)]
    uvs: Vec<[f32; 2]>,
    #[serde(default)]
    normals: Vec<[f32; 3]>,
    #[serde(default)]
    material: Option<MeshPayloadMaterial>,
    #[serde(default)]
    pbr_textures: Option<MeshPayloadPbrTextures>,
}

impl From<&SynthMesh> for MeshPayload {
    fn from(mesh: &SynthMesh) -> Self {
        Self {
            vertices: mesh.mesh.vertices.clone(),
            faces: mesh.mesh.faces.clone(),
            uvs: mesh.uvs.clone(),
            normals: mesh.normals.clone(),
            material: mesh.material.map(Into::into),
            pbr_textures: mesh.pbr_textures.clone().map(Into::into),
        }
    }
}

impl From<MeshPayload> for SynthMesh {
    fn from(payload: MeshPayload) -> Self {
        Self {
            mesh: crate::TripoMesh {
                vertices: payload.vertices,
                faces: payload.faces,
            },
            uvs: payload.uvs,
            normals: payload.normals,
            material: payload.material.map(Into::into),
            pbr_textures: payload.pbr_textures.map(Into::into),
        }
    }
}

fn mesh_local_aabb(mesh: &SynthMesh) -> Option<CachedAssetAabb> {
    let mut iter = mesh.mesh.vertices.iter();
    let first = *iter.next()?;
    let mut min = first;
    let mut max = first;
    for vertex in iter {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Some(CachedAssetAabb { min, max })
}

#[derive(Clone, Debug)]
pub struct MeshCache {
    index: CacheIndex,
    #[cfg(not(target_arch = "wasm32"))]
    root: PathBuf,
    #[cfg(target_arch = "wasm32")]
    prefix: String,
}

impl MeshCache {
    pub fn load_default() -> CacheResult<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            migrate_legacy_native_cache_root()?;
            Self::load_from_root(default_native_cache_root())
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::load_with_prefix(default_web_cache_prefix())
        }
    }

    pub fn empty_default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                index: CacheIndex::default(),
                root: default_native_cache_root(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                index: CacheIndex::default(),
                prefix: default_web_cache_prefix(),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_root(root: PathBuf) -> CacheResult<Self> {
        ensure_native_layout(&root)?;
        let mut cache = Self {
            index: read_native_index(&root)?,
            root,
        };
        cache.index.version = CACHE_VERSION;
        Ok(cache)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn load_with_prefix(prefix: impl Into<String>) -> CacheResult<Self> {
        let prefix = prefix.into();
        let mut cache = Self {
            index: read_web_index(&prefix)?,
            prefix,
        };
        cache.index.version = CACHE_VERSION;
        Ok(cache)
    }

    pub fn mesh_entries(&self) -> &[CachedMeshMetadata] {
        &self.index.meshes
    }

    pub fn asset_entries(&self) -> &[CachedMeshMetadata] {
        &self.index.meshes
    }

    pub fn scene_entries(&self) -> &[CachedSceneMetadata] {
        &self.index.scenes
    }

    pub fn world_items(&self) -> &[CachedWorldItem] {
        &self.index.world_items
    }

    pub fn camera_state(&self) -> Option<&CachedCameraState> {
        self.index.camera.as_ref()
    }

    pub fn load_mesh(&self, cache_key: &str) -> CacheResult<Option<SynthMesh>> {
        if let Some(glb_bytes) = self.read_glb_output(cache_key)? {
            let mesh = crate::io::mesh_from_glb_bytes(glb_bytes.as_slice())
                .map_err(|err| CacheError::InvalidData(err.to_string()))?;
            return Ok(Some(mesh));
        }

        // Backward compatibility: older cache versions stored the full mesh payload JSON.
        let mesh_payload = self.read_mesh_payload(cache_key)?;
        let Some(mesh_payload) = mesh_payload else {
            return Ok(None);
        };
        let payload: MeshPayload = serde_json::from_str(&mesh_payload)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        Ok(Some(payload.into()))
    }

    pub fn load_asset(&self, cache_key: &str) -> CacheResult<Option<SynthAsset>> {
        let kind = self
            .index
            .meshes
            .iter()
            .find(|entry| entry.cache_key == cache_key)
            .map(|entry| entry.asset_kind)
            .unwrap_or(CachedAssetKind::Mesh);
        match kind {
            CachedAssetKind::Mesh => self
                .load_mesh(cache_key)
                .map(|mesh| mesh.map(SynthAsset::Mesh)),
            CachedAssetKind::GaussianSplat => self
                .load_gaussian_splat(cache_key)
                .map(|splats| splats.map(SynthAsset::GaussianSplat)),
        }
    }

    pub fn load_source_image(
        &self,
        metadata: &CachedMeshMetadata,
    ) -> CacheResult<Option<CachedSourceImage>> {
        if let Some(bytes) = self.read_source_image_payload(metadata)? {
            return Ok(Some(CachedSourceImage {
                file_name: metadata.source_image_name.clone().unwrap_or_else(|| {
                    asset_label(&metadata.source_image_path, CachedAssetKind::Mesh)
                }),
                mime_type: metadata.source_image_mime.clone(),
                bytes,
            }));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = Path::new(&metadata.source_image_path);
            if path.is_file() {
                let bytes = fs::read(path).map_err(|err| CacheError::Io(err.to_string()))?;
                return Ok(Some(CachedSourceImage {
                    file_name: source_image_file_name(&metadata.source_image_path),
                    mime_type: source_image_mime(&metadata.source_image_path),
                    bytes,
                }));
            }
        }

        Ok(None)
    }

    pub fn load_scene_source_image(
        &self,
        metadata: &CachedSceneMetadata,
    ) -> CacheResult<Option<CachedSourceImage>> {
        if let Some(bytes) = self.read_scene_source_image_payload(metadata)? {
            return Ok(Some(CachedSourceImage {
                file_name: metadata
                    .source_image_name
                    .clone()
                    .unwrap_or_else(|| source_image_file_name(&metadata.source_scene_path)),
                mime_type: metadata.source_image_mime.clone(),
                bytes,
            }));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = Path::new(&metadata.source_scene_path);
            if path.is_file() {
                let bytes = fs::read(path).map_err(|err| CacheError::Io(err.to_string()))?;
                return Ok(Some(CachedSourceImage {
                    file_name: source_image_file_name(&metadata.source_scene_path),
                    mime_type: source_image_mime(&metadata.source_scene_path),
                    bytes,
                }));
            }
        }

        Ok(None)
    }

    pub fn load_gaussian_splat(&self, cache_key: &str) -> CacheResult<Option<GaussianSplatCloud>> {
        let Some(payload) = self.read_splat_payload(cache_key)? else {
            return Ok(None);
        };
        let splats: GaussianSplatCloud = serde_json::from_str(&payload)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        Ok(Some(splats))
    }

    pub fn upsert_mesh_for_image(
        &mut self,
        source_image_path: &Path,
        mesh: &SynthMesh,
    ) -> CacheResult<CachedMeshMetadata> {
        self.upsert_mesh_for_image_with_source_bytes(source_image_path, None, mesh)
    }

    pub fn upsert_mesh_for_image_with_source_bytes(
        &mut self,
        source_image_path: &Path,
        source_image_bytes: Option<&[u8]>,
        mesh: &SynthMesh,
    ) -> CacheResult<CachedMeshMetadata> {
        let source_image_path = normalize_source_image_path(source_image_path);
        let cache_key = cache_key_from_source(&source_image_path);
        let label = Path::new(&source_image_path)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("image")
            .to_string();

        // GLB is the canonical cache payload; remove any legacy mesh JSON sidecar.
        self.remove_mesh_payload(&cache_key)?;

        let glb = mesh_to_glb(mesh)?;
        self.write_glb_output(&cache_key, &glb)?;
        let glb_output_id = self.glb_output_id(&cache_key);
        let source_image_payload =
            self.write_source_image_payload(&cache_key, &source_image_path, source_image_bytes)?;

        let metadata = CachedMeshMetadata {
            cache_key: cache_key.clone(),
            source_image_path: source_image_path.clone(),
            label,
            source_image_payload_id: source_image_payload
                .as_ref()
                .map(|payload| payload.payload_id.clone()),
            source_image_name: source_image_payload
                .as_ref()
                .map(|payload| payload.file_name.clone()),
            source_image_mime: source_image_payload
                .as_ref()
                .and_then(|payload| payload.mime_type.clone()),
            asset_kind: CachedAssetKind::Mesh,
            mesh_payload_id: glb_output_id.clone(),
            gltf_output_id: None,
            glb_output_id,
            splat_payload_id: None,
            local_aabb: mesh_local_aabb(mesh),
            canonical_frame: None,
            updated_at_unix_ms: now_unix_ms(),
        };

        if let Some(position) = self.index.meshes.iter().position(|entry| {
            entry.source_image_path == source_image_path
                && entry.asset_kind == CachedAssetKind::Mesh
        }) {
            let old_cache_key = self.index.meshes[position].cache_key.clone();
            if old_cache_key != cache_key {
                self.remove_mesh_payload(&old_cache_key)?;
                self.remove_gltf_output(&old_cache_key)?;
                self.remove_glb_output(&old_cache_key)?;
                self.remove_splat_payload(&old_cache_key)?;
                self.remove_source_image_payload(&old_cache_key)?;
            }
            self.index.meshes[position] = metadata.clone();
        } else {
            self.index.meshes.push(metadata.clone());
        }

        self.save_index()?;
        Ok(metadata)
    }

    pub fn upsert_gaussian_splat_for_image(
        &mut self,
        source_image_path: &Path,
        splats: &GaussianSplatCloud,
    ) -> CacheResult<CachedMeshMetadata> {
        self.upsert_gaussian_splat_for_image_with_source_bytes(source_image_path, None, splats)
    }

    pub fn upsert_gaussian_splat_for_image_with_source_bytes(
        &mut self,
        source_image_path: &Path,
        source_image_bytes: Option<&[u8]>,
        splats: &GaussianSplatCloud,
    ) -> CacheResult<CachedMeshMetadata> {
        let source_image_path = normalize_source_image_path(source_image_path);
        let cache_key =
            cache_key_from_source_and_kind(&source_image_path, CachedAssetKind::GaussianSplat);
        let label = asset_label(&source_image_path, CachedAssetKind::GaussianSplat);

        self.remove_mesh_payload(&cache_key)?;
        self.remove_gltf_output(&cache_key)?;
        self.remove_glb_output(&cache_key)?;

        let payload = serde_json::to_string(splats)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        self.write_splat_payload(&cache_key, &payload)?;
        let splat_payload_id = self.splat_payload_id(&cache_key);
        let source_image_payload =
            self.write_source_image_payload(&cache_key, &source_image_path, source_image_bytes)?;

        let metadata = CachedMeshMetadata {
            cache_key: cache_key.clone(),
            source_image_path: source_image_path.clone(),
            label,
            source_image_payload_id: source_image_payload
                .as_ref()
                .map(|payload| payload.payload_id.clone()),
            source_image_name: source_image_payload
                .as_ref()
                .map(|payload| payload.file_name.clone()),
            source_image_mime: source_image_payload
                .as_ref()
                .and_then(|payload| payload.mime_type.clone()),
            asset_kind: CachedAssetKind::GaussianSplat,
            mesh_payload_id: splat_payload_id.clone(),
            gltf_output_id: None,
            glb_output_id: splat_payload_id.clone(),
            splat_payload_id: Some(splat_payload_id),
            local_aabb: None,
            canonical_frame: None,
            updated_at_unix_ms: now_unix_ms(),
        };

        if let Some(position) = self.index.meshes.iter().position(|entry| {
            entry.source_image_path == source_image_path
                && entry.asset_kind == CachedAssetKind::GaussianSplat
        }) {
            let old_cache_key = self.index.meshes[position].cache_key.clone();
            if old_cache_key != cache_key {
                self.remove_mesh_payload(&old_cache_key)?;
                self.remove_gltf_output(&old_cache_key)?;
                self.remove_glb_output(&old_cache_key)?;
                self.remove_splat_payload(&old_cache_key)?;
                self.remove_source_image_payload(&old_cache_key)?;
            }
            self.index.meshes[position] = metadata.clone();
        } else {
            self.index.meshes.push(metadata.clone());
        }

        self.save_index()?;
        Ok(metadata)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_scene_snapshot(
        &mut self,
        source_scene_path: &Path,
        source_image_bytes: Option<&[u8]>,
        label: impl Into<String>,
        pipeline: impl Into<String>,
        payload: &CachedScenePayload,
        artifact_dir: Option<String>,
        metrics: Option<CachedSceneMetrics>,
    ) -> CacheResult<CachedSceneMetadata> {
        let source_scene_path = normalize_source_image_path(source_scene_path);
        let label = label.into();
        let pipeline = pipeline.into();
        let scene_key = scene_cache_key(&source_scene_path, &label);

        let payload_json = serde_json::to_string_pretty(payload)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        self.write_scene_payload(&scene_key, &payload_json)?;
        let scene_payload_id = self.scene_payload_id(&scene_key);
        let source_image_payload =
            self.write_source_image_payload(&scene_key, &source_scene_path, source_image_bytes)?;

        let metadata = CachedSceneMetadata {
            scene_key: scene_key.clone(),
            label,
            source_scene_path: source_scene_path.clone(),
            source_image_payload_id: source_image_payload
                .as_ref()
                .map(|payload| payload.payload_id.clone()),
            source_image_name: source_image_payload
                .as_ref()
                .map(|payload| payload.file_name.clone()),
            source_image_mime: source_image_payload
                .as_ref()
                .and_then(|payload| payload.mime_type.clone()),
            scene_payload_id,
            pipeline,
            artifact_dir,
            metrics,
            updated_at_unix_ms: now_unix_ms(),
        };

        if let Some(position) = self
            .index
            .scenes
            .iter()
            .position(|entry| entry.scene_key == scene_key)
        {
            self.index.scenes[position] = metadata.clone();
        } else {
            self.index.scenes.push(metadata.clone());
        }
        self.save_index()?;
        Ok(metadata)
    }

    pub fn load_scene(&self, scene_key: &str) -> CacheResult<Option<CachedScenePayload>> {
        let Some(payload) = self.read_scene_payload(scene_key)? else {
            return Ok(None);
        };
        serde_json::from_str(&payload)
            .map(Some)
            .map_err(|err| CacheError::Serialization(err.to_string()))
    }

    pub fn rename_scene(&mut self, scene_key: &str, label: impl Into<String>) -> CacheResult<bool> {
        let Some(entry) = self
            .index
            .scenes
            .iter_mut()
            .find(|entry| entry.scene_key == scene_key)
        else {
            return Ok(false);
        };
        entry.label = label.into();
        entry.updated_at_unix_ms = now_unix_ms();
        self.save_index()?;
        Ok(true)
    }

    pub fn remove_scene_entry(&mut self, scene_key: &str) -> CacheResult<bool> {
        let Some(position) = self
            .index
            .scenes
            .iter()
            .position(|entry| entry.scene_key == scene_key)
        else {
            return Ok(false);
        };
        self.index.scenes.remove(position);
        self.remove_scene_payload(scene_key)?;
        self.remove_source_image_payload(scene_key)?;
        self.save_index()?;
        Ok(true)
    }

    pub fn remove_mesh_entry(&mut self, cache_key: &str) -> CacheResult<bool> {
        let Some(position) = self
            .index
            .meshes
            .iter()
            .position(|entry| entry.cache_key == cache_key)
        else {
            return Ok(false);
        };

        self.index.meshes.remove(position);
        self.remove_mesh_payload(cache_key)?;
        self.remove_gltf_output(cache_key)?;
        self.remove_glb_output(cache_key)?;
        self.remove_splat_payload(cache_key)?;
        self.remove_source_image_payload(cache_key)?;
        self.index
            .world_items
            .retain(|item| item.cache_key != cache_key);
        self.save_index()?;
        Ok(true)
    }

    pub fn set_world_items(&mut self, mut items: Vec<CachedWorldItem>) -> CacheResult<()> {
        items.sort_by(|left, right| left.cache_key.cmp(&right.cache_key));
        self.index.world_items = items;
        self.save_index()
    }

    pub fn set_camera_state(&mut self, camera: Option<CachedCameraState>) -> CacheResult<()> {
        self.index.camera = camera;
        self.save_index()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_index(&self) -> CacheResult<()> {
        ensure_native_layout(&self.root)?;
        let index_json = serde_json::to_string_pretty(&self.index)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        fs::write(self.index_path(), index_json).map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn save_index(&self) -> CacheResult<()> {
        let index_json = serde_json::to_string(&self.index)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        web_storage_set(&self.index_key(), &index_json)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_mesh_payload(&self, cache_key: &str) -> CacheResult<Option<String>> {
        let path = self.mesh_payload_path(cache_key);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(path).map_err(|err| CacheError::Io(err.to_string()))?;
        Ok(Some(content))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_mesh_payload(&self, cache_key: &str) -> CacheResult<Option<String>> {
        web_storage_get(&self.mesh_payload_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_splat_payload(&self, cache_key: &str) -> CacheResult<Option<String>> {
        let path = self.splat_payload_path(cache_key);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(path).map_err(|err| CacheError::Io(err.to_string()))?;
        Ok(Some(content))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_splat_payload(&self, cache_key: &str) -> CacheResult<Option<String>> {
        web_storage_get(&self.splat_payload_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_glb_output(&self, cache_key: &str) -> CacheResult<Option<Vec<u8>>> {
        let path = self.glb_output_path(cache_key);
        if !path.exists() {
            return Ok(None);
        }
        fs::read(path)
            .map(Some)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_source_image_payload(
        &self,
        metadata: &CachedMeshMetadata,
    ) -> CacheResult<Option<Vec<u8>>> {
        let Some(payload_id) = metadata.source_image_payload_id.as_ref() else {
            return Ok(None);
        };
        let path = Path::new(payload_id);
        if !path.exists() {
            return Ok(None);
        }
        fs::read(path)
            .map(Some)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_source_image_payload(
        &self,
        metadata: &CachedMeshMetadata,
    ) -> CacheResult<Option<Vec<u8>>> {
        let Some(payload_id) = metadata.source_image_payload_id.as_ref() else {
            return Ok(None);
        };
        let Some(encoded) = web_storage_get(payload_id)? else {
            return Ok(None);
        };
        BASE64_STANDARD
            .decode(encoded)
            .map(Some)
            .map_err(|err| CacheError::InvalidData(err.to_string()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_scene_source_image_payload(
        &self,
        metadata: &CachedSceneMetadata,
    ) -> CacheResult<Option<Vec<u8>>> {
        let Some(payload_id) = metadata.source_image_payload_id.as_ref() else {
            return Ok(None);
        };
        let path = Path::new(payload_id);
        if !path.exists() {
            return Ok(None);
        }
        fs::read(path)
            .map(Some)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_scene_source_image_payload(
        &self,
        metadata: &CachedSceneMetadata,
    ) -> CacheResult<Option<Vec<u8>>> {
        let Some(payload_id) = metadata.source_image_payload_id.as_ref() else {
            return Ok(None);
        };
        let Some(encoded) = web_storage_get(payload_id)? else {
            return Ok(None);
        };
        BASE64_STANDARD
            .decode(encoded)
            .map(Some)
            .map_err(|err| CacheError::InvalidData(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_glb_output(&self, cache_key: &str) -> CacheResult<Option<Vec<u8>>> {
        let Some(encoded) = web_storage_get(&self.glb_output_storage_key(cache_key))? else {
            return Ok(None);
        };
        let bytes = BASE64_STANDARD
            .decode(encoded)
            .map_err(|err| CacheError::InvalidData(err.to_string()))?;
        Ok(Some(bytes))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_mesh_payload(&self, cache_key: &str) -> CacheResult<()> {
        let path = self.mesh_payload_path(cache_key);
        if path.exists() {
            fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_mesh_payload(&self, cache_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.mesh_payload_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_splat_payload(&self, cache_key: &str, payload: &str) -> CacheResult<()> {
        fs::write(self.splat_payload_path(cache_key), payload)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn write_splat_payload(&self, cache_key: &str, payload: &str) -> CacheResult<()> {
        web_storage_set(&self.splat_payload_storage_key(cache_key), payload)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_splat_payload(&self, cache_key: &str) -> CacheResult<()> {
        let path = self.splat_payload_path(cache_key);
        if path.exists() {
            fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_source_image_payload(&self, cache_key: &str) -> CacheResult<()> {
        let dir = self.source_image_dir();
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir).map_err(|err| CacheError::Io(err.to_string()))? {
            let entry = entry.map_err(|err| CacheError::Io(err.to_string()))?;
            let path = entry.path();
            if path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem == cache_key)
                && path.is_file()
            {
                fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
            }
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_source_image_payload(&self, cache_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.source_image_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_scene_payload(&self, scene_key: &str) -> CacheResult<Option<String>> {
        let path = self.scene_payload_path(scene_key);
        if !path.exists() {
            return Ok(None);
        }
        fs::read_to_string(path)
            .map(Some)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_scene_payload(&self, scene_key: &str) -> CacheResult<Option<String>> {
        web_storage_get(&self.scene_payload_storage_key(scene_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_scene_payload(&self, scene_key: &str, payload: &str) -> CacheResult<()> {
        fs::write(self.scene_payload_path(scene_key), payload)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn write_scene_payload(&self, scene_key: &str, payload: &str) -> CacheResult<()> {
        web_storage_set(&self.scene_payload_storage_key(scene_key), payload)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_scene_payload(&self, scene_key: &str) -> CacheResult<()> {
        let path = self.scene_payload_path(scene_key);
        if path.exists() {
            fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_scene_payload(&self, scene_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.scene_payload_storage_key(scene_key))
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_splat_payload(&self, cache_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.splat_payload_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_gltf_output(&self, cache_key: &str) -> CacheResult<()> {
        let path = self.gltf_output_path(cache_key);
        if path.exists() {
            fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_gltf_output(&self, cache_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.gltf_output_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_glb_output(&self, cache_key: &str, glb: &[u8]) -> CacheResult<()> {
        fs::write(self.glb_output_path(cache_key), glb)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn write_glb_output(&self, cache_key: &str, glb: &[u8]) -> CacheResult<()> {
        let encoded = BASE64_STANDARD.encode(glb);
        web_storage_set(&self.glb_output_storage_key(cache_key), &encoded)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn remove_glb_output(&self, cache_key: &str) -> CacheResult<()> {
        let path = self.glb_output_path(cache_key);
        if path.exists() {
            fs::remove_file(path).map_err(|err| CacheError::Io(err.to_string()))?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_glb_output(&self, cache_key: &str) -> CacheResult<()> {
        web_storage_remove(&self.glb_output_storage_key(cache_key))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn index_path(&self) -> PathBuf {
        self.root.join(INDEX_FILE_NAME)
    }

    #[cfg(target_arch = "wasm32")]
    fn index_key(&self) -> String {
        format!("{}/{}", self.prefix, INDEX_FILE_NAME)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_payload_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(MESH_DIR_NAME)
            .join(format!("{cache_key}.mesh.json"))
    }

    #[cfg(target_arch = "wasm32")]
    fn mesh_payload_storage_key(&self, cache_key: &str) -> String {
        format!("{}/mesh/{cache_key}", self.prefix)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn splat_payload_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(MESH_DIR_NAME)
            .join(format!("{cache_key}.splat.json"))
    }

    #[cfg(target_arch = "wasm32")]
    fn splat_payload_storage_key(&self, cache_key: &str) -> String {
        format!("{}/splat/{cache_key}", self.prefix)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn gltf_output_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(MESH_DIR_NAME)
            .join(format!("{cache_key}.gltf"))
    }

    #[cfg(target_arch = "wasm32")]
    fn gltf_output_storage_key(&self, cache_key: &str) -> String {
        format!("{}/gltf/{cache_key}", self.prefix)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn glb_output_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(MESH_DIR_NAME)
            .join(format!("{cache_key}.glb"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn scene_payload_path(&self, scene_key: &str) -> PathBuf {
        self.root
            .join(SCENE_DIR_NAME)
            .join(format!("{scene_key}.scene.json"))
    }

    #[cfg(target_arch = "wasm32")]
    fn scene_payload_storage_key(&self, scene_key: &str) -> String {
        format!("{}/scene/{scene_key}", self.prefix)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn scene_payload_id(&self, scene_key: &str) -> String {
        self.scene_payload_path(scene_key)
            .to_string_lossy()
            .to_string()
    }

    #[cfg(target_arch = "wasm32")]
    fn scene_payload_id(&self, scene_key: &str) -> String {
        self.scene_payload_storage_key(scene_key)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn source_image_dir(&self) -> PathBuf {
        self.root.join(SOURCE_IMAGE_DIR_NAME)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn source_image_payload_path(&self, cache_key: &str, extension: &str) -> PathBuf {
        self.source_image_dir()
            .join(format!("{cache_key}.{extension}"))
    }

    #[cfg(target_arch = "wasm32")]
    fn source_image_storage_key(&self, cache_key: &str) -> String {
        format!("{}/source_image/{cache_key}", self.prefix)
    }

    fn write_source_image_payload(
        &self,
        cache_key: &str,
        source_image_path: &str,
        source_image_bytes: Option<&[u8]>,
    ) -> CacheResult<Option<SourceImagePayloadInfo>> {
        let bytes = if let Some(bytes) = source_image_bytes {
            if bytes.is_empty() {
                None
            } else {
                Some(bytes.to_vec())
            }
        } else {
            self.read_source_image_from_path(source_image_path)?
        };
        let Some(bytes) = bytes else {
            self.remove_source_image_payload(cache_key)?;
            return Ok(None);
        };
        let extension = source_image_extension(source_image_path);
        let file_name = source_image_file_name(source_image_path);
        let mime_type = source_image_mime(source_image_path);
        let payload_id = self.write_source_image_bytes(cache_key, extension, bytes.as_slice())?;
        Ok(Some(SourceImagePayloadInfo {
            payload_id,
            file_name,
            mime_type,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_source_image_from_path(&self, source_image_path: &str) -> CacheResult<Option<Vec<u8>>> {
        let path = Path::new(source_image_path);
        if !path.is_file() {
            return Ok(None);
        }
        fs::read(path)
            .map(Some)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn read_source_image_from_path(
        &self,
        _source_image_path: &str,
    ) -> CacheResult<Option<Vec<u8>>> {
        Ok(None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_source_image_bytes(
        &self,
        cache_key: &str,
        extension: &str,
        bytes: &[u8],
    ) -> CacheResult<String> {
        fs::create_dir_all(self.source_image_dir())
            .map_err(|err| CacheError::Io(err.to_string()))?;
        self.remove_source_image_payload(cache_key)?;
        let path = self.source_image_payload_path(cache_key, extension);
        fs::write(&path, bytes).map_err(|err| CacheError::Io(err.to_string()))?;
        Ok(path.to_string_lossy().to_string())
    }

    #[cfg(target_arch = "wasm32")]
    fn write_source_image_bytes(
        &self,
        cache_key: &str,
        _extension: &str,
        bytes: &[u8],
    ) -> CacheResult<String> {
        let key = self.source_image_storage_key(cache_key);
        let encoded = BASE64_STANDARD.encode(bytes);
        web_storage_set(&key, &encoded)?;
        Ok(key)
    }

    #[cfg(target_arch = "wasm32")]
    fn glb_output_storage_key(&self, cache_key: &str) -> String {
        format!("{}/glb/{cache_key}", self.prefix)
    }

    fn glb_output_id(&self, cache_key: &str) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.glb_output_path(cache_key)
                .to_string_lossy()
                .to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.glb_output_storage_key(cache_key)
        }
    }

    fn splat_payload_id(&self, cache_key: &str) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.splat_payload_path(cache_key)
                .to_string_lossy()
                .to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.splat_payload_storage_key(cache_key)
        }
    }
}

pub fn cache_key_from_image_path(path: &Path) -> String {
    let normalized = normalize_source_image_path(path);
    cache_key_from_source(&normalized)
}

fn cache_key_from_source_and_kind(source: &str, kind: CachedAssetKind) -> String {
    match kind {
        CachedAssetKind::Mesh => cache_key_from_source(source),
        CachedAssetKind::GaussianSplat => {
            cache_key_from_source(&format!("{source}#{TRIPOSPLAT_SPLAT_CACHE_NAMESPACE}"))
        }
    }
}

fn scene_cache_key(source: &str, label: &str) -> String {
    cache_key_from_source(&format!("{source}#{SCENE_CACHE_NAMESPACE}#{label}"))
}

fn asset_label(source_image_path: &str, kind: CachedAssetKind) -> String {
    let label = Path::new(source_image_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("image")
        .to_string();
    match kind {
        CachedAssetKind::Mesh => label,
        CachedAssetKind::GaussianSplat => format!("{label} (splat)"),
    }
}

#[derive(Clone, Debug)]
struct SourceImagePayloadInfo {
    payload_id: String,
    file_name: String,
    mime_type: Option<String>,
}

fn source_image_file_name(source_image_path: &str) -> String {
    Path::new(source_image_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("source_image")
        .to_string()
}

fn source_image_extension(source_image_path: &str) -> &str {
    match Path::new(source_image_path)
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "jpg",
        Some("png") => "png",
        Some("webp") => "webp",
        Some("bmp") => "bmp",
        Some("gif") => "gif",
        Some("tif" | "tiff") => "tiff",
        _ => "img",
    }
}

fn source_image_mime(source_image_path: &str) -> Option<String> {
    let mime = match source_image_extension(source_image_path) {
        "jpg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tiff" => "image/tiff",
        _ => return None,
    };
    Some(mime.to_string())
}

fn cache_key_from_source(source: &str) -> String {
    format!("{:016x}", fnv1a_hash(source.as_bytes()))
}

fn normalize_source_image_path(path: &Path) -> String {
    let source = path.to_string_lossy().replace('\\', "/");
    #[cfg(windows)]
    {
        source.to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        source
    }
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn now_unix_ms() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    #[cfg(target_arch = "wasm32")]
    {
        // wasm32-unknown-unknown std::time::SystemTime panics at runtime.
        // Use browser wall-clock time for cache metadata timestamps.
        js_sys::Date::now() as u64
    }
}

fn mesh_to_glb(mesh: &SynthMesh) -> CacheResult<Vec<u8>> {
    crate::io::mesh_to_glb_bytes(mesh).map_err(|err| CacheError::Serialization(err.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn default_native_cache_root() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".burn_synth").join("cache"))
        .unwrap_or_else(legacy_native_cache_root)
}

#[cfg(not(target_arch = "wasm32"))]
fn legacy_native_cache_root() -> PathBuf {
    PathBuf::from(".burn_synth_cache")
}

#[cfg(not(target_arch = "wasm32"))]
fn migrate_legacy_native_cache_root() -> CacheResult<()> {
    let legacy_root = legacy_native_cache_root();
    if !legacy_root.exists() {
        return Ok(());
    }
    let target_root = default_native_cache_root();
    if target_root == legacy_root || target_root.exists() {
        return Ok(());
    }
    if let Some(parent) = target_root.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            CacheError::Io(format!(
                "failed to create cache directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    fs::rename(&legacy_root, &target_root).map_err(|err| {
        CacheError::Io(format!(
            "failed to migrate legacy cache {} -> {}: {err}",
            legacy_root.display(),
            target_root.display()
        ))
    })?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn default_web_cache_prefix() -> String {
    "burn_synth/cache/v6".to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_native_layout(root: &Path) -> CacheResult<()> {
    fs::create_dir_all(root.join(MESH_DIR_NAME)).map_err(|err| CacheError::Io(err.to_string()))?;
    fs::create_dir_all(root.join(SCENE_DIR_NAME)).map_err(|err| CacheError::Io(err.to_string()))?;
    fs::create_dir_all(root.join(SOURCE_IMAGE_DIR_NAME))
        .map_err(|err| CacheError::Io(err.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_native_index(root: &Path) -> CacheResult<CacheIndex> {
    let path = root.join(INDEX_FILE_NAME);
    if !path.exists() {
        return Ok(CacheIndex::default());
    }
    let content = fs::read_to_string(path).map_err(|err| CacheError::Io(err.to_string()))?;
    serde_json::from_str(&content).map_err(|err| CacheError::Serialization(err.to_string()))
}

#[cfg(target_arch = "wasm32")]
fn read_web_index(prefix: &str) -> CacheResult<CacheIndex> {
    let index_key = format!("{prefix}/{INDEX_FILE_NAME}");
    let content = web_storage_get(&index_key)?;
    let Some(content) = content else {
        return Ok(CacheIndex::default());
    };
    serde_json::from_str(&content).map_err(|err| CacheError::Serialization(err.to_string()))
}

#[cfg(target_arch = "wasm32")]
fn web_storage() -> CacheResult<web_sys::Storage> {
    let window = web_sys::window().ok_or_else(|| {
        CacheError::Storage("window is unavailable; cannot access localStorage".to_string())
    })?;
    let storage = window
        .local_storage()
        .map_err(|err| CacheError::Storage(format!("localStorage access failed: {err:?}")))?
        .ok_or_else(|| CacheError::Storage("localStorage is unavailable".to_string()))?;
    Ok(storage)
}

#[cfg(target_arch = "wasm32")]
fn web_storage_get(key: &str) -> CacheResult<Option<String>> {
    web_storage()?
        .get_item(key)
        .map_err(|err| CacheError::Storage(format!("failed reading key '{key}': {err:?}")))
}

#[cfg(target_arch = "wasm32")]
fn web_storage_set(key: &str, value: &str) -> CacheResult<()> {
    web_storage()?
        .set_item(key, value)
        .map_err(|err| CacheError::Storage(format!("failed writing key '{key}': {err:?}")))
}

#[cfg(target_arch = "wasm32")]
fn web_storage_remove(key: &str) -> CacheResult<()> {
    web_storage()?
        .remove_item(key)
        .map_err(|err| CacheError::Storage(format!("failed removing key '{key}': {err:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn dummy_mesh(scale: f32) -> SynthMesh {
        SynthMesh {
            mesh: crate::TripoMesh {
                vertices: vec![
                    [0.0 * scale, 0.0 * scale, 0.0 * scale],
                    [1.0 * scale, 0.0 * scale, 0.0 * scale],
                    [0.0 * scale, 1.0 * scale, 0.0 * scale],
                ],
                faces: vec![[0, 1, 2]],
            },
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            normals: Vec::new(),
            material: Some(SynthMeshMaterial {
                base_color: [0.5, 0.7, 0.9],
                metallic: 0.15,
                roughness: 0.6,
                alpha: 0.92,
            }),
            pbr_textures: Some(crate::SynthMeshPbrTextures {
                base_color: crate::SynthMeshTexture {
                    width: 2,
                    height: 2,
                    rgba8: vec![
                        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                    ],
                },
                metallic_roughness: crate::SynthMeshTexture {
                    width: 2,
                    height: 2,
                    rgba8: vec![
                        0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255,
                    ],
                },
                normal: None,
                emissive: None,
                occlusion: None,
            }),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn temp_root(name: &str) -> PathBuf {
        let unique = format!(
            "burn_synth_cache_test_{name}_{}_{}",
            std::process::id(),
            now_unix_ms()
        );
        std::env::temp_dir().join(unique)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn upsert_mesh_updates_existing_entry_for_same_image() {
        let root = temp_root("upsert");
        let image = PathBuf::from("C:/data/input/plant.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let first = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert first");
        let second = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(2.0))
            .expect("update existing");

        assert_eq!(cache.mesh_entries().len(), 1);
        assert_eq!(first.cache_key, second.cache_key);
        assert_eq!(
            second.local_aabb,
            Some(CachedAssetAabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 2.0, 0.0],
            })
        );

        let loaded = cache
            .load_mesh(&second.cache_key)
            .expect("read mesh")
            .expect("mesh exists");
        assert_eq!(loaded.mesh.vertices[1], [2.0, 0.0, 0.0]);
        assert!(loaded.material.is_some());

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn upsert_mesh_stores_glb_without_redundant_mesh_payload_json() {
        let root = temp_root("glb_only");
        let image = PathBuf::from("C:/data/input/chair.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");

        let payload_path = cache.mesh_payload_path(&entry.cache_key);
        assert!(
            !payload_path.exists(),
            "legacy mesh payload JSON should not be written"
        );
        let glb_path = PathBuf::from(&entry.glb_output_id);
        assert!(glb_path.exists(), "GLB cache artifact should exist");

        let loaded = cache
            .load_mesh(&entry.cache_key)
            .expect("read mesh")
            .expect("mesh exists");
        assert_eq!(loaded.mesh.vertices.len(), 3);
        assert_eq!(loaded.mesh.faces.len(), 1);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn upsert_mesh_retains_source_image_payload() {
        let root = temp_root("source_image");
        let image = PathBuf::from("C:/data/input/chair.png");
        let source_bytes = [137, 80, 78, 71, 13, 10, 26, 10];

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image_with_source_bytes(&image, Some(&source_bytes), &dummy_mesh(1.0))
            .expect("insert mesh with source image");

        assert!(entry.source_image_payload_id.is_some());
        assert_eq!(entry.source_image_name.as_deref(), Some("chair.png"));
        assert_eq!(entry.source_image_mime.as_deref(), Some("image/png"));

        let loaded = cache
            .load_source_image(&entry)
            .expect("load source image")
            .expect("source image exists");
        assert_eq!(loaded.file_name, "chair.png");
        assert_eq!(loaded.mime_type.as_deref(), Some("image/png"));
        assert_eq!(loaded.bytes, source_bytes);

        let cache = MeshCache::load_from_root(root.clone()).expect("reload cache");
        let reloaded = cache
            .asset_entries()
            .iter()
            .find(|metadata| metadata.cache_key == entry.cache_key)
            .expect("reloaded metadata");
        let loaded = cache
            .load_source_image(reloaded)
            .expect("load source image after reload")
            .expect("source image exists after reload");
        assert_eq!(loaded.bytes, source_bytes);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn remove_mesh_entry_cleans_source_image_payload() {
        let root = temp_root("remove_source_image");
        let image = PathBuf::from("C:/data/input/remove_source.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image_with_source_bytes(&image, Some(&[1, 2, 3, 4]), &dummy_mesh(1.0))
            .expect("insert mesh with source image");
        let payload = entry
            .source_image_payload_id
            .as_ref()
            .map(PathBuf::from)
            .expect("source image payload path");
        assert!(payload.exists());

        let removed = cache
            .remove_mesh_entry(&entry.cache_key)
            .expect("remove mesh entry");
        assert!(removed);
        assert!(!payload.exists());

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn mesh_and_splat_entries_coexist_for_same_image() {
        let root = temp_root("mesh_splat");
        let image = PathBuf::from("C:/data/input/object.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let mesh_entry = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");
        let splat_entry = cache
            .upsert_gaussian_splat_for_image(&image, &GaussianSplatCloud::canonical_debug_cloud())
            .expect("insert splat");

        assert_eq!(cache.asset_entries().len(), 2);
        assert_ne!(mesh_entry.cache_key, splat_entry.cache_key);
        assert_eq!(mesh_entry.asset_kind, CachedAssetKind::Mesh);
        assert_eq!(splat_entry.asset_kind, CachedAssetKind::GaussianSplat);

        let mesh = cache
            .load_asset(&mesh_entry.cache_key)
            .expect("load mesh asset")
            .expect("mesh asset");
        assert!(matches!(mesh, SynthAsset::Mesh(_)));
        let splats = cache
            .load_asset(&splat_entry.cache_key)
            .expect("load splat asset")
            .expect("splat asset");
        match splats {
            SynthAsset::GaussianSplat(cloud) => assert_eq!(cloud.len(), 2),
            SynthAsset::Mesh(_) => panic!("expected splat asset"),
        }

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[test]
    fn gaussian_splat_cache_key_is_version_namespaced() {
        let source = "C:/data/input/object.png";

        let mesh_key = cache_key_from_source_and_kind(source, CachedAssetKind::Mesh);
        let splat_key = cache_key_from_source_and_kind(source, CachedAssetKind::GaussianSplat);

        assert_eq!(mesh_key, cache_key_from_source(source));
        assert_eq!(
            splat_key,
            cache_key_from_source(&format!("{source}#{TRIPOSPLAT_SPLAT_CACHE_NAMESPACE}"))
        );
        assert_ne!(mesh_key, splat_key);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn world_items_round_trip_in_index() {
        let root = temp_root("world");
        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");

        cache
            .set_world_items(vec![CachedWorldItem {
                cache_key: "abcd".to_string(),
                translation: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }])
            .expect("write world items");

        let cache = MeshCache::load_from_root(root.clone()).expect("reload cache");
        assert_eq!(cache.world_items().len(), 1);
        assert_eq!(cache.world_items()[0].cache_key, "abcd");
        assert_eq!(cache.world_items()[0].translation, [1.0, 2.0, 3.0]);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn camera_state_round_trip_in_index() {
        let root = temp_root("camera");
        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let state = CachedCameraState {
            translation: [3.0, 4.0, 5.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            focus: [0.5, 0.25, -0.75],
            yaw: 1.0,
            pitch: -0.25,
            radius: 6.5,
            vertical_fov_degrees: Some(72.0),
        };
        cache
            .set_camera_state(Some(state.clone()))
            .expect("write camera state");

        let cache = MeshCache::load_from_root(root.clone()).expect("reload cache");
        assert_eq!(cache.camera_state(), Some(&state));

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn scene_snapshot_round_trips_with_source_image_and_metrics() {
        let root = temp_root("scene_snapshot");
        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let source = PathBuf::from("C:/data/scenes/room.png");
        let source_bytes = [137, 80, 78, 71, 13, 10, 26, 10];
        let payload = CachedScenePayload {
            world_items: vec![CachedWorldItem {
                cache_key: "chair".to_string(),
                translation: [1.0, 0.0, 2.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }],
            camera: Some(CachedCameraState {
                translation: [0.0, 2.0, 5.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                focus: [0.0, 0.0, 0.0],
                yaw: 0.0,
                pitch: -0.2,
                radius: 5.0,
                vertical_fov_degrees: Some(70.0),
            }),
            bsn: Some("synth_scene_v1 {}".to_string()),
            asset_bindings: Some(serde_json::json!([])),
            e2e_summary: Some(serde_json::json!({ "ok": true })),
            response_summary: None,
        };
        let metrics = CachedSceneMetrics {
            ok: Some(true),
            elapsed_ms: Some(1234),
            object_count: Some(1),
            asset_count: Some(1),
            placement_count: Some(1),
            feedback_accepted: Some(true),
            feedback_iteration: Some(2),
            failed_stage: None,
        };

        let metadata = cache
            .upsert_scene_snapshot(
                &source,
                Some(&source_bytes),
                "demo scene",
                "explicit",
                &payload,
                Some("tmp/runs/demo".to_string()),
                Some(metrics.clone()),
            )
            .expect("write scene");

        assert_eq!(cache.scene_entries().len(), 1);
        assert_eq!(cache.scene_entries()[0].pipeline, "explicit");
        assert_eq!(cache.scene_entries()[0].metrics, Some(metrics));
        let loaded = cache
            .load_scene(&metadata.scene_key)
            .expect("load scene")
            .expect("scene exists");
        assert_eq!(loaded, payload);
        let image = cache
            .load_scene_source_image(&metadata)
            .expect("load scene source")
            .expect("source exists");
        assert_eq!(image.file_name, "room.png");
        assert_eq!(image.bytes, source_bytes);

        let cache = MeshCache::load_from_root(root.clone()).expect("reload cache");
        assert_eq!(cache.scene_entries().len(), 1);
        let reloaded = cache
            .load_scene(&metadata.scene_key)
            .expect("load scene after reload")
            .expect("scene exists after reload");
        assert_eq!(reloaded.world_items.len(), 1);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn scene_rename_and_delete_do_not_delete_object_assets() {
        let root = temp_root("scene_delete");
        let image = PathBuf::from("C:/data/input/chair.png");
        let source = PathBuf::from("C:/data/scenes/room.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let asset = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");
        let payload = CachedScenePayload {
            world_items: vec![CachedWorldItem {
                cache_key: asset.cache_key.clone(),
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            }],
            ..Default::default()
        };
        let scene = cache
            .upsert_scene_snapshot(&source, None, "before", "explicit", &payload, None, None)
            .expect("write scene");

        assert!(
            cache
                .rename_scene(&scene.scene_key, "after")
                .expect("rename scene")
        );
        assert_eq!(cache.scene_entries()[0].label, "after");
        assert!(
            cache
                .remove_scene_entry(&scene.scene_key)
                .expect("remove scene")
        );
        assert!(cache.scene_entries().is_empty());
        assert!(
            cache
                .load_asset(&asset.cache_key)
                .expect("load object asset")
                .is_some(),
            "scene deletion must not remove reusable object assets"
        );

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn remove_mesh_entry_cleans_payload_and_world_refs() {
        let root = temp_root("remove");
        let image = PathBuf::from("C:/data/input/delete_me.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");
        cache
            .set_world_items(vec![
                CachedWorldItem {
                    cache_key: entry.cache_key.clone(),
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                CachedWorldItem {
                    cache_key: "keep".to_string(),
                    translation: [1.0, 2.0, 3.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
            ])
            .expect("write world items");

        let removed = cache
            .remove_mesh_entry(&entry.cache_key)
            .expect("remove mesh entry");
        assert!(removed);
        assert!(cache.mesh_entries().is_empty());
        assert!(
            cache
                .world_items()
                .iter()
                .all(|item| item.cache_key != entry.cache_key)
        );
        let maybe_mesh = cache
            .load_mesh(&entry.cache_key)
            .expect("load mesh after removal");
        assert!(maybe_mesh.is_none());
        assert!(!PathBuf::from(&entry.glb_output_id).exists());

        let removed_again = cache
            .remove_mesh_entry(&entry.cache_key)
            .expect("remove mesh entry second time");
        assert!(!removed_again);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn glb_output_is_written_with_metadata_reference() {
        let root = temp_root("glb");
        let image = PathBuf::from("C:/data/input/tree.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");

        assert!(entry.gltf_output_id.is_none());
        let glb_path = PathBuf::from(&entry.glb_output_id);
        assert!(glb_path.exists());
        let bytes = fs::read(glb_path).expect("read glb");
        assert!(bytes.starts_with(&[0x67, 0x6C, 0x54, 0x46]));
        let parsed = gltf::Glb::from_slice(&bytes).expect("parse glb");
        assert!(!parsed.json.is_empty());
        assert!(parsed.bin.is_some());

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[test]
    fn cache_key_is_stable_for_same_path() {
        let key_a = cache_key_from_image_path(Path::new("assets/image/example.png"));
        let key_b = cache_key_from_image_path(Path::new("assets/image/example.png"));
        assert_eq!(key_a, key_b);
    }
}
