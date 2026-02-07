use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};

use burn_tripo::pipeline::mesh::Mesh as TripoMesh;

const CACHE_VERSION: u32 = 1;
const INDEX_FILE_NAME: &str = "index.json";
#[cfg(not(target_arch = "wasm32"))]
const MESH_DIR_NAME: &str = "meshes";

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CachedMeshMetadata {
    pub cache_key: String,
    pub source_image_path: String,
    pub label: String,
    pub mesh_payload_id: String,
    pub gltf_output_id: String,
    pub updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedWorldItem {
    pub cache_key: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheIndex {
    version: u32,
    #[serde(default)]
    meshes: Vec<CachedMeshMetadata>,
    #[serde(default)]
    world_items: Vec<CachedWorldItem>,
}

impl Default for CacheIndex {
    fn default() -> Self {
        Self {
            version: CACHE_VERSION,
            meshes: Vec::new(),
            world_items: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeshPayload {
    vertices: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
}

impl From<&TripoMesh> for MeshPayload {
    fn from(mesh: &TripoMesh) -> Self {
        Self {
            vertices: mesh.vertices.clone(),
            faces: mesh.faces.clone(),
        }
    }
}

impl From<MeshPayload> for TripoMesh {
    fn from(payload: MeshPayload) -> Self {
        Self {
            vertices: payload.vertices,
            faces: payload.faces,
        }
    }
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

    pub fn world_items(&self) -> &[CachedWorldItem] {
        &self.index.world_items
    }

    pub fn load_mesh(&self, cache_key: &str) -> CacheResult<Option<TripoMesh>> {
        let mesh_payload = self.read_mesh_payload(cache_key)?;
        let Some(mesh_payload) = mesh_payload else {
            return Ok(None);
        };
        let payload: MeshPayload = serde_json::from_str(&mesh_payload)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        Ok(Some(payload.into()))
    }

    pub fn upsert_mesh_for_image(
        &mut self,
        source_image_path: &Path,
        mesh: &TripoMesh,
    ) -> CacheResult<CachedMeshMetadata> {
        let source_image_path = normalize_source_image_path(source_image_path);
        let cache_key = cache_key_from_source(&source_image_path);
        let label = Path::new(&source_image_path)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("image")
            .to_string();

        let payload = MeshPayload::from(mesh);
        let payload_json = serde_json::to_string(&payload)
            .map_err(|err| CacheError::Serialization(err.to_string()))?;
        self.write_mesh_payload(&cache_key, &payload_json)?;

        let gltf_text = mesh_to_embedded_gltf(mesh)?;
        self.write_gltf_output(&cache_key, &gltf_text)?;

        let metadata = CachedMeshMetadata {
            cache_key: cache_key.clone(),
            source_image_path: source_image_path.clone(),
            label,
            mesh_payload_id: self.mesh_payload_id(&cache_key),
            gltf_output_id: self.gltf_output_id(&cache_key),
            updated_at_unix_ms: now_unix_ms(),
        };

        if let Some(position) = self
            .index
            .meshes
            .iter()
            .position(|entry| entry.source_image_path == source_image_path)
        {
            let old_cache_key = self.index.meshes[position].cache_key.clone();
            if old_cache_key != cache_key {
                self.remove_mesh_payload(&old_cache_key)?;
                self.remove_gltf_output(&old_cache_key)?;
            }
            self.index.meshes[position] = metadata.clone();
        } else {
            self.index.meshes.push(metadata.clone());
        }

        self.save_index()?;
        Ok(metadata)
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
    fn write_mesh_payload(&self, cache_key: &str, payload_json: &str) -> CacheResult<()> {
        fs::write(self.mesh_payload_path(cache_key), payload_json)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn write_mesh_payload(&self, cache_key: &str, payload_json: &str) -> CacheResult<()> {
        web_storage_set(&self.mesh_payload_storage_key(cache_key), payload_json)
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
    fn write_gltf_output(&self, cache_key: &str, gltf_text: &str) -> CacheResult<()> {
        fs::write(self.gltf_output_path(cache_key), gltf_text)
            .map_err(|err| CacheError::Io(err.to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    fn write_gltf_output(&self, cache_key: &str, gltf_text: &str) -> CacheResult<()> {
        web_storage_set(&self.gltf_output_storage_key(cache_key), gltf_text)
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

    fn mesh_payload_id(&self, cache_key: &str) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.mesh_payload_path(cache_key)
                .to_string_lossy()
                .to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.mesh_payload_storage_key(cache_key)
        }
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

    fn gltf_output_id(&self, cache_key: &str) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.gltf_output_path(cache_key)
                .to_string_lossy()
                .to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.gltf_output_storage_key(cache_key)
        }
    }
}

pub fn cache_key_from_image_path(path: &Path) -> String {
    let normalized = normalize_source_image_path(path);
    cache_key_from_source(&normalized)
}

fn cache_key_from_source(source: &str) -> String {
    format!("{:016x}", fnv1a_hash(source.as_bytes()))
}

fn normalize_source_image_path(path: &Path) -> String {
    let source = path.to_string_lossy().replace('\\', "/");
    #[cfg(windows)]
    {
        return source.to_ascii_lowercase();
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
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn mesh_to_embedded_gltf(mesh: &TripoMesh) -> CacheResult<String> {
    if mesh.vertices.is_empty() {
        return Err(CacheError::InvalidData(
            "cannot write glTF for mesh with no vertices".to_string(),
        ));
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }

    let mut buffer = Vec::with_capacity(mesh.vertices.len() * 12 + mesh.faces.len() * 12);
    for vertex in &mesh.vertices {
        for component in vertex {
            buffer.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_byte_length = buffer.len();
    for face in &mesh.faces {
        for index in face {
            buffer.extend_from_slice(&index.to_le_bytes());
        }
    }
    let indices_byte_length = buffer.len() - positions_byte_length;

    let uri = format!(
        "data:application/octet-stream;base64,{}",
        BASE64_STANDARD.encode(buffer)
    );

    let gltf = serde_json::json!({
        "asset": {
            "version": "2.0",
            "generator": "burn_synth"
        },
        "scene": 0,
        "scenes": [
            { "nodes": [0] }
        ],
        "nodes": [
            { "mesh": 0 }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0
                        },
                        "indices": 1,
                        "mode": 4
                    }
                ]
            }
        ],
        "buffers": [
            {
                "byteLength": positions_byte_length + indices_byte_length,
                "uri": uri
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": positions_byte_length,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": positions_byte_length,
                "byteLength": indices_byte_length,
                "target": 34963
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": mesh.vertices.len(),
                "type": "VEC3",
                "min": min,
                "max": max
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": mesh.faces.len() * 3,
                "type": "SCALAR"
            }
        ]
    });

    serde_json::to_string_pretty(&gltf).map_err(|err| CacheError::Serialization(err.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn default_native_cache_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("BURN_SYNTH_CACHE_DIR") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("burn_synth")
            .join("mesh_cache");
    }
    PathBuf::from(".burn_synth_cache")
}

#[cfg(target_arch = "wasm32")]
fn default_web_cache_prefix() -> String {
    if let Ok(explicit) = std::env::var("BURN_SYNTH_CACHE_PREFIX") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(prefix) = option_env!("BURN_SYNTH_CACHE_PREFIX") {
        let trimmed = prefix.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "burn_synth/cache/v1".to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_native_layout(root: &Path) -> CacheResult<()> {
    fs::create_dir_all(root.join(MESH_DIR_NAME)).map_err(|err| CacheError::Io(err.to_string()))
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
    fn dummy_mesh(scale: f32) -> TripoMesh {
        TripoMesh {
            vertices: vec![
                [0.0 * scale, 0.0 * scale, 0.0 * scale],
                [1.0 * scale, 0.0 * scale, 0.0 * scale],
                [0.0 * scale, 1.0 * scale, 0.0 * scale],
            ],
            faces: vec![[0, 1, 2]],
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

        let loaded = cache
            .load_mesh(&second.cache_key)
            .expect("read mesh")
            .expect("mesh exists");
        assert_eq!(loaded.vertices[1], [2.0, 0.0, 0.0]);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
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
        assert!(!PathBuf::from(&entry.gltf_output_id).exists());

        let removed_again = cache
            .remove_mesh_entry(&entry.cache_key)
            .expect("remove mesh entry second time");
        assert!(!removed_again);

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gltf_output_is_written_with_metadata_reference() {
        let root = temp_root("gltf");
        let image = PathBuf::from("C:/data/input/tree.png");

        let mut cache = MeshCache::load_from_root(root.clone()).expect("create cache");
        let entry = cache
            .upsert_mesh_for_image(&image, &dummy_mesh(1.0))
            .expect("insert mesh");

        let gltf_path = PathBuf::from(&entry.gltf_output_id);
        assert!(gltf_path.exists());
        let gltf = fs::read_to_string(gltf_path).expect("read gltf");
        assert!(gltf.contains("\"asset\""));
        assert!(gltf.contains("\"meshes\""));

        fs::remove_dir_all(root).expect("cleanup temp cache root");
    }

    #[test]
    fn cache_key_is_stable_for_same_path() {
        let key_a = cache_key_from_image_path(Path::new("assets/image/example.png"));
        let key_b = cache_key_from_image_path(Path::new("assets/image/example.png"));
        assert_eq!(key_a, key_b);
    }
}
