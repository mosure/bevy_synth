use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use burn::prelude::Backend;
use burn::tensor::{BoolStore, Bytes, DType, Shape, TensorData};
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, TensorSnapshot,
    TensorSnapshotError,
};
use serde::Deserialize;
use serde_json::Value;

use super::weight_parts::burnpack_parts_manifest_path;
use crate::blob_burnpack::load_blob_bytes_from_burnpack_bytes;
use crate::virtual_fs;

const BURNPACK_PART_ALIGNMENT: u64 = 256;

#[derive(Debug, Deserialize)]
struct BurnpackPartEntry {
    path: String,
    #[serde(default)]
    bytes: u64,
}

#[derive(Debug, Deserialize)]
struct BurnpackPartsManifest {
    #[serde(default)]
    total_bytes: u64,
    #[serde(default)]
    parts: Vec<BurnpackPartEntry>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Clone)]
enum PartSource {
    Path(PathBuf),
    Url(String),
}

#[derive(Debug, Clone)]
struct PartDescriptor {
    source: PartSource,
    declared_file_bytes: u64,
    payload_bytes: Option<u64>,
}

#[derive(Debug, Default)]
struct PartPayloadCache {
    order: VecDeque<usize>,
    entries: HashMap<usize, Arc<Vec<u8>>>,
}

impl PartPayloadCache {
    // TRELLIS web burnpacks currently shard as high as 77 parts. Keep one full
    // component resident during lazy snapshot application to avoid synchronous
    // browser re-fetch loops, while still bounding multi-component cache growth.
    const MAX_ENTRIES: usize = 96;

    fn get(&mut self, index: usize) -> Option<Arc<Vec<u8>>> {
        let payload = self.entries.get(&index)?.clone();
        if let Some(pos) = self.order.iter().position(|value| *value == index) {
            self.order.remove(pos);
        }
        self.order.push_back(index);
        Some(payload)
    }

    fn insert(&mut self, index: usize, payload: Arc<Vec<u8>>) {
        if self.entries.contains_key(&index) {
            self.entries.insert(index, payload);
            if let Some(pos) = self.order.iter().position(|value| *value == index) {
                self.order.remove(pos);
            }
            self.order.push_back(index);
            return;
        }

        while self.entries.len() >= Self::MAX_ENTRIES {
            let Some(evict) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&evict);
        }

        self.entries.insert(index, payload);
        self.order.push_back(index);
    }
}

#[derive(Debug)]
struct BlobPartsArchive {
    total_hint: Option<u64>,
    parts: Vec<PartDescriptor>,
    cache: PartPayloadCache,
}

impl BlobPartsArchive {
    fn from_parts_manifest(burnpack_path: &Path) -> Result<Self, String> {
        let manifest_path = burnpack_parts_manifest_path(burnpack_path);
        let manifest_bytes = virtual_fs::read(&manifest_path).map_err(|err| {
            format!(
                "failed to read burnpack parts manifest '{}': {err}",
                manifest_path.display()
            )
        })?;
        let manifest: BurnpackPartsManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|err| {
                format!(
                    "failed to parse burnpack parts manifest '{}': {err}",
                    manifest_path.display()
                )
            })?;
        if manifest.parts.is_empty() {
            return Err(format!(
                "burnpack parts manifest '{}' contains no parts",
                manifest_path.display()
            ));
        }

        let manifest_source_url = virtual_fs::source_url(&manifest_path);
        let mut parts = Vec::with_capacity(manifest.parts.len());
        for part in manifest.parts {
            let part_path = resolve_manifest_part_path(&manifest_path, part.path.as_str())?;
            let source = if virtual_fs::has_virtual_file(&part_path) || part_path.exists() {
                PartSource::Path(part_path)
            } else if let Some(url) = manifest_source_url.as_deref() {
                PartSource::Url(resolve_manifest_part_url(url, part.path.as_str()))
            } else if let Some(url) = virtual_fs::source_url(&part_path) {
                PartSource::Url(url)
            } else {
                return Err(format!(
                    "burnpack part '{}' is missing and no manifest URL is available",
                    part_path.display()
                ));
            };

            parts.push(PartDescriptor {
                source,
                declared_file_bytes: part.bytes,
                payload_bytes: None,
            });
        }

        Ok(Self {
            total_hint: (manifest.total_bytes > 0).then_some(manifest.total_bytes),
            parts,
            cache: PartPayloadCache::default(),
        })
    }

    fn total_hint(&self) -> Option<u64> {
        self.total_hint
    }

    fn payload_len_for_part(&mut self, index: usize) -> Result<u64, String> {
        if let Some(len) = self
            .parts
            .get(index)
            .and_then(|descriptor| descriptor.payload_bytes)
        {
            return Ok(len);
        }

        let payload = self.load_part_payload(index)?;
        Ok(payload.len() as u64)
    }

    fn load_part_payload(&mut self, index: usize) -> Result<Arc<Vec<u8>>, String> {
        if let Some(payload) = self.cache.get(index) {
            return Ok(payload);
        }

        let Some(descriptor) = self.parts.get(index).cloned() else {
            return Err(format!("invalid burnpack part index {index}"));
        };

        let burnpack_bytes = match descriptor.source {
            PartSource::Path(path) => virtual_fs::read(&path).map_err(|err| {
                format!("failed to read burnpack part '{}': {err}", path.display())
            })?,
            PartSource::Url(url) => {
                return Err(format!(
                    "burnpack part URL '{url}' was not preloaded; async wasm loaders must download and register shard bytes before materialization"
                ));
            }
        };

        if descriptor.declared_file_bytes > 0 {
            let actual_file_bytes = burnpack_bytes.len() as u64;
            let matches_declared = actual_file_bytes == descriptor.declared_file_bytes;
            let matches_padded = actual_file_bytes >= descriptor.declared_file_bytes
                && actual_file_bytes.saturating_sub(descriptor.declared_file_bytes)
                    <= BURNPACK_PART_ALIGNMENT;
            if !matches_declared && !matches_padded {
                return Err(format!(
                    "burnpack part {} file-size mismatch: manifest={} actual={}",
                    index, descriptor.declared_file_bytes, actual_file_bytes
                ));
            }
        }

        let payload = load_blob_bytes_from_burnpack_bytes(&burnpack_bytes).map_err(|err| {
            format!(
                "failed to decode burnpack part {} into blob payload bytes: {err}",
                index
            )
        })?;

        let payload = Arc::new(payload);
        if let Some(target) = self.parts.get_mut(index) {
            target.payload_bytes = Some(payload.len() as u64);
        }
        self.cache.insert(index, payload.clone());
        Ok(payload)
    }

    fn read_range(&mut self, start: u64, end: u64) -> Result<Vec<u8>, String> {
        if end < start {
            return Err(format!(
                "invalid blob range [{start}, {end}) where end < start"
            ));
        }
        if end == start {
            return Ok(Vec::new());
        }

        if let Some(total) = self.total_hint
            && end > total
        {
            return Err(format!(
                "requested blob range [{start}, {end}) exceeds manifest total bytes {total}"
            ));
        }

        let requested = end
            .checked_sub(start)
            .ok_or_else(|| format!("blob range overflow: start={start} end={end}"))?;
        let requested_usize = usize::try_from(requested)
            .map_err(|_| format!("blob range too large for host allocation: {requested} bytes"))?;
        let mut out = Vec::with_capacity(requested_usize);

        let mut cursor = 0u64;
        for index in 0..self.parts.len() {
            let part_len = self.payload_len_for_part(index)?;
            let part_start = cursor;
            let part_end = cursor
                .checked_add(part_len)
                .ok_or_else(|| format!("blob part offset overflow at index {index}"))?;

            if part_end <= start {
                cursor = part_end;
                continue;
            }
            if part_start >= end {
                break;
            }

            let payload = self.load_part_payload(index)?;
            let copy_start = start.max(part_start);
            let copy_end = end.min(part_end);
            let local_start = usize::try_from(copy_start - part_start)
                .map_err(|_| format!("range local-start overflow for part {index}"))?;
            let local_end = usize::try_from(copy_end - part_start)
                .map_err(|_| format!("range local-end overflow for part {index}"))?;
            if local_end > payload.len() || local_start > local_end {
                return Err(format!(
                    "invalid local range [{local_start}, {local_end}) for part {index} with payload len {}",
                    payload.len()
                ));
            }
            out.extend_from_slice(&payload[local_start..local_end]);
            if copy_end >= end {
                break;
            }
            cursor = part_end;
        }

        if out.len() != requested_usize {
            return Err(format!(
                "incomplete blob read [{start}, {end}); expected {} bytes, assembled {} bytes",
                requested_usize,
                out.len()
            ));
        }

        Ok(out)
    }
}

#[derive(Clone)]
struct SharedBlobPartsArchive {
    inner: Rc<RefCell<BlobPartsArchive>>,
}

impl SharedBlobPartsArchive {
    fn from_parts_manifest(burnpack_path: &Path) -> Result<Self, String> {
        Ok(Self {
            inner: Rc::new(RefCell::new(BlobPartsArchive::from_parts_manifest(
                burnpack_path,
            )?)),
        })
    }

    fn total_hint(&self) -> Option<u64> {
        self.inner.borrow().total_hint()
    }

    fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, String> {
        self.inner.borrow_mut().read_range(start, end)
    }
}

pub(crate) fn chunked_blob_parts_manifest_exists(path: &Path) -> bool {
    virtual_fs::exists(&burnpack_parts_manifest_path(path))
}

pub(crate) struct ChunkedBlobSafetensorsStore {
    snapshots: BTreeMap<String, TensorSnapshot>,
    from_adapter: Box<dyn ModuleAdapter>,
    validate: bool,
    allow_partial: bool,
}

impl ChunkedBlobSafetensorsStore {
    pub(crate) fn from_blob_burnpack_parts(
        burnpack_path: &Path,
        key_remap_rules: &[(&str, &str)],
    ) -> Result<Self, String> {
        let archive = SharedBlobPartsArchive::from_parts_manifest(burnpack_path)?;
        let (header_len, header_json) = read_safetensors_header(&archive)?;
        let mut snapshots = build_snapshots_from_header(&archive, header_len, &header_json)?;

        if !key_remap_rules.is_empty() {
            let mut remapper = KeyRemapper::new();
            for &(from, to) in key_remap_rules {
                remapper = remapper
                    .add_pattern(from, to)
                    .map_err(|err| format!("invalid sparse flow remap rule {from}->{to}: {err}"))?;
            }
            let (remapped, _) = remapper.remap(snapshots);
            snapshots = remapped;
        }

        let snapshots = snapshots
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot))
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            snapshots,
            from_adapter: Box::new(PyTorchToBurnAdapter),
            validate: true,
            allow_partial: false,
        })
    }
}

impl ModuleStore for ChunkedBlobSafetensorsStore {
    type Error = String;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        _module: &M,
    ) -> Result<(), Self::Error> {
        Err("chunked blob safetensors store does not support save/collect".to_string())
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<burn_store::ApplyResult, Self::Error> {
        let snapshots = self.snapshots.values().cloned().collect::<Vec<_>>();
        let result = module.apply(snapshots, None, Some(self.from_adapter.clone()), false);

        if self.validate && !result.errors.is_empty() {
            return Err(format!("Import errors: {:?}", result.errors));
        }
        if !self.allow_partial && !result.missing.is_empty() {
            return Err(format!("\n{result}"));
        }

        Ok(result)
    }

    fn get_snapshot(&mut self, name: &str) -> Result<Option<&TensorSnapshot>, Self::Error> {
        Ok(self.snapshots.get(name))
    }

    fn get_all_snapshots(&mut self) -> Result<&BTreeMap<String, TensorSnapshot>, Self::Error> {
        Ok(&self.snapshots)
    }

    fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
        Ok(self.snapshots.keys().cloned().collect())
    }
}

fn read_safetensors_header(archive: &SharedBlobPartsArchive) -> Result<(u64, Value), String> {
    let prefix = archive.read_range(0, 8)?;
    if prefix.len() != 8 {
        return Err(format!(
            "invalid safetensors header prefix size {}; expected 8 bytes",
            prefix.len()
        ));
    }
    let mut length_bytes = [0u8; 8];
    length_bytes.copy_from_slice(prefix.as_slice());
    let header_len = u64::from_le_bytes(length_bytes);
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| format!("safetensors header length overflow: {header_len}"))?;

    if let Some(total) = archive.total_hint()
        && header_end > total
    {
        return Err(format!(
            "invalid safetensors header: 8+header_len={} exceeds total bytes {}",
            header_end, total
        ));
    }

    let header = archive.read_range(8, header_end)?;
    let header_json: Value = serde_json::from_slice(header.as_slice())
        .map_err(|err| format!("failed to parse safetensors JSON header: {err}"))?;
    Ok((header_len, header_json))
}

fn build_snapshots_from_header(
    archive: &SharedBlobPartsArchive,
    header_len: u64,
    header_json: &Value,
) -> Result<Vec<TensorSnapshot>, String> {
    let object = header_json
        .as_object()
        .ok_or_else(|| "invalid safetensors header: expected top-level object".to_string())?;
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| format!("safetensors data start overflow: header_len={header_len}"))?;

    let mut snapshots = Vec::new();
    for (name, entry_value) in object {
        if name == "__metadata__" {
            continue;
        }

        let entry: SafetensorsHeaderEntry = serde_json::from_value(entry_value.clone())
            .map_err(|err| format!("invalid safetensors tensor metadata for '{name}': {err}"))?;
        let source_dtype = parse_safetensors_dtype(&entry.dtype)?;
        let dtype = chunked_snapshot_dtype(source_dtype);
        let [offset_start, offset_end] = entry.data_offsets;
        if offset_end < offset_start {
            return Err(format!(
                "invalid safetensors data_offsets for '{name}': [{offset_start}, {offset_end}]"
            ));
        }

        let absolute_start = data_start
            .checked_add(offset_start)
            .ok_or_else(|| format!("tensor '{name}' start offset overflow"))?;
        let absolute_end = data_start
            .checked_add(offset_end)
            .ok_or_else(|| format!("tensor '{name}' end offset overflow"))?;

        if let Some(total) = archive.total_hint()
            && absolute_end > total
        {
            return Err(format!(
                "tensor '{name}' end offset {} exceeds total bytes {}",
                absolute_end, total
            ));
        }

        let shape: Shape = entry.shape.into();
        let shape_for_data = shape.clone();
        let archive_for_data = archive.clone();
        let source_dtype_for_data = source_dtype;
        let dtype_for_data = dtype;
        let name_for_data = name.clone();
        let data_fn = Rc::new(move || {
            let bytes = archive_for_data
                .read_range(absolute_start, absolute_end)
                .map_err(TensorSnapshotError::IoError)?;
            let data = TensorData {
                bytes: Bytes::from_bytes_vec(bytes),
                shape: shape_for_data.clone(),
                dtype: source_dtype_for_data,
            };
            Ok(promote_chunked_tensor_data_if_needed(data, dtype_for_data))
        });

        let path_stack = name_for_data
            .split('.')
            .map(|segment| segment.to_string())
            .collect::<Vec<_>>();
        snapshots.push(TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape,
            path_stack,
            Vec::new(),
            burn::module::ParamId::new(),
        ));
    }

    if snapshots.is_empty() {
        return Err("safetensors header contained no tensor entries".to_string());
    }

    Ok(snapshots)
}

fn parse_safetensors_dtype(value: &str) -> Result<DType, String> {
    match value {
        "F64" => Ok(DType::F64),
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I64" => Ok(DType::I64),
        "I32" => Ok(DType::I32),
        "I16" => Ok(DType::I16),
        "I8" => Ok(DType::I8),
        "U64" => Ok(DType::U64),
        "U32" => Ok(DType::U32),
        "U8" => Ok(DType::U8),
        "BOOL" => Ok(DType::Bool(BoolStore::Native)),
        other => Err(format!("unsupported safetensors dtype '{other}'")),
    }
}

fn chunked_snapshot_dtype(dtype: DType) -> DType {
    match dtype {
        #[cfg(target_arch = "wasm32")]
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

fn promote_chunked_tensor_data_if_needed(data: TensorData, target_dtype: DType) -> TensorData {
    if data.dtype == target_dtype {
        return data;
    }
    match target_dtype {
        DType::F32 => data.convert::<f32>(),
        _ => data,
    }
}

fn resolve_manifest_part_url(manifest_url: &str, entry: &str) -> String {
    if entry.contains("://") || entry.starts_with('/') {
        return entry.to_string();
    }
    let normalized = entry.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

fn resolve_manifest_part_path(manifest_path: &Path, entry: &str) -> Result<PathBuf, String> {
    let entry_path = Path::new(entry);
    if entry_path.is_absolute() {
        return Ok(entry_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| {
            format!(
                "invalid burnpack parts manifest path '{}'",
                manifest_path.display()
            )
        })
}
