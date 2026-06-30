use super::neighbor_kernels::*;
use super::*;

fn resolve_neighbor_backend(_rows: usize, _kernel_rows: usize) -> NeighborBuildBackend {
    // Canonical path: device-resident neighbor map generation.
    NeighborBuildBackend::Device
}

pub(super) fn resolve_neighbor_device_algo(
    rows: usize,
    kernel_rows: usize,
    preference: NeighborDeviceAlgoPreference,
) -> NeighborDeviceAlgo {
    #[cfg(target_arch = "wasm32")]
    if matches!(preference, NeighborDeviceAlgoPreference::SortedHash) {
        if kernel_rows <= 64 && rows >= DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K {
            return NeighborDeviceAlgo::BucketHash;
        }
        return NeighborDeviceAlgo::Hash;
    }
    match preference {
        NeighborDeviceAlgoPreference::Auto => {}
        NeighborDeviceAlgoPreference::Scan => return NeighborDeviceAlgo::Scan,
        NeighborDeviceAlgoPreference::SortedHash => return NeighborDeviceAlgo::SortedHash,
        NeighborDeviceAlgoPreference::HashTableSerial => return NeighborDeviceAlgo::Hash,
        NeighborDeviceAlgoPreference::BucketHash => return NeighborDeviceAlgo::BucketHash,
    }
    let work = rows.saturating_mul(kernel_rows);
    // Tuned from bounded stage-only bench runs in docs/reports/parity_gap:
    // small kernels cross earlier; very large kernels need more rows before
    // sort+search amortizes launch/sort overhead over scan.
    let sorted_threshold = if kernel_rows <= 64 {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_SMALL_K
    } else if kernel_rows <= 256 {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_MEDIUM_K
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_LARGE_K
    };
    // Bucket-hash beats sorted-hash on large decode-like small-k workloads by
    // avoiding sort_with_indices overhead; keep routing conservative so
    // mid-row shapes that still favor sorted-hash remain unchanged.
    if kernel_rows <= 64
        && rows >= DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K
        && work >= sorted_threshold
    {
        return NeighborDeviceAlgo::BucketHash;
    }
    #[cfg(target_arch = "wasm32")]
    if work >= sorted_threshold {
        return NeighborDeviceAlgo::Hash;
    }
    if work >= sorted_threshold {
        NeighborDeviceAlgo::SortedHash
    } else {
        NeighborDeviceAlgo::Scan
    }
}

fn resolve_neighbor_hash_load_factor() -> usize {
    DEFAULT_NEIGHBOR_HASH_LOAD_FACTOR
}

fn resolve_neighbor_hash_table_size(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let load_factor = resolve_neighbor_hash_load_factor();
    let min_capacity = rows.saturating_mul(load_factor);
    let capacity = min_capacity.next_power_of_two();
    capacity.max(64)
}

fn resolve_neighbor_hash_max_probe(table_size: usize) -> usize {
    // Keep probe work bounded for the current kernel form; the loop-break form
    // triggers cubecl-opt uniformity panics in this path.
    table_size.clamp(1, DEFAULT_NEIGHBOR_HASH_MAX_PROBE)
}

fn resolve_neighbor_bucket_hash_bucket_size(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    rows.next_power_of_two().max(64)
}

fn resolve_neighbor_sorted_hash_match_scan(rows: usize, kernel_rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let cap = if kernel_rows <= 64 {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_SMALL_K
    } else if kernel_rows <= 256 {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_MEDIUM_K
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_LARGE_K
    };
    cap.min(rows).max(1)
}

pub(super) fn resolve_neighbor_sorted_hash_search_steps(rows: usize) -> usize {
    // Keep search-step routing compile-time per kernel variant. Runtime-gated
    // loop bounds regressed parity on CubeCL/WGSL in this path. Keep the
    // decode-hot mid bucket tighter (2^16..2^18 => 18) to avoid 24-step
    // over-iteration on common 512-quality row counts.
    if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL
    } else if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MEDIUM_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM
    } else if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_MEDIUM_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_LARGE
    }
}

fn record_neighbor_device_build(algo: NeighborDeviceAlgo, elapsed_ns: u64) {
    NEIGHBOR_BUILDS_DEVICE.fetch_add(1, Ordering::Relaxed);
    match algo {
        NeighborDeviceAlgo::Scan => {
            NEIGHBOR_DEVICE_SCAN_BUILDS.fetch_add(1, Ordering::Relaxed);
            NEIGHBOR_DEVICE_SCAN_BUILD_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        NeighborDeviceAlgo::Hash
        | NeighborDeviceAlgo::SortedHash
        | NeighborDeviceAlgo::BucketHash => {
            NEIGHBOR_DEVICE_HASH_BUILDS.fetch_add(1, Ordering::Relaxed);
            NEIGHBOR_DEVICE_HASH_BUILD_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }
}

fn neighbor_cache_max_entries() -> usize {
    DEFAULT_NEIGHBOR_CACHE_MAX
}

fn trim_cache(cache: &mut HashMap<NeighborRowsCacheKey, BurnTensor<DefaultWgpuBackend, 2, Int>>) {
    let max = neighbor_cache_max_entries();
    while cache.len() > max {
        let Some(key) = cache.keys().next().cloned() else {
            break;
        };
        cache.remove(&key);
    }
}

fn hash_coords(coords: &[[u32; 4]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for coord in coords {
        for value in coord {
            hash ^= *value as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash ^= coords.len() as u64;
    hash.wrapping_mul(0x0000_0100_0000_01b3)
}

#[cfg(feature = "wgpu-kernel")]
fn hash_tensor_identity(coords_t: &BurnTensor<DefaultWgpuBackend, 2, Int>) -> u64 {
    let primitive = coords_t.clone().into_primitive();
    // Hash buffer handle metadata without host readback so tensor-native paths can
    // reuse cached neighbor tensors across repeated decode calls. CubeCL's public
    // handle debug includes mutable memory-location state after first binding, so
    // keep only the allocation id plus static view metadata.
    let memory_debug = format!("{:?}", primitive.handle.memory);
    let memory_id = stable_memory_id_from_debug(&memory_debug);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&memory_id, &mut hasher);
    std::hash::Hash::hash(&primitive.handle.offset_start, &mut hasher);
    std::hash::Hash::hash(&primitive.handle.offset_end, &mut hasher);
    std::hash::Hash::hash(&format!("{:?}", primitive.handle.stream), &mut hasher);
    std::hash::Hash::hash(&primitive.meta.shape, &mut hasher);
    std::hash::Hash::hash(&primitive.meta.strides, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

#[cfg(feature = "wgpu-kernel")]
fn stable_memory_id_from_debug(debug: &str) -> String {
    const PREFIX: &str = "ManagedMemoryId { value: ";
    let Some(start) = debug.find(PREFIX) else {
        return debug.to_string();
    };
    let rest = &debug[start + PREFIX.len()..];
    let Some(end) = rest.find('}') else {
        return debug.to_string();
    };
    rest[..end].trim().to_string()
}

fn kernel_offsets(config: &SparseSubmConvConfig) -> Vec<[i32; 3]> {
    let center_d = (config.kernel_d / 2) as i32;
    let center_h = (config.kernel_h / 2) as i32;
    let center_w = (config.kernel_w / 2) as i32;
    let mut offsets = Vec::with_capacity(
        config
            .kernel_d
            .saturating_mul(config.kernel_h)
            .saturating_mul(config.kernel_w),
    );
    for kd_idx in 0..config.kernel_d {
        for kh_idx in 0..config.kernel_h {
            for kw_idx in 0..config.kernel_w {
                let deltas = [
                    config.axis_sign[0] * (kd_idx as i32 - center_d),
                    config.axis_sign[1] * (kh_idx as i32 - center_h),
                    config.axis_sign[2] * (kw_idx as i32 - center_w),
                ];
                let mut offset = [0i32; 3];
                offset[config.axis_order[0]] = deltas[0];
                offset[config.axis_order[1]] = deltas[1];
                offset[config.axis_order[2]] = deltas[2];
                offsets.push(offset);
            }
        }
    }
    offsets
}

fn neighbor_cache_key(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    backend: NeighborBuildBackend,
) -> NeighborRowsCacheKey {
    NeighborRowsCacheKey {
        config: NeighborConfigCacheKey::from(config),
        backend,
        rows: coords.len(),
        coords_hash: hash_coords(coords),
        device_key: format!("{device:?}"),
    }
}

#[cfg(feature = "wgpu-kernel")]
fn neighbor_cache_key_tensor(
    config: &SparseSubmConvConfig,
    coords_t: &BurnTensor<DefaultWgpuBackend, 2, Int>,
    backend: NeighborBuildBackend,
) -> NeighborRowsCacheKey {
    let [rows, _] = coords_t.dims();
    let device = coords_t.device();
    NeighborRowsCacheKey {
        config: NeighborConfigCacheKey::from(config),
        backend,
        rows,
        coords_hash: hash_tensor_identity(coords_t),
        // Keep tensor-path keys disjoint from host-path keys in shared cache map.
        device_key: format!("{device:?}:tensor"),
    }
}

fn flatten_coords_i32(coords: &[[u32; 4]]) -> Result<Vec<i32>, String> {
    let mut coords_flat = Vec::with_capacity(coords.len() * 4);
    for coord in coords.iter().copied() {
        for value in coord {
            let converted = i32::try_from(value).map_err(|_| {
                format!("coord value {value} exceeds i32::MAX for device neighbor kernel")
            })?;
            coords_flat.push(converted);
        }
    }
    Ok(coords_flat)
}

pub(super) fn build_neighbor_rows_tensor_device_scan(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let coords_flat = flatten_coords_i32(coords)?;
    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    )
    .reshape([rows, 4]);
    build_neighbor_rows_tensor_device_scan_tensor(config, coords_t)
}

fn build_neighbor_rows_tensor_device_scan_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let coords_p = coords_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    )
    .reshape([kernel_rows, 3]);
    let offsets_p = offsets_t.into_primitive();
    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([rows, kernel_rows]),
        coords_p.client.empty(output_bytes),
        DType::I32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_rows_from_coords_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            cube_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            kernel_rows,
        )
        .map_err(|err| format!("neighbor_rows_from_coords_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::from_primitive(output))
}

#[allow(dead_code)]
fn build_neighbor_rows_tensor_device_hash_wgsl_table(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let coords_flat = flatten_coords_i32(coords)?;
    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    )
    .reshape([rows, 4]);
    build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(config, coords_t)
}

#[allow(dead_code)]
fn build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let table_size = resolve_neighbor_hash_table_size(rows);
    if table_size > i32::MAX as usize {
        return Err("neighbor hash table size exceeds i32::MAX entries".to_string());
    }
    let table_coords_elements = table_size
        .checked_mul(4)
        .ok_or_else(|| "neighbor hash coordinate table size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let table_rows_bytes = table_size
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash row table byte size overflow".to_string())?;
    let table_coords_bytes = table_coords_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash coord table byte size overflow".to_string())?;

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();
    let table_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([table_size]),
        coords_p.client.empty(table_rows_bytes),
        DType::U32,
    );
    let table_coords = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([table_coords_elements]),
        coords_p.client.empty(table_coords_bytes),
        DType::I32,
    );
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );
    let hash_build_stats = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([HASH_BUILD_STAT_LEN]),
        coords_p
            .client
            .empty(HASH_BUILD_STAT_LEN * core::mem::size_of::<i32>()),
        DType::I32,
    );
    let table_mask = table_size - 1;
    let max_probe = resolve_neighbor_hash_max_probe(table_size);

    let cube_dim = resolve_cube_dim();
    let reset_count = calculate_cube_count_elemwise(&coords_p.client, table_size, cube_dim);
    unsafe {
        neighbor_hash_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_count,
            cube_dim,
            table_rows.clone().into_array_arg(),
            HASH_SLOT_EMPTY,
        )
        .map_err(|err| format!("neighbor_hash_reset_kernel launch failed: {err:?}"))?;
    }
    let counter_reset_count =
        calculate_cube_count_elemwise(&coords_p.client, HASH_BUILD_STAT_LEN, cube_dim);
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            counter_reset_count,
            cube_dim,
            hash_build_stats.clone().into_array_arg(),
            0,
        )
        .map_err(|err| format!("neighbor_hash_stats_reset_kernel launch failed: {err:?}"))?;
    }

    let build_count = calculate_cube_count_elemwise(&coords_p.client, 1, cube_dim);
    unsafe {
        neighbor_hash_build_serial_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            build_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            table_rows.clone().into_array_arg(),
            table_coords.clone().into_array_arg(),
            hash_build_stats.clone().into_array_arg(),
            rows,
            table_mask,
            max_probe,
        )
        .map_err(|err| format!("neighbor_hash_build_serial_kernel launch failed: {err:?}"))?;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let hash_build_stats_t: BurnTensor<DefaultWgpuBackend, 1, Int> =
            BurnTensor::from_primitive(hash_build_stats);
        let hash_build_stats_data = hash_build_stats_t.to_data();
        let hash_build_stats = hash_build_stats_data
            .as_slice::<i32>()
            .map_err(|err| format!("neighbor hash build stats readback failed: {err:?}"))?;
        let fail_rows = hash_build_stats
            .get(HASH_BUILD_STAT_FAIL_ROWS)
            .copied()
            .unwrap_or(0);
        let total_probes = hash_build_stats
            .get(HASH_BUILD_STAT_TOTAL_PROBES)
            .copied()
            .unwrap_or(0)
            .max(0) as u64;
        let max_probe_used = hash_build_stats
            .get(HASH_BUILD_STAT_MAX_PROBE)
            .copied()
            .unwrap_or(0)
            .max(0) as u64;
        NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(total_probes, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(max_probe_used, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.fetch_add(fail_rows.max(0) as u64, Ordering::Relaxed);
        if fail_rows != 0 {
            return Err(format!(
                "neighbor hash build failed to insert {fail_rows} row(s); rows={rows} table_size={table_size} max_probe={max_probe} probe_total={total_probes} probe_max={max_probe_used}"
            ));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Browser WebGPU cannot synchronously read the diagnostic build-stat
        // tensor. Keep the wasm path device-resident and route auto selection to
        // this low-load-factor serial hash builder; native still validates the
        // failure counter above.
        let _ = hash_build_stats;
        NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
            (rows as u64).saturating_mul(max_probe as u64),
            Ordering::Relaxed,
        );
        NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(max_probe as u64, Ordering::Relaxed);
    }

    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            table_rows.clone().into_array_arg(),
            table_coords.clone().into_array_arg(),
            output.clone().into_array_arg(),
            kernel_rows,
            table_mask,
            max_probe,
        )
        .map_err(|err| format!("neighbor_hash_query_kernel launch failed: {err:?}"))?;
    }

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device_hash(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    build_neighbor_rows_tensor_device_hash_wgsl_table(config, coords, device)
}

fn build_neighbor_rows_tensor_device_sorted_hash_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let hash_bytes = rows
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash key byte size overflow".to_string())?;

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();

    let hash_keys = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([rows]),
        coords_p.client.empty(hash_bytes),
        DType::I32,
    );
    let cube_dim = resolve_cube_dim();
    let hash_count = calculate_cube_count_elemwise(&coords_p.client, rows, cube_dim);
    unsafe {
        neighbor_coord_hash_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            hash_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            hash_keys.clone().into_array_arg(),
            rows,
        )
        .map_err(|err| format!("neighbor_coord_hash_kernel launch failed: {err:?}"))?;
    }

    let hash_keys_t: BurnTensor<DefaultWgpuBackend, 1, Int> = BurnTensor::from_primitive(hash_keys);
    let (sorted_hashes_t, sorted_idx_t) = hash_keys_t.sort_with_indices(0);
    let sorted_hashes_p = sorted_hashes_t.into_primitive();
    let sorted_idx_p = sorted_idx_t.into_primitive();

    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );
    let search_steps = resolve_neighbor_sorted_hash_search_steps(rows);
    let match_scan = resolve_neighbor_sorted_hash_match_scan(rows, kernel_rows);
    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        match search_steps {
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL => {
                neighbor_rows_from_sorted_hash_kernel_16::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_16 launch failed: {err:?}")
                })?;
            }
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM => {
                neighbor_rows_from_sorted_hash_kernel_18::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_18 launch failed: {err:?}")
                })?;
            }
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM => {
                neighbor_rows_from_sorted_hash_kernel_24::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_24 launch failed: {err:?}")
                })?;
            }
            _ => {
                neighbor_rows_from_sorted_hash_kernel_32::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_32 launch failed: {err:?}")
                })?;
            }
        }
    }

    NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
        (output_elements as u64).saturating_mul(search_steps as u64),
        Ordering::Relaxed,
    );
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max((search_steps + match_scan) as u64, Ordering::Relaxed);

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device_bucket_hash_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let bucket_count = resolve_neighbor_bucket_hash_bucket_size(rows);
    let bucket_mask = bucket_count - 1;
    let bucket_rows_len = bucket_count
        .checked_mul(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP)
        .ok_or_else(|| "neighbor bucket-hash row table size overflow".to_string())?;
    let bucket_counts_bytes = bucket_count
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or_else(|| "neighbor bucket-hash counts byte size overflow".to_string())?;
    let bucket_rows_bytes = bucket_rows_len
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor bucket-hash rows byte size overflow".to_string())?;

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();

    let bucket_counts = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([bucket_count]),
        coords_p.client.empty(bucket_counts_bytes),
        DType::U32,
    );
    let bucket_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([bucket_rows_len]),
        coords_p.client.empty(bucket_rows_bytes),
        DType::I32,
    );
    let overflow_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([1]),
        coords_p.client.empty(core::mem::size_of::<i32>()),
        DType::I32,
    );
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );

    let cube_dim = resolve_cube_dim();
    let reset_counts = calculate_cube_count_elemwise(&coords_p.client, bucket_count, cube_dim);
    unsafe {
        neighbor_hash_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_counts,
            cube_dim,
            bucket_counts.clone().into_array_arg(),
            0u32,
        )
        .map_err(|err| format!("neighbor bucket-hash counts reset launch failed: {err:?}"))?;
    }
    let reset_rows = calculate_cube_count_elemwise(&coords_p.client, bucket_rows_len, cube_dim);
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_rows,
            cube_dim,
            bucket_rows.clone().into_array_arg(),
            INVALID_NEIGHBOR,
        )
        .map_err(|err| format!("neighbor bucket-hash rows reset launch failed: {err:?}"))?;
    }
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            calculate_cube_count_elemwise(&coords_p.client, 1, cube_dim),
            cube_dim,
            overflow_rows.clone().into_array_arg(),
            0i32,
        )
        .map_err(|err| format!("neighbor bucket-hash overflow reset launch failed: {err:?}"))?;
    }

    let build_count = calculate_cube_count_elemwise(&coords_p.client, rows, cube_dim);
    unsafe {
        neighbor_bucket_hash_build_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            build_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            bucket_counts.clone().into_array_arg(),
            bucket_rows.clone().into_array_arg(),
            overflow_rows.clone().into_array_arg(),
            rows,
            bucket_mask,
        )
        .map_err(|err| format!("neighbor_bucket_hash_build_kernel launch failed: {err:?}"))?;
    }

    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_bucket_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            bucket_counts.clone().into_array_arg(),
            bucket_rows.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            kernel_rows,
            bucket_mask,
        )
        .map_err(|err| format!("neighbor_bucket_hash_query_kernel launch failed: {err:?}"))?;
    }

    NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
        (output_elements as u64).saturating_mul(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP as u64),
        Ordering::Relaxed,
    );
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(
        DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP as u64,
        Ordering::Relaxed,
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let overflow_rows_t: BurnTensor<DefaultWgpuBackend, 1, Int> =
            BurnTensor::from_primitive(overflow_rows);
        let overflow_rows_data = overflow_rows_t.to_data();
        let overflow_rows = overflow_rows_data
            .as_slice::<i32>()
            .map_err(|err| format!("neighbor bucket-hash overflow readback failed: {err:?}"))?
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as u64;

        NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.fetch_add(overflow_rows, Ordering::Relaxed);
        if overflow_rows != 0 {
            return Err(format!(
                "neighbor bucket-hash overflowed {} row(s); rows={} buckets={} slot_cap={}",
                overflow_rows, rows, bucket_count, DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP
            ));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = overflow_rows;
    }

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    if rows == 0 || kernel_rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2, Int>::zeros(
            [rows, kernel_rows],
            device,
        ));
    }
    if rows > i32::MAX as usize {
        return Err("sparse conv row count exceeds i32::MAX for neighbor kernel".to_string());
    }

    let algo = resolve_neighbor_device_algo(rows, kernel_rows, preference);
    let build_start = Instant::now();
    let result = match algo {
        NeighborDeviceAlgo::Scan => build_neighbor_rows_tensor_device_scan(config, coords, device),
        NeighborDeviceAlgo::Hash => build_neighbor_rows_tensor_device_hash(config, coords, device),
        NeighborDeviceAlgo::SortedHash => {
            let coords_flat = flatten_coords_i32(coords)?;
            let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(coords_flat, [rows * 4]),
                device,
            )
            .reshape([rows, 4]);
            build_neighbor_rows_tensor_device_sorted_hash_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::BucketHash => {
            let coords_flat = flatten_coords_i32(coords)?;
            let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(coords_flat, [rows * 4]),
                device,
            )
            .reshape([rows, 4]);
            build_neighbor_rows_tensor_device_bucket_hash_tensor(config, coords_t)
        }
    };
    record_neighbor_device_build(algo, elapsed_ns_u64(build_start));
    result
}

fn build_neighbor_rows_tensor_device_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    if rows == 0 || kernel_rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2, Int>::zeros(
            [rows, kernel_rows],
            &coords_t.device(),
        ));
    }
    if rows > i32::MAX as usize {
        return Err("sparse conv row count exceeds i32::MAX for neighbor kernel".to_string());
    }

    let algo = resolve_neighbor_device_algo(rows, kernel_rows, preference);
    let build_start = Instant::now();
    let result = match algo {
        NeighborDeviceAlgo::Scan => build_neighbor_rows_tensor_device_scan_tensor(config, coords_t),
        NeighborDeviceAlgo::Hash => {
            build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::SortedHash => {
            build_neighbor_rows_tensor_device_sorted_hash_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::BucketHash => {
            build_neighbor_rows_tensor_device_bucket_hash_tensor(config, coords_t)
        }
    };
    record_neighbor_device_build(algo, elapsed_ns_u64(build_start));
    result
}

/// Build a device neighbor-row tensor directly from a device-resident coords tensor.
pub fn neighbor_rows_tensor_from_coords_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let backend = resolve_neighbor_backend(rows, kernel_rows);
    let key = neighbor_cache_key_tensor(config, &coords_t, backend);
    if let Some(hit) = NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        NEIGHBOR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(hit);
    }
    NEIGHBOR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    let tensor = build_neighbor_rows_tensor_device_tensor(
        config,
        coords_t,
        NeighborDeviceAlgoPreference::Auto,
    )?;
    NEIGHBOR_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(key, tensor.clone());
        trim_cache(&mut cache);
    });
    Ok(tensor)
}

/// Build a device neighbor-row tensor from a device-resident coords tensor with explicit algo selection.
pub fn neighbor_rows_tensor_from_coords_tensor_with_algo(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    build_neighbor_rows_tensor_device_tensor(config, coords_t, preference)
}

pub fn clear_neighbor_rows_tensor_cache() {
    NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow_mut().clear());
}

pub fn reset_neighbor_rows_build_stats() {
    NEIGHBOR_CACHE_HITS.store(0, Ordering::Relaxed);
    NEIGHBOR_CACHE_MISSES.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_HOST.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_DEVICE.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_SCAN_BUILDS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_BUILDS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_SCAN_BUILD_NS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_BUILD_NS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_ROWS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.store(0, Ordering::Relaxed);
}

pub fn neighbor_rows_build_stats() -> NeighborRowsBuildStats {
    NeighborRowsBuildStats {
        cache_hits: NEIGHBOR_CACHE_HITS.load(Ordering::Relaxed),
        cache_misses: NEIGHBOR_CACHE_MISSES.load(Ordering::Relaxed),
        host_builds: NEIGHBOR_BUILDS_HOST.load(Ordering::Relaxed),
        device_builds: NEIGHBOR_BUILDS_DEVICE.load(Ordering::Relaxed),
        device_scan_builds: NEIGHBOR_DEVICE_SCAN_BUILDS.load(Ordering::Relaxed),
        device_hash_builds: NEIGHBOR_DEVICE_HASH_BUILDS.load(Ordering::Relaxed),
        device_scan_build_ns: NEIGHBOR_DEVICE_SCAN_BUILD_NS.load(Ordering::Relaxed),
        device_hash_build_ns: NEIGHBOR_DEVICE_HASH_BUILD_NS.load(Ordering::Relaxed),
        device_hash_rows: NEIGHBOR_DEVICE_HASH_ROWS.load(Ordering::Relaxed),
        device_hash_probe_total: NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.load(Ordering::Relaxed),
        device_hash_probe_max: NEIGHBOR_DEVICE_HASH_PROBE_MAX.load(Ordering::Relaxed),
        device_hash_insert_fail_rows: NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.load(Ordering::Relaxed),
    }
}

pub fn reset_sparse_wgpu_kernel_stats() {
    SPARSE_WGPU_CONV_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_SPLITK_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_FUSED_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_DISPATCHES.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_ROWS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_NS.store(0, Ordering::Relaxed);
}

pub fn sparse_wgpu_kernel_stats() -> SparseWgpuKernelStats {
    SparseWgpuKernelStats {
        calls: SPARSE_WGPU_CONV_CALLS.load(Ordering::Relaxed),
        splitk_calls: SPARSE_WGPU_CONV_SPLITK_CALLS.load(Ordering::Relaxed),
        fused_variant_calls: SPARSE_WGPU_CONV_FUSED_CALLS.load(Ordering::Relaxed),
        single_group_specialized_calls: SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS
            .load(Ordering::Relaxed),
        total_dispatches: SPARSE_WGPU_CONV_TOTAL_DISPATCHES.load(Ordering::Relaxed),
        total_rows: SPARSE_WGPU_CONV_TOTAL_ROWS.load(Ordering::Relaxed),
        total_output_elements: SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS.load(Ordering::Relaxed),
        total_elapsed_ns: SPARSE_WGPU_CONV_TOTAL_NS.load(Ordering::Relaxed),
    }
}

/// Build a device tensor containing sparse neighbor row indices.
pub fn neighbor_rows_tensor_from_coords(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let kernel_rows = kernel_rows(config)?;
    let backend = resolve_neighbor_backend(coords.len(), kernel_rows);
    let key = neighbor_cache_key(config, coords, device, backend);
    if let Some(hit) = NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        NEIGHBOR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(hit);
    }
    NEIGHBOR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);

    let tensor = match backend {
        NeighborBuildBackend::Device => build_neighbor_rows_tensor_device(
            config,
            coords,
            device,
            NeighborDeviceAlgoPreference::Auto,
        )?,
    };

    NEIGHBOR_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(key, tensor.clone());
        trim_cache(&mut cache);
    });
    Ok(tensor)
}

/// Build a device tensor containing sparse neighbor rows with explicit algorithm selection.
pub fn neighbor_rows_tensor_from_coords_with_algo(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    build_neighbor_rows_tensor_device(config, coords, device, preference)
}
