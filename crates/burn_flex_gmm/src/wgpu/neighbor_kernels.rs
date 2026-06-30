use super::*;

#[allow(clippy::eq_op)]
#[cube(launch_unchecked)]
pub(super) fn neighbor_rows_from_coords_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    rows: &usize,
    kernel_rows: &usize,
) {
    if ABSOLUTE_POS >= neighbor_rows.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_row = out_idx / *kernel_rows;
    let kernel_idx = out_idx % *kernel_rows;
    let coord_base = out_row * 4;
    let batch = coords[coord_base];
    let ox = coords[coord_base + 1];
    let oy = coords[coord_base + 2];
    let oz = coords[coord_base + 3];

    let offset_base = kernel_idx * 3;
    let nx = ox + offsets[offset_base];
    let ny = oy + offsets[offset_base + 1];
    let nz = oz + offsets[offset_base + 2];

    let mut found = batch - batch - 1;
    for in_row in 0..*rows {
        let src = in_row * 4;
        let same = coords[src] == batch
            && coords[src + 1] == nx
            && coords[src + 2] == ny
            && coords[src + 3] == nz;
        if same && found == INVALID_NEIGHBOR {
            found = i32::cast_from(in_row);
        }
    }

    if nx < 0 || ny < 0 || nz < 0 {
        found = INVALID_NEIGHBOR;
    }
    neighbor_rows[out_idx] = found;
}

#[cube]
pub(super) fn spatial_hash_u32(batch: i32, x: i32, y: i32, z: i32) -> usize {
    let b = usize::cast_from(batch);
    let xx = usize::cast_from(x);
    let yy = usize::cast_from(y);
    let zz = usize::cast_from(z);
    let mut hash = b * 0x9e37_79b1usize;
    hash ^= xx * 0x85eb_ca77usize;
    hash ^= yy * 0xc2b2_ae3dusize;
    hash ^= zz * 0x27d4_eb2fusize;
    hash
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_coord_hash_kernel(
    coords: &Array<i32>,
    hashes: &mut Array<i32>,
    rows: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let coord_base = row * 4;
    let batch = coords[coord_base];
    let x = coords[coord_base + 1];
    let y = coords[coord_base + 2];
    let z = coords[coord_base + 3];
    hashes[row] = i32::cast_from(spatial_hash_u32(batch, x, y, z));
}

macro_rules! define_neighbor_rows_from_sorted_hash_kernel {
    ($name:ident, $binary_steps:expr) => {
        #[cube(launch_unchecked)]
        pub(super) fn $name(
            coords: &Array<i32>,
            offsets: &Array<i32>,
            sorted_hashes: &Array<i32>,
            sorted_rows: &Array<i32>,
            neighbor_rows: &mut Array<i32>,
            rows: &usize,
            kernel_rows: &usize,
            max_match_scan: &usize,
        ) {
            if ABSOLUTE_POS >= neighbor_rows.len() {
                terminate!();
            }

            let out_idx = ABSOLUTE_POS;
            let out_row = out_idx / *kernel_rows;
            let kernel_idx = out_idx % *kernel_rows;
            let coord_base = out_row * 4;
            let batch = coords[coord_base];
            let ox = coords[coord_base + 1];
            let oy = coords[coord_base + 2];
            let oz = coords[coord_base + 3];

            let offset_base = kernel_idx * 3;
            let nx = ox + offsets[offset_base];
            let ny = oy + offsets[offset_base + 1];
            let nz = oz + offsets[offset_base + 2];
            if nx < 0 || ny < 0 || nz < 0 {
                neighbor_rows[out_idx] = INVALID_NEIGHBOR;
                terminate!();
            }

            let query_hash = i32::cast_from(spatial_hash_u32(batch, nx, ny, nz));
            let lo = RuntimeCell::<usize>::new(0);
            let hi = RuntimeCell::<usize>::new(*rows);
            for _ in 0..$binary_steps {
                let lo_v = lo.read();
                let hi_v = hi.read();
                if lo_v < hi_v {
                    let mid = lo_v + (hi_v - lo_v) / 2;
                    let mid_hash = sorted_hashes[mid];
                    if mid_hash < query_hash {
                        lo.store(mid + 1);
                    } else {
                        hi.store(mid);
                    }
                }
            }

            let start = lo.read();
            if start >= *rows || sorted_hashes[start] != query_hash {
                neighbor_rows[out_idx] = INVALID_NEIGHBOR;
                terminate!();
            }

            let best = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);
            let active = RuntimeCell::<i32>::new(1);
            for scan in 0..*max_match_scan {
                if active.read() == 1 {
                    let idx = start + scan;
                    if idx < *rows {
                        if sorted_hashes[idx] == query_hash {
                            let candidate = sorted_rows[idx];
                            if candidate >= 0 {
                                let candidate_base = usize::cast_from(candidate) * 4;
                                let same = coords[candidate_base] == batch
                                    && coords[candidate_base + 1] == nx
                                    && coords[candidate_base + 2] == ny
                                    && coords[candidate_base + 3] == nz;
                                if same {
                                    let prev = best.read();
                                    if prev == INVALID_NEIGHBOR || candidate < prev {
                                        best.store(candidate);
                                    }
                                }
                            }
                        } else {
                            // Sorted hashes are contiguous by key; once the run
                            // ends we can stop scanning without losing matches.
                            active.store(0);
                        }
                    } else {
                        active.store(0);
                    }
                }
            }

            neighbor_rows[out_idx] = best.read();
        }
    };
}

// Keep binary-search loop bounds compile-time static. Runtime-gated loop steps
// regressed sorted-hash parity on current CubeCL/WGSL lowering.
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_16, 16);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_18, 18);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_24, 24);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_32, 32);

#[cube(launch_unchecked)]
pub(super) fn neighbor_hash_reset_kernel(table_rows: &mut Array<u32>, fill: &u32) {
    if ABSOLUTE_POS >= table_rows.len() {
        terminate!();
    }
    table_rows[ABSOLUTE_POS] = *fill;
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_hash_stats_reset_kernel(stats: &mut Array<i32>, fill: &i32) {
    if ABSOLUTE_POS >= stats.len() {
        terminate!();
    }
    stats[ABSOLUTE_POS] = *fill;
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_hash_build_serial_kernel(
    coords: &Array<i32>,
    table_rows: &mut Array<u32>,
    table_coords: &mut Array<i32>,
    build_stats: &mut Array<i32>,
    rows: &usize,
    table_mask: &usize,
    max_probe: &usize,
) {
    // NOTE: compare_exchange atomics currently panic on cubecl-spirv for this
    // path ("Atomic should have a scope registered"), so we keep a deterministic
    // single-lane device-side insertion kernel until upstream atomic-scope
    // support is fixed.
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    let total_probes = RuntimeCell::<i32>::new(0);
    let max_probe_seen = RuntimeCell::<i32>::new(0);
    let fail_rows = RuntimeCell::<i32>::new(0);

    for row in 0..*rows {
        let coord_base = row * 4;
        let batch = coords[coord_base];
        let x = coords[coord_base + 1];
        let y = coords[coord_base + 2];
        let z = coords[coord_base + 3];
        let hash = spatial_hash_u32(batch, x, y, z);
        let row_u32 = u32::cast_from(row);
        let inserted = RuntimeCell::<i32>::new(0);
        let row_probe_steps = RuntimeCell::<i32>::new(0);

        for probe in 0..*max_probe {
            if inserted.read() == 0 {
                let slot = (hash + probe) & *table_mask;
                let slot_state = table_rows[slot];
                if slot_state == HASH_SLOT_EMPTY {
                    let dst = slot * 4;
                    table_coords[dst] = batch;
                    table_coords[dst + 1] = x;
                    table_coords[dst + 2] = y;
                    table_coords[dst + 3] = z;
                    table_rows[slot] = row_u32;
                    row_probe_steps.store(i32::cast_from(probe + 1));
                    inserted.store(1);
                } else {
                    let dst = slot * 4;
                    let same = table_coords[dst] == batch
                        && table_coords[dst + 1] == x
                        && table_coords[dst + 2] == y
                        && table_coords[dst + 3] == z;
                    if same {
                        // Duplicate coords can appear in malformed inputs; keep
                        // query semantics deterministic by retaining lowest row.
                        let current = table_rows[slot];
                        table_rows[slot] = current.min(row_u32);
                        row_probe_steps.store(i32::cast_from(probe + 1));
                        inserted.store(1);
                    }
                }
            }
        }

        if inserted.read() == 0 {
            fail_rows.store(fail_rows.read() + 1);
            let max_probe_i32 = i32::cast_from(*max_probe);
            total_probes.store(total_probes.read() + max_probe_i32);
            max_probe_seen.store(max_probe_seen.read().max(max_probe_i32));
        } else {
            let used = row_probe_steps.read();
            total_probes.store(total_probes.read() + used);
            max_probe_seen.store(max_probe_seen.read().max(used));
        }
    }

    build_stats[HASH_BUILD_STAT_FAIL_ROWS] = fail_rows.read();
    build_stats[HASH_BUILD_STAT_TOTAL_PROBES] = total_probes.read();
    build_stats[HASH_BUILD_STAT_MAX_PROBE] = max_probe_seen.read();
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_hash_query_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    table_rows: &Array<u32>,
    table_coords: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    kernel_rows: &usize,
    table_mask: &usize,
    max_probe: &usize,
) {
    if ABSOLUTE_POS >= neighbor_rows.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_row = out_idx / *kernel_rows;
    let kernel_idx = out_idx % *kernel_rows;
    let coord_base = out_row * 4;
    let batch = coords[coord_base];
    let ox = coords[coord_base + 1];
    let oy = coords[coord_base + 2];
    let oz = coords[coord_base + 3];

    let offset_base = kernel_idx * 3;
    let nx = ox + offsets[offset_base];
    let ny = oy + offsets[offset_base + 1];
    let nz = oz + offsets[offset_base + 2];
    if nx < 0 || ny < 0 || nz < 0 {
        neighbor_rows[out_idx] = INVALID_NEIGHBOR;
        terminate!();
    }

    let hash = spatial_hash_u32(batch, nx, ny, nz);
    let found = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);
    let active = RuntimeCell::<i32>::new(1);
    for probe in 0..*max_probe {
        if active.read() == 1 {
            let slot = (hash + probe) & *table_mask;
            let state = table_rows[slot];
            if state == HASH_SLOT_EMPTY {
                active.store(0);
            } else {
                let table_base = slot * 4;
                if table_coords[table_base] == batch
                    && table_coords[table_base + 1] == nx
                    && table_coords[table_base + 2] == ny
                    && table_coords[table_base + 3] == nz
                {
                    found.store(i32::cast_from(state));
                    active.store(0);
                }
            }
        }
    }

    neighbor_rows[out_idx] = found.read();
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_bucket_hash_build_kernel(
    coords: &Array<i32>,
    bucket_counts: &mut Array<Atomic<u32>>,
    bucket_rows: &mut Array<i32>,
    overflow_rows: &mut Array<Atomic<i32>>,
    rows: &usize,
    bucket_mask: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let coord_base = row * 4;
    let batch = coords[coord_base];
    let x = coords[coord_base + 1];
    let y = coords[coord_base + 2];
    let z = coords[coord_base + 3];
    let hash = spatial_hash_u32(batch, x, y, z);
    let bucket = hash & *bucket_mask;
    let slot = usize::cast_from(bucket_counts[bucket].fetch_add(u32::cast_from(1usize)));

    if slot < DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP {
        let dst = bucket * DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP + slot;
        bucket_rows[dst] = i32::cast_from(row);
    } else {
        overflow_rows[0].fetch_add(i32::cast_from(1usize));
    }
}

#[cube(launch_unchecked)]
pub(super) fn neighbor_bucket_hash_query_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    bucket_counts: &Array<u32>,
    bucket_rows: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    rows: &usize,
    kernel_rows: &usize,
    bucket_mask: &usize,
) {
    if ABSOLUTE_POS >= neighbor_rows.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_row = out_idx / *kernel_rows;
    let kernel_idx = out_idx % *kernel_rows;
    let coord_base = out_row * 4;
    let batch = coords[coord_base];
    let ox = coords[coord_base + 1];
    let oy = coords[coord_base + 2];
    let oz = coords[coord_base + 3];

    let offset_base = kernel_idx * 3;
    let nx = ox + offsets[offset_base];
    let ny = oy + offsets[offset_base + 1];
    let nz = oz + offsets[offset_base + 2];
    if nx < 0 || ny < 0 || nz < 0 {
        neighbor_rows[out_idx] = INVALID_NEIGHBOR;
        terminate!();
    }

    let hash = spatial_hash_u32(batch, nx, ny, nz);
    let bucket = hash & *bucket_mask;
    let bucket_count =
        usize::cast_from(bucket_counts[bucket]).min(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP);
    let best = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);

    for slot in 0..DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP {
        if slot < bucket_count {
            let candidate = bucket_rows[bucket * DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP + slot];
            if candidate >= 0 {
                let candidate_row = usize::cast_from(candidate);
                if candidate_row < *rows {
                    let base = candidate_row * 4;
                    let same = coords[base] == batch
                        && coords[base + 1] == nx
                        && coords[base + 2] == ny
                        && coords[base + 3] == nz;
                    if same {
                        let prev = best.read();
                        if prev == INVALID_NEIGHBOR || candidate < prev {
                            best.store(candidate);
                        }
                    }
                }
            }
        }
    }

    neighbor_rows[out_idx] = best.read();
}
