use crate::pipeline::dmc_tables::{
    DMCEDGEOFFSET, DMCQUAD, MCCORNERS, MCEDGEINDEX, MCEDGELOCATIONS, MCFIRSTEDGEINDEX,
    MCFIRSTPATCHINDEX, PROBLEMATICCONFIGS,
};
use fast_surface_nets::ndshape::RuntimeShape;
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use marching_cubes::tables::{EDGE_TABLE, TRI_TABLE};

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone)]
pub struct DenseGrid {
    pub values: Vec<f32>,
    pub size: [usize; 3],
    pub bounds: [f32; 6],
}

type DmcOutput = Option<(Vec<[f32; 3]>, Vec<[u32; 4]>)>;

impl DenseGrid {
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let [nx, ny, _] = self.size;
        (z * ny + y) * nx + x
    }

    fn value_at(&self, x: usize, y: usize, z: usize) -> f32 {
        let idx = self.index(x, y, z);
        self.values[idx]
    }
}

pub fn grid_to_mesh(grid: &DenseGrid, iso: f32) -> Option<Mesh> {
    let [nx, ny, nz] = grid.size;
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }

    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    let mut x_edges = vec![-1_i32; (nx - 1) * ny * nz];
    let mut y_edges = vec![-1_i32; nx * (ny - 1) * nz];
    let mut z_edges = vec![-1_i32; nx * ny * (nz - 1)];

    for z in 0..(nz - 1) {
        for y in 0..(ny - 1) {
            for x in 0..(nx - 1) {
                let v0 = grid.value_at(x, y, z);
                let v1 = grid.value_at(x + 1, y, z);
                let v2 = grid.value_at(x + 1, y + 1, z);
                let v3 = grid.value_at(x, y + 1, z);
                let v4 = grid.value_at(x, y, z + 1);
                let v5 = grid.value_at(x + 1, y, z + 1);
                let v6 = grid.value_at(x + 1, y + 1, z + 1);
                let v7 = grid.value_at(x, y + 1, z + 1);

                let mut cube_index = 0usize;
                if v0 < iso {
                    cube_index |= 1;
                }
                if v1 < iso {
                    cube_index |= 2;
                }
                if v2 < iso {
                    cube_index |= 4;
                }
                if v3 < iso {
                    cube_index |= 8;
                }
                if v4 < iso {
                    cube_index |= 16;
                }
                if v5 < iso {
                    cube_index |= 32;
                }
                if v6 < iso {
                    cube_index |= 64;
                }
                if v7 < iso {
                    cube_index |= 128;
                }

                let edge_mask = EDGE_TABLE[cube_index] as u16;
                if edge_mask == 0 {
                    continue;
                }

                let mut edge_vertices = [0u32; 12];
                for (edge, slot) in edge_vertices.iter_mut().enumerate() {
                    if edge_mask & (1u16 << edge) == 0 {
                        continue;
                    }
                    *slot = vertex_for_edge(
                        edge,
                        x,
                        y,
                        z,
                        nx,
                        ny,
                        nz,
                        grid,
                        &mut x_edges,
                        &mut y_edges,
                        &mut z_edges,
                        &mut vertices,
                        iso,
                    );
                }

                let tri = &TRI_TABLE[cube_index];
                let mut i = 0usize;
                while tri[i] != -1 {
                    let a = tri[i] as usize;
                    let b = tri[i + 1] as usize;
                    let c = tri[i + 2] as usize;
                    faces.push([edge_vertices[a], edge_vertices[b], edge_vertices[c]]);
                    i += 3;
                }
            }
        }
    }

    if vertices.is_empty() {
        return None;
    }

    let bounds = grid.bounds;
    let size_x = nx as f32;
    let size_y = ny as f32;
    let size_z = nz as f32;
    let scale_x = (bounds[3] - bounds[0]) / size_x;
    let scale_y = (bounds[4] - bounds[1]) / size_y;
    let scale_z = (bounds[5] - bounds[2]) / size_z;

    for v in &mut vertices {
        v[0] = bounds[0] + v[0] * scale_x;
        v[1] = bounds[1] + v[1] * scale_y;
        v[2] = bounds[2] + v[2] * scale_z;
    }

    Some(Mesh { vertices, faces })
}

pub fn sdf_to_mesh_surface_nets(grid: &DenseGrid) -> Option<Mesh> {
    let [nx, ny, nz] = grid.size;
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }

    let shape = RuntimeShape::<u32, 3>::new([nx as u32, ny as u32, nz as u32]);
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(
        grid.values.as_slice(),
        &shape,
        [0, 0, 0],
        [nx as u32 - 1, ny as u32 - 1, nz as u32 - 1],
        &mut buffer,
    );

    if buffer.indices.is_empty() {
        return None;
    }

    let bounds = grid.bounds;
    let scale_x = (bounds[3] - bounds[0]) / (nx as f32 - 1.0);
    let scale_y = (bounds[4] - bounds[1]) / (ny as f32 - 1.0);
    let scale_z = (bounds[5] - bounds[2]) / (nz as f32 - 1.0);

    let mut vertices = Vec::with_capacity(buffer.positions.len());
    for pos in &buffer.positions {
        vertices.push([
            bounds[0] + pos[0] * scale_x,
            bounds[1] + pos[1] * scale_y,
            bounds[2] + pos[2] * scale_z,
        ]);
    }

    let mut faces = Vec::with_capacity(buffer.indices.len() / 3);
    for chunk in buffer.indices.chunks(3) {
        if let [a, b, c] = *chunk {
            faces.push([a, b, c]);
        }
    }

    Some(Mesh { vertices, faces })
}

pub fn sdf_to_mesh_diff_dmc(grid: &DenseGrid) -> Option<Mesh> {
    let [nx, ny, nz] = grid.size;
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }

    let iso = 0.0f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &grid.values {
        if !v.is_finite() {
            continue;
        }
        min = min.min(v);
        max = max.max(v);
    }
    if min == f32::INFINITY || max == f32::NEG_INFINITY || min >= iso || max <= iso {
        return None;
    }

    let pad = iso + 1.0;
    let px = nx + 2;
    let py = ny + 2;
    let pz = nz + 2;
    let mut padded = vec![pad; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            let src_base = (z * ny + y) * nx;
            let dst_base = ((z + 1) * py + (y + 1)) * px;
            for x in 0..nx {
                padded[dst_base + x + 1] = grid.values[src_base + x];
            }
        }
    }

    let (mut vertices, quads) = diff_dmc(&padded, [px, py, pz], iso)?;
    if vertices.is_empty() || quads.is_empty() {
        return None;
    }

    for v in &mut vertices {
        v[0] -= 1.0;
        v[1] -= 1.0;
        v[2] -= 1.0;
    }

    let mut faces = triangulate_quads(&vertices, &quads);
    if faces.is_empty() {
        return None;
    }
    for face in &mut faces {
        face.swap(0, 2);
    }

    let bounds = grid.bounds;
    let scale_x = (bounds[3] - bounds[0]) / (nx as f32 - 1.0);
    let scale_y = (bounds[4] - bounds[1]) / (ny as f32 - 1.0);
    let scale_z = (bounds[5] - bounds[2]) / (nz as f32 - 1.0);

    for v in &mut vertices {
        v[0] = bounds[0] + v[0] * scale_x;
        v[1] = bounds[1] + v[1] * scale_y;
        v[2] = bounds[2] + v[2] * scale_z;
    }

    Some(Mesh { vertices, faces })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_dmc_treats_nan_background_as_outside() {
        let size = [7usize, 7, 7];
        let mut values = vec![f32::NAN; size[0] * size[1] * size[2]];
        let c = 3usize;
        for z in 0..size[2] {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    let dx = x as f32 - c as f32;
                    let dy = y as f32 - c as f32;
                    let dz = z as f32 - c as f32;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 <= 9.0 {
                        let idx = (z * size[1] + y) * size[0] + x;
                        values[idx] = r2.sqrt() - 2.0;
                    }
                }
            }
        }

        let grid = DenseGrid {
            values,
            size,
            bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        };
        let mesh = sdf_to_mesh_diff_dmc(&grid).expect("expected non-empty mesh");
        let (min, max) = mesh_bounds(&mesh);
        for axis in 0..3 {
            assert!(
                min[axis] > -0.8,
                "min bound too low on axis {axis}: {}",
                min[axis]
            );
            assert!(
                max[axis] < 0.8,
                "max bound too high on axis {axis}: {}",
                max[axis]
            );
        }
    }

    #[test]
    fn diff_dmc_skips_cells_with_nan_corners() {
        let size = [3usize, 3, 3];
        let mut values = vec![f32::NAN; size[0] * size[1] * size[2]];
        let center = (size[1] + 1) * size[0] + 1;
        values[center] = -1.0;

        let grid = DenseGrid {
            values,
            size,
            bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        };
        let mesh = sdf_to_mesh_diff_dmc(&grid);
        assert!(
            mesh.is_none(),
            "expected no surface when every candidate cell includes NaN corners"
        );
    }

    fn mesh_bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &mesh.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        (min, max)
    }
}

fn edge_index_x(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * (nx - 1) + x
}

fn edge_index_y(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * (ny - 1) + y) * nx + x
}

fn edge_index_z(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * nx + x
}

#[allow(clippy::too_many_arguments)]
fn vertex_for_edge(
    edge: usize,
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
    _nz: usize,
    grid: &DenseGrid,
    x_edges: &mut [i32],
    y_edges: &mut [i32],
    z_edges: &mut [i32],
    vertices: &mut Vec<[f32; 3]>,
    iso: f32,
) -> u32 {
    let (p1, p2, v1, v2, slot) = match edge {
        0 => {
            let slot = edge_index_x(x, y, z, nx, ny);
            (
                [x as f32, y as f32, z as f32],
                [(x + 1) as f32, y as f32, z as f32],
                grid.value_at(x, y, z),
                grid.value_at(x + 1, y, z),
                EdgeSlot::X(slot),
            )
        }
        1 => {
            let slot = edge_index_y(x + 1, y, z, nx, ny);
            (
                [(x + 1) as f32, y as f32, z as f32],
                [(x + 1) as f32, (y + 1) as f32, z as f32],
                grid.value_at(x + 1, y, z),
                grid.value_at(x + 1, y + 1, z),
                EdgeSlot::Y(slot),
            )
        }
        2 => {
            let slot = edge_index_x(x, y + 1, z, nx, ny);
            (
                [(x + 1) as f32, (y + 1) as f32, z as f32],
                [x as f32, (y + 1) as f32, z as f32],
                grid.value_at(x + 1, y + 1, z),
                grid.value_at(x, y + 1, z),
                EdgeSlot::X(slot),
            )
        }
        3 => {
            let slot = edge_index_y(x, y, z, nx, ny);
            (
                [x as f32, (y + 1) as f32, z as f32],
                [x as f32, y as f32, z as f32],
                grid.value_at(x, y + 1, z),
                grid.value_at(x, y, z),
                EdgeSlot::Y(slot),
            )
        }
        4 => {
            let slot = edge_index_x(x, y, z + 1, nx, ny);
            (
                [x as f32, y as f32, (z + 1) as f32],
                [(x + 1) as f32, y as f32, (z + 1) as f32],
                grid.value_at(x, y, z + 1),
                grid.value_at(x + 1, y, z + 1),
                EdgeSlot::X(slot),
            )
        }
        5 => {
            let slot = edge_index_y(x + 1, y, z + 1, nx, ny);
            (
                [(x + 1) as f32, y as f32, (z + 1) as f32],
                [(x + 1) as f32, (y + 1) as f32, (z + 1) as f32],
                grid.value_at(x + 1, y, z + 1),
                grid.value_at(x + 1, y + 1, z + 1),
                EdgeSlot::Y(slot),
            )
        }
        6 => {
            let slot = edge_index_x(x, y + 1, z + 1, nx, ny);
            (
                [(x + 1) as f32, (y + 1) as f32, (z + 1) as f32],
                [x as f32, (y + 1) as f32, (z + 1) as f32],
                grid.value_at(x + 1, y + 1, z + 1),
                grid.value_at(x, y + 1, z + 1),
                EdgeSlot::X(slot),
            )
        }
        7 => {
            let slot = edge_index_y(x, y, z + 1, nx, ny);
            (
                [x as f32, (y + 1) as f32, (z + 1) as f32],
                [x as f32, y as f32, (z + 1) as f32],
                grid.value_at(x, y + 1, z + 1),
                grid.value_at(x, y, z + 1),
                EdgeSlot::Y(slot),
            )
        }
        8 => {
            let slot = edge_index_z(x, y, z, nx, ny);
            (
                [x as f32, y as f32, z as f32],
                [x as f32, y as f32, (z + 1) as f32],
                grid.value_at(x, y, z),
                grid.value_at(x, y, z + 1),
                EdgeSlot::Z(slot),
            )
        }
        9 => {
            let slot = edge_index_z(x + 1, y, z, nx, ny);
            (
                [(x + 1) as f32, y as f32, z as f32],
                [(x + 1) as f32, y as f32, (z + 1) as f32],
                grid.value_at(x + 1, y, z),
                grid.value_at(x + 1, y, z + 1),
                EdgeSlot::Z(slot),
            )
        }
        10 => {
            let slot = edge_index_z(x + 1, y + 1, z, nx, ny);
            (
                [(x + 1) as f32, (y + 1) as f32, z as f32],
                [(x + 1) as f32, (y + 1) as f32, (z + 1) as f32],
                grid.value_at(x + 1, y + 1, z),
                grid.value_at(x + 1, y + 1, z + 1),
                EdgeSlot::Z(slot),
            )
        }
        11 => {
            let slot = edge_index_z(x, y + 1, z, nx, ny);
            (
                [x as f32, (y + 1) as f32, z as f32],
                [x as f32, (y + 1) as f32, (z + 1) as f32],
                grid.value_at(x, y + 1, z),
                grid.value_at(x, y + 1, z + 1),
                EdgeSlot::Z(slot),
            )
        }
        _ => unreachable!("invalid edge"),
    };

    if let Some(existing) = slot.get(x_edges, y_edges, z_edges) {
        return existing as u32;
    }

    let vertex = interpolate_vertex(iso, p1, p2, v1, v2);
    let index = vertices.len() as u32;
    vertices.push(vertex);
    slot.set(x_edges, y_edges, z_edges, index as i32);
    index
}

fn interpolate_vertex(iso: f32, p1: [f32; 3], p2: [f32; 3], v1: f32, v2: f32) -> [f32; 3] {
    const EPS: f32 = 1e-6;
    if (iso - v1).abs() < EPS {
        return p1;
    }
    if (iso - v2).abs() < EPS {
        return p2;
    }
    if (v1 - v2).abs() < EPS {
        return p1;
    }
    let t = (iso - v1) / (v2 - v1);
    [
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    ]
}

#[derive(Clone, Copy)]
enum EdgeSlot {
    X(usize),
    Y(usize),
    Z(usize),
}

impl EdgeSlot {
    fn get(self, x_edges: &[i32], y_edges: &[i32], z_edges: &[i32]) -> Option<i32> {
        let value = match self {
            EdgeSlot::X(idx) => x_edges[idx],
            EdgeSlot::Y(idx) => y_edges[idx],
            EdgeSlot::Z(idx) => z_edges[idx],
        };
        if value >= 0 { Some(value) } else { None }
    }

    fn set(self, x_edges: &mut [i32], y_edges: &mut [i32], z_edges: &mut [i32], value: i32) {
        match self {
            EdgeSlot::X(idx) => x_edges[idx] = value,
            EdgeSlot::Y(idx) => y_edges[idx] = value,
            EdgeSlot::Z(idx) => z_edges[idx] = value,
        }
    }
}

#[derive(Clone, Copy)]
struct UsedCell {
    x: usize,
    y: usize,
    z: usize,
    code: u8,
    num_patches: usize,
    num_mc_verts: usize,
}

fn diff_dmc(grid: &[f32], dims: [usize; 3], iso: f32) -> DmcOutput {
    let [nx, ny, nz] = dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }
    let cell_nx = nx - 1;
    let cell_ny = ny - 1;
    let cell_nz = nz - 1;
    let n_cells = cell_nx * cell_ny * cell_nz;

    let mut cell_codes = vec![0u8; n_cells];
    for z in 0..cell_nz {
        for y in 0..cell_ny {
            for x in 0..cell_nx {
                let idx = cell_index(x, y, z, cell_nx, cell_ny);
                cell_codes[idx] = get_cell_code(grid, dims, x, y, z, iso);
            }
        }
    }

    let mut used_cells = Vec::new();
    let mut cell_to_used = vec![usize::MAX; n_cells];

    for z in 0..cell_nz {
        for y in 0..cell_ny {
            for x in 0..cell_nx {
                let cell_idx = cell_index(x, y, z, cell_nx, cell_ny);
                let raw = cell_codes[cell_idx];
                if raw == 0 || raw == 255 {
                    continue;
                }
                let good =
                    get_good_cell_code(&cell_codes, [cell_nx, cell_ny, cell_nz], x, y, z, raw);

                let num_patches = (MCFIRSTPATCHINDEX[good as usize + 1]
                    - MCFIRSTPATCHINDEX[good as usize]) as usize;

                let d0 = grid[grid_index(x, y, z, nx, ny)];
                let dx = grid[grid_index(x + 1, y, z, nx, ny)];
                let dy = grid[grid_index(x, y + 1, z, nx, ny)];
                let dz = grid[grid_index(x, y, z + 1, nx, ny)];
                let mut num_mc_verts = 0usize;
                if (d0 < iso && dx >= iso) || (dx < iso && d0 >= iso) {
                    num_mc_verts += 1;
                }
                if (d0 < iso && dy >= iso) || (dy < iso && d0 >= iso) {
                    num_mc_verts += 1;
                }
                if (d0 < iso && dz >= iso) || (dz < iso && d0 >= iso) {
                    num_mc_verts += 1;
                }

                let used_index = used_cells.len();
                used_cells.push(UsedCell {
                    x,
                    y,
                    z,
                    code: good,
                    num_patches,
                    num_mc_verts,
                });
                cell_to_used[cell_idx] = used_index;
            }
        }
    }

    if used_cells.is_empty() {
        return None;
    }

    let mut patch_start = Vec::with_capacity(used_cells.len());
    let mut vert_start = Vec::with_capacity(used_cells.len());
    let mut total_patches = 0usize;
    let mut total_mc_verts = 0usize;
    for cell in &used_cells {
        patch_start.push(total_patches);
        vert_start.push(total_mc_verts);
        total_patches += cell.num_patches;
        total_mc_verts += cell.num_mc_verts;
    }

    if total_patches == 0 || total_mc_verts == 0 {
        return None;
    }

    let mut verts = vec![[0.0f32; 3]; total_patches];
    for (used_index, cell) in used_cells.iter().enumerate() {
        let cube = cell.code as usize;
        let start_patch = MCFIRSTPATCHINDEX[cube] as usize;
        let end_patch = MCFIRSTPATCHINDEX[cube + 1] as usize;
        for (out, patch_index) in (patch_start[used_index]..).zip(start_patch..end_patch) {
            let edge_start = MCFIRSTEDGEINDEX[patch_index] as usize;
            let edge_end = MCFIRSTEDGEINDEX[patch_index + 1] as usize;
            let mut p = [0.0f32; 3];
            let mut count = 0.0f32;
            for &edge in MCEDGEINDEX[edge_start..edge_end].iter() {
                let eid = edge as usize;
                let loc = MCEDGELOCATIONS[eid];
                let ex = loc[0] as isize;
                let ey = loc[1] as isize;
                let ez = loc[2] as isize;
                let en = loc[3] as usize;
                let vx = (cell.x as isize + ex) as usize;
                let vy = (cell.y as isize + ey) as usize;
                let vz = (cell.z as isize + ez) as usize;
                let v = compute_mc_vert(grid, dims, vx, vy, vz, en, iso);
                p[0] += v[0];
                p[1] += v[1];
                p[2] += v[2];
                count += 1.0;
            }
            if count > 0.0 {
                p[0] /= count;
                p[1] /= count;
                p[2] /= count;
            }
            verts[out] = p;
        }
    }

    let mut mc_vert_to_cell = vec![0usize; total_mc_verts];
    let mut mc_vert_type = vec![0u8; total_mc_verts];
    for (used_index, cell) in used_cells.iter().enumerate() {
        let d0 = grid[grid_index(cell.x, cell.y, cell.z, nx, ny)];
        let ds = [
            grid[grid_index(cell.x + 1, cell.y, cell.z, nx, ny)],
            grid[grid_index(cell.x, cell.y + 1, cell.z, nx, ny)],
            grid[grid_index(cell.x, cell.y, cell.z + 1, nx, ny)],
        ];
        let mut out = vert_start[used_index];
        for (dim, &d) in ds.iter().enumerate() {
            let entering = d0 < iso && d >= iso;
            let exiting = d < iso && d0 >= iso;
            if entering || exiting {
                mc_vert_to_cell[out] = used_index;
                mc_vert_type[out] = if exiting { (3 + dim) as u8 } else { dim as u8 };
                out += 1;
            }
        }
    }

    let mut quads = Vec::with_capacity(total_mc_verts);
    for quad_index in 0..total_mc_verts {
        let used_index = mc_vert_to_cell[quad_index];
        let cell = used_cells[used_index];
        let vert_type = mc_vert_type[quad_index] as usize;
        if vert_type >= DMCQUAD.len() {
            continue;
        }
        let mut quad = [0u32; 4];
        let mut valid = true;
        for i in 0..4 {
            let entry = DMCQUAD[vert_type][i];
            let nx = cell.x as isize + entry[0] as isize;
            let ny = cell.y as isize + entry[1] as isize;
            let nz = cell.z as isize + entry[2] as isize;
            if nx < 0 || ny < 0 || nz < 0 {
                valid = false;
                break;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            let nzu = nz as usize;
            if nxu >= cell_nx || nyu >= cell_ny || nzu >= cell_nz {
                valid = false;
                break;
            }

            let neighbor_cell = cell_index(nxu, nyu, nzu, cell_nx, cell_ny);
            let neighbor_used = cell_to_used[neighbor_cell];
            if neighbor_used == usize::MAX {
                valid = false;
                break;
            }
            let neighbor_code = used_cells[neighbor_used].code as usize;
            let eid = entry[3] as usize;
            let offset = DMCEDGEOFFSET[neighbor_code][eid] as isize;
            if offset < 0 {
                valid = false;
                break;
            }
            let v_index = patch_start[neighbor_used] + offset as usize;
            if v_index >= total_patches {
                valid = false;
                break;
            }
            quad[i] = v_index as u32;
        }
        if valid {
            quads.push(quad);
        }
    }

    Some((verts, quads))
}

fn cell_index(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * nx + x
}

fn grid_index(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * nx + x
}

fn get_cell_code(grid: &[f32], dims: [usize; 3], x: usize, y: usize, z: usize, iso: f32) -> u8 {
    let mut code = 0u8;
    for (i, corner) in MCCORNERS.iter().enumerate() {
        let cx = (x as i32 + corner[0]) as usize;
        let cy = (y as i32 + corner[1]) as usize;
        let cz = (z as i32 + corner[2]) as usize;
        let v = grid[grid_index(cx, cy, cz, dims[0], dims[1])];
        if !v.is_finite() {
            return 0;
        }
        if v >= iso {
            code |= 1u8 << i;
        }
    }
    code
}

fn get_good_cell_code(
    cell_codes: &[u8],
    cell_dims: [usize; 3],
    x: usize,
    y: usize,
    z: usize,
    raw: u8,
) -> u8 {
    let direction = PROBLEMATICCONFIGS[raw as usize];
    if direction == 255 {
        return raw;
    }

    let component = (direction >> 1) as isize;
    let delta = if (direction & 1) == 1 {
        1isize
    } else {
        -1isize
    };
    let mut nx = x as isize;
    let mut ny = y as isize;
    let mut nz = z as isize;
    match component {
        0 => nx += delta,
        1 => ny += delta,
        2 => nz += delta,
        _ => {}
    }
    if nx < 0
        || ny < 0
        || nz < 0
        || nx >= cell_dims[0] as isize
        || ny >= cell_dims[1] as isize
        || nz >= cell_dims[2] as isize
    {
        return raw;
    }
    let neighbor_idx = cell_index(
        nx as usize,
        ny as usize,
        nz as usize,
        cell_dims[0],
        cell_dims[1],
    );
    let neighbor_code = cell_codes[neighbor_idx];
    if PROBLEMATICCONFIGS[neighbor_code as usize] != 255 {
        raw ^ 0xff
    } else {
        raw
    }
}

fn compute_mc_vert(
    grid: &[f32],
    dims: [usize; 3],
    x: usize,
    y: usize,
    z: usize,
    dim: usize,
    iso: f32,
) -> [f32; 3] {
    let (ox, oy, oz) = match dim {
        0 => (1, 0, 0),
        1 => (0, 1, 0),
        2 => (0, 0, 1),
        _ => (0, 0, 0),
    };
    let v0 = grid[grid_index(x, y, z, dims[0], dims[1])];
    let v1 = grid[grid_index(x + ox, y + oy, z + oz, dims[0], dims[1])];
    let mut t = if (v1 - v0).abs() > 0.0 {
        (iso - v0) / (v1 - v0)
    } else {
        0.5
    };
    t = t.clamp(0.0, 1.0);
    let p0 = [x as f32, y as f32, z as f32];
    let p1 = [(x + ox) as f32, (y + oy) as f32, (z + oz) as f32];
    [
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        p0[2] + (p1[2] - p0[2]) * t,
    ]
}

fn triangulate_quads(vertices: &[[f32; 3]], quads: &[[u32; 4]]) -> Vec<[u32; 3]> {
    let mut faces = Vec::with_capacity(quads.len() * 2);
    for quad in quads {
        let [a, b, c, d] = *quad;
        let va = vertices[a as usize];
        let vb = vertices[b as usize];
        let vc = vertices[c as usize];
        let vd = vertices[d as usize];

        let config1 = triangle_max_cos(va, vb, vd).max(triangle_max_cos(vb, vc, vd));
        let config2 = triangle_max_cos(va, vb, vc).max(triangle_max_cos(va, vc, vd));

        if config1 < config2 {
            faces.push([a, b, d]);
            faces.push([b, c, d]);
        } else {
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    faces
}

fn triangle_max_cos(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let v01 = normalize(sub(b, a));
    let v02 = normalize(sub(c, a));
    let v12 = normalize(sub(c, b));
    let v10 = normalize(sub(a, b));
    let v20 = normalize(sub(a, c));
    let v21 = normalize(sub(b, c));
    let cos1 = dot(v01, v02);
    let cos2 = dot(v12, v10);
    let cos3 = dot(v20, v21);
    cos1.max(cos2).max(cos3)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let denom = len.max(1e-12);
    [v[0] / denom, v[1] / denom, v[2] / denom]
}
