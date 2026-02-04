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
                if v0 < iso { cube_index |= 1; }
                if v1 < iso { cube_index |= 2; }
                if v2 < iso { cube_index |= 4; }
                if v3 < iso { cube_index |= 8; }
                if v4 < iso { cube_index |= 16; }
                if v5 < iso { cube_index |= 32; }
                if v6 < iso { cube_index |= 64; }
                if v7 < iso { cube_index |= 128; }

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
                    faces.push([
                        edge_vertices[a],
                        edge_vertices[b],
                        edge_vertices[c],
                    ]);
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
