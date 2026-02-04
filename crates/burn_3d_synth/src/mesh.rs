#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshStats {
    pub vertices: usize,
    pub faces: usize,
}

pub trait MeshLike {
    fn vertices(&self) -> &[[f32; 3]];
    fn faces(&self) -> &[[u32; 3]];
}

impl MeshLike for Mesh {
    fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}

pub fn mesh_stats<M: MeshLike>(mesh: &M) -> MeshStats {
    MeshStats {
        vertices: mesh.vertices().len(),
        faces: mesh.faces().len(),
    }
}

pub fn mesh_bounds<M: MeshLike>(mesh: &M) -> Option<([f32; 3], [f32; 3])> {
    let vertices = mesh.vertices();
    let first = vertices.first()?;
    let mut min = *first;
    let mut max = *first;
    for v in vertices.iter().skip(1) {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    Some((min, max))
}

#[cfg(feature = "triposg")]
impl From<burn_3d_synth_tripo::pipeline::mesh::Mesh> for Mesh {
    fn from(value: burn_3d_synth_tripo::pipeline::mesh::Mesh) -> Self {
        Self {
            vertices: value.vertices,
            faces: value.faces,
        }
    }
}

#[cfg(feature = "triposg")]
impl MeshLike for burn_3d_synth_tripo::pipeline::mesh::Mesh {
    fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}
