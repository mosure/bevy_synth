use burn::prelude::Backend;

use crate::model::triposg::load_policy::{BpkPrecisionPreference, BurnpackLoadPolicy};
use crate::pipeline::mesh::Mesh;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DinoBackendChoice {
    Auto,
    Cpu,
    Gpu,
}

impl DinoBackendChoice {
    pub fn resolve<B: Backend>(self) -> Self {
        resolve_dino_backend::<B>(self)
    }
}

pub fn resolve_dino_backend<B: Backend>(requested: DinoBackendChoice) -> DinoBackendChoice {
    match requested {
        DinoBackendChoice::Auto => {
            if cfg!(target_arch = "wasm32") {
                if is_gpu_backend::<B>() {
                    DinoBackendChoice::Gpu
                } else {
                    DinoBackendChoice::Cpu
                }
            } else if is_gpu_backend::<B>() {
                DinoBackendChoice::Gpu
            } else {
                DinoBackendChoice::Cpu
            }
        }
        other => other,
    }
}

pub fn should_use_cpu_dino_backend<B: Backend>(requested: DinoBackendChoice) -> bool {
    matches!(resolve_dino_backend::<B>(requested), DinoBackendChoice::Cpu)
}

pub fn is_wgpu_backend<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("wgpu")
}

pub fn is_cuda_backend<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("cuda")
}

pub fn is_gpu_backend<B: Backend>() -> bool {
    is_wgpu_backend::<B>() || is_cuda_backend::<B>()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripoSGRuntimeParityProfile {
    pub strict_dino_preprocess: bool,
    pub strict_rmbg_interp: bool,
    pub max_image_dim: Option<usize>,
    pub burnpack_policy: BurnpackLoadPolicy,
}

pub fn triposg_runtime_profile(
    fallback_max_image_dim: Option<usize>,
) -> TripoSGRuntimeParityProfile {
    let max_image_dim = fallback_max_image_dim.filter(|value| *value > 0 && *value != usize::MAX);
    let burnpack_policy =
        BurnpackLoadPolicy::default().with_precision(BpkPrecisionPreference::PreferF32);

    TripoSGRuntimeParityProfile {
        strict_dino_preprocess: true,
        strict_rmbg_interp: true,
        max_image_dim,
        burnpack_policy,
    }
}

pub fn should_prefer_f16_triposg_weights(parity: TripoSGRuntimeParityProfile) -> bool {
    parity.burnpack_policy.precision.prefer_f16()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn decimate_tripo_mesh(mesh: &Mesh, target_faces: usize) -> Result<Mesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(mesh.vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    // Start from low-error simplification and only relax if we can't reach target count.
    // This avoids aggressive topology collapse at moderate face budgets (e.g. 10k).
    let mut result_error = 0.0f32;
    let mut simplified = Vec::<u32>::new();
    for error_limit in [0.02f32, 0.05, 0.1, 0.25, 0.5, 1.0] {
        let mut stage_error = 0.0f32;
        let candidate = meshopt::simplify(
            &indices,
            &adapter,
            target_index_count,
            error_limit,
            meshopt::SimplifyOptions::None,
            Some(&mut stage_error),
        );
        if candidate.len() < 3 {
            continue;
        }
        result_error = stage_error;
        simplified = candidate;
        if simplified.len() <= target_index_count {
            break;
        }
    }
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(
            &indices,
            &adapter,
            target_index_count,
            result_error.max(0.25),
            None,
        );
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(mesh.vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(mesh.vertices.as_slice(), vertex_count, &remap);
    let indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();
    Ok(Mesh { vertices, faces })
}

#[cfg(target_arch = "wasm32")]
pub fn decimate_tripo_mesh(mesh: &Mesh, target_faces: usize) -> Result<Mesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    // Keep wasm path free of meshopt C++ symbols that force wasm `env` shims.
    // Use deterministic face-stride selection and compact vertices afterward.
    let step = mesh.faces.len().div_ceil(target_faces).max(1);
    let mut selected = Vec::with_capacity(target_faces.min(mesh.faces.len()));
    let mut face_index = 0usize;
    while face_index < mesh.faces.len() && selected.len() < target_faces {
        selected.push(mesh.faces[face_index]);
        face_index += step;
    }
    if selected.is_empty() {
        return Err("wasm decimation produced empty face set".to_string());
    }

    remap_decimated_faces(mesh, &selected)
}

#[cfg(target_arch = "wasm32")]
fn remap_decimated_faces(mesh: &Mesh, selected: &[[u32; 3]]) -> Result<Mesh, String> {
    use std::collections::HashMap;

    let mut remap = HashMap::<u32, u32>::new();
    let mut vertices = Vec::<[f32; 3]>::new();
    let mut faces = Vec::<[u32; 3]>::with_capacity(selected.len());

    for face in selected {
        let mut out = [0u32; 3];
        for (slot, src_index) in face.iter().enumerate() {
            let src = *src_index as usize;
            if src >= mesh.vertices.len() {
                return Err(format!(
                    "wasm decimation index out of bounds: {src} >= {}",
                    mesh.vertices.len()
                ));
            }
            let dst = if let Some(existing) = remap.get(src_index) {
                *existing
            } else {
                let new_index = vertices.len() as u32;
                vertices.push(mesh.vertices[src]);
                remap.insert(*src_index, new_index);
                new_index
            };
            out[slot] = dst;
        }
        faces.push(out);
    }

    if faces.is_empty() || vertices.is_empty() {
        return Err("wasm decimation produced empty mesh".to_string());
    }

    Ok(Mesh { vertices, faces })
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::{
        DinoBackendChoice, decimate_tripo_mesh, resolve_dino_backend,
        should_prefer_f16_triposg_weights, triposg_runtime_profile,
    };

    #[test]
    fn auto_dino_backend_resolves_to_cpu_on_ndarray() {
        assert_eq!(
            resolve_dino_backend::<NdArray<f32>>(DinoBackendChoice::Auto),
            DinoBackendChoice::Cpu
        );
    }

    #[test]
    fn parity_profile_has_strict_defaults() {
        let profile = triposg_runtime_profile(Some(777));
        assert!(profile.strict_dino_preprocess);
        assert!(profile.strict_rmbg_interp);
        assert_eq!(profile.max_image_dim, Some(777));
        assert!(!profile.burnpack_policy.precision.prefer_f16());
    }

    #[test]
    fn default_profile_prefers_f32_weights() {
        let profile = triposg_runtime_profile(None);
        assert!(!should_prefer_f16_triposg_weights(profile));
    }

    #[test]
    fn explicit_f16_profile_is_respected() {
        use crate::model::triposg::load_policy::BpkPrecisionPreference;

        let mut profile = triposg_runtime_profile(None);
        profile.burnpack_policy = profile
            .burnpack_policy
            .with_precision(BpkPrecisionPreference::PreferF16);
        assert!(should_prefer_f16_triposg_weights(profile));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn auto_dino_backend_uses_gpu_on_wgpu() {
        type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;
        assert_eq!(
            resolve_dino_backend::<WgpuBackend>(DinoBackendChoice::Auto),
            DinoBackendChoice::Gpu
        );
    }

    #[test]
    fn decimation_reduces_faces_and_preserves_index_bounds() {
        let n = 24usize;
        let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
        let mut faces = Vec::with_capacity(n * n * 2);
        for y in 0..=n {
            for x in 0..=n {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let i0 = (y * (n + 1) + x) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + (n + 1) as u32;
                let i3 = i2 + 1;
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }
        let original_faces = faces.len();
        let mesh = crate::pipeline::mesh::Mesh { vertices, faces };
        let decimated = decimate_tripo_mesh(&mesh, 200).expect("decimation should succeed");
        assert!(decimated.faces.len() <= 200);
        assert!(!decimated.faces.is_empty());
        assert!(decimated.faces.len() < original_faces);
        let vertex_count = decimated.vertices.len() as u32;
        for face in &decimated.faces {
            assert!(face[0] < vertex_count);
            assert!(face[1] < vertex_count);
            assert!(face[2] < vertex_count);
        }
    }
}
