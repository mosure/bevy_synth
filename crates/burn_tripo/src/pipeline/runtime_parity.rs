use burn::prelude::Backend;

use crate::pipeline::mesh::Mesh;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DinoBackendChoice {
    Auto,
    Cpu,
    Gpu,
}

impl DinoBackendChoice {
    pub fn resolve<B: Backend>(self, match_python: bool) -> Self {
        resolve_dino_backend::<B>(self, match_python)
    }
}

pub fn resolve_dino_backend<B: Backend>(
    requested: DinoBackendChoice,
    match_python: bool,
) -> DinoBackendChoice {
    match requested {
        DinoBackendChoice::Auto => {
            if cfg!(target_arch = "wasm32") {
                if is_gpu_backend::<B>() {
                    DinoBackendChoice::Gpu
                } else {
                    DinoBackendChoice::Cpu
                }
            } else if is_gpu_backend::<B>() {
                if match_python && is_wgpu_backend::<B>() {
                    DinoBackendChoice::Cpu
                } else {
                    DinoBackendChoice::Gpu
                }
            } else {
                DinoBackendChoice::Cpu
            }
        }
        other => other,
    }
}

pub fn should_use_cpu_dino_backend<B: Backend>(
    requested: DinoBackendChoice,
    match_python: bool,
) -> bool {
    matches!(
        resolve_dino_backend::<B>(requested, match_python),
        DinoBackendChoice::Cpu
    )
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

pub fn configure_triposg_parity_env(match_python: bool, fallback_max_image_dim: Option<usize>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::env::var("DINO_STRICT_PREPROCESS").is_err() {
            unsafe {
                std::env::set_var("DINO_STRICT_PREPROCESS", "1");
            }
        }
        if std::env::var("RMBG_STRICT_INTERP").is_err() {
            unsafe {
                std::env::set_var("RMBG_STRICT_INTERP", "1");
            }
        }
        if std::env::var("TRIPOSG_MAX_IMAGE_DIM").is_err() {
            if match_python {
                unsafe {
                    std::env::set_var("TRIPOSG_MAX_IMAGE_DIM", "2000");
                }
            } else if let Some(max_dim) =
                fallback_max_image_dim.filter(|value| *value > 0 && *value != usize::MAX)
            {
                unsafe {
                    std::env::set_var("TRIPOSG_MAX_IMAGE_DIM", max_dim.to_string());
                }
            }
        }
    }
}

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

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        1.0,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(&indices, &adapter, target_index_count, 1.0, None);
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

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::{DinoBackendChoice, resolve_dino_backend};

    #[test]
    fn auto_dino_backend_resolves_to_cpu_on_ndarray() {
        assert_eq!(
            resolve_dino_backend::<NdArray<f32>>(DinoBackendChoice::Auto, true),
            DinoBackendChoice::Cpu
        );
        assert_eq!(
            resolve_dino_backend::<NdArray<f32>>(DinoBackendChoice::Auto, false),
            DinoBackendChoice::Cpu
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn auto_dino_backend_uses_cpu_only_for_python_parity_on_wgpu() {
        type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;
        assert_eq!(
            resolve_dino_backend::<WgpuBackend>(DinoBackendChoice::Auto, true),
            DinoBackendChoice::Cpu
        );
        assert_eq!(
            resolve_dino_backend::<WgpuBackend>(DinoBackendChoice::Auto, false),
            DinoBackendChoice::Gpu
        );
    }
}
