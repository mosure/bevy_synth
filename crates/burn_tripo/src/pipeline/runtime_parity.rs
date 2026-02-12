use burn::prelude::Backend;
use std::collections::HashMap;

use crate::model::triposg::load_policy::{BpkPrecisionPreference, BurnpackLoadPolicy};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripoSGRuntimeParityProfile {
    pub strict_dino_preprocess: bool,
    pub strict_rmbg_interp: bool,
    pub max_image_dim: Option<usize>,
    pub burnpack_policy: BurnpackLoadPolicy,
}

pub fn triposg_runtime_parity_profile(
    match_python: bool,
    fallback_max_image_dim: Option<usize>,
) -> TripoSGRuntimeParityProfile {
    let max_image_dim = if match_python {
        Some(2000)
    } else {
        fallback_max_image_dim.filter(|value| *value > 0 && *value != usize::MAX)
    };
    let burnpack_policy = if match_python {
        BurnpackLoadPolicy::default().with_precision(BpkPrecisionPreference::PreferF32)
    } else {
        BurnpackLoadPolicy::default()
    };

    TripoSGRuntimeParityProfile {
        strict_dino_preprocess: true,
        strict_rmbg_interp: true,
        max_image_dim,
        burnpack_policy,
    }
}

pub fn configure_triposg_parity_env(match_python: bool, fallback_max_image_dim: Option<usize>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let profile = triposg_runtime_parity_profile(match_python, fallback_max_image_dim);
        if std::env::var("DINO_STRICT_PREPROCESS").is_err() && profile.strict_dino_preprocess {
            unsafe {
                std::env::set_var("DINO_STRICT_PREPROCESS", "1");
            }
        }
        if std::env::var("RMBG_STRICT_INTERP").is_err() && profile.strict_rmbg_interp {
            unsafe {
                std::env::set_var("RMBG_STRICT_INTERP", "1");
            }
        }
        if std::env::var("TRIPOSG_MAX_IMAGE_DIM").is_err()
            && let Some(max_dim) = profile.max_image_dim
        {
            unsafe {
                std::env::set_var("TRIPOSG_MAX_IMAGE_DIM", max_dim.to_string());
            }
        }
        if !profile.burnpack_policy.precision.prefer_f16() {
            if std::env::var("TRIPOSG_BPK_PRECISION").is_err() {
                unsafe {
                    std::env::set_var("TRIPOSG_BPK_PRECISION", "f32");
                }
            }
            if std::env::var("BURN_SYNTH_BPK_PRECISION").is_err() {
                unsafe {
                    std::env::set_var("BURN_SYNTH_BPK_PRECISION", "f32");
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
    let (welded_vertices, welded_indices) = weld_close_vertices(&mesh.vertices, &indices);
    if welded_vertices.is_empty() || welded_indices.len() < 3 {
        return Ok(mesh.clone());
    }

    let indices = welded_indices;
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(welded_vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let target_error = std::env::var("TRIPOSG_DECIMATE_TARGET_ERROR")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.02)
        .clamp(1e-6, 1.0);
    let allow_sloppy = std::env::var("TRIPOSG_DECIMATE_ALLOW_SLOPPY")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false);

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        target_error,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if allow_sloppy && simplified.len() > target_index_count {
        simplified =
            meshopt::simplify_sloppy(&indices, &adapter, target_index_count, target_error, None);
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(welded_vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(welded_vertices.as_slice(), vertex_count, &remap);
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

fn weld_close_vertices(vertices: &[[f32; 3]], indices: &[u32]) -> (Vec<[f32; 3]>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let diag =
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    let tolerance = std::env::var("TRIPOSG_DECIMATE_WELD_TOLERANCE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or_else(|| (diag * 1e-6).max(1e-8));
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return (vertices.to_vec(), indices.to_vec());
    }

    let inv_tol = 1.0 / tolerance;
    let mut key_to_new = HashMap::<(i64, i64, i64), u32>::new();
    let mut old_to_new = vec![0u32; vertices.len()];
    let mut welded = Vec::<[f32; 3]>::with_capacity(vertices.len());

    for (idx, vertex) in vertices.iter().enumerate() {
        let key = (
            (vertex[0] * inv_tol).round() as i64,
            (vertex[1] * inv_tol).round() as i64,
            (vertex[2] * inv_tol).round() as i64,
        );
        let entry = key_to_new.entry(key).or_insert_with(|| {
            let next = welded.len() as u32;
            welded.push(*vertex);
            next
        });
        old_to_new[idx] = *entry;
    }

    let mut welded_indices = Vec::<u32>::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let a = old_to_new[tri[0] as usize];
        let b = old_to_new[tri[1] as usize];
        let c = old_to_new[tri[2] as usize];
        if a == b || b == c || a == c {
            continue;
        }
        welded_indices.push(a);
        welded_indices.push(b);
        welded_indices.push(c);
    }

    (welded, welded_indices)
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

    #[test]
    fn parity_profile_match_python_prefers_f32() {
        let profile = super::triposg_runtime_parity_profile(true, Some(777));
        assert!(profile.strict_dino_preprocess);
        assert!(profile.strict_rmbg_interp);
        assert_eq!(profile.max_image_dim, Some(2000));
        assert!(!profile.burnpack_policy.precision.prefer_f16());
    }

    #[test]
    fn parity_profile_non_python_prefers_f16() {
        let profile = super::triposg_runtime_parity_profile(false, Some(777));
        assert!(profile.strict_dino_preprocess);
        assert!(profile.strict_rmbg_interp);
        assert_eq!(profile.max_image_dim, Some(777));
        assert!(profile.burnpack_policy.precision.prefer_f16());
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
