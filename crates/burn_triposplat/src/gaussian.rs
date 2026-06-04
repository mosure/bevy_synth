use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

const SH_C0: f32 = 0.282_094_8;
const EPS: f32 = 1.0e-6;
const DEFAULT_TRANSFORM: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianSplat {
    pub position: [f32; 3],
    pub features_dc: [f32; 3],
    pub opacity: f32,
    pub scale: [f32; 3],
    /// Unit quaternion in upstream TripoSplat/3DGS order: `[w, x, y, z]`.
    pub rotation: [f32; 4],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GaussianSplatCloud {
    pub splats: Vec<GaussianSplat>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GaussianSplatStats {
    pub splats: usize,
    pub splat_bytes: usize,
}

impl GaussianSplatCloud {
    pub fn new(splats: Vec<GaussianSplat>) -> Self {
        Self { splats }
    }

    pub fn is_empty(&self) -> bool {
        self.splats.is_empty()
    }

    pub fn len(&self) -> usize {
        self.splats.len()
    }

    pub fn stats(&self) -> GaussianSplatStats {
        GaussianSplatStats {
            splats: self.splats.len(),
            splat_bytes: self.splats.len() * 32,
        }
    }

    pub fn canonical_debug_cloud() -> Self {
        Self::new(vec![
            GaussianSplat {
                position: [-0.25, 0.0, 0.0],
                features_dc: [0.9, 0.1, -0.1],
                opacity: 0.85,
                scale: [0.05, 0.05, 0.05],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            GaussianSplat {
                position: [0.25, 0.0, 0.0],
                features_dc: [-0.1, 0.3, 0.9],
                opacity: 0.65,
                scale: [0.08, 0.04, 0.04],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
        ])
    }

    pub fn to_splat_bytes(&self) -> Result<Vec<u8>, String> {
        self.to_splat_bytes_with_transform(DEFAULT_TRANSFORM)
    }

    pub fn to_splat_bytes_with_transform(
        &self,
        transform: [[f32; 3]; 3],
    ) -> Result<Vec<u8>, String> {
        validate_non_empty(self)?;
        let mut records = self
            .splats
            .iter()
            .enumerate()
            .map(|(index, splat)| transformed_record(index, splat, transform))
            .collect::<Result<Vec<_>, _>>()?;
        records.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out = Vec::with_capacity(records.len() * 32);
        for record in records {
            for value in record.position {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for value in record.scale {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&record.rgba);
            out.extend_from_slice(&record.rotation_u8);
        }
        Ok(out)
    }

    pub fn write_splat(&self, path: impl AsRef<Path>) -> Result<(), String> {
        ensure_parent(path.as_ref())?;
        fs::write(path.as_ref(), self.to_splat_bytes()?)
            .map_err(|err| format!("failed to write {}: {err}", path.as_ref().display()))
    }

    pub fn to_ply_bytes(&self) -> Result<Vec<u8>, String> {
        self.to_ply_bytes_with_transform(DEFAULT_TRANSFORM)
    }

    pub fn to_ply_bytes_with_transform(&self, transform: [[f32; 3]; 3]) -> Result<Vec<u8>, String> {
        validate_non_empty(self)?;
        let mut out = Vec::new();
        out.extend_from_slice(ply_header(self.splats.len()).as_bytes());
        for (index, splat) in self.splats.iter().enumerate() {
            let record = transformed_record(index, splat, transform)?;
            let opacity_logit = logit(splat.opacity);
            let scale_log = [
                splat.scale[0].max(EPS).ln(),
                splat.scale[1].max(EPS).ln(),
                splat.scale[2].max(EPS).ln(),
            ];
            for value in record.position {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for value in [0.0f32, 0.0, 0.0] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for value in splat.features_dc {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&opacity_logit.to_le_bytes());
            for value in scale_log {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for value in record.rotation {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(out)
    }

    pub fn write_ply(&self, path: impl AsRef<Path>) -> Result<(), String> {
        ensure_parent(path.as_ref())?;
        fs::write(path.as_ref(), self.to_ply_bytes()?)
            .map_err(|err| format!("failed to write {}: {err}", path.as_ref().display()))
    }
}

#[derive(Clone, Copy, Debug)]
struct SplatRecord {
    position: [f32; 3],
    scale: [f32; 3],
    rgba: [u8; 4],
    rotation: [f32; 4],
    rotation_u8: [u8; 4],
    weight: f32,
}

fn transformed_record(
    index: usize,
    splat: &GaussianSplat,
    transform: [[f32; 3]; 3],
) -> Result<SplatRecord, String> {
    validate_splat(index, splat)?;
    let position = mat3_vec3(transform, splat.position);
    let rotation = matrix_to_quat(mat3_mul(transform, quat_to_matrix(splat.rotation)));
    let rotation = normalize_quat(rotation);
    let opacity = splat.opacity.clamp(0.0, 1.0);
    let rgb = [
        to_u8((splat.features_dc[0] * SH_C0 + 0.5) * 255.0),
        to_u8((splat.features_dc[1] * SH_C0 + 0.5) * 255.0),
        to_u8((splat.features_dc[2] * SH_C0 + 0.5) * 255.0),
    ];
    let rgba = [rgb[0], rgb[1], rgb[2], to_u8(opacity * 255.0)];
    let rotation_u8 = [
        to_u8(rotation[0] * 128.0 + 128.0),
        to_u8(rotation[1] * 128.0 + 128.0),
        to_u8(rotation[2] * 128.0 + 128.0),
        to_u8(rotation[3] * 128.0 + 128.0),
    ];
    Ok(SplatRecord {
        position,
        scale: splat.scale,
        rgba,
        rotation,
        rotation_u8,
        weight: opacity * splat.scale[0] * splat.scale[1] * splat.scale[2],
    })
}

fn validate_non_empty(cloud: &GaussianSplatCloud) -> Result<(), String> {
    if cloud.splats.is_empty() {
        return Err("cannot export an empty Gaussian splat cloud".to_string());
    }
    Ok(())
}

fn validate_splat(index: usize, splat: &GaussianSplat) -> Result<(), String> {
    for (axis, value) in splat.position.iter().enumerate() {
        if !value.is_finite() {
            return invalid_splat(index, "position", axis, *value);
        }
    }
    for (axis, value) in splat.features_dc.iter().enumerate() {
        if !value.is_finite() {
            return invalid_splat(index, "features_dc", axis, *value);
        }
    }
    if !splat.opacity.is_finite() {
        return invalid_splat(index, "opacity", 0, splat.opacity);
    }
    for (axis, value) in splat.scale.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return invalid_splat(index, "scale", axis, *value);
        }
    }
    for (axis, value) in splat.rotation.iter().enumerate() {
        if !value.is_finite() {
            return invalid_splat(index, "rotation", axis, *value);
        }
    }
    Ok(())
}

fn invalid_splat<T>(index: usize, field: &str, component: usize, value: f32) -> Result<T, String> {
    Err(format!(
        "Gaussian splat contains non-finite or invalid values: splat_index={index} field={field} component={component} value={value}"
    ))
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    Ok(())
}

fn ply_header(num_vertices: usize) -> String {
    let mut header = String::from("ply\nformat binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {num_vertices}\n"));
    for name in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    header
}

fn to_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0) as u8
}

fn logit(value: f32) -> f32 {
    let value = value.clamp(EPS, 1.0 - EPS);
    (value / (1.0 - value)).ln()
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm <= EPS {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

fn quat_to_matrix(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = normalize_quat(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn matrix_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        return normalize_quat([
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]);
    }
    if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        return normalize_quat([
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]);
    }
    if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        return normalize_quat([
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]);
    }
    let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
    normalize_quat([
        (m[1][0] - m[0][1]) / s,
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
    ])
}

fn mat3_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_export_uses_32_byte_records() {
        let cloud = GaussianSplatCloud::canonical_debug_cloud();
        let bytes = cloud.to_splat_bytes().expect("splat bytes");
        assert_eq!(bytes.len(), cloud.len() * 32);
    }

    #[test]
    fn splat_export_sorts_by_opacity_scale_volume() {
        let cloud = GaussianSplatCloud::canonical_debug_cloud();
        let bytes = cloud.to_splat_bytes().expect("splat bytes");
        let first_alpha = bytes[27];
        let second_alpha = bytes[59];
        assert!(first_alpha >= second_alpha);
    }

    #[test]
    fn ply_export_has_gaussian_properties() {
        let cloud = GaussianSplatCloud::canonical_debug_cloud();
        let bytes = cloud.to_ply_bytes().expect("ply bytes");
        let text = String::from_utf8_lossy(
            &bytes[..bytes
                .iter()
                .position(|b| *b == 0)
                .unwrap_or(bytes.len())
                .min(512)],
        );
        assert!(text.contains("element vertex 2"));
        assert!(text.contains("property float f_dc_0"));
        assert!(text.contains("property float rot_3"));
    }
}
