use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::Path;

const NORMAL_RENDER_SIZE: u32 = 256;
const NORMAL_ALPHA_THRESHOLD: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl RenderAabb {
    pub fn size(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DepthNormalIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DepthMapView<'a> {
    pub depth_m: &'a [f32],
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryMaskView<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshNormalInput<'a> {
    pub vertices: &'a [[f32; 3]],
    pub faces: &'a [[u32; 3]],
    pub normals: Option<&'a [[f32; 3]]>,
}

#[derive(Clone, Copy, Debug)]
pub struct SourceDepthNormalInput<'a> {
    pub depth_map: DepthMapView<'a>,
    pub mask: BinaryMaskView<'a>,
    pub bbox: [f32; 4],
    pub intrinsics: DepthNormalIntrinsics,
    pub depth_sidecar_label: Option<&'a str>,
    pub mask_kind: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct CanonicalPoseCamera {
    pub eye: [f32; 3],
    pub focus: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub vertical_fov_degrees: f32,
}

impl CanonicalPoseCamera {
    pub fn from_asset_aabb(local_aabb: RenderAabb) -> Self {
        let size = local_aabb.size();
        let radius = size[0].max(size[1]).max(size[2]).max(0.75) * 2.45;
        let focus = [0.0, (size[1] * 0.48).max(0.15), 0.0];
        let eye = [0.0, focus[1] + 0.25, radius];
        Self::look_at(eye, focus, 42.0)
    }

    pub fn look_at(eye: [f32; 3], focus: [f32; 3], vertical_fov_degrees: f32) -> Self {
        let forward = normalize3(sub3(focus, eye)).unwrap_or([0.0, 0.0, -1.0]);
        let world_up = [0.0, 1.0, 0.0];
        let right = normalize3(cross3(forward, world_up)).unwrap_or([1.0, 0.0, 0.0]);
        let up = normalize3(cross3(right, forward)).unwrap_or(world_up);
        Self {
            eye,
            focus,
            right,
            up,
            forward,
            vertical_fov_degrees,
        }
    }

    pub fn world_to_camera(self, point: [f32; 3]) -> [f32; 3] {
        let rel = sub3(point, self.eye);
        [
            dot3(rel, self.right),
            dot3(rel, self.up),
            dot3(rel, self.forward),
        ]
    }

    pub fn world_normal_to_encoded_camera(self, normal: [f32; 3]) -> Option<[f32; 3]> {
        let normal = normalize3(normal)?;
        normalize3([
            dot3(normal, self.right),
            dot3(normal, self.up),
            dot3(normal, mul3(self.forward, -1.0)),
        ])
    }
}

pub fn canonical_pose_asset_translation(local_aabb: RenderAabb) -> [f32; 3] {
    let center = [
        (local_aabb.min[0] + local_aabb.max[0]) * 0.5,
        (local_aabb.min[1] + local_aabb.max[1]) * 0.5,
        (local_aabb.min[2] + local_aabb.max[2]) * 0.5,
    ];
    [-center[0], -local_aabb.min[1], -center[2]]
}

pub fn write_source_depth_normal_evidence(
    input: SourceDepthNormalInput<'_>,
    output_path: &Path,
) -> Result<Value, String> {
    let render =
        render_source_depth_normals(input.depth_map, input.mask, input.bbox, input.intrinsics);
    if render.covered_px == 0 {
        return Err("source depth normal render produced no covered pixels".to_string());
    }
    write_normal_map_png(output_path, &render)?;
    Ok(json!({
        "kind": "source_depth_normal",
        "path": output_path.display().to_string(),
        "width": render.width,
        "height": render.height,
        "covered_px": render.covered_px,
        "coverage": render.covered_px as f32 / (render.width * render.height).max(1) as f32,
        "bbox": input.bbox,
        "mean_normal": render.mean_normal,
        "source": {
            "depth_sidecar": input.depth_sidecar_label,
            "mask_kind": input.mask_kind,
        },
    }))
}

pub fn write_candidate_mesh_normal_render(
    mesh: MeshNormalInput<'_>,
    local_aabb: RenderAabb,
    yaw_degrees: f32,
    output_path: &Path,
) -> Result<Value, String> {
    let render = render_candidate_mesh_normals(mesh, local_aabb, yaw_degrees);
    if render.covered_px == 0 {
        return Err("candidate normal render produced no covered pixels".to_string());
    }
    write_normal_map_png(output_path, &render)?;
    Ok(json!({
        "kind": "candidate_mesh_normal",
        "path": output_path.display().to_string(),
        "width": render.width,
        "height": render.height,
        "covered_px": render.covered_px,
        "coverage": render.covered_px as f32 / (render.width * render.height).max(1) as f32,
        "mean_normal": render.mean_normal,
        "yaw_degrees": yaw_degrees,
        "camera": render.camera,
    }))
}

pub fn normal_map_similarity(source_path: &Path, candidate_path: &Path) -> Result<Value, String> {
    let source = load_normal_map(source_path)?;
    let candidate = load_normal_map(candidate_path)?;
    let source_mean = source.mean_normal;
    let candidate_mean = candidate.mean_normal;
    let mean_alignment = normal_dot_score(source_mean, candidate_mean);
    let width = source.width.min(candidate.width);
    let height = source.height.min(candidate.height);
    let mut overlap = 0usize;
    let mut source_count = 0usize;
    let mut candidate_count = 0usize;
    let mut dot_sum = 0.0f32;
    let mut mse_sum = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let source_index = (y * source.width + x) as usize;
            let candidate_index = (y * candidate.width + x) as usize;
            let source_valid = source.mask[source_index] != 0;
            let candidate_valid = candidate.mask[candidate_index] != 0;
            source_count += usize::from(source_valid);
            candidate_count += usize::from(candidate_valid);
            if !source_valid || !candidate_valid {
                continue;
            }
            let source_normal = source.normals[source_index];
            let candidate_normal = candidate.normals[candidate_index];
            overlap += 1;
            dot_sum += dot3(source_normal, candidate_normal).clamp(-1.0, 1.0);
            let diff = sub3(source_normal, candidate_normal);
            mse_sum += dot3(diff, diff) / 3.0;
        }
    }
    let overlap_fraction = overlap as f32 / source_count.max(candidate_count).max(1) as f32;
    let pixel_alignment = if overlap > 0 {
        ((dot_sum / overlap as f32) + 1.0) * 0.5
    } else {
        mean_alignment
    };
    let normal_mse = if overlap > 0 {
        mse_sum / overlap as f32
    } else {
        2.0
    };
    let score =
        (0.56 * pixel_alignment + 0.34 * mean_alignment + 0.10 * overlap_fraction).clamp(0.0, 1.0);
    Ok(json!({
        "score": score,
        "pixel_alignment": pixel_alignment,
        "mean_alignment": mean_alignment,
        "overlap_fraction": overlap_fraction,
        "normal_mse": normal_mse,
        "overlap_px": overlap,
        "source_px": source_count,
        "candidate_px": candidate_count,
        "source_mean_normal": source_mean,
        "candidate_mean_normal": candidate_mean,
        "descriptor": "depth_normal_vs_mesh_normal",
    }))
}

#[derive(Clone)]
struct NormalRender {
    width: u32,
    height: u32,
    normals: Vec<[f32; 3]>,
    mask: Vec<u8>,
    covered_px: usize,
    mean_normal: [f32; 3],
    camera: Value,
}

fn render_source_depth_normals(
    depth_map: DepthMapView<'_>,
    mask: BinaryMaskView<'_>,
    bbox: [f32; 4],
    intrinsics: DepthNormalIntrinsics,
) -> NormalRender {
    let width = NORMAL_RENDER_SIZE;
    let height = NORMAL_RENDER_SIZE;
    let len = (width * height) as usize;
    let mut normals = vec![[0.0, 0.0, 1.0]; len];
    let mut out_mask = vec![0_u8; len];
    let mut mean = [0.0f32; 3];
    let mut covered = 0usize;
    for y in 0..height {
        for x in 0..width {
            let u = bbox[0] + ((x as f32 + 0.5) / width as f32) * (bbox[2] - bbox[0]);
            let v = bbox[1] + ((y as f32 + 0.5) / height as f32) * (bbox[3] - bbox[1]);
            if !mask_contains_normalized(mask, u, v) {
                continue;
            }
            let px =
                (u.clamp(0.0, 1.0) * (depth_map.width.saturating_sub(1)) as f32).round() as usize;
            let py =
                (v.clamp(0.0, 1.0) * (depth_map.height.saturating_sub(1)) as f32).round() as usize;
            let Some(normal) = source_depth_normal_at(depth_map, intrinsics, px, py) else {
                continue;
            };
            let encoded =
                normalize3([normal[0], -normal[1], -normal[2]]).unwrap_or([0.0, 0.0, 1.0]);
            let index = (y * width + x) as usize;
            normals[index] = encoded;
            out_mask[index] = 255;
            mean = add3(mean, encoded);
            covered += 1;
        }
    }
    let mean_normal = normalize3(mean).unwrap_or([0.0, 0.0, 1.0]);
    NormalRender {
        width,
        height,
        normals,
        mask: out_mask,
        covered_px: covered,
        mean_normal,
        camera: json!({
            "type": "source_depth_pro_camera",
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        }),
    }
}

fn render_candidate_mesh_normals(
    mesh: MeshNormalInput<'_>,
    local_aabb: RenderAabb,
    yaw_degrees: f32,
) -> NormalRender {
    let width = NORMAL_RENDER_SIZE;
    let height = NORMAL_RENDER_SIZE;
    let len = (width * height) as usize;
    let mut normals = vec![[0.0, 0.0, 1.0]; len];
    let mut out_mask = vec![0_u8; len];
    let mut depth = vec![f32::INFINITY; len];
    let mut mean = [0.0f32; 3];
    let mut covered = 0usize;
    let camera = CanonicalPoseCamera::from_asset_aabb(local_aabb);
    let translation = canonical_pose_asset_translation(local_aabb);
    let vertex_normals = mesh_vertex_normals(mesh);
    for face in mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let mut camera_points = [[0.0f32; 3]; 3];
        let mut pixels = [[0.0f32; 2]; 3];
        let mut world_normals = [[0.0f32; 3]; 3];
        let mut valid = true;
        for (slot, index) in indices.iter().copied().enumerate() {
            let world =
                canonical_pose_transform_point(mesh.vertices[index], yaw_degrees, translation);
            let camera_point = camera.world_to_camera(world);
            let Some(pixel) = project_camera_point(camera_point, camera, width, height) else {
                valid = false;
                break;
            };
            camera_points[slot] = camera_point;
            pixels[slot] = pixel;
            world_normals[slot] =
                canonical_pose_transform_normal(vertex_normals[index], yaw_degrees);
        }
        if !valid {
            continue;
        }
        rasterize_normal_triangle(
            pixels,
            [
                camera_points[0][2],
                camera_points[1][2],
                camera_points[2][2],
            ],
            world_normals,
            camera,
            &mut normals,
            &mut out_mask,
            &mut depth,
            width,
            height,
        );
    }
    for (normal, valid) in normals.iter().copied().zip(&out_mask) {
        if *valid == 0 {
            continue;
        }
        mean = add3(mean, normal);
        covered += 1;
    }
    let mean_normal = normalize3(mean).unwrap_or([0.0, 0.0, 1.0]);
    NormalRender {
        width,
        height,
        normals,
        mask: out_mask,
        covered_px: covered,
        mean_normal,
        camera: json!({
            "type": "canonical_pose_asset_camera",
            "eye": camera.eye,
            "focus": camera.focus,
            "vertical_fov_degrees": camera.vertical_fov_degrees,
        }),
    }
}

fn rasterize_normal_triangle(
    pixels: [[f32; 2]; 3],
    depths: [f32; 3],
    normals_world: [[f32; 3]; 3],
    camera: CanonicalPoseCamera,
    normal_buffer: &mut [[f32; 3]],
    mask: &mut [u8],
    depth_buffer: &mut [f32],
    width: u32,
    height: u32,
) {
    let min_x = pixels
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = pixels
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(width.saturating_sub(1) as f32) as u32;
    let min_y = pixels
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = pixels
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(height.saturating_sub(1) as f32) as u32;
    if max_x < min_x || max_y < min_y {
        return;
    }
    let area = edge2(pixels[0], pixels[1], pixels[2]);
    if area.abs() <= 1.0e-6 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge2(pixels[1], pixels[2], p) / area;
            let w1 = edge2(pixels[2], pixels[0], p) / area;
            let w2 = edge2(pixels[0], pixels[1], p) / area;
            if w0 < -1.0e-4 || w1 < -1.0e-4 || w2 < -1.0e-4 {
                continue;
            }
            let z = depths[0] * w0 + depths[1] * w1 + depths[2] * w2;
            if !z.is_finite() || z <= 0.0 {
                continue;
            }
            let index = (y * width + x) as usize;
            if z >= depth_buffer[index] {
                continue;
            }
            let normal_world = normalize3(add3(
                add3(mul3(normals_world[0], w0), mul3(normals_world[1], w1)),
                mul3(normals_world[2], w2),
            ))
            .unwrap_or([0.0, 1.0, 0.0]);
            let Some(encoded) = camera.world_normal_to_encoded_camera(normal_world) else {
                continue;
            };
            depth_buffer[index] = z;
            normal_buffer[index] = encoded;
            mask[index] = 255;
        }
    }
}

fn write_normal_map_png(path: &Path, render: &NormalRender) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut image = image::RgbaImage::new(render.width, render.height);
    for y in 0..render.height {
        for x in 0..render.width {
            let index = (y * render.width + x) as usize;
            let n = render.normals[index];
            image.put_pixel(
                x,
                y,
                image::Rgba([
                    encode_normal_channel(n[0]),
                    encode_normal_channel(n[1]),
                    encode_normal_channel(n[2]),
                    render.mask[index],
                ]),
            );
        }
    }
    image
        .save(path)
        .map_err(|err| format!("failed to write normal map {}: {err}", path.display()))
}

fn load_normal_map(path: &Path) -> Result<NormalRender, String> {
    let image = image::open(path)
        .map_err(|err| format!("failed to open normal map {}: {err}", path.display()))?
        .to_rgba8();
    let width = image.width();
    let height = image.height();
    let len = (width * height) as usize;
    let mut normals = vec![[0.0, 0.0, 1.0]; len];
    let mut mask = vec![0_u8; len];
    let mut mean = [0.0f32; 3];
    let mut covered = 0usize;
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = image.get_pixel(x, y).0;
            if pixel[3] <= NORMAL_ALPHA_THRESHOLD {
                continue;
            }
            let normal = normalize3([
                decode_normal_channel(pixel[0]),
                decode_normal_channel(pixel[1]),
                decode_normal_channel(pixel[2]),
            ])
            .unwrap_or([0.0, 0.0, 1.0]);
            normals[index] = normal;
            mask[index] = 255;
            mean = add3(mean, normal);
            covered += 1;
        }
    }
    Ok(NormalRender {
        width,
        height,
        normals,
        mask,
        covered_px: covered,
        mean_normal: normalize3(mean).unwrap_or([0.0, 0.0, 1.0]),
        camera: Value::Null,
    })
}

fn source_depth_normal_at(
    depth_map: DepthMapView<'_>,
    intrinsics: DepthNormalIntrinsics,
    px: usize,
    py: usize,
) -> Option<[f32; 3]> {
    let x0 = px.saturating_sub(1);
    let x1 = (px + 1).min(depth_map.width.saturating_sub(1));
    let y0 = py.saturating_sub(1);
    let y1 = (py + 1).min(depth_map.height.saturating_sub(1));
    if x0 == x1 || y0 == y1 {
        return None;
    }
    let center = depth_point(depth_map, intrinsics, px, py)?;
    let right = depth_point(depth_map, intrinsics, x1, py)?;
    let down = depth_point(depth_map, intrinsics, px, y1)?;
    let left = depth_point(depth_map, intrinsics, x0, py).unwrap_or(center);
    let up = depth_point(depth_map, intrinsics, px, y0).unwrap_or(center);
    let dx = sub3(right, left);
    let dy = sub3(down, up);
    let normal = normalize3(cross3(dy, dx))?;
    if normal.iter().all(|value| value.is_finite()) {
        Some(normal)
    } else {
        None
    }
}

fn depth_point(
    depth_map: DepthMapView<'_>,
    intrinsics: DepthNormalIntrinsics,
    px: usize,
    py: usize,
) -> Option<[f32; 3]> {
    let depth = depth_map
        .depth_m
        .get(py.checked_mul(depth_map.width)?.checked_add(px)?)?;
    if !depth.is_finite() || *depth <= 0.0 {
        return None;
    }
    let x = ((px as f32 + 0.5) - intrinsics.cx) / intrinsics.fx * *depth;
    let y = ((py as f32 + 0.5) - intrinsics.cy) / intrinsics.fy * *depth;
    Some([x, y, *depth])
}

fn mask_contains_normalized(mask: BinaryMaskView<'_>, u: f32, v: f32) -> bool {
    if mask.width == 0 || mask.height == 0 {
        return false;
    }
    let x = (u.clamp(0.0, 1.0) * (mask.width.saturating_sub(1)) as f32).round() as u32;
    let y = (v.clamp(0.0, 1.0) * (mask.height.saturating_sub(1)) as f32).round() as u32;
    mask.data
        .get(y as usize * mask.width as usize + x as usize)
        .copied()
        .unwrap_or(0)
        != 0
}

fn mesh_vertex_normals(mesh: MeshNormalInput<'_>) -> Vec<[f32; 3]> {
    if let Some(normals) = mesh.normals
        && normals.len() == mesh.vertices.len()
        && normals
            .iter()
            .all(|normal| normal.iter().all(|value| value.is_finite()))
    {
        return normals
            .iter()
            .copied()
            .map(|normal| normalize3(normal).unwrap_or([0.0, 1.0, 0.0]))
            .collect();
    }
    let mut normals = vec![[0.0f32; 3]; mesh.vertices.len()];
    for face in mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let a = mesh.vertices[indices[0]];
        let b = mesh.vertices[indices[1]];
        let c = mesh.vertices[indices[2]];
        let Some(normal) = normalize3(cross3(sub3(b, a), sub3(c, a))) else {
            continue;
        };
        for index in indices {
            normals[index] = add3(normals[index], normal);
        }
    }
    normals
        .into_iter()
        .map(|normal| normalize3(normal).unwrap_or([0.0, 1.0, 0.0]))
        .collect()
}

fn canonical_pose_transform_point(
    local: [f32; 3],
    yaw_degrees: f32,
    translation: [f32; 3],
) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let cos = yaw.cos();
    let sin = yaw.sin();
    [
        translation[0] + local[0] * cos + local[2] * sin,
        translation[1] + local[1],
        translation[2] - local[0] * sin + local[2] * cos,
    ]
}

fn canonical_pose_transform_normal(normal: [f32; 3], yaw_degrees: f32) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let cos = yaw.cos();
    let sin = yaw.sin();
    [
        normal[0] * cos + normal[2] * sin,
        normal[1],
        -normal[0] * sin + normal[2] * cos,
    ]
}

fn project_camera_point(
    point: [f32; 3],
    camera: CanonicalPoseCamera,
    width: u32,
    height: u32,
) -> Option<[f32; 2]> {
    if !point[2].is_finite() || point[2] <= 1.0e-4 {
        return None;
    }
    let aspect = width as f32 / height.max(1) as f32;
    let tan_half = (camera.vertical_fov_degrees.to_radians() * 0.5).tan();
    let ndc_x = point[0] / (point[2] * tan_half * aspect);
    let ndc_y = point[1] / (point[2] * tan_half);
    if !ndc_x.is_finite() || !ndc_y.is_finite() {
        return None;
    }
    Some([
        (ndc_x * 0.5 + 0.5) * (width.saturating_sub(1)) as f32,
        (0.5 - ndc_y * 0.5) * (height.saturating_sub(1)) as f32,
    ])
}

fn normal_dot_score(left: [f32; 3], right: [f32; 3]) -> f32 {
    ((dot3(left, right).clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn encode_normal_channel(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
}

fn decode_normal_channel(value: u8) -> f32 {
    (value as f32 / 255.0) * 2.0 - 1.0
}

fn edge2(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn mul3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = dot3(value, value).sqrt();
    (len.is_finite() && len > 1.0e-8).then(|| [value[0] / len, value[1] / len, value[2] / len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_map_similarity_prefers_matching_normals() {
        let root = std::env::temp_dir().join(format!(
            "burn_synth_mcp_normal_similarity_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let source_path = root.join("source.png");
        let same_path = root.join("same.png");
        let opposite_path = root.join("opposite.png");
        write_constant_normal(&source_path, [0.0, 0.0, 1.0]);
        write_constant_normal(&same_path, [0.0, 0.0, 1.0]);
        write_constant_normal(&opposite_path, [0.0, 0.0, -1.0]);

        let same = normal_map_similarity(&source_path, &same_path).unwrap();
        let opposite = normal_map_similarity(&source_path, &opposite_path).unwrap();
        assert!(same["score"].as_f64().unwrap() > 0.98);
        assert!(opposite["score"].as_f64().unwrap() < 0.15);
        let _ = fs::remove_dir_all(root);
    }

    fn write_constant_normal(path: &Path, normal: [f32; 3]) {
        let render = NormalRender {
            width: 16,
            height: 16,
            normals: vec![normal; 16 * 16],
            mask: vec![255; 16 * 16],
            covered_px: 16 * 16,
            mean_normal: normal,
            camera: Value::Null,
        };
        write_normal_map_png(path, &render).unwrap();
    }
}
