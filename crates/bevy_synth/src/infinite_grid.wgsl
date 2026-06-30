// Local adaptation of fslabs/bevy_infinite_grid for the pinned Bevy render API.

struct InfiniteGridPosition {
    planar_rotation_matrix: mat3x3<f32>,
    origin: vec3<f32>,
    normal: vec3<f32>,
};

struct InfiniteGridSettings {
    scale: f32,
    dist_fadeout_const: f32,
    dot_fadeout_const: f32,
    x_axis_col: vec3<f32>,
    z_axis_col: vec3<f32>,
    minor_line_col: vec4<f32>,
    major_line_col: vec4<f32>,
};

struct View {
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    world_position: vec3<f32>,
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> grid_position: InfiniteGridPosition;
@group(1) @binding(1) var<uniform> grid_settings: InfiniteGridSettings;

struct Vertex {
    @builtin(vertex_index) index: u32,
};

fn unproject_point(p: vec3<f32>) -> vec3<f32> {
    let unprojected = view.view * view.inverse_projection * vec4<f32>(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) near_point: vec3<f32>,
    @location(1) far_point: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var grid_plane = array<vec3<f32>, 4>(
        vec3<f32>(-1.0, -1.0, 1.0),
        vec3<f32>(-1.0, 1.0, 1.0),
        vec3<f32>(1.0, -1.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0)
    );
    let p = grid_plane[vertex.index].xyz;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(p, 1.0);
    out.near_point = unproject_point(p);
    out.far_point = unproject_point(vec3<f32>(p.xy, 0.001));
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let ray_origin = in.near_point;
    let ray_direction = normalize(in.far_point - in.near_point);
    let plane_normal = grid_position.normal;
    let plane_origin = grid_position.origin;

    let denominator = dot(ray_direction, plane_normal);
    let point_to_point = plane_origin - ray_origin;
    let t = dot(plane_normal, point_to_point) / denominator;
    let frag_pos_3d = ray_direction * t + ray_origin;

    let planar_offset = frag_pos_3d - plane_origin;
    let plane_coords = (grid_position.planar_rotation_matrix * planar_offset).xz;

    let view_space_pos = view.inverse_view * vec4<f32>(frag_pos_3d, 1.0);
    let clip_space_pos = view.projection * view_space_pos;
    let clip_depth = clip_space_pos.z / clip_space_pos.w;
    let real_depth = -view_space_pos.z;

    var out: FragmentOutput;
    out.depth = clip_depth;

    let scale = grid_settings.scale;
    let coord = plane_coords * scale;
    let derivative = fwidth(coord);
    let grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    let line = min(grid.x, grid.y);

    let major_derivative = fwidth(coord * 0.1);
    let major_grid = abs(fract(coord * 0.1 - 0.5) - 0.5) / major_derivative;
    let major_line = min(major_grid.x, major_grid.y);

    let axis_grid = abs(coord) / derivative;
    let axis_line = min(axis_grid.x, axis_grid.y);

    var alpha = vec3<f32>(1.0) - min(vec3<f32>(axis_line, major_line, line), vec3<f32>(1.0));
    alpha.y *= (1.0 - alpha.x) * grid_settings.major_line_col.a;
    alpha.z *= (1.0 - (alpha.x + alpha.y)) * grid_settings.minor_line_col.a;

    let dist_fadeout = min(1.0, 1.0 - grid_settings.dist_fadeout_const * real_depth);
    let dot_fadeout = abs(dot(grid_position.normal, normalize(view.world_position - frag_pos_3d)));
    let alpha_fadeout = mix(dist_fadeout, 1.0, dot_fadeout) * min(grid_settings.dot_fadeout_const * dot_fadeout, 1.0);

    let total_alpha = alpha.x + alpha.y + alpha.z;
    alpha /= total_alpha;
    alpha = clamp(alpha, vec3<f32>(0.0), vec3<f32>(1.0));

    let axis_color = mix(grid_settings.x_axis_col, grid_settings.z_axis_col, step(axis_grid.x, axis_grid.y));
    out.color = vec4<f32>(
        axis_color * alpha.x + grid_settings.major_line_col.rgb * alpha.y + grid_settings.minor_line_col.rgb * alpha.z,
        max(total_alpha * alpha_fadeout, 0.0),
    );

    return out;
}
