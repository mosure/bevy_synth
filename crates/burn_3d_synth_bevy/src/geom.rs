use bevy::camera::primitives::Aabb;
use bevy::math::Ray3d;
use bevy::prelude::{GlobalTransform, Vec3};

use crate::state::DraggableMesh;

pub(crate) fn aabb_min_max(aabb: &Aabb) -> (Vec3, Vec3) {
    let min = Vec3::from(aabb.min());
    let max = Vec3::from(aabb.max());
    (min, max)
}

pub(crate) fn world_aabb(bounds: &DraggableMesh, transform: &GlobalTransform) -> (Vec3, Vec3) {
    let min = bounds.local_min;
    let max = bounds.local_max;
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
    ];

    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let world = transform.transform_point(corner);
        world_min = world_min.min(world);
        world_max = world_max.max(world);
    }
    (world_min, world_max)
}

pub(crate) fn ray_aabb_intersection(ray: Ray3d, min: Vec3, max: Vec3) -> Option<f32> {
    let mut tmin = 0.0f32;
    let mut tmax = f32::INFINITY;
    let origin = ray.origin;
    let dir = ray.direction.as_vec3();

    for i in 0..3 {
        let origin_axis = origin[i];
        let dir_axis = dir[i];
        let min_axis = min[i];
        let max_axis = max[i];

        if dir_axis.abs() < 1e-6 {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
        } else {
            let inv = 1.0 / dir_axis;
            let mut t1 = (min_axis - origin_axis) * inv;
            let mut t2 = (max_axis - origin_axis) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmax < tmin {
                return None;
            }
        }
    }

    if tmax >= 0.0 {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

pub(crate) fn ray_plane_intersection(ray: Ray3d, plane_y: f32) -> Option<Vec3> {
    let dir = ray.direction.as_vec3();
    let denom = dir.y;
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_y - ray.origin.y) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + dir * t)
}
