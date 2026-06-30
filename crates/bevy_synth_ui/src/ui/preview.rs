use super::*;

pub(super) fn spawn_preview_scene(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    asset: PreviewAsset,
    layer_index: usize,
    fit: PreviewFit,
) -> PreviewScene {
    let layer = RenderLayers::layer(layer_index);
    let size = Extent3d {
        width: PREVIEW_SIZE,
        height: PREVIEW_SIZE,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[14, 16, 20, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let image_handle = images.add(image);

    let half_fov = PREVIEW_CAMERA_FOV * 0.5;
    let camera_distance =
        (fit.radius / half_fov.tan()).max(fit.radius + 0.35) + PREVIEW_CAMERA_MARGIN;

    let mut asset_entities = Vec::new();
    match asset {
        PreviewAsset::Mesh { mesh, material } => {
            asset_entities.push(
                commands
                    .spawn((
                        Pickable::IGNORE,
                        Mesh3d(mesh),
                        MeshMaterial3d(material),
                        Transform {
                            translation: fit.mesh_translation,
                            scale: Vec3::splat(fit.mesh_scale),
                            ..default()
                        },
                        layer.clone(),
                        ThumbnailSpin,
                    ))
                    .id(),
            );
        }
        PreviewAsset::GaussianSplat { cloud } => {
            asset_entities.push(
                commands
                    .spawn((
                        Pickable::IGNORE,
                        PlanarGaussian3dHandle(cloud),
                        triposplat_preview_cloud_settings(),
                        Transform {
                            translation: fit.mesh_translation,
                            scale: Vec3::splat(fit.mesh_scale),
                            ..default()
                        },
                        layer.clone(),
                        ThumbnailSpin,
                    ))
                    .id(),
            );
        }
        PreviewAsset::Scene { items } => {
            for item in items {
                let mut transform = item.transform;
                transform.translation =
                    fit.mesh_translation + transform.translation * fit.mesh_scale;
                transform.scale *= fit.mesh_scale;
                match item.asset {
                    CatalogSpawnAsset::Mesh { mesh, material } => {
                        asset_entities.push(
                            commands
                                .spawn((
                                    Pickable::IGNORE,
                                    Mesh3d(mesh),
                                    MeshMaterial3d(material),
                                    transform,
                                    layer.clone(),
                                ))
                                .id(),
                        );
                    }
                    CatalogSpawnAsset::GaussianSplat { cloud } => {
                        asset_entities.push(
                            commands
                                .spawn((
                                    Pickable::IGNORE,
                                    PlanarGaussian3dHandle(cloud),
                                    triposplat_preview_cloud_settings(),
                                    transform,
                                    layer.clone(),
                                ))
                                .id(),
                        );
                    }
                }
            }
        }
    };
    let light_entities = vec![
        commands
            .spawn((
                DirectionalLight {
                    color: Color::srgb(1.0, 0.98, 0.95),
                    illuminance: 18_000.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(2.8, 4.0, 3.4).looking_at(Vec3::ZERO, Vec3::Y),
                layer.clone(),
            ))
            .id(),
        commands
            .spawn((
                PointLight {
                    color: Color::srgb(0.76, 0.86, 1.0),
                    intensity: 12_000.0,
                    range: 7.0,
                    radius: 0.35,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(-2.5, 2.6, 2.2),
                layer.clone(),
            ))
            .id(),
    ];

    let camera_entity = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 2,
                output_mode: CameraOutputMode::Write {
                    blend_state: None,
                    clear_color: ClearColorConfig::Default,
                },
                ..default()
            },
            RenderTarget::Image(image_handle.clone().into()),
            Projection::Perspective(PerspectiveProjection {
                fov: PREVIEW_CAMERA_FOV,
                near: 0.01,
                ..default()
            }),
            Transform::from_translation(Vec3::new(0.0, fit.radius * 0.35, camera_distance))
                .looking_at(Vec3::ZERO, Vec3::Y),
            GaussianCamera::default(),
            layer.clone(),
        ))
        .id();

    PreviewScene {
        image: image_handle,
        asset_entities,
        camera_entity,
        light_entities,
        layer_index,
    }
}

pub(super) fn triposplat_preview_cloud_settings() -> CloudSettings {
    CloudSettings {
        sort_mode: SortMode::Std,
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

pub(super) fn preview_fit_for_scene_items(
    items: &[CatalogScenePreviewItem],
    meshes: &Assets<BevyMesh>,
    gaussian_clouds: &Assets<PlanarGaussian3d>,
) -> PreviewFit {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for item in items {
        match &item.asset {
            CatalogSpawnAsset::Mesh { mesh, .. } => {
                if let Some(mesh) = meshes.get(mesh) {
                    accumulate_mesh_bounds(mesh, item.transform, &mut min, &mut max, &mut any);
                }
            }
            CatalogSpawnAsset::GaussianSplat { cloud } => {
                if let Some(cloud) = gaussian_clouds.get(cloud) {
                    accumulate_gaussian_bounds(cloud, item.transform, &mut min, &mut max, &mut any);
                }
            }
        }
    }
    if !any || !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }
    preview_fit_from_bounds(min, max)
}

pub(super) fn preview_fit_for_mesh(mesh: &BevyMesh) -> PreviewFit {
    let Some(positions) = mesh
        .attribute(BevyMesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
    else {
        return PreviewFit::fallback();
    };
    if positions.is_empty() {
        return PreviewFit::fallback();
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        let point = Vec3::new(position[0], position[1], position[2]);
        min = min.min(point);
        max = max.max(point);
    }

    preview_fit_from_bounds(min, max)
}

pub(super) fn preview_fit_from_bounds(min: Vec3, max: Vec3) -> PreviewFit {
    if !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;
    let raw_radius = half_extents.length();
    if !raw_radius.is_finite() || raw_radius <= 0.000_1 {
        return PreviewFit::fallback();
    }

    let mesh_scale = PREVIEW_TARGET_RADIUS / raw_radius;
    if !mesh_scale.is_finite() || mesh_scale <= 0.000_1 {
        return PreviewFit::fallback();
    }

    PreviewFit {
        mesh_translation: -center * mesh_scale,
        mesh_scale,
        radius: (raw_radius * mesh_scale).max(0.05),
    }
}

pub(super) fn accumulate_mesh_bounds(
    mesh: &BevyMesh,
    transform: Transform,
    min: &mut Vec3,
    max: &mut Vec3,
    any: &mut bool,
) {
    let Some(positions) = mesh
        .attribute(BevyMesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
    else {
        return;
    };
    let matrix = transform.to_matrix();
    for position in positions {
        let point = matrix.transform_point3(Vec3::new(position[0], position[1], position[2]));
        if point.is_finite() {
            *min = min.min(point);
            *max = max.max(point);
            *any = true;
        }
    }
}

pub(super) fn accumulate_gaussian_bounds(
    cloud: &PlanarGaussian3d,
    transform: Transform,
    min: &mut Vec3,
    max: &mut Vec3,
    any: &mut bool,
) {
    let matrix = transform.to_matrix();
    for position_visibility in cloud.position_visibility.iter() {
        let point = matrix.transform_point3(Vec3::new(
            position_visibility.position[0],
            position_visibility.position[1],
            position_visibility.position[2],
        ));
        if point.is_finite() {
            *min = min.min(point);
            *max = max.max(point);
            *any = true;
        }
    }
}

pub(super) fn preview_fit_for_gaussian_cloud(cloud: &PlanarGaussian3d) -> PreviewFit {
    if cloud.position_visibility.is_empty() {
        return PreviewFit::fallback();
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position_visibility in cloud.position_visibility.iter() {
        let point = Vec3::new(
            position_visibility.position[0],
            position_visibility.position[1],
            position_visibility.position[2],
        );
        if !point.is_finite() {
            return PreviewFit::fallback();
        }
        min = min.min(point);
        max = max.max(point);
    }

    if !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;
    let raw_radius = half_extents.length();
    if !raw_radius.is_finite() || raw_radius <= 0.000_1 {
        return PreviewFit::fallback();
    }

    let mesh_scale = PREVIEW_TARGET_RADIUS / raw_radius;
    if !mesh_scale.is_finite() || mesh_scale <= 0.000_1 {
        return PreviewFit::fallback();
    }

    PreviewFit {
        mesh_translation: -center * mesh_scale,
        mesh_scale,
        radius: (raw_radius * mesh_scale).max(0.05),
    }
}

pub fn preview_light_layers() -> RenderLayers {
    RenderLayers::layer(0)
}
