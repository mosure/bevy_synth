use super::*;
use bevy_gaussian_splatting::{Gaussian3d, SphericalHarmonicCoefficients};
use bevy_synth_runtime::state::InferenceQueue;

fn ui_test_app(args: Option<AppArgs>) -> App {
    let mut app = App::new();
    if let Some(args) = args {
        app.insert_resource(args);
    }
    app.insert_resource(InferenceQueue::default());
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(Assets::<PlanarGaussian3d>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(Time::<()>::default());
    app.add_plugins(BurnSynthUiPlugin);
    app
}

#[test]
fn ui_root_is_pass_through_for_world_picking() {
    let mut app = ui_test_app(None);

    app.update();

    let world = app.world_mut();
    let mut query = world.query::<(&UiRootNode, &Pickable)>();
    let pickables: Vec<Pickable> = query.iter(world).map(|(_, pickable)| *pickable).collect();
    assert_eq!(pickables.len(), 1, "expected exactly one UI root node");
    assert_eq!(pickables[0], Pickable::IGNORE);
}

#[test]
fn pipeline_selector_is_owned_by_settings_modal_for_single_launch_model() {
    let args = AppArgs {
        backend: BackendKind::Wgpu,
        synthesis_models: vec![SynthesisModel::Triposplat],
        available_synthesis_models: vec![SynthesisModel::Triposplat],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));

    app.update();

    let world = app.world_mut();
    assert_eq!(
        world.query::<&PipelineSelectorButton>().iter(world).count(),
        0
    );
    assert_eq!(
        world.query::<&PipelineOptionButton>().iter(world).count(),
        0
    );
    assert_eq!(
        world.resource::<AvailablePipelines>().object_models,
        vec![SynthesisModel::Triposplat]
    );

    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    let world = app.world_mut();
    assert_eq!(
        world.query::<&PipelineOptionButton>().iter(world).count(),
        1
    );
}

#[test]
fn scene_source_modal_has_image_and_stats_tabs() {
    let mut app = ui_test_app(None);
    let metrics = CachedSceneMetrics {
        ok: Some(true),
        elapsed_ms: Some(42_000),
        object_count: Some(3),
        asset_count: Some(2),
        placement_count: Some(3),
        feedback_accepted: Some(true),
        feedback_iteration: Some(2),
        failed_stage: None,
        category_breakdown: vec![
            CachedSceneCategoryMetric {
                label: "chair".to_string(),
                object_count: Some(2),
                detection_count: Some(4),
                asset_count: Some(1),
                placement_count: Some(2),
            },
            CachedSceneCategoryMetric {
                label: "table".to_string(),
                object_count: Some(1),
                detection_count: Some(1),
                asset_count: Some(1),
                placement_count: Some(1),
            },
        ],
    };
    app.world_mut()
        .resource_mut::<CatalogState>()
        .add_ready_scene(
            42,
            "meeting scene".to_string(),
            Some("scene_cache_key".to_string()),
            Vec::new(),
            Some("/tmp/source_scene.jpg".to_string()),
            None,
            Some("explicit".to_string()),
            Some(metrics),
            Some("tmp/runs/demo_scene".to_string()),
        );
    app.world_mut()
        .resource_mut::<CatalogSourceImageModalState>()
        .entry_id = Some(42);

    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert_eq!(
            world
                .query::<&CatalogSourceImageTabButton>()
                .iter(world)
                .count(),
            2
        );
        assert_eq!(
            world
                .query::<&CatalogSourceImageTabPanel>()
                .iter(world)
                .count(),
            2
        );
        let mut texts = world.query::<&Text>();
        let values: Vec<_> = texts.iter(world).map(|text| text.0.clone()).collect();
        assert!(values.iter().any(|text| text == "categories"));
        assert!(
            values.iter().any(|text| {
                text.contains("chair | 2 planned / 4 detected / 1 assets / 2 placed")
            })
        );
    }

    let stats_button = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &CatalogSourceImageTabButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.tab == CatalogSourceImageTab::Stats).then_some(entity)
            })
            .expect("stats tab button")
    };
    app.world_mut()
        .entity_mut(stats_button)
        .insert(Interaction::Pressed);
    app.update();
    app.update();

    let world = app.world_mut();
    let mut panels = world.query::<(&CatalogSourceImageTabPanel, &Node)>();
    let visible_tabs: Vec<_> = panels
        .iter(world)
        .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
        .collect();
    assert_eq!(visible_tabs, vec![CatalogSourceImageTab::Stats]);
}

#[test]
fn settings_modal_spawns_enabled_model_options() {
    let args = AppArgs {
        backend: BackendKind::Wgpu,
        synthesis_models: vec![SynthesisModel::Triposg],
        available_synthesis_models: vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));

    app.update();
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    let world = app.world_mut();
    let mut query = world.query::<&PipelineOptionButton>();
    let models: Vec<_> = query
        .iter(world)
        .filter_map(|button| match button.choice {
            CatalogPipelineChoice::Object(model) => Some(model),
            CatalogPipelineChoice::Scene(_) => None,
        })
        .collect();
    assert_eq!(
        models,
        vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat
        ]
    );
}

#[test]
fn scene_settings_model_buttons_update_scene_mesh_asset_model() {
    let args = AppArgs {
        backend: BackendKind::Wgpu,
        synthesis_models: vec![SynthesisModel::Triposg],
        available_synthesis_models: vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));
    app.world_mut()
        .resource_mut::<CatalogState>()
        .set_active_mode(CatalogMode::Scene);
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    {
        let world = app.world_mut();
        let mut query = world.query::<&PipelineOptionButton>();
        let models: Vec<_> = query
            .iter(world)
            .filter_map(|button| match button.choice {
                CatalogPipelineChoice::Object(model) => Some(model),
                CatalogPipelineChoice::Scene(_) => None,
            })
            .collect();
        assert_eq!(
            models,
            vec![SynthesisModel::Triposg, SynthesisModel::Trellis]
        );
    }

    let triposg_button = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &PipelineOptionButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                matches!(
                    button.choice,
                    CatalogPipelineChoice::Object(SynthesisModel::Triposg)
                )
                .then_some(entity)
            })
            .expect("TripoSG scene image-to-3d button")
    };
    app.world_mut()
        .entity_mut(triposg_button)
        .insert(Interaction::Pressed);
    app.update();

    assert_eq!(
        app.world()
            .resource::<ScenePipelineUiSettings>()
            .image_to_3d_model,
        SynthesisModel::Triposg
    );
    assert_eq!(
        app.world().resource::<AppArgs>().synthesis_models,
        vec![SynthesisModel::Triposg],
        "scene image-to-3d selection must not mutate the object catalog pipeline"
    );
}

#[test]
fn scene_optional_stage_toggle_labels_track_settings() {
    let mut settings = ScenePipelineUiSettings::default();
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::LocateAnything),
        "on"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::Depth),
        "on"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::Segmentation),
        "on"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::PoseFit),
        "on"
    );

    settings.locate_anything_enabled = false;
    settings.depth_enabled = false;
    settings.segmentation_enabled = false;
    settings.pose_fit_enabled = false;

    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::LocateAnything),
        "off"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::Depth),
        "off"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::Segmentation),
        "off"
    );
    assert_eq!(
        scene_toggle_value_text(&settings, SceneToggleSetting::PoseFit),
        "off"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn processing_artifact_classifier_prioritizes_scene_intermediates() {
    assert_eq!(
        visual_artifact_kind(std::path::Path::new("tmp/runs/demo/detections_overlay.png")),
        Some(ProcessingArtifactVisualKind::Locate)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new("tmp/runs/demo/depth_map.png")),
        Some(ProcessingArtifactVisualKind::Depth)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new(
            "tmp/runs/demo/objects/crops/chair_0.jpg"
        )),
        Some(ProcessingArtifactVisualKind::Crop)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new(
            "tmp/runs/demo/canonical_pose/chair_selection.png"
        )),
        Some(ProcessingArtifactVisualKind::Canonical)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new(
            "tmp/runs/demo/iterations/iter_03/screenshot.png"
        )),
        Some(ProcessingArtifactVisualKind::Feedback)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new(
            "tmp/runs/demo/iterations/iter_03/rotation_candidates/chair/current_isolated_full_frame.png"
        )),
        Some(ProcessingArtifactVisualKind::IsolatedRender)
    );
    assert_eq!(
        visual_artifact_kind(std::path::Path::new(
            "tmp/runs/demo/iterations/iter_03/rotation_candidates/chair/candidate_00_yaw_pos0_screenshot.png"
        )),
        Some(ProcessingArtifactVisualKind::IsolatedRender)
    );
    assert_eq!(
        artifact_kind_label("tmp/runs/demo/iterations/iter_03/scene.bsn"),
        "bsn  "
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn processing_artifacts_sort_latest_generation_first() {
    let root = temp_visual_artifact_dir("latest_sort");
    let older = root.join("iterations/iter_00/screenshot.png");
    let newer = root.join("iterations/iter_03/screenshot.png");
    std::fs::create_dir_all(older.parent().expect("older parent")).expect("older dir");
    std::fs::create_dir_all(newer.parent().expect("newer parent")).expect("newer dir");
    std::fs::write(&older, &[] as &[u8]).expect("older artifact");
    std::thread::sleep(std::time::Duration::from_millis(10));
    std::fs::write(&newer, &[] as &[u8]).expect("newer artifact");

    let mut state = SceneProcessingState::default();
    state
        .recent_artifacts
        .push_front(root.display().to_string());
    let discovered = discover_processing_visual_artifacts(&state);

    assert_eq!(
        discovered.first().map(|(path, _)| path),
        Some(&newer),
        "latest pipeline artifact should be shown first"
    );
    std::fs::remove_dir_all(root).expect("remove temp artifacts");
}

#[test]
fn developer_visual_tab_renders_artifact_previews() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
    app.world_mut().resource_mut::<DeveloperPanelState>().tab = DeveloperPanelTab::Visuals;
    let artifact_path = std::env::temp_dir().join(format!(
        "bevy_synth_ui_{}_detections_overlay.png",
        std::process::id()
    ));
    image::save_buffer_with_format(
        &artifact_path,
        &[
            255, 255, 255, 255, 64, 64, 64, 255, 64, 64, 64, 255, 255, 255, 255, 255,
        ],
        2,
        2,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .expect("write preview artifact");
    {
        let mut state = app.world_mut().resource_mut::<SceneProcessingState>();
        state
            .recent_artifacts
            .push_front(artifact_path.display().to_string());
    }

    app.update();
    app.update();

    let world = app.world_mut();
    let mut grids = world.query::<(&SettingsDeveloperVisualGrid, &Children)>();
    let child_counts = grids
        .iter(world)
        .map(|(_, children)| children.len())
        .collect::<Vec<_>>();
    assert_eq!(child_counts, vec![1]);
    let mut texts = world.query::<&Text>();
    let values = texts
        .iter(world)
        .map(|text| text.0.clone())
        .collect::<Vec<_>>();
    assert!(values.iter().any(|text| text == "locate"));
    assert!(
        values
            .iter()
            .any(|text| text.contains("detections_overlay.png"))
    );
    std::fs::remove_file(artifact_path).expect("remove preview artifact");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn developer_visual_tab_paginates_artifact_previews() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
    app.world_mut().resource_mut::<DeveloperPanelState>().tab = DeveloperPanelTab::Visuals;
    let root = temp_visual_artifact_dir("pagination");
    std::fs::create_dir_all(&root).expect("preview dir");
    for index in 0..(DEVELOPER_VISUAL_ROWS + 1) {
        write_preview_png(&root.join(format!("iter_{index:02}_detections_overlay.png")));
    }
    {
        let mut state = app.world_mut().resource_mut::<SceneProcessingState>();
        state
            .recent_artifacts
            .push_front(root.display().to_string());
    }

    app.update();
    app.update();

    {
        let cache = app.world().resource::<ProcessingArtifactPreviewCache>();
        assert_eq!(cache.total_count, DEVELOPER_VISUAL_ROWS + 1);
        assert_eq!(cache.page, 0);
        assert_eq!(cache.page_count, 2);
        assert_eq!(cache.previews.len(), DEVELOPER_VISUAL_ROWS);
        assert!(
            cache.previews[0]
                .path
                .contains(&format!("iter_{:02}", DEVELOPER_VISUAL_ROWS)),
            "first page should be latest-first"
        );
    }
    assert_visual_grid_child_count(&mut app, DEVELOPER_VISUAL_ROWS);

    let next_button = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &SettingsDeveloperVisualPageButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.direction == DeveloperVisualPageDirection::Next).then_some(entity)
            })
            .expect("next visual page button")
    };
    app.world_mut()
        .entity_mut(next_button)
        .insert(Interaction::Pressed);
    app.update();
    app.update();

    {
        let cache = app.world().resource::<ProcessingArtifactPreviewCache>();
        assert_eq!(cache.page, 1);
        assert_eq!(cache.previews.len(), 1);
    }
    assert_visual_grid_child_count(&mut app, 1);
    let world = app.world_mut();
    let mut pager_texts = world.query_filtered::<&Text, With<SettingsDeveloperVisualPagerText>>();
    assert!(
        pager_texts
            .iter(world)
            .any(|text| text.0.contains("page 2/2")),
        "pager text should report the active page"
    );
    std::fs::remove_dir_all(root).expect("remove preview artifacts");
}

#[cfg(not(target_arch = "wasm32"))]
fn temp_visual_artifact_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "bevy_synth_ui_{label}_{}_{}",
        std::process::id(),
        nanos
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_preview_png(path: &std::path::Path) {
    image::save_buffer_with_format(
        path,
        &[
            255, 255, 255, 255, 64, 64, 64, 255, 64, 64, 64, 255, 255, 255, 255, 255,
        ],
        2,
        2,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .expect("write preview artifact");
}

fn assert_visual_grid_child_count(app: &mut App, expected: usize) {
    let world = app.world_mut();
    let mut grids = world.query::<(&SettingsDeveloperVisualGrid, &Children)>();
    let child_counts = grids
        .iter(world)
        .map(|(_, children)| children.len())
        .collect::<Vec<_>>();
    assert_eq!(child_counts, vec![expected]);
}

#[test]
fn unavailable_launch_models_are_not_selectable() {
    let available = AvailablePipelines {
        object_models: vec![SynthesisModel::Triposplat],
        scene_pipelines: vec![ScenePipelineKind::Explicit],
    };
    assert!(pipeline_available(
        Some(&available),
        CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
    ));
    assert!(!pipeline_available(
        Some(&available),
        CatalogPipelineChoice::Object(SynthesisModel::Triposg)
    ));
    assert!(pipeline_available(
        Some(&available),
        CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit)
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn triposplat_pipeline_is_supported_on_native_wgpu() {
    let args = AppArgs {
        backend: BackendKind::Wgpu,
        ..Default::default()
    };
    assert!(pipeline_supported(
        Some(&args),
        CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
    ));
    assert!(pipeline_supported(
        Some(&args),
        CatalogPipelineChoice::Object(SynthesisModel::Trellis)
    ));
}

#[test]
fn ready_cube_detection_allows_cached_cube_entries() {
    let mut catalog = CatalogState::default();
    assert!(!catalog.has_ready_cube_entry());

    catalog.add_ready(
        1,
        "cube".to_string(),
        Handle::default(),
        Handle::default(),
        Some("builtin/cube".to_string()),
        Some("builtin-cube-cache-key".to_string()),
    );
    assert!(catalog.has_ready_cube_entry());
}

#[test]
fn catalog_labels_are_ellipsized_to_fixed_width() {
    assert_eq!(ellipsize_text("short", 12), "short");
    assert_eq!(
        ellipsize_text("very-long-catalog-entry-name", 13),
        "very-long-..."
    );
    assert_eq!(ellipsize_text("abcdef", 2), "..");
}

#[test]
fn splat_catalog_entry_creates_gaussian_preview_scene() {
    let mut app = App::new();
    app.insert_resource(CatalogState::default());
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(Assets::<PlanarGaussian3d>::default());
    app.add_systems(Update, sync_catalog_previews);

    let cloud = PlanarGaussian3d::from(vec![
        Gaussian3d {
            position_visibility: [-0.25, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.04, 0.04, 0.04, 0.8].into(),
        },
        Gaussian3d {
            position_visibility: [0.25, 0.2, 0.0, 1.0].into(),
            spherical_harmonic: SphericalHarmonicCoefficients::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.04, 0.04, 0.04, 0.8].into(),
        },
    ]);
    let cloud_handle = app
        .world_mut()
        .resource_mut::<Assets<PlanarGaussian3d>>()
        .add(cloud);
    app.world_mut()
        .resource_mut::<CatalogState>()
        .add_ready_gaussian_splat(
            7,
            "splat".to_string(),
            cloud_handle,
            Some("input.png".to_string()),
            Some("cache-key".to_string()),
        );

    app.update();

    let world = app.world_mut();
    let (has_preview, has_gaussian, has_mesh, has_material) = {
        let catalog = world.resource::<CatalogState>();
        let entry = catalog.entry(7).expect("splat catalog entry");
        (
            entry.preview.is_some(),
            entry.gaussian.is_some(),
            entry.mesh.is_some(),
            entry.material.is_some(),
        )
    };
    assert!(has_preview, "splat entry should get a preview");
    assert_eq!(
        world
            .resource::<CatalogState>()
            .entry(7)
            .and_then(|entry| entry.preview.as_ref())
            .map(|preview| preview.light_entities.len()),
        Some(2),
        "preview scene should own isolated light entities"
    );
    assert!(has_gaussian);
    assert!(!has_mesh);
    assert!(!has_material);
    assert_eq!(
        world.query::<&PlanarGaussian3dHandle>().iter(world).count(),
        1
    );
    let mut settings = world.query::<&CloudSettings>();
    let settings = settings
        .single(world)
        .expect("one Gaussian preview settings");
    assert_eq!(settings.sort_mode, SortMode::Std);
    assert_eq!(settings.color_space, GaussianColorSpace::SrgbRec709Display);
    assert_eq!(world.query::<&GaussianCamera>().iter(world).count(), 1);
    assert_eq!(world.query::<&DirectionalLight>().iter(world).count(), 1);
    assert_eq!(world.query::<&PointLight>().iter(world).count(), 1);
}

#[test]
fn settings_modal_opens_for_all_pipeline_settings() {
    let triposg_args = AppArgs {
        synthesis_models: vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(triposg_args));

    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(
            world.query::<&SettingsModalRoot>().iter(world).count(),
            1,
            "TripoSG settings should open when TripoSG is active"
        );
        assert_eq!(
            world
                .query::<&TripoSgSettingValueLabel>()
                .iter(world)
                .count(),
            4
        );
        assert_eq!(
            world
                .query::<&TripoSplatProfileButton>()
                .iter(world)
                .count(),
            0
        );
    }

    app.world_mut().resource_mut::<AppArgs>().synthesis_models =
        vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(
            world.query::<&SettingsModalRoot>().iter(world).count(),
            1,
            "TripoSplat settings should open when TripoSplat is active"
        );
        assert_eq!(
            world
                .query::<&TripoSplatSettingValueLabel>()
                .iter(world)
                .count(),
            3
        );
        assert_eq!(
            world
                .query::<&TripoSplatProfileButton>()
                .iter(world)
                .count(),
            3
        );
        assert_eq!(
            world
                .query::<&TripoSgSettingValueLabel>()
                .iter(world)
                .count(),
            0
        );
    }

    app.world_mut().resource_mut::<AppArgs>().synthesis_models =
        vec![SynthesisModel::Trellis, SynthesisModel::Triposg];
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(
            world.query::<&SettingsModalRoot>().iter(world).count(),
            1,
            "Trellis.2 settings should open when Trellis.2 is active"
        );
        assert_eq!(
            world.query::<&TrellisQualityButton>().iter(world).count(),
            3
        );
        assert_eq!(
            world
                .query::<&TrellisSettingValueLabel>()
                .iter(world)
                .count(),
            5
        );
    }
}

#[test]
fn settings_modal_rebuilds_or_closes_when_pipeline_changes() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposplat, SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));

    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();
    {
        let world = app.world_mut();
        assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
        assert_eq!(
            world
                .query::<&TripoSplatSettingValueLabel>()
                .iter(world)
                .count(),
            3
        );
    }

    app.world_mut().resource_mut::<AppArgs>().synthesis_models =
        vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
        assert_eq!(
            world
                .query::<&TripoSgSettingValueLabel>()
                .iter(world)
                .count(),
            4
        );
        assert_eq!(
            world
                .query::<&TripoSplatSettingValueLabel>()
                .iter(world)
                .count(),
            0
        );
    }

    app.world_mut().resource_mut::<AppArgs>().synthesis_models = vec![SynthesisModel::Trellis];
    app.update();
    app.update();

    let world = app.world_mut();
    assert!(world.resource::<SettingsModalState>().open);
    assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
    assert_eq!(
        world
            .query::<&TrellisSettingValueLabel>()
            .iter(world)
            .count(),
        5
    );
}

#[test]
fn settings_modal_uses_tabs_for_pipeline_general_physics_and_developer() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));

    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    {
        let world = app.world_mut();
        assert_eq!(world.query::<&SettingsTabButton>().iter(world).count(), 4);
        assert_eq!(world.query::<&SettingsTabPanel>().iter(world).count(), 4);
        let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
        let visible_tabs: Vec<_> = panels
            .iter(world)
            .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
            .collect();
        assert_eq!(visible_tabs, vec![SettingsModalTab::Pipeline]);
    }

    app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::General;
    app.update();

    let world = app.world_mut();
    let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
    let visible_tabs: Vec<_> = panels
        .iter(world)
        .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
        .collect();
    assert_eq!(visible_tabs, vec![SettingsModalTab::General]);
    assert_eq!(
        world.query::<&ViewerAabbModeButton>().iter(world).count(),
        3
    );

    app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
    app.update();

    let world = app.world_mut();
    let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
    let visible_tabs: Vec<_> = panels
        .iter(world)
        .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
        .collect();
    assert_eq!(visible_tabs, vec![SettingsModalTab::Developer]);
    assert_eq!(
        world
            .query::<&SettingsDeveloperEventsText>()
            .iter(world)
            .count(),
        1
    );
}

#[test]
fn scene_settings_modal_splits_long_pipeline_controls() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Trellis, SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));
    app.world_mut()
        .resource_mut::<CatalogState>()
        .set_active_mode(CatalogMode::Scene);
    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    let world = app.world_mut();
    assert_eq!(world.query::<&SettingsTabButton>().iter(world).count(), 7);
    assert_eq!(world.query::<&SettingsTabPanel>().iter(world).count(), 7);
    assert_eq!(world.query::<&SettingsScrollArea>().iter(world).count(), 7);
    assert_eq!(
        settings_tabs_for_pipeline(CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit)),
        vec![
            SettingsModalTab::Pipeline,
            SettingsModalTab::Generation,
            SettingsModalTab::Grounding,
            SettingsModalTab::Debug,
            SettingsModalTab::General,
            SettingsModalTab::Physics,
            SettingsModalTab::Developer,
        ]
    );

    let mut scroll_panels = world.query::<(&SettingsScrollArea, &Node)>();
    for (_, node) in scroll_panels.iter(world) {
        assert_eq!(node.max_height, Val::Vh(SETTINGS_TAB_BODY_MAX_HEIGHT_VH));
        assert_eq!(node.overflow.y, OverflowAxis::Scroll);
    }
}

#[test]
fn worker_status_text_is_compacted_for_menu_bar() {
    let text = compact_worker_status_text(
        "scene progress: images_to_assets - running TRELLIS batch for 2 image(s)",
    );
    assert_eq!(text, "scene progress: images_to_assets");
    assert!(text.len() <= 34);
}

#[test]
fn scene_processing_heartbeat_advances_elapsed_without_new_events() {
    let mut state = SceneProcessingState::default();
    state.begin("source.jpg");
    state.wall_started_at = Some(Instant::now() - Duration::from_millis(2500));
    state.tick();

    assert!(
        state.elapsed_ms >= 2400,
        "active processing elapsed time should advance from wall clock even without worker events"
    );
    assert!(format_developer_current_block(&state).contains("last event:"));
}

#[test]
fn developer_processing_blocks_include_artifacts_and_recent_events() {
    let mut state = SceneProcessingState::default();
    state.begin("scene.png");
    state.push_event(
        "run_001".to_string(),
        SceneProcessingEvent {
            stage: "images_to_assets".to_string(),
            phase: "progress".to_string(),
            execution: "gpu".to_string(),
            message: "running TRELLIS batch for 2 image(s)".to_string(),
            elapsed_ms: 42_000,
            item_index: Some(1),
            item_count: Some(2),
            artifact_path: Some("tmp/runs/run_001/assets".to_string()),
            token_usage: None,
            is_failure: false,
        },
    );

    let event_text = format_developer_event_block(&state);
    assert!(event_text.contains("[gpu] progress / images_to_assets"));
    assert!(event_text.contains("[1/2]"));
    let artifact_text = format_developer_artifact_block(&state);
    assert!(artifact_text.contains("dir"));
    assert!(artifact_text.contains("tmp/runs/run_001/assets"));
}

#[test]
fn viewer_debug_buttons_update_shared_settings() {
    let args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposg],
        ..Default::default()
    };
    let mut app = ui_test_app(Some(args));

    app.world_mut().resource_mut::<SettingsModalState>().open = true;
    app.update();
    app.update();

    let all_button = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &ViewerAabbModeButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.mode == ViewerAabbOverlayMode::All).then_some(entity)
            })
            .expect("all AABB button")
    };
    app.world_mut()
        .entity_mut(all_button)
        .insert(Interaction::Pressed);
    app.update();
    assert_eq!(
        app.world().resource::<ViewerDebugSettings>().aabb_overlay,
        ViewerAabbOverlayMode::All
    );

    let tolerance_step = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.setting == ViewerDebugNumericSetting::ContactTolerance
                    && button.delta > 0.0)
                    .then_some(entity)
            })
            .expect("contact tolerance step button")
    };
    app.world_mut()
        .entity_mut(tolerance_step)
        .insert(Interaction::Pressed);
    app.update();
    assert!(
        app.world()
            .resource::<ViewerDebugSettings>()
            .contact_tolerance
            > 0.02
    );

    let frustum_length_step = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.setting == ViewerDebugNumericSetting::SceneCameraFrustumLength
                    && button.delta > 0.0)
                    .then_some(entity)
            })
            .expect("frustum length step button")
    };
    app.world_mut()
        .entity_mut(frustum_length_step)
        .insert(Interaction::Pressed);
    app.update();
    assert_eq!(
        app.world()
            .resource::<ViewerDebugSettings>()
            .scene_camera_frustum_length,
        DEFAULT_VIEWER_FRUSTUM_LENGTH + VIEWER_FRUSTUM_LENGTH_STEP
    );

    let depth_cloud_toggle = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &ViewerDebugToggleButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.setting == ViewerDebugToggleSetting::DepthCloud).then_some(entity)
            })
            .expect("depth cloud debug toggle")
    };
    app.world_mut()
        .entity_mut(depth_cloud_toggle)
        .insert(Interaction::Pressed);
    app.update();
    assert!(
        app.world()
            .resource::<ViewerDebugSettings>()
            .depth_cloud_overlay
    );

    let depth_cap_step = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
        query
            .iter(world)
            .find_map(|(entity, button)| {
                (button.setting == ViewerDebugNumericSetting::DepthCloudMaxGaussians
                    && button.delta > 0.0)
                    .then_some(entity)
            })
            .expect("depth cloud cap step button")
    };
    app.world_mut()
        .entity_mut(depth_cap_step)
        .insert(Interaction::Pressed);
    app.update();
    assert_eq!(
        app.world()
            .resource::<ViewerDebugSettings>()
            .depth_cloud_max_gaussians,
        VIEWER_DEPTH_CLOUD_DEFAULT_GAUSSIANS + VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP
    );
}

#[test]
fn triposplat_profile_buttons_apply_canonical_settings() {
    let mut args = AppArgs::default();
    args.apply_triposplat_profile(TripoSplatProfile::Low);
    assert_eq!(args.num_steps, 5);
    assert_eq!(args.guidance_scale, 3.0);
    assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MIN_NUM_GAUSSIANS);

    args.apply_triposplat_profile(TripoSplatProfile::High);
    assert_eq!(args.num_steps, 50);
    assert_eq!(args.guidance_scale, 3.0);
    assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MAX_NUM_GAUSSIANS);
}

#[test]
fn triposplat_manual_steps_mark_profile_custom_and_clamp() {
    let mut args = AppArgs::default();
    args.apply_triposplat_profile(TripoSplatProfile::Low);
    adjust_triposplat_setting(
        &mut args,
        TripoSplatSetting::Steps,
        TripoSplatSettingDelta::Integer(-100),
    );
    assert_eq!(args.num_steps, TRIPOSPLAT_MIN_STEPS);
    assert_eq!(args.triposplat_profile, TripoSplatProfile::Custom);

    adjust_triposplat_setting(
        &mut args,
        TripoSplatSetting::Steps,
        TripoSplatSettingDelta::Integer(100),
    );
    assert_eq!(args.num_steps, TRIPOSPLAT_MAX_STEPS);
}

#[test]
fn triposplat_manual_gaussian_count_stays_in_supported_range() {
    let mut args = AppArgs::default();
    args.apply_triposplat_profile(TripoSplatProfile::High);
    adjust_triposplat_setting(
        &mut args,
        TripoSplatSetting::Gaussians,
        TripoSplatSettingDelta::Integer(TRIPOSPLAT_GAUSSIAN_STEP as isize),
    );
    assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MAX_NUM_GAUSSIANS);

    args.apply_triposplat_profile(TripoSplatProfile::Low);
    adjust_triposplat_setting(
        &mut args,
        TripoSplatSetting::Gaussians,
        TripoSplatSettingDelta::Integer(-(TRIPOSPLAT_GAUSSIAN_STEP as isize)),
    );
    assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MIN_NUM_GAUSSIANS);
}

#[test]
fn triposplat_gaussian_value_text_uses_exact_grouped_count() {
    let args = AppArgs {
        triposplat_num_gaussians: TRIPOSPLAT_MAX_NUM_GAUSSIANS,
        ..Default::default()
    };
    assert_eq!(
        triposplat_setting_value_text(&args, TripoSplatSetting::Gaussians),
        "262,144"
    );
}

#[test]
fn triposg_manual_settings_clamp_and_format_values() {
    let mut args = AppArgs {
        num_steps: 1,
        ..Default::default()
    };
    adjust_triposg_setting(
        &mut args,
        TripoSgSetting::Steps,
        TripoSgSettingDelta::Integer(-100),
    );
    assert_eq!(args.num_steps, TRIPOSG_MIN_STEPS);

    adjust_triposg_setting(
        &mut args,
        TripoSgSetting::Steps,
        TripoSgSettingDelta::Integer(100),
    );
    assert_eq!(args.num_steps, TRIPOSG_MAX_STEPS);

    args.num_tokens = 1024;
    adjust_triposg_setting(
        &mut args,
        TripoSgSetting::Tokens,
        TripoSgSettingDelta::Integer(TRIPOSG_TOKEN_STEP as isize),
    );
    assert_eq!(args.num_tokens, 1152);
    assert_eq!(
        triposg_setting_value_text(&args, TripoSgSetting::Tokens),
        "1,152"
    );

    args.target_faces = None;
    adjust_triposg_setting(
        &mut args,
        TripoSgSetting::TargetFaces,
        TripoSgSettingDelta::Integer(TRIPOSG_FACE_STEP as isize),
    );
    assert_eq!(args.target_faces, Some(TRIPOSG_FACE_STEP));
    adjust_triposg_setting(
        &mut args,
        TripoSgSetting::TargetFaces,
        TripoSgSettingDelta::Integer(-(TRIPOSG_FACE_STEP as isize)),
    );
    assert_eq!(args.target_faces, None);
    assert_eq!(
        triposg_setting_value_text(&args, TripoSgSetting::TargetFaces),
        "disabled"
    );
}

#[test]
fn trellis_settings_clamp_and_format_values() {
    let mut args = AppArgs {
        trellis_quality: TrellisQuality::Low,
        ..Default::default()
    };
    assert_eq!(
        trellis_setting_value_text(&args, TrellisSetting::Resolution),
        "512"
    );
    args.trellis_quality = TrellisQuality::High;
    assert_eq!(
        trellis_quality_value_text(args.trellis_quality),
        "high / 1024"
    );

    args.trellis_pbr_enabled = false;
    assert_eq!(
        trellis_setting_value_text(&args, TrellisSetting::PbrTextureSize),
        "disabled"
    );
    args.trellis_pbr_enabled = true;
    args.trellis_pbr_texture_size = Some(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE);
    adjust_trellis_setting(
        &mut args,
        TrellisSetting::PbrTextureSize,
        TrellisSettingDelta::Integer(10_000),
    );
    assert_eq!(args.trellis_pbr_texture_size, Some(TRELLIS_PBR_TEXTURE_MAX));

    args.trellis_target_faces = None;
    adjust_trellis_setting(
        &mut args,
        TrellisSetting::TargetFaces,
        TrellisSettingDelta::Integer(TRELLIS_FACE_STEP as isize),
    );
    assert_eq!(args.trellis_target_faces, Some(TRELLIS_FACE_STEP));
    adjust_trellis_setting(
        &mut args,
        TrellisSetting::TargetFaces,
        TrellisSettingDelta::Integer(-(TRELLIS_FACE_STEP as isize)),
    );
    assert_eq!(args.trellis_target_faces, None);
    assert_eq!(
        trellis_setting_value_text(&args, TrellisSetting::TargetFaces),
        "disabled"
    );

    args.trellis_max_sparse_coords = None;
    adjust_trellis_setting(
        &mut args,
        TrellisSetting::MaxSparseCoords,
        TrellisSettingDelta::Integer(TRELLIS_SPARSE_COORD_STEP as isize),
    );
    assert_eq!(
        args.trellis_max_sparse_coords,
        Some(TRELLIS_SPARSE_COORD_STEP)
    );
    adjust_trellis_setting(
        &mut args,
        TrellisSetting::MaxSparseCoords,
        TrellisSettingDelta::Integer(-(TRELLIS_SPARSE_COORD_STEP as isize)),
    );
    assert_eq!(args.trellis_max_sparse_coords, None);
    assert_eq!(
        trellis_setting_value_text(&args, TrellisSetting::MaxSparseCoords),
        "uncapped"
    );
}

#[test]
fn pipeline_setting_gate_tracks_active_pipeline() {
    let mut catalog = CatalogState::default();
    let mut args = AppArgs {
        synthesis_models: vec![SynthesisModel::Triposg, SynthesisModel::Triposplat],
        ..Default::default()
    };
    assert_eq!(
        active_settings_pipeline(Some(&args)),
        Some(SynthesisModel::Triposg)
    );
    assert!(pipeline_settings_enabled(&catalog, Some(&args)));

    args.synthesis_models = vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
    assert_eq!(
        active_settings_pipeline(Some(&args)),
        Some(SynthesisModel::Triposplat)
    );
    assert!(pipeline_settings_enabled(&catalog, Some(&args)));

    args.synthesis_models = vec![SynthesisModel::Trellis];
    assert_eq!(
        active_settings_pipeline(Some(&args)),
        Some(SynthesisModel::Trellis)
    );
    assert!(pipeline_settings_enabled(&catalog, Some(&args)));

    catalog.set_active_mode(CatalogMode::Scene);
    assert!(pipeline_settings_enabled(&catalog, Some(&args)));
}
