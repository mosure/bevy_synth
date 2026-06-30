use super::*;

pub(super) fn handle_settings_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SettingsButton>)>,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    mut modal: ResMut<SettingsModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            if pipeline_settings_enabled(&catalog, args.as_deref()) {
                modal.open = !modal.open;
            } else {
                modal.open = false;
            }
        }
    }
}

pub(super) fn handle_settings_close_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SettingsCloseButton>)>,
    mut modal: ResMut<SettingsModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.open = false;
        }
    }
}

pub(super) fn handle_triposplat_profile_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSplatProfileButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposplat) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.apply_triposplat_profile(button.profile);
        log::info!(
            "selected TripoSplat profile: {}",
            triposplat_profile_label(button.profile)
        );
    }
}

pub(super) fn handle_triposplat_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSplatSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposplat) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_triposplat_setting(&mut args, button.setting, button.delta);
    }
}

pub(super) fn handle_triposg_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSgSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposg) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_triposg_setting(&mut args, button.setting, button.delta);
    }
}

pub(super) fn handle_trellis_quality_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TrellisQualityButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.trellis_quality = button.quality;
        log::info!(
            "Trellis.2 settings: quality={} resolution={}",
            trellis_quality_label(button.quality),
            trellis_resolution_text(button.quality)
        );
    }
}

pub(super) fn handle_trellis_pbr_toggle_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<TrellisPbrToggleButton>)>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.trellis_pbr_enabled = !args.trellis_pbr_enabled;
        log::info!(
            "Trellis.2 settings: pbr={}",
            if args.trellis_pbr_enabled {
                "on"
            } else {
                "off"
            }
        );
    }
}

pub(super) fn handle_trellis_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TrellisSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_trellis_setting(&mut args, button.setting, button.delta);
    }
}

pub(super) fn handle_scene_quality_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneQualityButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        scene_settings.quality_profile = button.quality;
    }
}

pub(super) fn handle_scene_setting_step_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneSettingStepButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_scene_setting(&mut scene_settings, button.setting, button.delta);
    }
}

pub(super) fn handle_scene_setting_toggle_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneSettingToggleButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            SceneToggleSetting::Pbr => scene_settings.pbr_enabled = !scene_settings.pbr_enabled,
            SceneToggleSetting::CatalogReuse => {
                scene_settings.allow_catalog_reuse = !scene_settings.allow_catalog_reuse
            }
            SceneToggleSetting::LiftAssets => {
                scene_settings.lift_assets = !scene_settings.lift_assets
            }
            SceneToggleSetting::LocateAnything => {
                scene_settings.locate_anything_enabled = !scene_settings.locate_anything_enabled
            }
            SceneToggleSetting::Depth => {
                scene_settings.depth_enabled = !scene_settings.depth_enabled
            }
            SceneToggleSetting::Segmentation => {
                scene_settings.segmentation_enabled = !scene_settings.segmentation_enabled
            }
            SceneToggleSetting::PoseFit => {
                scene_settings.pose_fit_enabled = !scene_settings.pose_fit_enabled
            }
            SceneToggleSetting::Feedback => {
                scene_settings.feedback_enabled = !scene_settings.feedback_enabled
            }
            SceneToggleSetting::WriteArtifacts => {
                scene_settings.write_artifacts = !scene_settings.write_artifacts
            }
            SceneToggleSetting::PromoteToCatalog => {
                scene_settings.promote_to_catalog = !scene_settings.promote_to_catalog
            }
        }
    }
}

pub(super) fn handle_settings_tab_button(
    mut modal: ResMut<SettingsModalState>,
    mut interactions: Query<(&Interaction, &SettingsTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed && modal.tab != button.tab {
            modal.tab = button.tab;
        }
    }
}

pub(super) fn handle_settings_scroll(
    mut scroll: On<Pointer<PointerScroll>>,
    mut query: Query<(&Node, &ComputedNode, &mut ScrollPosition), With<SettingsScrollArea>>,
) {
    let Ok((node, computed, mut scroll_position)) = query.get_mut(scroll.entity) else {
        return;
    };
    if node.overflow.y != OverflowAxis::Scroll || scroll.y == 0.0 {
        return;
    }

    scroll.propagate(false);
    let visible_size = computed.size() * computed.inverse_scale_factor();
    let content_size = computed.content_size() * computed.inverse_scale_factor();
    let max_offset = (content_size - visible_size).max(Vec2::ZERO);
    let unit = match scroll.unit {
        MouseScrollUnit::Line => MouseScrollUnit::SCROLL_UNIT_CONVERSION_FACTOR,
        MouseScrollUnit::Pixel => 1.0,
    };
    scroll_position.y = (scroll_position.y - scroll.y * unit).clamp(0.0, max_offset.y);
}

pub(super) fn handle_developer_panel_tab_button(
    mut state: ResMut<DeveloperPanelState>,
    mut interactions: Query<(&Interaction, &SettingsDeveloperTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed && state.tab != button.tab {
            state.tab = button.tab;
        }
    }
}

pub(super) fn handle_developer_visual_page_button(
    mut state: ResMut<DeveloperPanelState>,
    cache: Res<ProcessingArtifactPreviewCache>,
    mut interactions: Query<
        (&Interaction, &SettingsDeveloperVisualPageButton),
        Changed<Interaction>,
    >,
) {
    let max_page = cache.page_count.saturating_sub(1);
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.direction {
            DeveloperVisualPageDirection::Previous => {
                state.visual_page = state.visual_page.saturating_sub(1);
            }
            DeveloperVisualPageDirection::Next => {
                state.visual_page = state.visual_page.saturating_add(1).min(max_page);
            }
        }
    }
}

pub(super) fn handle_viewer_aabb_mode_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerAabbModeButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            settings.aabb_overlay = button.mode;
        }
    }
}

pub(super) fn handle_viewer_debug_toggle_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerDebugToggleButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            ViewerDebugToggleSetting::GroundContact => {
                settings.draw_ground_contact = !settings.draw_ground_contact;
            }
            ViewerDebugToggleSetting::SceneCameraFrustum => {
                settings.draw_scene_camera_frustum = !settings.draw_scene_camera_frustum;
            }
            ViewerDebugToggleSetting::DepthCloud => {
                settings.depth_cloud_overlay = !settings.depth_cloud_overlay;
            }
        }
    }
}

pub(super) fn handle_viewer_debug_step_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerDebugStepButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            ViewerDebugNumericSetting::GroundY => {
                settings.ground_y = (settings.ground_y + button.delta)
                    .clamp(VIEWER_GROUND_Y_MIN, VIEWER_GROUND_Y_MAX);
            }
            ViewerDebugNumericSetting::ContactTolerance => {
                settings.contact_tolerance = (settings.contact_tolerance + button.delta)
                    .clamp(VIEWER_CONTACT_TOLERANCE_MIN, VIEWER_CONTACT_TOLERANCE_MAX);
            }
            ViewerDebugNumericSetting::SceneCameraFrustumLength => {
                settings.scene_camera_frustum_length = (settings.scene_camera_frustum_length
                    + button.delta)
                    .clamp(VIEWER_FRUSTUM_LENGTH_MIN, VIEWER_FRUSTUM_LENGTH_MAX);
            }
            ViewerDebugNumericSetting::DepthCloudMaxGaussians => {
                let next = settings.depth_cloud_max_gaussians as f32 + button.delta;
                let stepped = (next / VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32).round()
                    * VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32;
                settings.depth_cloud_max_gaussians = stepped
                    .clamp(
                        VIEWER_DEPTH_CLOUD_MIN_GAUSSIANS as f32,
                        VIEWER_DEPTH_CLOUD_MAX_GAUSSIANS as f32,
                    )
                    .round() as usize;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sync_settings_modal(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut modal: ResMut<SettingsModalState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
) {
    let active_pipeline = active_pipeline_choice(&catalog, args.as_deref(), Some(&scene_settings));
    if active_pipeline.is_none() || !pipeline_settings_enabled(&catalog, args.as_deref()) {
        modal.open = false;
    }
    if modal.entity.is_some() && modal.pipeline != active_pipeline {
        if let Some(entity) = modal.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        modal.pipeline = None;
        modal.tab = SettingsModalTab::Pipeline;
    }
    ui.settings_modal_open = modal.open;
    match (modal.open, modal.entity) {
        (true, None) => {
            if let Some(pipeline) = active_pipeline {
                modal.entity = Some(spawn_settings_modal(
                    &mut commands,
                    pipeline,
                    modal.tab,
                    available.as_deref(),
                ));
                modal.pipeline = Some(pipeline);
            }
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            modal.entity = None;
            modal.pipeline = None;
        }
        _ => {}
    }
}

pub(super) fn sync_settings_tab_visuals(
    modal: Res<SettingsModalState>,
    mut panels: Query<(&SettingsTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &SettingsTabButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (panel, mut node, mut visibility) in panels.iter_mut() {
        let (next_visibility, next_display) = if panel.tab == modal.tab {
            (Visibility::Visible, Display::Flex)
        } else {
            (Visibility::Hidden, Display::None)
        };
        if *visibility != next_visibility {
            *visibility = next_visibility;
        }
        if node.display != next_display {
            node.display = next_display;
        }
    }
    for (tab, interaction, children, mut bg, mut border) in tabs.iter_mut() {
        let active = tab.tab == modal.tab;
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Secondary, *interaction, false, active);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

pub(super) fn sync_developer_panel_tab_visuals(
    state: Res<DeveloperPanelState>,
    mut panels: Query<(&SettingsDeveloperTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &SettingsDeveloperTabButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (panel, mut node, mut visibility) in panels.iter_mut() {
        let (next_visibility, next_display) = if panel.tab == state.tab {
            (Visibility::Visible, Display::Flex)
        } else {
            (Visibility::Hidden, Display::None)
        };
        if *visibility != next_visibility {
            *visibility = next_visibility;
        }
        if node.display != next_display {
            node.display = next_display;
        }
    }
    for (tab, interaction, children, mut bg, mut border) in tabs.iter_mut() {
        let active = tab.tab == state.tab;
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Secondary, *interaction, false, active);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

pub(super) fn sync_developer_visual_page_controls(
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut buttons: Query<
        (
            &SettingsDeveloperVisualPageButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
    mut pager_texts: Query<&mut Text, With<SettingsDeveloperVisualPagerText>>,
) {
    let page_text = if artifact_previews.total_count == 0 {
        "page 0/0 | 0 images".to_string()
    } else {
        format!(
            "page {}/{} | {} images | latest first",
            artifact_previews.page + 1,
            artifact_previews.page_count.max(1),
            artifact_previews.total_count
        )
    };
    for mut text in &mut pager_texts {
        text.0 = page_text.clone();
    }

    for (button, interaction, children, mut bg, mut border) in &mut buttons {
        let disabled = artifact_previews.total_count == 0
            || match button.direction {
                DeveloperVisualPageDirection::Previous => artifact_previews.page == 0,
                DeveloperVisualPageDirection::Next => {
                    artifact_previews.page + 1 >= artifact_previews.page_count.max(1)
                }
            };
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Nav, *interaction, disabled, false);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

pub(super) fn sync_settings_developer_visual_grid(
    mut commands: Commands,
    children: Query<&Children>,
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut grids: Query<(Entity, &mut SettingsDeveloperVisualGrid)>,
) {
    for (entity, mut grid) in &mut grids {
        if grid.signature == artifact_previews.signature {
            continue;
        }
        despawn_children_recursive(entity, &mut commands, &children);
        grid.signature = artifact_previews.signature.clone();
        commands.entity(entity).with_children(|parent| {
            if artifact_previews.previews.is_empty() {
                parent.spawn((
                    Text::new("no image artifacts discovered for the active run"),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.62, 0.66, 0.74)),
                ));
                return;
            }
            for preview in artifact_previews.previews.iter() {
                spawn_developer_visual_preview_row(parent, preview);
            }
        });
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn update_settings_labels(
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    mut profile_labels: Query<
        &mut Text,
        (
            With<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut value_labels: Query<
        (&TripoSplatSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut triposg_value_labels: Query<
        (&TripoSgSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut trellis_quality_labels: Query<
        &mut Text,
        (
            With<TrellisQualityValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut trellis_value_labels: Query<
        (&TrellisSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_quality_labels: Query<
        &mut Text,
        (
            With<SceneQualityValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneToggleValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_value_labels: Query<
        (&SceneSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneToggleValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_toggle_labels: Query<
        (&SceneToggleValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_image_model_labels: Query<
        &mut Text,
        (
            With<SceneImageTo3dModelValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneToggleValueLabel>,
        ),
    >,
) {
    if let Some(args) = args {
        for mut label in profile_labels.iter_mut() {
            let next = triposplat_profile_label(args.triposplat_profile).to_string();
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in value_labels.iter_mut() {
            let next = triposplat_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in triposg_value_labels.iter_mut() {
            let next = triposg_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
        for mut label in trellis_quality_labels.iter_mut() {
            let next = trellis_quality_value_text(args.trellis_quality);
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in trellis_value_labels.iter_mut() {
            let next = trellis_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
    }
    for mut label in scene_quality_labels.iter_mut() {
        let next = scene_settings.quality_profile.label().to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in scene_value_labels.iter_mut() {
        let next = scene_setting_value_text(&scene_settings, value.setting);
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in scene_toggle_labels.iter_mut() {
        let next = scene_toggle_value_text(&scene_settings, value.setting);
        if label.0 != next {
            label.0 = next;
        }
    }
    for mut label in scene_image_model_labels.iter_mut() {
        let next = pipeline_label(scene_settings.image_to_3d_model).to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn update_viewer_debug_labels(
    viewer_debug: Res<ViewerDebugSettings>,
    mut aabb_labels: Query<
        &mut Text,
        (
            With<ViewerAabbModeValueLabel>,
            Without<ViewerDebugToggleValueLabel>,
            Without<ViewerDebugNumericValueLabel>,
        ),
    >,
    mut toggle_labels: Query<
        (&ViewerDebugToggleValueLabel, &mut Text),
        (
            Without<ViewerAabbModeValueLabel>,
            Without<ViewerDebugNumericValueLabel>,
        ),
    >,
    mut numeric_labels: Query<
        (&ViewerDebugNumericValueLabel, &mut Text),
        (
            Without<ViewerAabbModeValueLabel>,
            Without<ViewerDebugToggleValueLabel>,
        ),
    >,
) {
    for mut label in aabb_labels.iter_mut() {
        let next = viewer_debug.aabb_overlay.label().to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in toggle_labels.iter_mut() {
        let next = match value.setting {
            ViewerDebugToggleSetting::GroundContact => {
                if viewer_debug.draw_ground_contact {
                    "on"
                } else {
                    "off"
                }
            }
            ViewerDebugToggleSetting::SceneCameraFrustum => {
                if viewer_debug.draw_scene_camera_frustum {
                    "on"
                } else {
                    "off"
                }
            }
            ViewerDebugToggleSetting::DepthCloud => {
                if viewer_debug.depth_cloud_overlay {
                    "on"
                } else {
                    "off"
                }
            }
        }
        .to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in numeric_labels.iter_mut() {
        let next = match value.setting {
            ViewerDebugNumericSetting::GroundY => format!("{:.2}", viewer_debug.ground_y),
            ViewerDebugNumericSetting::ContactTolerance => {
                format!("{:.2}", viewer_debug.contact_tolerance)
            }
            ViewerDebugNumericSetting::SceneCameraFrustumLength => {
                format!("{:.2}", viewer_debug.scene_camera_frustum_length)
            }
            ViewerDebugNumericSetting::DepthCloudMaxGaussians => {
                format!("{}", viewer_debug.depth_cloud_max_gaussians)
            }
        };
        if label.0 != next {
            label.0 = next;
        }
    }
}
