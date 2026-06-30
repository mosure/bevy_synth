use super::*;

pub(super) fn handle_catalog_toggle(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogToggleButton>)>,
    mut catalog: ResMut<CatalogState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            catalog.expanded = !catalog.expanded;
            catalog.bump_revision();
        }
    }
}

pub(super) fn handle_catalog_mode_selector_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogModeSelectorButton>)>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            dropdown.open = !dropdown.open;
        }
    }
}

pub(super) fn handle_catalog_mode_option_button(
    mut interactions: Query<(&Interaction, &CatalogModeOptionButton), Changed<Interaction>>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
    mut pipeline_dropdown: ResMut<PipelineDropdownState>,
    mut settings: ResMut<SettingsModalState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut catalog: ResMut<CatalogState>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        catalog.set_active_mode(button.mode);
        selection.selected = None;
        selection.last_pressed = None;
        drag.active = None;
        drag.ghost_entry = None;
        dropdown.open = false;
        pipeline_dropdown.open = false;
        settings.open = false;
    }
}

pub(super) fn sync_catalog_mode_dropdown(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<CatalogModeDropdownHost>>,
    children: Query<&Children>,
) {
    ui.catalog_mode_menu_open = dropdown.open;
    match (dropdown.open, dropdown.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                dropdown.open = false;
                ui.catalog_mode_menu_open = false;
                return;
            };
            dropdown.entity = Some(spawn_catalog_mode_dropdown(
                &mut commands,
                host,
                catalog.active_mode(),
            ));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            dropdown.entity = None;
        }
        _ => {}
    }
}

pub(super) fn update_catalog_mode_value_label(
    catalog: Res<CatalogState>,
    mut labels: Query<&mut Text, With<CatalogModeValueLabel>>,
) {
    let next = catalog.active_mode().label().to_string();
    for mut label in labels.iter_mut() {
        if label.0 != next {
            label.0 = next.clone();
        }
    }
}

pub(super) fn spawn_catalog_mode_dropdown(
    commands: &mut Commands,
    host: Entity,
    active: CatalogMode,
) -> Entity {
    let mut menu_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        menu_entity = host
            .spawn((
                CatalogModeDropdownRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(32.0),
                    left: Val::Px(0.0),
                    width: Val::Px(104.0),
                    padding: UiRect::all(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    ..default()
                },
                ZIndex(100),
                GlobalZIndex(20_000),
                BorderColor::all(PANEL_BORDER),
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|menu| {
                for mode in [CatalogMode::Object, CatalogMode::Scene] {
                    menu.spawn((
                        Button,
                        CatalogModeOptionButton { mode },
                        ControlButton(ControlButtonKind::Secondary),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(26.0),
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BorderColor::all(if mode == active {
                            BUTTON_ACTIVE_BORDER
                        } else {
                            BUTTON_BORDER
                        }),
                        BackgroundColor(if mode == active {
                            BUTTON_ACTIVE_BG
                        } else {
                            BUTTON_BG
                        }),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(mode.label()),
                            TextFont::from_font_size(12.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                }
            })
            .id();
    });
    menu_entity
}

pub(super) fn rebuild_catalog_list(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
    mut toggle_query: Query<&mut Text, (With<ToggleLabel>, Without<PageLabel>)>,
    mut page_labels: Query<&mut Text, (With<PageLabel>, Without<ToggleLabel>)>,
) {
    if catalog.revision == ui.last_revision && catalog.expanded == ui.last_expanded {
        return;
    }

    ui.last_revision = catalog.revision;
    ui.last_expanded = catalog.expanded;

    for mut label in toggle_query.iter_mut() {
        label.0 = if catalog.expanded {
            "hide".to_string()
        } else {
            "show".to_string()
        };
    }
    let page_count = catalog.page_count();
    let page_index = catalog.page().saturating_add(1);
    for mut label in page_labels.iter_mut() {
        label.0 = format!("{}/{}", page_index, page_count);
    }

    despawn_children_recursive(ui.list_entity, &mut commands, &children);
    if !catalog.expanded {
        return;
    }

    let indices = catalog.visible_indices();
    commands.entity(ui.list_entity).with_children(|parent| {
        if indices.is_empty() {
            let (empty_title, empty_hint) = match catalog.active_mode() {
                CatalogMode::Object => (
                    "No object catalog items yet",
                    "Drop an image, or click open image to queue one.",
                ),
                CatalogMode::Scene => (
                    "No saved scenes yet",
                    "Open an image in scene mode to run the explicit scene pipeline.",
                ),
            };
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        padding: UiRect::all(Val::Px(12.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(4.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.08, 0.09, 0.12)),
                    BorderColor::all(Color::srgb(0.2, 0.22, 0.28)),
                ))
                .with_children(|empty| {
                    empty.spawn((
                        Text::new(empty_title),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.88, 0.9, 0.95)),
                    ));
                    empty.spawn((
                        Text::new(empty_hint),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.66, 0.7, 0.78)),
                    ));
                });
            return;
        }

        for &index in indices.iter() {
            let entry = &catalog.entries[index];
            let (status_label, status_color) = match &entry.status {
                CatalogStatus::Pending => ("pending".to_string(), Color::srgb(0.9, 0.7, 0.2)),
                CatalogStatus::Ready if entry.kind == CatalogEntryKind::Scene => {
                    let mut label = scene_entry_status_text(entry);
                    label = ellipsize_text(&label, CATALOG_STATUS_MAX_CHARS);
                    (label, Color::srgb(0.4, 0.85, 0.55))
                }
                CatalogStatus::Ready => ("ready".to_string(), Color::srgb(0.4, 0.85, 0.55)),
                CatalogStatus::Failed(err) => {
                    let mut label = if err.is_empty() {
                        "failed".to_string()
                    } else {
                        format!("failed: {err}")
                    };
                    label = ellipsize_text(&label, CATALOG_STATUS_MAX_CHARS);
                    (label, Color::srgb(0.9, 0.3, 0.3))
                }
            };
            let display_label = ellipsize_text(&entry.label, CATALOG_LABEL_MAX_CHARS);
            parent
                .spawn((
                    Button,
                    CatalogEntryButton { id: entry.id },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(THUMB_SIZE + 16.0),
                        padding: UiRect::all(Val::Px(8.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        column_gap: Val::Px(10.0),
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(ENTRY_BG),
                    BorderColor::all(ENTRY_BORDER),
                ))
                .with_children(|row| {
                    if let Some(preview) = entry.preview.as_ref() {
                        row.spawn((
                            Node {
                                width: Val::Px(THUMB_SIZE),
                                height: Val::Px(THUMB_SIZE),
                                ..default()
                            },
                            ImageNode::new(preview.image.clone()),
                        ));
                    } else {
                        row.spawn((
                            Node {
                                width: Val::Px(THUMB_SIZE),
                                height: Val::Px(THUMB_SIZE),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.08, 0.09, 0.12)),
                            BorderColor::all(Color::srgb(0.22, 0.24, 0.3)),
                        ))
                        .with_children(|pending| {
                            pending.spawn((
                                Text::new("pending"),
                                TextFont::from_font_size(12.0),
                                TextColor(Color::srgb(0.7, 0.73, 0.8)),
                            ));
                        });
                    }

                    row.spawn(Node {
                        width: Val::Px(PANEL_WIDTH - THUMB_SIZE - 96.0),
                        min_width: Val::Px(0.0),
                        flex_direction: FlexDirection::Column,
                        justify_content: JustifyContent::Center,
                        padding: UiRect::left(Val::Px(4.0)),
                        row_gap: Val::Px(4.0),
                        overflow: Overflow::clip(),
                        ..default()
                    })
                    .with_children(|text_col| {
                        text_col.spawn((
                            Text::new(display_label.clone()),
                            TextFont::from_font_size(13.0),
                            TextColor(Color::srgb(0.9, 0.92, 0.97)),
                        ));
                        text_col.spawn((
                            Text::new(status_label.clone()),
                            TextFont::from_font_size(12.0),
                            TextColor(status_color),
                        ));
                    });

                    if entry.kind == CatalogEntryKind::Scene && !entry.is_unsaved_scene() {
                        row.spawn((
                            Button,
                            CatalogSceneLoadButton { id: entry.id },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                width: Val::Px(46.0),
                                height: Val::Px(26.0),
                                margin: UiRect::left(Val::Auto),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("load"),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    } else {
                        row.spawn((
                            Node {
                                width: Val::Px(10.0),
                                height: Val::Px(10.0),
                                margin: UiRect::left(Val::Auto),
                                ..default()
                            },
                            BackgroundColor(status_color),
                        ));
                    }
                });
        }
    });
}

pub(super) fn scene_entry_status_text(entry: &CatalogEntry) -> String {
    if entry.is_unsaved_scene() {
        return "current world".to_string();
    }
    let Some(metrics) = entry.scene_metrics.as_ref() else {
        return entry
            .scene_pipeline
            .as_deref()
            .unwrap_or("scene")
            .to_string();
    };
    let mut parts = Vec::new();
    if let Some(count) = metrics.object_count.or(metrics.placement_count) {
        parts.push(format!("{count} objects"));
    }
    if let Some(elapsed_ms) = metrics.elapsed_ms {
        parts.push(format!("{:.1}s", elapsed_ms as f32 / 1000.0));
    }
    if metrics.ok == Some(false) {
        parts.push("needs review".to_string());
    }
    if parts.is_empty() {
        entry
            .scene_pipeline
            .as_deref()
            .unwrap_or("scene")
            .to_string()
    } else {
        parts.join(" | ")
    }
}

pub(super) fn despawn_children_recursive(
    entity: Entity,
    commands: &mut Commands,
    children_query: &Query<&Children>,
) {
    let Ok(children) = children_query.get(entity) else {
        return;
    };
    for child in children.iter() {
        despawn_children_recursive(child, commands, children_query);
        commands.entity(child).despawn();
    }
}

pub(super) fn handle_catalog_entry_interaction(
    mut interactions: Query<(&Interaction, &CatalogEntryButton), Changed<Interaction>>,
    time: Res<Time>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut source_modal: ResMut<CatalogSourceImageModalState>,
) {
    for (interaction, entry) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            let now = time.elapsed_secs_f64();
            let double_click = selection.last_pressed.is_some_and(|(id, last)| {
                id == entry.id && now - last <= CATALOG_DOUBLE_CLICK_SECONDS
            });
            selection.last_pressed = Some((entry.id, now));
            if double_click {
                drag.active = None;
                drag.ghost_entry = None;
                source_modal.entry_id = Some(entry.id);
                source_modal.tab = CatalogSourceImageTab::Image;
                return;
            } else {
                selection.selected = Some(entry.id);
                let is_object = catalog
                    .entry(entry.id)
                    .is_some_and(|entry| entry.kind == CatalogEntryKind::Object);
                drag.active = is_object.then_some(entry.id);
                drag.ghost_entry = None;
            }
        }
    }
}

pub(super) fn handle_catalog_scene_load_button(
    mut interactions: Query<(&Interaction, &CatalogSceneLoadButton), Changed<Interaction>>,
    catalog: Res<CatalogState>,
    mut requests: MessageWriter<SceneLoadRequest>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(entry) = catalog.entry(button.id) else {
            continue;
        };
        let Some(scene_key) = entry.scene_key.clone() else {
            continue;
        };
        requests.write(SceneLoadRequest { scene_key });
    }
}

pub(super) fn handle_source_image_modal_close_button(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<CatalogSourceImageCloseButton>),
    >,
    mut modal: ResMut<CatalogSourceImageModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.entry_id = None;
        }
    }
}

pub(super) fn handle_source_image_modal_tab_button(
    mut modal: ResMut<CatalogSourceImageModalState>,
    mut interactions: Query<(&Interaction, &CatalogSourceImageTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.tab = button.tab;
        }
    }
}

pub(super) fn handle_source_image_modal_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut modal: ResMut<CatalogSourceImageModalState>,
) {
    if modal.entry_id.is_some() && keys.just_pressed(KeyCode::Escape) {
        modal.entry_id = None;
    }
}

pub(super) fn sync_source_image_modal(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut modal: ResMut<CatalogSourceImageModalState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
) {
    if modal.entry_id.and_then(|id| catalog.entry(id)).is_none() {
        modal.entry_id = None;
    }
    ui.source_modal_open = modal.entry_id.is_some();

    if modal.entity.is_some() && modal.rendered_entry_id != modal.entry_id {
        if let Some(entity) = modal.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        modal.rendered_entry_id = None;
    }

    match (modal.entry_id, modal.entity) {
        (Some(id), None) => {
            if let Some(entry) = catalog.entry(id) {
                modal.entity = Some(spawn_source_image_modal(&mut commands, entry));
                modal.rendered_entry_id = Some(id);
            }
        }
        (None, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            modal.entity = None;
            modal.rendered_entry_id = None;
            modal.tab = CatalogSourceImageTab::Image;
        }
        (Some(_), Some(_)) => {}
        _ => {}
    }
}

pub(super) fn sync_source_image_modal_tab_visuals(
    modal: Res<CatalogSourceImageModalState>,
    mut panels: Query<(&CatalogSourceImageTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &CatalogSourceImageTabButton,
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

    for (button, interaction, children, mut background, mut border) in tabs.iter_mut() {
        let active = button.tab == modal.tab;
        let (bg, br) = if active {
            (BUTTON_ACTIVE_BG, BUTTON_ACTIVE_BORDER)
        } else {
            match *interaction {
                Interaction::Pressed => (BUTTON_BG_PRESSED, BUTTON_BORDER_PRESSED),
                Interaction::Hovered => (BUTTON_BG_HOVER, BUTTON_BORDER_HOVER),
                Interaction::None => (BUTTON_BG, BUTTON_BORDER),
            }
        };
        *background = BackgroundColor(bg);
        *border = BorderColor::all(br);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child) {
                label.0 = if active { Color::WHITE } else { BUTTON_TEXT };
            }
        }
    }
}

pub(super) fn delete_catalog_entry(
    id: u32,
    catalog: &mut CatalogState,
    selection: &mut CatalogSelectionState,
    drag: &mut DragState,
    commands: &mut Commands,
    delete_requests: &mut MessageWriter<CatalogDeleteRequest>,
    scene_delete_requests: &mut MessageWriter<SceneDeleteRequest>,
) {
    let Some(entry) = catalog.remove_entry(id) else {
        if selection.selected == Some(id) {
            selection.selected = None;
        }
        if drag.active == Some(id) {
            drag.active = None;
        }
        if drag.ghost_entry == Some(id) {
            clear_drag_ghost(drag, commands);
        }
        return;
    };
    if entry.is_unsaved_scene() {
        catalog.entries.push(entry);
        catalog.clamp_page();
        catalog.bump_revision();
        return;
    }

    if let Some(preview) = entry.preview {
        for entity in preview.asset_entities {
            commands.entity(entity).despawn();
        }
        commands.entity(preview.camera_entity).despawn();
        for light in preview.light_entities {
            commands.entity(light).despawn();
        }
        catalog.release_preview_layer(preview.layer_index);
    }
    if selection.selected == Some(id) {
        selection.selected = None;
    }
    if drag.active == Some(id) {
        drag.active = None;
    }
    if drag.ghost_entry == Some(id) {
        clear_drag_ghost(drag, commands);
    }

    match entry.kind {
        CatalogEntryKind::Object => {
            delete_requests.write(CatalogDeleteRequest {
                cache_key: entry.cache_key,
            });
        }
        CatalogEntryKind::Scene => {
            if let Some(scene_key) = entry.scene_key {
                scene_delete_requests.write(SceneDeleteRequest { scene_key });
            }
        }
    }
}

pub(super) fn handle_catalog_delete_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogDeleteButton>)>,
    mut catalog: ResMut<CatalogState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut delete_requests: MessageWriter<CatalogDeleteRequest>,
    mut scene_delete_requests: MessageWriter<SceneDeleteRequest>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(id) = selection.selected else {
            continue;
        };
        delete_catalog_entry(
            id,
            &mut catalog,
            &mut selection,
            &mut drag,
            &mut commands,
            &mut delete_requests,
            &mut scene_delete_requests,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_catalog_delete_shortcut(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    ui_state: Res<CatalogUiState>,
    mut catalog: ResMut<CatalogState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut delete_requests: MessageWriter<CatalogDeleteRequest>,
    mut scene_delete_requests: MessageWriter<SceneDeleteRequest>,
) {
    if !keys.just_pressed(KeyCode::Delete) && !keys.just_pressed(KeyCode::Backspace) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    if !ui_state.cursor_over_ui(window) {
        return;
    }
    let Some(id) = selection.selected else {
        return;
    };
    delete_catalog_entry(
        id,
        &mut catalog,
        &mut selection,
        &mut drag,
        &mut commands,
        &mut delete_requests,
        &mut scene_delete_requests,
    );
}

pub(super) fn handle_page_buttons(
    mut prev: Query<&Interaction, (Changed<Interaction>, With<CatalogPrevButton>)>,
    mut next: Query<&Interaction, (Changed<Interaction>, With<CatalogNextButton>)>,
    mut catalog: ResMut<CatalogState>,
) {
    let mut changed = false;
    for interaction in prev.iter_mut() {
        if *interaction == Interaction::Pressed {
            let page = catalog.page();
            if page > 0 {
                catalog.set_page(page - 1);
                changed = true;
            }
        }
    }
    for interaction in next.iter_mut() {
        if *interaction == Interaction::Pressed {
            let page = catalog.page();
            if page + 1 < catalog.page_count() {
                catalog.set_page(page + 1);
                changed = true;
            }
        }
    }
    if changed {
        catalog.bump_revision();
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn update_button_visuals(
    catalog: Res<CatalogState>,
    drag: Res<DragState>,
    args: Option<Res<AppArgs>>,
    available: Option<Res<AvailablePipelines>>,
    modal: Res<SettingsModalState>,
    viewer_debug: Res<ViewerDebugSettings>,
    mut selection: ResMut<CatalogSelectionState>,
    mut controls: Query<
        (
            &Interaction,
            &ControlButton,
            Option<&CatalogPrevButton>,
            Option<&CatalogNextButton>,
            Option<&CatalogDeleteButton>,
            Option<&PipelineSelectorButton>,
            Option<&PipelineOptionButton>,
            Option<&SettingsButton>,
            Option<&TripoSplatProfileButton>,
            Option<&TrellisQualityButton>,
            Option<&TrellisPbrToggleButton>,
            Option<&ViewerAabbModeButton>,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
    mut entries: Query<
        (
            &Interaction,
            &CatalogEntryButton,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        (With<Button>, Without<ControlButton>),
    >,
) {
    if let Some(selected) = selection.selected
        && catalog.entry(selected).is_none()
    {
        selection.selected = None;
    }

    let args_ref = args.as_deref();
    let available_ref = available.as_deref();
    let selected_pipeline = active_pipeline_choice(&catalog, args_ref, None);
    let settings_enabled = pipeline_settings_enabled(&catalog, args_ref);

    for (
        interaction,
        button,
        prev,
        next,
        delete,
        pipeline_selector,
        pipeline_option,
        settings_button,
        profile,
        trellis_quality,
        trellis_pbr,
        viewer_aabb,
        children,
        mut bg,
        mut border,
    ) in controls.iter_mut()
    {
        let disabled = if prev.is_some() {
            catalog.page() == 0
        } else if next.is_some() {
            catalog.page() + 1 >= catalog.page_count()
        } else if delete.is_some() {
            selection.selected.is_none()
        } else if let Some(pipeline_option) = pipeline_option {
            !pipeline_available(available_ref, pipeline_option.choice)
                || !pipeline_supported(args_ref, pipeline_option.choice)
        } else if settings_button.is_some() {
            !settings_enabled
        } else {
            false
        };
        let active = pipeline_selector.is_some()
            || pipeline_option.is_some_and(|pipeline| Some(pipeline.choice) == selected_pipeline)
            || settings_button.is_some_and(|_| settings_enabled && modal.open)
            || profile
                .zip(args_ref)
                .is_some_and(|(profile, args)| profile.profile == args.triposplat_profile)
            || trellis_quality
                .zip(args_ref)
                .is_some_and(|(button, args)| button.quality == args.trellis_quality)
            || trellis_pbr
                .zip(args_ref)
                .is_some_and(|(_, args)| args.trellis_pbr_enabled)
            || viewer_aabb.is_some_and(|button| button.mode == viewer_debug.aabb_overlay);
        let (button_bg, button_border, text_color) =
            control_button_palette(button.0, *interaction, disabled, active);
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

    for (interaction, entry, mut bg, mut border) in entries.iter_mut() {
        let dragging_this_entry = drag.active == Some(entry.id);
        let selected = selection.selected == Some(entry.id);
        let (entry_bg, entry_border) = if dragging_this_entry {
            (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER)
        } else if selected {
            (ENTRY_BG_HOVER, ENTRY_BORDER_HOVER)
        } else {
            match *interaction {
                Interaction::Pressed => (ENTRY_BG_PRESSED, ENTRY_BORDER_PRESSED),
                Interaction::Hovered => (ENTRY_BG_HOVER, ENTRY_BORDER_HOVER),
                Interaction::None => (ENTRY_BG, ENTRY_BORDER),
            }
        };
        if bg.0 != entry_bg {
            bg.0 = entry_bg;
        }
        *border = BorderColor::all(entry_border);
    }
}

pub(super) fn control_button_palette(
    kind: ControlButtonKind,
    interaction: Interaction,
    disabled: bool,
    active: bool,
) -> (Color, Color, Color) {
    if disabled {
        return (
            BUTTON_BG_DISABLED,
            BUTTON_BORDER_DISABLED,
            BUTTON_TEXT_DISABLED,
        );
    }
    if active {
        return match interaction {
            Interaction::Pressed => (
                BUTTON_OPEN_BG_PRESSED,
                BUTTON_OPEN_BORDER_PRESSED,
                BUTTON_TEXT,
            ),
            Interaction::Hovered => (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_ACTIVE_BG, BUTTON_ACTIVE_BORDER, BUTTON_TEXT),
        };
    }

    match kind {
        ControlButtonKind::Primary => match interaction {
            Interaction::Pressed => (
                BUTTON_OPEN_BG_PRESSED,
                BUTTON_OPEN_BORDER_PRESSED,
                BUTTON_TEXT,
            ),
            Interaction::Hovered => (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_OPEN_BG, BUTTON_OPEN_BORDER, BUTTON_TEXT),
        },
        ControlButtonKind::Secondary | ControlButtonKind::Nav => match interaction {
            Interaction::Pressed => (BUTTON_BG_PRESSED, BUTTON_BORDER_PRESSED, BUTTON_TEXT),
            Interaction::Hovered => (BUTTON_BG_HOVER, BUTTON_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_BG, BUTTON_BORDER, BUTTON_TEXT),
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_drag_release(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    mut selection: ResMut<CatalogSelectionState>,
    ui_state: Res<CatalogUiState>,
    mut commands: Commands,
    mut spawn_requests: MessageWriter<CatalogSpawnRequest>,
) {
    if !buttons.just_released(MouseButton::Left) {
        return;
    }
    clear_drag_ghost(&mut drag, &mut commands);
    let Some(id) = drag.active.take() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(position) = cursor_spawn_position(window, &ui_state, camera, camera_transform) else {
        return;
    };
    let Some(entry) = catalog.entry(id) else {
        return;
    };
    let asset = if let (Some(mesh), Some(material)) = (entry.mesh.clone(), entry.material.clone()) {
        CatalogSpawnAsset::Mesh { mesh, material }
    } else if let Some(cloud) = entry.gaussian.clone() {
        CatalogSpawnAsset::GaussianSplat { cloud }
    } else {
        return;
    };
    spawn_requests.write(CatalogSpawnRequest {
        asset,
        transform: Transform::from_translation(position),
        cache_key: entry.cache_key.clone(),
        select_spawned: true,
    });
    if selection.selected == Some(id) {
        selection.selected = None;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_drag_ghost(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    ui_state: Res<CatalogUiState>,
    mut commands: Commands,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let Some(id) = drag.active else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    }

    let Ok(window) = windows.single() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(position) = cursor_spawn_position(window, &ui_state, camera, camera_transform) else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };

    let Some(entry) = catalog.entry(id) else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(mesh_handle) = entry.mesh.clone() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(materials) = materials.as_mut() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };

    if drag.ghost.is_none() || drag.ghost_entry != Some(id) {
        clear_drag_ghost(&mut drag, &mut commands);
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(0.78, 0.84, 0.92, DRAG_GHOST_ALPHA),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        let ghost = commands
            .spawn((
                DragGhost,
                Pickable::IGNORE,
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
                Transform::from_translation(position),
                RenderLayers::layer(0),
            ))
            .id();
        drag.ghost = Some(ghost);
        drag.ghost_entry = Some(id);
        return;
    }

    if let Some(ghost) = drag.ghost {
        commands
            .entity(ghost)
            .insert(Transform::from_translation(position));
    }
}

pub(super) fn clear_drag_ghost(drag: &mut DragState, commands: &mut Commands) {
    if let Some(entity) = drag.ghost.take() {
        commands.entity(entity).despawn();
    }
    drag.ghost_entry = None;
}

pub(super) fn cleanup_drag_ghosts(
    buttons: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    ghosts: Query<Entity, With<DragGhost>>,
    mut commands: Commands,
) {
    if drag.active.is_some() && buttons.pressed(MouseButton::Left) {
        return;
    }
    drag.ghost = None;
    drag.ghost_entry = None;
    for entity in ghosts.iter() {
        commands.entity(entity).despawn();
    }
}

pub(super) fn cursor_spawn_position(
    window: &Window,
    ui_state: &CatalogUiState,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<Vec3> {
    if ui_state.cursor_over_ui(window) {
        return None;
    }
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    let hit = if ray.direction.y.abs() > 0.0001 {
        let t = -ray.origin.y / ray.direction.y;
        if t.is_finite() && t > 0.0 {
            Some(ray.origin + ray.direction * t)
        } else {
            None
        }
    } else {
        None
    };
    Some(hit.unwrap_or(ray.origin + ray.direction * 4.0))
}

pub(super) fn sync_catalog_previews(
    mut catalog: ResMut<CatalogState>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    meshes: Res<Assets<BevyMesh>>,
    gaussian_clouds: Res<Assets<PlanarGaussian3d>>,
) {
    enum PreviewAction {
        Create {
            index: usize,
            asset: PreviewAsset,
            fit: PreviewFit,
        },
        Remove {
            index: usize,
        },
    }

    let visible_ids: Vec<u32> = catalog
        .visible_indices()
        .into_iter()
        .filter_map(|index| catalog.entries.get(index).map(|entry| entry.id))
        .collect();

    let mut actions = Vec::new();
    for (index, entry) in catalog.entries.iter().enumerate() {
        let should_show = visible_ids.contains(&entry.id)
            && matches!(entry.status, CatalogStatus::Ready)
            && ((entry.mesh.is_some() && entry.material.is_some())
                || entry.gaussian.is_some()
                || (entry.kind == CatalogEntryKind::Scene && !entry.scene_items.is_empty()));

        match (should_show, entry.preview.is_some()) {
            (true, false) => {
                if let (Some(mesh), Some(material)) = (entry.mesh.clone(), entry.material.clone()) {
                    let fit = meshes
                        .get(&mesh)
                        .map(preview_fit_for_mesh)
                        .unwrap_or_else(PreviewFit::fallback);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::Mesh { mesh, material },
                        fit,
                    });
                } else if let Some(cloud) = entry.gaussian.clone() {
                    let fit = gaussian_clouds
                        .get(&cloud)
                        .map(preview_fit_for_gaussian_cloud)
                        .unwrap_or_else(PreviewFit::fallback);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::GaussianSplat { cloud },
                        fit,
                    });
                } else if entry.kind == CatalogEntryKind::Scene && !entry.scene_items.is_empty() {
                    let fit =
                        preview_fit_for_scene_items(&entry.scene_items, &meshes, &gaussian_clouds);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::Scene {
                            items: entry.scene_items.clone(),
                        },
                        fit,
                    });
                }
            }
            (false, true) => actions.push(PreviewAction::Remove { index }),
            _ => {}
        }
    }

    let mut changed = false;
    for action in actions {
        match action {
            PreviewAction::Create { index, asset, fit } => {
                if let Some(layer_index) = catalog.alloc_preview_layer() {
                    let preview =
                        spawn_preview_scene(&mut commands, &mut images, asset, layer_index, fit);
                    if let Some(entry) = catalog.entries.get_mut(index) {
                        entry.preview = Some(preview);
                    }
                    changed = true;
                }
            }
            PreviewAction::Remove { index } => {
                let preview = catalog
                    .entries
                    .get_mut(index)
                    .and_then(|entry| entry.preview.take());
                if let Some(preview) = preview {
                    for entity in preview.asset_entities {
                        commands.entity(entity).despawn();
                    }
                    commands.entity(preview.camera_entity).despawn();
                    for light in preview.light_entities {
                        commands.entity(light).despawn();
                    }
                    catalog.release_preview_layer(preview.layer_index);
                    changed = true;
                }
            }
        }
    }

    if changed {
        catalog.bump_revision();
    }
}

pub(super) fn spin_thumbnails(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<ThumbnailSpin>>,
) {
    for mut transform in query.iter_mut() {
        transform.rotate_y(time.delta_secs() * 0.8);
        transform.rotate_x(time.delta_secs() * 0.3);
    }
}
