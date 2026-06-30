use super::*;

pub(super) fn handle_open_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<OpenImageButton>)>,
    mut commands: Commands,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            commands
                .dialog()
                .set_title("Open Image")
                .add_filter(
                    "Images",
                    &[
                        "png", "jpg", "jpeg", "bmp", "gif", "webp", "tga", "tif", "tiff",
                    ],
                )
                .load_multiple_files::<ImagePickDialog>();
        }
    }
}

pub(super) fn handle_pipeline_selector_button(
    catalog: Res<CatalogState>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<PipelineSelectorButton>)>,
) {
    let option_count = active_pipeline_choices(catalog.active_mode(), available.as_deref()).len();
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        dropdown.open = option_count > 1 && !dropdown.open;
    }
}

pub(super) fn handle_pipeline_option_button(
    mut args: Option<ResMut<AppArgs>>,
    catalog: Res<CatalogState>,
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut modal: ResMut<SettingsModalState>,
    mut interactions: Query<(&Interaction, &PipelineOptionButton), Changed<Interaction>>,
) {
    let available_ref = available.as_deref();
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if !pipeline_available(available_ref, button.choice) {
            log::info!(
                "synthesis pipeline {} is not enabled for this app launch",
                button.choice.label()
            );
            continue;
        }
        if matches!(catalog.active_mode(), CatalogMode::Scene)
            && let CatalogPipelineChoice::Object(model) = button.choice
        {
            if !pipeline_supported(args.as_deref(), button.choice) {
                if let Some(args) = args.as_deref() {
                    log::info!(
                        "scene image-to-3d model {} is unavailable for backend {:?}",
                        pipeline_label(model),
                        args.backend
                    );
                }
                continue;
            }
            scene_settings.image_to_3d_model = model;
            dropdown.open = false;
            log::info!(
                "selected scene image-to-3d model: {}",
                pipeline_label(model)
            );
            continue;
        }
        match button.choice {
            CatalogPipelineChoice::Object(model) => {
                let Some(args) = args.as_deref_mut() else {
                    return;
                };
                if args
                    .synthesis_models
                    .first()
                    .is_some_and(|current| *current == model)
                {
                    dropdown.open = false;
                    continue;
                }
                if !pipeline_supported(Some(&*args), button.choice) {
                    log::info!(
                        "synthesis pipeline {} is unavailable for backend {:?}",
                        pipeline_label(model),
                        args.backend
                    );
                    continue;
                }
                args.synthesis_models = vec![model];
                if !pipeline_has_settings(model) {
                    modal.open = false;
                }
                if matches!(model, SynthesisModel::Triposplat)
                    && args.triposplat_profile != TripoSplatProfile::Custom
                {
                    let profile = args.triposplat_profile;
                    args.apply_triposplat_profile(profile);
                }
            }
            CatalogPipelineChoice::Scene(pipeline) => {
                scene_settings.pipeline = pipeline;
            }
        }
        dropdown.open = false;
        log::info!("selected synthesis pipeline: {}", button.choice.label());
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sync_pipeline_dropdown(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<PipelineDropdownHost>>,
    children: Query<&Children>,
) {
    ui.pipeline_menu_open = dropdown.open;
    let Some(available) = available else {
        dropdown.open = false;
        ui.pipeline_menu_open = false;
        if let Some(entity) = dropdown.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        return;
    };
    let choices = active_pipeline_choices(catalog.active_mode(), Some(&available));
    if choices.len() <= 1 {
        dropdown.open = false;
        ui.pipeline_menu_open = false;
    }

    match (dropdown.open, dropdown.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                dropdown.open = false;
                return;
            };
            dropdown.entity = Some(spawn_pipeline_dropdown(
                &mut commands,
                host,
                catalog.active_mode(),
                args.as_deref(),
                &scene_settings,
                &available,
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

pub(super) fn handle_save_scene_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SaveSceneButton>)>,
    mut menu: ResMut<SaveSceneMenuState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            menu.open = !menu.open;
        }
    }
}

pub(super) fn handle_save_scene_option_button(
    mut interactions: Query<(&Interaction, &SaveSceneOptionButton), Changed<Interaction>>,
    mut menu: ResMut<SaveSceneMenuState>,
    mut requests: MessageWriter<SceneSaveRequest>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        menu.open = false;
        requests.write(SceneSaveRequest { kind: button.kind });
    }
}

pub(super) fn sync_save_scene_menu(
    mut commands: Commands,
    mut menu: ResMut<SaveSceneMenuState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<SaveSceneButton>>,
    children: Query<&Children>,
) {
    ui.save_menu_open = menu.open;
    match (menu.open, menu.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                menu.open = false;
                ui.save_menu_open = false;
                return;
            };
            menu.entity = Some(spawn_save_scene_menu(&mut commands, host));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            menu.entity = None;
        }
        _ => {}
    }
}

pub(super) fn spawn_save_scene_menu(commands: &mut Commands, host: Entity) -> Entity {
    let mut menu_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        menu_entity = host
            .spawn((
                SaveSceneMenuRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(32.0),
                    right: Val::Px(0.0),
                    width: Val::Px(154.0),
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
                spawn_save_scene_option(menu, SceneSaveKind::Catalog, "save to catalog");
                spawn_save_scene_option(menu, SceneSaveKind::Bsn, "save BSN");
                spawn_save_scene_option(menu, SceneSaveKind::Glb, "export GLB");
            })
            .id();
    });
    menu_entity
}

pub(super) fn spawn_save_scene_option(
    parent: &mut ChildSpawnerCommands<'_>,
    kind: SceneSaveKind,
    label: &str,
) {
    parent
        .spawn((
            Button,
            SaveSceneOptionButton { kind },
            ControlButton(ControlButtonKind::Secondary),
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(26.0),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(label),
                TextFont::from_font_size(12.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn update_pipeline_value_label(
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut labels: Query<&mut Text, With<PipelineValueLabel>>,
) {
    let selected = active_pipeline_choice(&catalog, args.as_deref(), Some(&scene_settings))
        .unwrap_or(CatalogPipelineChoice::Object(SynthesisModel::Triposg));
    let option_count = active_pipeline_choices(catalog.active_mode(), available.as_deref()).len();
    let next = pipeline_selector_value_text(selected, option_count);
    for mut label in labels.iter_mut() {
        if label.0 != next {
            label.0 = next.clone();
        }
    }
}

pub(super) fn spawn_pipeline_dropdown(
    commands: &mut Commands,
    host: Entity,
    mode: CatalogMode,
    args: Option<&AppArgs>,
    scene_settings: &ScenePipelineUiSettings,
    available: &AvailablePipelines,
) -> Entity {
    let selected_pipeline = active_pipeline_choice_for_mode(mode, args, Some(scene_settings));
    let choices = active_pipeline_choices(mode, Some(available));
    let mut dropdown_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        dropdown_entity = host
            .spawn((
                PipelineDropdownRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(PIPELINE_SELECTOR_HEIGHT + 4.0),
                    left: Val::Px(0.0),
                    width: Val::Px(PIPELINE_SELECTOR_WIDTH),
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
                for choice in choices {
                    menu.spawn((
                        Button,
                        PipelineOptionButton { choice },
                        ControlButton(ControlButtonKind::Secondary),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(26.0),
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::SpaceBetween,
                            ..default()
                        },
                        BorderColor::all(if Some(choice) == selected_pipeline {
                            BUTTON_ACTIVE_BORDER
                        } else {
                            BUTTON_BORDER
                        }),
                        BackgroundColor(if Some(choice) == selected_pipeline {
                            BUTTON_ACTIVE_BG
                        } else {
                            BUTTON_BG
                        }),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(choice.label()),
                            TextFont::from_font_size(12.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                }
            })
            .id();
    });
    dropdown_entity
}
