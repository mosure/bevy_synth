use super::*;

pub(super) fn setup_ui(mut commands: Commands, args: Option<Res<AppArgs>>) {
    let mut list_entity = Entity::PLACEHOLDER;
    let pipeline_models = available_pipeline_models(args.as_deref());
    commands.insert_resource(AvailablePipelines {
        object_models: pipeline_models,
        scene_pipelines: vec![ScenePipelineKind::Explicit],
    });

    let root = commands
        .spawn((
            UiRootNode,
            // Keep world-space picking active in empty viewport regions.
            Pickable::IGNORE,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        parent
            .spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(MENU_HEIGHT),
                    padding: UiRect::horizontal(Val::Px(14.0)),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(MENU_BG),
            ))
            .with_children(|menu| {
                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(12.0),
                    ..default()
                })
                .with_children(|left| {
                    left.spawn((
                        Text::new("bevy_synth"),
                        TextFont::from_font_size(16.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ));

                    left.spawn((
                        Button,
                        OpenImageButton,
                        ControlButton(ControlButtonKind::Primary),
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BorderColor::all(BUTTON_OPEN_BORDER),
                        BackgroundColor(BUTTON_OPEN_BG),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new("open image"),
                            TextFont::from_font_size(13.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                });

                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|right| {
                    right
                        .spawn((
                            Button,
                            SaveSceneButton,
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                position_type: PositionType::Relative,
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                overflow: Overflow::visible(),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("save scene"),
                                TextFont::from_font_size(13.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });

                    right
                        .spawn((
                            Button,
                            SettingsButton,
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("settings"),
                                TextFont::from_font_size(13.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });

                    right
                        .spawn((
                            Node {
                                width: Val::Px(STATUS_BADGE_WIDTH),
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::FlexStart,
                                column_gap: Val::Px(7.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                overflow: Overflow::clip_x(),
                                ..default()
                            },
                            BackgroundColor(STATUS_BADGE_BG),
                            BorderColor::all(STATUS_BADGE_BORDER),
                            QueueStatusBadge,
                        ))
                        .with_children(|badge| {
                            badge.spawn((
                                Node {
                                    width: Val::Px(8.0),
                                    height: Val::Px(8.0),
                                    ..default()
                                },
                                BackgroundColor(STATUS_IDLE),
                                QueueStatusDot,
                            ));
                            badge.spawn((
                                Text::new("idle"),
                                TextFont::from_font_size(13.0),
                                TextColor(Color::srgb(0.72, 0.76, 0.84)),
                                QueueText,
                            ));
                        });
                });
            });

        parent
            .spawn((
                ProcessingPanelRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(MENU_HEIGHT + 12.0),
                    right: Val::Px(12.0),
                    width: Val::Px(330.0),
                    max_height: Val::Px(220.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(8.0),
                    padding: UiRect::all(Val::Px(12.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                Visibility::Hidden,
                BackgroundColor(Color::srgba(0.045, 0.05, 0.065, 0.94)),
                BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("processing"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.88, 0.91, 0.96)),
                    ProcessingCurrentText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.72, 0.78, 0.88)),
                    ProcessingTimelineText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(10.0),
                    TextColor(Color::srgb(0.58, 0.66, 0.76)),
                    ProcessingArtifactText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.96, 0.62, 0.62)),
                    ProcessingErrorText,
                ));
            });

        parent
            .spawn((
                Node {
                    width: Val::Px(PANEL_WIDTH),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(12.0),
                    padding: UiRect::all(Val::Px(14.0)),
                    border: UiRect::right(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(PANEL_BG),
                BorderColor::all(PANEL_BORDER),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::FlexStart,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|header| {
                        header
                            .spawn((
                                CatalogModeDropdownHost,
                                Node {
                                    position_type: PositionType::Relative,
                                    width: Val::Px(CATALOG_MODE_SELECTOR_WIDTH),
                                    height: Val::Px(28.0),
                                    overflow: Overflow::visible(),
                                    ..default()
                                },
                            ))
                            .with_children(|host| {
                                host.spawn((
                                    Button,
                                    CatalogModeSelectorButton,
                                    ControlButton(ControlButtonKind::Secondary),
                                    Node {
                                        width: Val::Percent(100.0),
                                        height: Val::Percent(100.0),
                                        padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                        border: UiRect::all(Val::Px(1.0)),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::Center,
                                        ..default()
                                    },
                                    BorderColor::all(BUTTON_ACTIVE_BORDER),
                                    BackgroundColor(BUTTON_ACTIVE_BG),
                                ))
                                .with_children(|button| {
                                    button.spawn((
                                        Text::new(CatalogMode::Object.label()),
                                        TextFont::from_font_size(13.0),
                                        TextColor(BUTTON_TEXT),
                                        CatalogModeValueLabel,
                                        ButtonLabel,
                                    ));
                                });
                            });
                        header
                            .spawn(Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(6.0),
                                ..default()
                            })
                            .with_children(|controls| {
                                controls
                                    .spawn((
                                        Button,
                                        CatalogPrevButton,
                                        ControlButton(ControlButtonKind::Nav),
                                        Node {
                                            width: Val::Px(CATALOG_NAV_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("<"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls.spawn((
                                    Text::new("1/1"),
                                    TextFont::from_font_size(12.0),
                                    TextColor(Color::srgb(0.78, 0.82, 0.9)),
                                    Node {
                                        width: Val::Px(CATALOG_PAGE_LABEL_WIDTH),
                                        justify_content: JustifyContent::Center,
                                        ..default()
                                    },
                                    PageLabel,
                                ));
                                controls
                                    .spawn((
                                        Button,
                                        CatalogNextButton,
                                        ControlButton(ControlButtonKind::Nav),
                                        Node {
                                            width: Val::Px(CATALOG_NAV_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new(">"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls
                                    .spawn((
                                        Button,
                                        CatalogDeleteButton,
                                        ControlButton(ControlButtonKind::Secondary),
                                        Node {
                                            width: Val::Px(CATALOG_DELETE_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("delete"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls
                                    .spawn((
                                        Button,
                                        CatalogToggleButton,
                                        ControlButton(ControlButtonKind::Secondary),
                                        Node {
                                            width: Val::Px(CATALOG_TOGGLE_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("hide"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ToggleLabel,
                                            ButtonLabel,
                                        ));
                                    });
                            });
                    });

                let list = panel
                    .spawn((
                        Node {
                            width: Val::Percent(100.0),
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(ENTRY_GAP),
                            flex_grow: 1.0,
                            overflow: Overflow::clip_y(),
                            ..default()
                        },
                        CatalogList,
                    ))
                    .id();
                list_entity = list;
            });
    });

    commands.insert_resource(CatalogUiState {
        list_entity,
        last_revision: 0,
        last_expanded: true,
        panel_width: PANEL_WIDTH,
        catalog_mode_menu_open: false,
        settings_modal_open: false,
        source_modal_open: false,
        pipeline_menu_open: false,
        save_menu_open: false,
    });
}

#[allow(clippy::type_complexity)]
pub(super) fn update_queue_text(
    queue: Res<InferenceQueue>,
    status: Option<Res<UiStatus>>,
    mut query: Query<(&mut Text, &mut TextColor), With<QueueText>>,
    mut dots: Query<&mut BackgroundColor, (With<QueueStatusDot>, Without<QueueStatusBadge>)>,
    mut badges: Query<
        (&mut BackgroundColor, &mut BorderColor),
        (With<QueueStatusBadge>, Without<QueueStatusDot>),
    >,
) {
    let (text, text_color, dot_color, badge_bg, badge_border) = if let Some(worker_message) = status
        .as_ref()
        .and_then(|state| state.worker_message.as_ref())
    {
        let is_failure = worker_message.to_ascii_lowercase().contains("failed");
        (
            compact_worker_status_text(worker_message),
            if is_failure {
                Color::srgb(0.96, 0.72, 0.72)
            } else {
                Color::srgb(0.76, 0.86, 0.98)
            },
            if is_failure {
                Color::srgb(0.86, 0.28, 0.28)
            } else {
                STATUS_PENDING
            },
            if is_failure {
                Color::srgb(0.22, 0.1, 0.1)
            } else {
                Color::srgb(0.08, 0.15, 0.2)
            },
            if is_failure {
                Color::srgb(0.58, 0.23, 0.23)
            } else {
                Color::srgb(0.2, 0.4, 0.55)
            },
        )
    } else if let Some(active) = queue.active.as_ref() {
        let name = if active.len() == 1 {
            active[0]
                .image_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image")
                .to_string()
        } else {
            format!("{} images", active.len())
        };
        (
            format!("processing: {name} | queued: {}", queue.pending.len()),
            Color::srgb(0.95, 0.89, 0.74),
            STATUS_PROCESSING,
            Color::srgb(0.21, 0.15, 0.08),
            Color::srgb(0.58, 0.41, 0.18),
        )
    } else if !queue.pending.is_empty() {
        (
            format!("queued: {}", queue.pending.len()),
            Color::srgb(0.76, 0.86, 0.98),
            STATUS_PENDING,
            Color::srgb(0.08, 0.15, 0.2),
            Color::srgb(0.2, 0.4, 0.55),
        )
    } else {
        (
            "idle".to_string(),
            Color::srgb(0.72, 0.76, 0.84),
            STATUS_IDLE,
            STATUS_BADGE_BG,
            STATUS_BADGE_BORDER,
        )
    };

    for (mut node, mut color) in query.iter_mut() {
        if node.0 != text {
            node.0 = text.clone();
        }
        if color.0 != text_color {
            color.0 = text_color;
        }
    }
    for mut dot in dots.iter_mut() {
        if dot.0 != dot_color {
            dot.0 = dot_color;
        }
    }
    for (mut bg, mut border) in badges.iter_mut() {
        if bg.0 != badge_bg {
            bg.0 = badge_bg;
        }
        *border = BorderColor::all(badge_border);
    }
}
