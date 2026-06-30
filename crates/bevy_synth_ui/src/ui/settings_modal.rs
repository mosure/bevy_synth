use super::*;

pub(super) fn spawn_settings_modal(
    commands: &mut Commands,
    pipeline: CatalogPipelineChoice,
    active_tab: SettingsModalTab,
    available: Option<&AvailablePipelines>,
) -> Entity {
    commands
        .spawn((
            SettingsModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_SCRIM),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(SETTINGS_MODAL_WIDTH),
                    max_height: Val::Vh(SETTINGS_MODAL_MAX_HEIGHT_VH),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(14.0),
                    padding: UiRect::all(Val::Px(16.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                BackgroundColor(MODAL_BG),
                BorderColor::all(MODAL_BORDER),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new(settings_modal_title(pipeline)),
                            TextFont::from_font_size(16.0),
                            TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ));
                        header
                            .spawn((
                                Button,
                                SettingsCloseButton,
                                ControlButton(ControlButtonKind::Secondary),
                                Node {
                                    padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                                    border: UiRect::all(Val::Px(1.0)),
                                    ..default()
                                },
                                BorderColor::all(BUTTON_BORDER),
                                BackgroundColor(BUTTON_BG),
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    Text::new("close"),
                                    TextFont::from_font_size(12.0),
                                    TextColor(BUTTON_TEXT),
                                    ButtonLabel,
                                ));
                            });
                    });

                spawn_settings_tabs(panel, pipeline);
                for tab in settings_tabs_for_pipeline(pipeline) {
                    spawn_settings_tab_panel(panel, tab, active_tab == tab, |panel| {
                        spawn_settings_tab_content(panel, pipeline, tab, available);
                    });
                }
            });
        })
        .id()
}

pub(super) fn settings_modal_title(pipeline: CatalogPipelineChoice) -> &'static str {
    match pipeline {
        CatalogPipelineChoice::Object(SynthesisModel::Triposg) => "TripoSG settings",
        CatalogPipelineChoice::Object(SynthesisModel::Triposplat) => "TripoSplat settings",
        CatalogPipelineChoice::Object(SynthesisModel::Trellis) => "Trellis.2 settings",
        CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit) => "explicit scene settings",
    }
}

pub(super) fn settings_tabs_for_pipeline(pipeline: CatalogPipelineChoice) -> Vec<SettingsModalTab> {
    match pipeline {
        CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit) => vec![
            SettingsModalTab::Pipeline,
            SettingsModalTab::Generation,
            SettingsModalTab::Grounding,
            SettingsModalTab::Debug,
            SettingsModalTab::General,
            SettingsModalTab::Physics,
            SettingsModalTab::Developer,
        ],
        CatalogPipelineChoice::Object(_) => vec![
            SettingsModalTab::Pipeline,
            SettingsModalTab::General,
            SettingsModalTab::Physics,
            SettingsModalTab::Developer,
        ],
    }
}

pub(super) fn spawn_settings_tabs(
    panel: &mut ChildSpawnerCommands,
    pipeline: CatalogPipelineChoice,
) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            row_gap: Val::Px(6.0),
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .with_children(|row| {
            for tab in settings_tabs_for_pipeline(pipeline) {
                row.spawn((
                    Button,
                    SettingsTabButton { tab },
                    Node {
                        height: Val::Px(28.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
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
                        Text::new(tab.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

pub(super) fn spawn_settings_tab_content(
    panel: &mut ChildSpawnerCommands,
    pipeline: CatalogPipelineChoice,
    tab: SettingsModalTab,
    available: Option<&AvailablePipelines>,
) {
    match (pipeline, tab) {
        (CatalogPipelineChoice::Object(SynthesisModel::Triposg), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_triposg_settings(panel);
        }
        (CatalogPipelineChoice::Object(SynthesisModel::Triposplat), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_triposplat_settings(panel);
        }
        (CatalogPipelineChoice::Object(SynthesisModel::Trellis), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_trellis_settings(panel);
        }
        (CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit), SettingsModalTab::Pipeline) => {
            spawn_scene_pipeline_settings(panel, available);
        }
        (
            CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit),
            SettingsModalTab::Generation,
        ) => {
            spawn_scene_generation_settings(panel);
        }
        (
            CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit),
            SettingsModalTab::Grounding,
        ) => {
            spawn_scene_grounding_settings(panel);
        }
        (CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit), SettingsModalTab::Debug) => {
            spawn_scene_debug_settings(panel);
        }
        (_, SettingsModalTab::General) => spawn_general_settings(panel),
        (_, SettingsModalTab::Physics) => spawn_physics_settings(panel),
        (_, SettingsModalTab::Developer) => spawn_developer_settings(panel),
        _ => {}
    }
}

pub(super) fn spawn_settings_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: SettingsModalTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            SettingsTabPanel { tab },
            SettingsScrollArea,
            Node {
                width: Val::Percent(100.0),
                max_height: Val::Vh(SETTINGS_TAB_BODY_MAX_HEIGHT_VH),
                display: if visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                overflow: Overflow::scroll_y(),
                padding: UiRect::right(Val::Px(4.0)),
                ..default()
            },
            ScrollPosition::default(),
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ))
        .with_children(spawn_content);
}

pub(super) fn spawn_object_pipeline_selector(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
    selected: CatalogPipelineChoice,
) {
    let choices = active_pipeline_choices(CatalogMode::Object, available);
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("pipeline"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(selected.label()),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        PipelineValueLabel,
                    ));
                });
            spawn_pipeline_button_row(column, choices);
        });
}

pub(super) fn spawn_scene_pipeline_selector(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new("scene pipeline"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Text::new(ScenePipelineKind::Explicit.label()),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
            ));
        });

    let choices = scene_image_to_3d_pipeline_choices(available);
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("image to 3d"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(pipeline_label(SynthesisModel::Trellis)),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        SceneImageTo3dModelValueLabel,
                    ));
                });
            spawn_pipeline_button_row(column, choices);
        });
}

pub(super) fn scene_image_to_3d_pipeline_choices(
    available: Option<&AvailablePipelines>,
) -> Vec<CatalogPipelineChoice> {
    active_pipeline_choices(CatalogMode::Object, available)
        .into_iter()
        .filter(|choice| {
            !matches!(
                choice,
                CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
            )
        })
        .collect()
}

pub(super) fn spawn_pipeline_button_row(
    parent: &mut ChildSpawnerCommands,
    choices: Vec<CatalogPipelineChoice>,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            align_items: AlignItems::Center,
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .with_children(|row| {
            for choice in choices {
                row.spawn((
                    Button,
                    PipelineOptionButton { choice },
                    ControlButton(ControlButtonKind::Secondary),
                    Node {
                        height: Val::Px(28.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
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
                        Text::new(choice.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

pub(super) fn spawn_triposplat_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("profile"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for profile in [
                        TripoSplatProfile::Low,
                        TripoSplatProfile::Balanced,
                        TripoSplatProfile::High,
                    ] {
                        row.spawn((
                            Button,
                            TripoSplatProfileButton { profile },
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
                                Text::new(triposplat_profile_label(profile)),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new("balanced"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        TripoSplatProfileValueLabel,
                    ));
                });
        });

    spawn_triposplat_setting_row(panel, "steps", TripoSplatSetting::Steps);
    spawn_triposplat_setting_row(panel, "cfg guidance", TripoSplatSetting::Guidance);
    spawn_triposplat_setting_row(panel, "gaussian count", TripoSplatSetting::Gaussians);
}

pub(super) fn spawn_triposg_settings(panel: &mut ChildSpawnerCommands) {
    spawn_triposg_setting_row(panel, "steps", TripoSgSetting::Steps);
    spawn_triposg_setting_row(panel, "tokens", TripoSgSetting::Tokens);
    spawn_triposg_setting_row(panel, "cfg guidance", TripoSgSetting::Guidance);
    spawn_triposg_setting_row(panel, "target faces", TripoSgSetting::TargetFaces);
}

pub(super) fn spawn_trellis_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("quality"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for quality in [
                        TrellisQuality::Low,
                        TrellisQuality::Medium,
                        TrellisQuality::High,
                    ] {
                        row.spawn((
                            Button,
                            TrellisQualityButton { quality },
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
                                Text::new(trellis_quality_label(quality)),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new(trellis_quality_value_text(TrellisQuality::Low)),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        TrellisQualityValueLabel,
                    ));
                });
        });

    spawn_trellis_value_row(panel, "resolution", TrellisSetting::Resolution);
    spawn_trellis_toggle_row(panel, "pbr textures");
    spawn_trellis_setting_row(panel, "pbr texture size", TrellisSetting::PbrTextureSize);
    spawn_trellis_setting_row(panel, "target faces", TrellisSetting::TargetFaces);
    spawn_trellis_setting_row(panel, "sparse cap", TrellisSetting::MaxSparseCoords);
}

pub(super) fn spawn_scene_pipeline_settings(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
) {
    spawn_scene_pipeline_selector(panel, available);
    spawn_scene_quality_settings(panel);
}

pub(super) fn spawn_scene_quality_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("quality"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for quality in [
                        SceneQualityProfileSetting::Fast,
                        SceneQualityProfileSetting::Balanced,
                        SceneQualityProfileSetting::Full,
                    ] {
                        row.spawn((
                            Button,
                            SceneQualityButton { quality },
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
                                Text::new(quality.label()),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new(SceneQualityProfileSetting::Fast.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        SceneQualityValueLabel,
                    ));
                });
        });
}

pub(super) fn spawn_scene_generation_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "asset generation");
    spawn_scene_setting_row(panel, "instances", SceneSetting::InstanceGeneration, 1);
    spawn_scene_toggle_row(panel, "lift assets", SceneToggleSetting::LiftAssets);
    spawn_scene_toggle_row(panel, "catalog reuse", SceneToggleSetting::CatalogReuse);
    spawn_scene_toggle_row(
        panel,
        "promote catalog",
        SceneToggleSetting::PromoteToCatalog,
    );
    spawn_scene_settings_section_label(panel, "mesh output");
    spawn_scene_toggle_row(panel, "pbr textures", SceneToggleSetting::Pbr);
    spawn_scene_setting_row(
        panel,
        "pbr texture size",
        SceneSetting::PbrTextureSize,
        TRELLIS_PBR_TEXTURE_STEP as isize,
    );
    spawn_scene_setting_row(
        panel,
        "target faces",
        SceneSetting::TargetFaces,
        TRELLIS_FACE_STEP as isize,
    );
}

pub(super) fn spawn_scene_grounding_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "layout search");
    spawn_scene_setting_row(panel, "candidates", SceneSetting::CandidateCount, 1);
    spawn_scene_setting_row(panel, "ground cal", SceneSetting::GroundCalibration, 1);
    spawn_scene_setting_row(panel, "pose refine", SceneSetting::ObjectPoseRefinement, 1);
    spawn_scene_settings_section_label(panel, "evidence");
    spawn_scene_toggle_row(panel, "locate bboxes", SceneToggleSetting::LocateAnything);
    spawn_scene_toggle_row(panel, "depth/floor", SceneToggleSetting::Depth);
    spawn_scene_toggle_row(panel, "sam masks", SceneToggleSetting::Segmentation);
    spawn_scene_toggle_row(panel, "visible fit", SceneToggleSetting::PoseFit);
}

pub(super) fn spawn_scene_debug_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "feedback");
    spawn_scene_setting_row(panel, "feedback iters", SceneSetting::FeedbackIterations, 1);
    spawn_scene_toggle_row(panel, "feedback loop", SceneToggleSetting::Feedback);
    spawn_scene_settings_section_label(panel, "artifacts");
    spawn_scene_toggle_row(panel, "write artifacts", SceneToggleSetting::WriteArtifacts);
}

pub(super) fn spawn_general_settings(panel: &mut ChildSpawnerCommands) {
    spawn_viewer_aabb_row(panel);
}

pub(super) fn spawn_physics_settings(panel: &mut ChildSpawnerCommands) {
    spawn_viewer_debug_toggle_row(
        panel,
        "ground/contact",
        ViewerDebugToggleSetting::GroundContact,
    );
    spawn_viewer_debug_numeric_row(
        panel,
        "ground y",
        ViewerDebugNumericSetting::GroundY,
        VIEWER_GROUND_Y_STEP,
    );
    spawn_viewer_debug_numeric_row(
        panel,
        "contact tolerance",
        ViewerDebugNumericSetting::ContactTolerance,
        VIEWER_CONTACT_TOLERANCE_STEP,
    );
}

pub(super) fn spawn_developer_settings(panel: &mut ChildSpawnerCommands) {
    spawn_developer_tabs(panel);
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Status, true, |panel| {
        spawn_developer_text_block::<SettingsDeveloperCurrentText>(panel, "current", "idle");
        spawn_developer_text_block::<SettingsDeveloperTokenText>(panel, "tokens", "no token usage");
        spawn_developer_section_label(panel, "debug overlays");
        spawn_viewer_debug_toggle_row(
            panel,
            "scene camera frustum",
            ViewerDebugToggleSetting::SceneCameraFrustum,
        );
        spawn_viewer_debug_numeric_row(
            panel,
            "frustum length",
            ViewerDebugNumericSetting::SceneCameraFrustumLength,
            VIEWER_FRUSTUM_LENGTH_STEP,
        );
        spawn_viewer_debug_toggle_row(
            panel,
            "depth rgb splats",
            ViewerDebugToggleSetting::DepthCloud,
        );
        spawn_viewer_debug_numeric_row(
            panel,
            "depth splat cap",
            ViewerDebugNumericSetting::DepthCloudMaxGaussians,
            VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32,
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Events, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperEventsText>(
            panel,
            "events",
            "no scene build events yet",
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Artifacts, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperArtifactText>(
            panel,
            "artifacts",
            "no artifacts yet",
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Visuals, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperVisualText>(
            panel,
            "visual artifacts",
            "no visual artifacts yet",
        );
        spawn_developer_visual_pager(panel);
        panel.spawn((
            SettingsDeveloperVisualGrid::default(),
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                ..default()
            },
        ));
    });
}

pub(super) fn spawn_developer_visual_pager(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(28.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            spawn_developer_visual_page_button(row, DeveloperVisualPageDirection::Previous, "<");
            row.spawn((
                Text::new("page 0/0 | 0 images"),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.72, 0.77, 0.86)),
                SettingsDeveloperVisualPagerText,
                Node {
                    flex_grow: 1.0,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
            ));
            spawn_developer_visual_page_button(row, DeveloperVisualPageDirection::Next, ">");
        });
}

pub(super) fn spawn_developer_visual_page_button(
    row: &mut ChildSpawnerCommands,
    direction: DeveloperVisualPageDirection,
    label: &'static str,
) {
    row.spawn((
        Button,
        SettingsDeveloperVisualPageButton { direction },
        Node {
            width: Val::Px(30.0),
            height: Val::Px(24.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderColor::all(BUTTON_BORDER_DISABLED),
        BackgroundColor(BUTTON_BG_DISABLED),
    ))
    .with_children(|button| {
        button.spawn((
            Text::new(label),
            TextFont::from_font_size(12.0),
            TextColor(BUTTON_TEXT_DISABLED),
            ButtonLabel,
        ));
    });
}

pub(super) fn spawn_developer_tabs(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|row| {
            for tab in [
                DeveloperPanelTab::Status,
                DeveloperPanelTab::Events,
                DeveloperPanelTab::Artifacts,
                DeveloperPanelTab::Visuals,
            ] {
                row.spawn((
                    Button,
                    SettingsDeveloperTabButton { tab },
                    Node {
                        height: Val::Px(26.0),
                        padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
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
                        Text::new(tab.label()),
                        TextFont::from_font_size(11.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

pub(super) fn spawn_developer_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: DeveloperPanelTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            SettingsDeveloperTabPanel { tab },
            Node {
                width: Val::Percent(100.0),
                display: if visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                ..default()
            },
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ))
        .with_children(spawn_content);
}

pub(super) fn spawn_developer_text_block<T: Component + Default>(
    panel: &mut ChildSpawnerCommands,
    label: &'static str,
    value: &'static str,
) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|block| {
            block.spawn((
                Text::new(label),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.58, 0.64, 0.74)),
            ));
            block.spawn((
                Text::new(value),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.76, 0.81, 0.9)),
                T::default(),
            ));
        });
}

pub(super) fn spawn_developer_visual_preview_row(
    parent: &mut ChildSpawnerCommands,
    preview: &ProcessingArtifactPreview,
) {
    let filename = preview
        .path
        .rsplit(['/', '\\'])
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or(preview.path.as_str());
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(8.0),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Node {
                    width: Val::Px(DEVELOPER_VISUAL_THUMB_WIDTH),
                    height: Val::Px(DEVELOPER_VISUAL_THUMB_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.04, 0.045, 0.055)),
                BorderColor::all(Color::srgb(0.22, 0.26, 0.34)),
                ImageNode::new(preview.image.clone()),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                flex_grow: 1.0,
                ..default()
            })
            .with_children(|labels| {
                labels.spawn((
                    Text::new(preview.kind.label()),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.62, 0.7, 0.86)),
                ));
                labels.spawn((
                    Text::new(ellipsize_text(filename, 42)),
                    TextFont::from_font_size(10.5),
                    TextColor(Color::srgb(0.78, 0.82, 0.9)),
                ));
            });
        });
}

pub(super) fn spawn_viewer_aabb_row(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("object AABB"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(ViewerAabbOverlayMode::Selected.label()),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ViewerAabbModeValueLabel,
                    ));
                });
            column
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for mode in [
                        ViewerAabbOverlayMode::Off,
                        ViewerAabbOverlayMode::Selected,
                        ViewerAabbOverlayMode::All,
                    ] {
                        row.spawn((
                            Button,
                            ViewerAabbModeButton { mode },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                height: Val::Px(28.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
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
                                Text::new(mode.label()),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                });
        });
}

pub(super) fn spawn_viewer_debug_toggle_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: ViewerDebugToggleSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                ViewerDebugToggleButton { setting },
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
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
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    ViewerDebugToggleValueLabel { setting },
                ));
            });
        });
}

pub(super) fn spawn_viewer_debug_numeric_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: ViewerDebugNumericSetting,
    step: f32,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_viewer_debug_step_button(control, setting, -step);
                control.spawn((
                    Text::new("0.00"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ViewerDebugNumericValueLabel { setting },
                ));
                spawn_viewer_debug_step_button(control, setting, step);
            });
        });
}

pub(super) fn spawn_viewer_debug_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: ViewerDebugNumericSetting,
    delta: f32,
) {
    parent
        .spawn((
            Button,
            ViewerDebugStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
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
                Text::new(if delta > 0.0 { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn spawn_triposg_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TripoSgSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_triposg_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TripoSgSettingValueLabel { setting },
                ));
                spawn_triposg_setting_step_button(control, setting, true);
            });
        });
}

pub(super) fn spawn_trellis_value_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TrellisSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Text::new("0"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
                TrellisSettingValueLabel { setting },
            ));
        });
}

pub(super) fn spawn_trellis_toggle_row(parent: &mut ChildSpawnerCommands, label: &'static str) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                TrellisPbrToggleButton,
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
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
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    TrellisSettingValueLabel {
                        setting: TrellisSetting::Pbr,
                    },
                ));
            });
        });
}

pub(super) fn spawn_trellis_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TrellisSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_trellis_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TrellisSettingValueLabel { setting },
                ));
                spawn_trellis_setting_step_button(control, setting, true);
            });
        });
}

pub(super) fn spawn_scene_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: SceneSetting,
    step: isize,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_scene_setting_step_button(control, setting, -step);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    SceneSettingValueLabel { setting },
                ));
                spawn_scene_setting_step_button(control, setting, step);
            });
        });
}

pub(super) fn spawn_scene_settings_section_label(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
) {
    parent.spawn((
        Text::new(label),
        TextFont::from_font_size(11.0),
        TextColor(Color::srgb(0.58, 0.64, 0.74)),
    ));
}

pub(super) fn spawn_developer_section_label(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
) {
    spawn_scene_settings_section_label(parent, label);
}

pub(super) fn spawn_scene_toggle_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: SceneToggleSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                SceneSettingToggleButton { setting },
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
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
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    SceneToggleValueLabel { setting },
                ));
            });
        });
}

pub(super) fn spawn_scene_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: SceneSetting,
    delta: isize,
) {
    parent
        .spawn((
            Button,
            SceneSettingStepButton {
                setting,
                delta: SceneSettingDelta::Integer(delta),
            },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
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
                Text::new(if delta > 0 { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn spawn_triposplat_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TripoSplatSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TripoSplatSettingValueLabel { setting },
                ));
                spawn_setting_step_button(control, setting, true);
            });
        });
}

pub(super) fn spawn_triposg_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TripoSgSetting,
    positive: bool,
) {
    let delta = match setting {
        TripoSgSetting::Steps => TripoSgSettingDelta::Integer(if positive { 1 } else { -1 }),
        TripoSgSetting::Tokens => TripoSgSettingDelta::Integer(if positive {
            TRIPOSG_TOKEN_STEP as isize
        } else {
            -(TRIPOSG_TOKEN_STEP as isize)
        }),
        TripoSgSetting::Guidance => TripoSgSettingDelta::Float(if positive {
            TRIPOSG_GUIDANCE_STEP
        } else {
            -TRIPOSG_GUIDANCE_STEP
        }),
        TripoSgSetting::TargetFaces => TripoSgSettingDelta::Integer(if positive {
            TRIPOSG_FACE_STEP as isize
        } else {
            -(TRIPOSG_FACE_STEP as isize)
        }),
    };
    parent
        .spawn((
            Button,
            TripoSgSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
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
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn spawn_trellis_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TrellisSetting,
    positive: bool,
) {
    let delta = match setting {
        TrellisSetting::PbrTextureSize => TrellisSettingDelta::Integer(if positive {
            TRELLIS_PBR_TEXTURE_STEP as isize
        } else {
            -(TRELLIS_PBR_TEXTURE_STEP as isize)
        }),
        TrellisSetting::TargetFaces => TrellisSettingDelta::Integer(if positive {
            TRELLIS_FACE_STEP as isize
        } else {
            -(TRELLIS_FACE_STEP as isize)
        }),
        TrellisSetting::MaxSparseCoords => TrellisSettingDelta::Integer(if positive {
            TRELLIS_SPARSE_COORD_STEP as isize
        } else {
            -(TRELLIS_SPARSE_COORD_STEP as isize)
        }),
        TrellisSetting::Resolution | TrellisSetting::Pbr => TrellisSettingDelta::Integer(0),
    };
    parent
        .spawn((
            Button,
            TrellisSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
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
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn spawn_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TripoSplatSetting,
    positive: bool,
) {
    let delta = match setting {
        TripoSplatSetting::Steps => TripoSplatSettingDelta::Integer(if positive { 1 } else { -1 }),
        TripoSplatSetting::Guidance => TripoSplatSettingDelta::Float(if positive {
            TRIPOSPLAT_GUIDANCE_STEP
        } else {
            -TRIPOSPLAT_GUIDANCE_STEP
        }),
        TripoSplatSetting::Gaussians => TripoSplatSettingDelta::Integer(if positive {
            TRIPOSPLAT_GAUSSIAN_STEP as isize
        } else {
            -(TRIPOSPLAT_GAUSSIAN_STEP as isize)
        }),
    };
    parent
        .spawn((
            Button,
            TripoSplatSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
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
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

pub(super) fn adjust_triposplat_setting(
    args: &mut AppArgs,
    setting: TripoSplatSetting,
    delta: TripoSplatSettingDelta,
) {
    match (setting, delta) {
        (TripoSplatSetting::Steps, TripoSplatSettingDelta::Integer(delta)) => {
            args.num_steps = apply_integer_delta(
                args.num_steps,
                delta,
                TRIPOSPLAT_MIN_STEPS,
                TRIPOSPLAT_MAX_STEPS,
            );
        }
        (TripoSplatSetting::Guidance, TripoSplatSettingDelta::Float(delta)) => {
            args.guidance_scale = (args.guidance_scale + delta)
                .clamp(TRIPOSPLAT_MIN_GUIDANCE, TRIPOSPLAT_MAX_GUIDANCE);
        }
        (TripoSplatSetting::Gaussians, TripoSplatSettingDelta::Integer(delta)) => {
            args.triposplat_num_gaussians = apply_integer_delta(
                args.triposplat_num_gaussians,
                delta,
                TRIPOSPLAT_MIN_NUM_GAUSSIANS,
                TRIPOSPLAT_MAX_NUM_GAUSSIANS,
            );
        }
        _ => {}
    }
    args.refresh_triposplat_profile_from_current_settings();
    log::info!(
        "TripoSplat settings: profile={} steps={} guidance_scale={:.3} gaussians={}",
        triposplat_profile_label(args.triposplat_profile),
        args.num_steps,
        args.guidance_scale,
        args.triposplat_num_gaussians
    );
}

pub(super) fn adjust_triposg_setting(
    args: &mut AppArgs,
    setting: TripoSgSetting,
    delta: TripoSgSettingDelta,
) {
    match (setting, delta) {
        (TripoSgSetting::Steps, TripoSgSettingDelta::Integer(delta)) => {
            args.num_steps =
                apply_integer_delta(args.num_steps, delta, TRIPOSG_MIN_STEPS, TRIPOSG_MAX_STEPS);
        }
        (TripoSgSetting::Tokens, TripoSgSettingDelta::Integer(delta)) => {
            args.num_tokens = apply_integer_delta(
                args.num_tokens,
                delta,
                TRIPOSG_MIN_TOKENS,
                TRIPOSG_MAX_TOKENS,
            );
        }
        (TripoSgSetting::Guidance, TripoSgSettingDelta::Float(delta)) => {
            args.guidance_scale =
                (args.guidance_scale + delta).clamp(TRIPOSG_MIN_GUIDANCE, TRIPOSG_MAX_GUIDANCE);
        }
        (TripoSgSetting::TargetFaces, TripoSgSettingDelta::Integer(delta)) => {
            let current = args.target_faces.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRIPOSG_MAX_FACES);
            args.target_faces = (next > 0).then_some(next);
        }
        _ => {}
    }
    log::info!(
        "TripoSG settings: steps={} tokens={} guidance_scale={:.3} target_faces={}",
        args.num_steps,
        args.num_tokens,
        args.guidance_scale,
        args.target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string())
    );
}

pub(super) fn adjust_trellis_setting(
    args: &mut AppArgs,
    setting: TrellisSetting,
    delta: TrellisSettingDelta,
) {
    match (setting, delta) {
        (TrellisSetting::PbrTextureSize, TrellisSettingDelta::Integer(delta)) => {
            let current = args
                .trellis_pbr_texture_size
                .unwrap_or(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE);
            args.trellis_pbr_texture_size = Some(apply_integer_delta(
                current,
                delta,
                TRELLIS_PBR_TEXTURE_MIN,
                TRELLIS_PBR_TEXTURE_MAX,
            ));
        }
        (TrellisSetting::TargetFaces, TrellisSettingDelta::Integer(delta)) => {
            let current = args.trellis_target_faces.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRELLIS_MAX_FACES);
            args.trellis_target_faces = (next > 0).then_some(next);
        }
        (TrellisSetting::MaxSparseCoords, TrellisSettingDelta::Integer(delta)) => {
            let current = args.trellis_max_sparse_coords.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRELLIS_MAX_SPARSE_COORDS);
            args.trellis_max_sparse_coords = (next > 0).then_some(next);
        }
        _ => {}
    }
    log::info!(
        "Trellis.2 settings: quality={} pbr={} texture_size={} target_faces={} max_sparse_coords={}",
        trellis_quality_label(args.trellis_quality),
        if args.trellis_pbr_enabled {
            "on"
        } else {
            "off"
        },
        args.trellis_pbr_texture_size
            .map(format_grouped_usize)
            .unwrap_or_else(|| "runtime".to_string()),
        args.trellis_target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
        args.trellis_max_sparse_coords
            .map(format_grouped_usize)
            .unwrap_or_else(|| "uncapped".to_string())
    );
}

pub(super) fn adjust_scene_setting(
    settings: &mut ScenePipelineUiSettings,
    setting: SceneSetting,
    delta: SceneSettingDelta,
) {
    match (setting, delta) {
        (SceneSetting::GroundCalibration, SceneSettingDelta::Integer(delta)) => {
            settings.ground_calibration = settings.ground_calibration.cycle(delta);
        }
        (SceneSetting::InstanceGeneration, SceneSettingDelta::Integer(delta)) => {
            settings.instance_generation = settings.instance_generation.cycle(delta);
        }
        (SceneSetting::ObjectPoseRefinement, SceneSettingDelta::Integer(delta)) => {
            settings.object_pose_refinement = settings.object_pose_refinement.cycle(delta);
        }
        (SceneSetting::CandidateCount, SceneSettingDelta::Integer(delta)) => {
            settings.candidate_count = apply_integer_delta(settings.candidate_count, delta, 1, 6);
        }
        (SceneSetting::FeedbackIterations, SceneSettingDelta::Integer(delta)) => {
            settings.feedback_iterations =
                apply_integer_delta(settings.feedback_iterations, delta, 0, 24);
        }
        (SceneSetting::PbrTextureSize, SceneSettingDelta::Integer(delta)) => {
            settings.pbr_texture_size = apply_integer_delta(
                settings.pbr_texture_size,
                delta,
                TRELLIS_PBR_TEXTURE_MIN,
                TRELLIS_PBR_TEXTURE_MAX,
            );
        }
        (SceneSetting::TargetFaces, SceneSettingDelta::Integer(delta)) => {
            settings.target_faces = apply_integer_delta(
                settings.target_faces,
                delta,
                TRELLIS_FACE_STEP,
                TRELLIS_MAX_FACES,
            );
        }
    }
    log::info!(
        "explicit scene settings: image_to_3d={} quality={} ground_calibration={} instances={} object_refine={} candidates={} feedback_iters={} pbr={} texture_size={} target_faces={} catalog_reuse={} lift_assets={} locate={} depth={} segmentation={} pose_fit={} feedback={} artifacts={} promote={}",
        pipeline_label(settings.image_to_3d_model),
        settings.quality_profile.label(),
        settings.ground_calibration.label(),
        settings.instance_generation.label(),
        settings.object_pose_refinement.label(),
        settings.candidate_count,
        settings.feedback_iterations,
        if settings.pbr_enabled { "on" } else { "off" },
        format_grouped_usize(settings.pbr_texture_size),
        format_grouped_usize(settings.target_faces),
        if settings.allow_catalog_reuse {
            "on"
        } else {
            "off"
        },
        if settings.lift_assets { "on" } else { "off" },
        if settings.locate_anything_enabled {
            "on"
        } else {
            "off"
        },
        if settings.depth_enabled { "on" } else { "off" },
        if settings.segmentation_enabled {
            "on"
        } else {
            "off"
        },
        if settings.pose_fit_enabled {
            "on"
        } else {
            "off"
        },
        if settings.feedback_enabled {
            "on"
        } else {
            "off"
        },
        if settings.write_artifacts {
            "on"
        } else {
            "off"
        },
        if settings.promote_to_catalog {
            "on"
        } else {
            "off"
        },
    );
}

pub(super) fn apply_integer_delta(value: usize, delta: isize, min: usize, max: usize) -> usize {
    value.saturating_add_signed(delta).clamp(min, max)
}

pub(super) fn triposplat_profile_label(profile: TripoSplatProfile) -> &'static str {
    match profile {
        TripoSplatProfile::Low => "low",
        TripoSplatProfile::Balanced => "balanced",
        TripoSplatProfile::High => "high",
        TripoSplatProfile::Custom => "custom",
    }
}

pub(super) fn trellis_quality_label(quality: TrellisQuality) -> &'static str {
    match quality {
        TrellisQuality::Low => "low",
        TrellisQuality::Medium => "medium",
        TrellisQuality::High => "high",
    }
}

pub(super) fn trellis_resolution_text(quality: TrellisQuality) -> &'static str {
    match quality {
        TrellisQuality::Low => "512",
        TrellisQuality::Medium | TrellisQuality::High => "1024",
    }
}

pub(super) fn trellis_quality_value_text(quality: TrellisQuality) -> String {
    format!(
        "{} / {}",
        trellis_quality_label(quality),
        trellis_resolution_text(quality)
    )
}

pub(super) fn triposplat_setting_value_text(args: &AppArgs, setting: TripoSplatSetting) -> String {
    match setting {
        TripoSplatSetting::Steps => args.num_steps.to_string(),
        TripoSplatSetting::Guidance => format!("{:.1}", args.guidance_scale),
        TripoSplatSetting::Gaussians => format_grouped_usize(args.triposplat_num_gaussians),
    }
}

pub(super) fn triposg_setting_value_text(args: &AppArgs, setting: TripoSgSetting) -> String {
    match setting {
        TripoSgSetting::Steps => args.num_steps.to_string(),
        TripoSgSetting::Tokens => format_grouped_usize(args.num_tokens),
        TripoSgSetting::Guidance => format!("{:.1}", args.guidance_scale),
        TripoSgSetting::TargetFaces => args
            .target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
    }
}

pub(super) fn trellis_setting_value_text(args: &AppArgs, setting: TrellisSetting) -> String {
    match setting {
        TrellisSetting::Resolution => trellis_resolution_text(args.trellis_quality).to_string(),
        TrellisSetting::Pbr => {
            if args.trellis_pbr_enabled {
                "on".to_string()
            } else {
                "off".to_string()
            }
        }
        TrellisSetting::PbrTextureSize => {
            if args.trellis_pbr_enabled {
                args.trellis_pbr_texture_size
                    .map(format_grouped_usize)
                    .unwrap_or_else(|| "runtime".to_string())
            } else {
                "disabled".to_string()
            }
        }
        TrellisSetting::TargetFaces => args
            .trellis_target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
        TrellisSetting::MaxSparseCoords => args
            .trellis_max_sparse_coords
            .map(format_grouped_usize)
            .unwrap_or_else(|| "uncapped".to_string()),
    }
}

pub(super) fn scene_setting_value_text(
    settings: &ScenePipelineUiSettings,
    setting: SceneSetting,
) -> String {
    match setting {
        SceneSetting::GroundCalibration => settings.ground_calibration.label().to_string(),
        SceneSetting::InstanceGeneration => settings.instance_generation.label().to_string(),
        SceneSetting::ObjectPoseRefinement => settings.object_pose_refinement.label().to_string(),
        SceneSetting::CandidateCount => settings.candidate_count.to_string(),
        SceneSetting::FeedbackIterations => settings.feedback_iterations.to_string(),
        SceneSetting::PbrTextureSize => {
            if settings.pbr_enabled {
                format_grouped_usize(settings.pbr_texture_size)
            } else {
                "disabled".to_string()
            }
        }
        SceneSetting::TargetFaces => format_grouped_usize(settings.target_faces),
    }
}

pub(super) fn scene_toggle_value_text(
    settings: &ScenePipelineUiSettings,
    setting: SceneToggleSetting,
) -> String {
    let enabled = match setting {
        SceneToggleSetting::Pbr => settings.pbr_enabled,
        SceneToggleSetting::CatalogReuse => settings.allow_catalog_reuse,
        SceneToggleSetting::LiftAssets => settings.lift_assets,
        SceneToggleSetting::LocateAnything => settings.locate_anything_enabled,
        SceneToggleSetting::Depth => settings.depth_enabled,
        SceneToggleSetting::Segmentation => settings.segmentation_enabled,
        SceneToggleSetting::PoseFit => settings.pose_fit_enabled,
        SceneToggleSetting::Feedback => settings.feedback_enabled,
        SceneToggleSetting::WriteArtifacts => settings.write_artifacts,
        SceneToggleSetting::PromoteToCatalog => settings.promote_to_catalog,
    };
    if enabled {
        "on".to_string()
    } else {
        "off".to_string()
    }
}

pub(super) fn pipeline_has_settings(model: SynthesisModel) -> bool {
    matches!(
        model,
        SynthesisModel::Triposg | SynthesisModel::Trellis | SynthesisModel::Triposplat
    )
}

pub(super) fn active_settings_pipeline(args: Option<&AppArgs>) -> Option<SynthesisModel> {
    args.and_then(|args| args.synthesis_models.first().copied())
        .filter(|model| pipeline_has_settings(*model))
}

pub(super) fn active_pipeline_choice(
    catalog: &CatalogState,
    args: Option<&AppArgs>,
    scene_settings: Option<&ScenePipelineUiSettings>,
) -> Option<CatalogPipelineChoice> {
    active_pipeline_choice_for_mode(catalog.active_mode(), args, scene_settings)
}

pub(super) fn active_pipeline_choice_for_mode(
    mode: CatalogMode,
    args: Option<&AppArgs>,
    scene_settings: Option<&ScenePipelineUiSettings>,
) -> Option<CatalogPipelineChoice> {
    match mode {
        CatalogMode::Object => args
            .and_then(|args| args.synthesis_models.first().copied())
            .map(CatalogPipelineChoice::Object),
        CatalogMode::Scene => Some(CatalogPipelineChoice::Scene(
            scene_settings
                .map(|settings| settings.pipeline)
                .unwrap_or(ScenePipelineKind::Explicit),
        )),
    }
}

pub(super) fn pipeline_settings_enabled(catalog: &CatalogState, args: Option<&AppArgs>) -> bool {
    match catalog.active_mode() {
        CatalogMode::Object => active_settings_pipeline(args).is_some(),
        CatalogMode::Scene => true,
    }
}

pub(super) fn ellipsize_text(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return "...".chars().take(max_chars).collect();
    }
    let keep = max_chars - 3;
    let mut out: String = value.chars().take(keep).collect();
    out.push_str("...");
    out
}

pub(super) fn format_grouped_usize(value: usize) -> String {
    let raw = value.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len().saturating_sub(1) / 3);
    let first_group_len = raw.len() % 3;
    for (index, ch) in raw.chars().enumerate() {
        if index > 0
            && (index == first_group_len
                || (index > first_group_len && (index - first_group_len).is_multiple_of(3)))
        {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
