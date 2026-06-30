use super::*;

pub(super) fn spawn_source_image_modal(commands: &mut Commands, entry: &CatalogEntry) -> Entity {
    let title = ellipsize_text(&entry.label, 56);
    let source_text = entry
        .source_image_path
        .as_deref()
        .map(|path| ellipsize_text(path, 72))
        .unwrap_or_else(|| "source image unknown".to_string());
    commands
        .spawn((
            CatalogSourceImageModalRoot,
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
            GlobalZIndex(30_000),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(560.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(12.0),
                    padding: UiRect::all(Val::Px(16.0)),
                    border: UiRect::all(Val::Px(1.0)),
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
                        column_gap: Val::Px(12.0),
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new(title.clone()),
                            TextFont::from_font_size(16.0),
                            TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ));
                        header
                            .spawn((
                                Button,
                                CatalogSourceImageCloseButton,
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

                panel.spawn((
                    Text::new(source_text),
                    TextFont::from_font_size(12.0),
                    TextColor(Color::srgb(0.7, 0.74, 0.82)),
                ));

                if entry.kind == CatalogEntryKind::Scene {
                    spawn_source_image_tabs(panel);
                    spawn_source_image_tab_panel(
                        panel,
                        CatalogSourceImageTab::Image,
                        true,
                        |panel| spawn_source_image_body(panel, entry),
                    );
                    spawn_source_image_tab_panel(
                        panel,
                        CatalogSourceImageTab::Stats,
                        false,
                        |panel| spawn_scene_details_stats(panel, entry),
                    );
                } else {
                    spawn_source_image_body(panel, entry);
                }
            });
        })
        .id()
}

pub(super) fn spawn_source_image_tabs(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|row| {
            for tab in [CatalogSourceImageTab::Image, CatalogSourceImageTab::Stats] {
                row.spawn((
                    Button,
                    CatalogSourceImageTabButton { tab },
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

pub(super) fn spawn_source_image_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: CatalogSourceImageTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            CatalogSourceImageTabPanel { tab },
            Node {
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

pub(super) fn spawn_source_image_body(parent: &mut ChildSpawnerCommands, entry: &CatalogEntry) {
    if let Some(image) = entry.source_image.as_ref() {
        parent.spawn((
            Node {
                width: Val::Px(512.0),
                height: Val::Px(512.0),
                border: UiRect::all(Val::Px(1.0)),
                align_self: AlignSelf::Center,
                ..default()
            },
            BorderColor::all(Color::srgb(0.24, 0.27, 0.34)),
            ImageNode::new(image.clone()),
        ));
    } else {
        parent
            .spawn((
                Node {
                    width: Val::Px(512.0),
                    height: Val::Px(220.0),
                    border: UiRect::all(Val::Px(1.0)),
                    align_self: AlignSelf::Center,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderColor::all(Color::srgb(0.24, 0.27, 0.34)),
                BackgroundColor(Color::srgb(0.05, 0.06, 0.08)),
            ))
            .with_children(|missing| {
                missing.spawn((
                    Text::new("source image unavailable"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.8, 0.84, 0.9)),
                ));
            });
    }
}

pub(super) fn spawn_scene_details_stats(parent: &mut ChildSpawnerCommands, entry: &CatalogEntry) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(12.0),
            ..default()
        })
        .with_children(|stats| {
            let pipeline = entry.scene_pipeline.as_deref().unwrap_or("scene");
            spawn_scene_stats_section(stats, "summary", |section| {
                spawn_scene_stat_row(section, "pipeline", pipeline.to_string());
                if let Some(scene_key) = entry.scene_key.as_deref() {
                    spawn_scene_stat_row(section, "scene key", ellipsize_text(scene_key, 52));
                }
                if let Some(metrics) = entry.scene_metrics.as_ref() {
                    spawn_scene_stat_row(section, "status", scene_metric_status(metrics));
                    if let Some(elapsed_ms) = metrics.elapsed_ms {
                        spawn_scene_stat_row(
                            section,
                            "runtime",
                            format!("{:.1}s", elapsed_ms as f32 / 1000.0),
                        );
                    }
                    spawn_scene_stat_row(section, "counts", scene_metric_counts_text(metrics));
                } else {
                    spawn_scene_stat_row(section, "status", "no cached metrics".to_string());
                }
            });

            if let Some(metrics) = entry.scene_metrics.as_ref() {
                spawn_scene_stats_section(stats, "categories", |section| {
                    if metrics.category_breakdown.is_empty() {
                        spawn_scene_stat_row(section, "breakdown", "unavailable".to_string());
                    } else {
                        for category in metrics.category_breakdown.iter().take(10) {
                            spawn_scene_category_row(section, category);
                        }
                    }
                });

                spawn_scene_stats_section(stats, "quality", |section| {
                    spawn_scene_stat_row(section, "feedback", scene_feedback_text(metrics));
                    if let Some(stage) = metrics.failed_stage.as_deref() {
                        spawn_scene_stat_row(section, "failed stage", stage.to_string());
                    }
                });
            }

            spawn_scene_stats_section(stats, "artifacts", |section| {
                if let Some(path) = entry.source_image_path.as_deref() {
                    spawn_scene_stat_row(section, "source", ellipsize_text(path, 58));
                }
                if let Some(dir) = entry.scene_artifact_dir.as_deref() {
                    spawn_scene_stat_row(section, "run dir", ellipsize_text(dir, 58));
                }
            });
        });
}

pub(super) fn spawn_scene_stats_section(
    parent: &mut ChildSpawnerCommands,
    title: &'static str,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(5.0),
            ..default()
        })
        .with_children(|section| {
            section.spawn((
                Text::new(title),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.58, 0.64, 0.74)),
            ));
            spawn_content(section);
        });
}

pub(super) fn spawn_scene_stat_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    value: String,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::FlexStart,
            column_gap: Val::Px(16.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            row.spawn((
                Text::new(value),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
        });
}

pub(super) fn spawn_scene_category_row(
    parent: &mut ChildSpawnerCommands,
    category: &CachedSceneCategoryMetric,
) {
    spawn_scene_stat_row(
        parent,
        "category",
        format!(
            "{} | {}",
            category.label,
            scene_category_counts_text(category)
        ),
    );
}

pub(super) fn scene_metric_status(metrics: &CachedSceneMetrics) -> String {
    match metrics.ok {
        Some(true) => "ok".to_string(),
        Some(false) => "needs review".to_string(),
        None => "unknown".to_string(),
    }
}

pub(super) fn scene_metric_counts_text(metrics: &CachedSceneMetrics) -> String {
    let mut parts = Vec::new();
    if let Some(count) = metrics.object_count {
        parts.push(format!("{count} objects"));
    }
    if let Some(count) = metrics.asset_count {
        parts.push(format!("{count} assets"));
    }
    if let Some(count) = metrics.placement_count {
        parts.push(format!("{count} placements"));
    }
    if parts.is_empty() {
        "unavailable".to_string()
    } else {
        parts.join(" | ")
    }
}

pub(super) fn scene_feedback_text(metrics: &CachedSceneMetrics) -> String {
    match (metrics.feedback_accepted, metrics.feedback_iteration) {
        (Some(true), Some(iteration)) => format!("accepted at iter {iteration}"),
        (Some(true), None) => "accepted".to_string(),
        (Some(false), Some(iteration)) => format!("failed after iter {iteration}"),
        (Some(false), None) => "failed".to_string(),
        (None, _) => "not recorded".to_string(),
    }
}

pub(super) fn scene_category_counts_text(category: &CachedSceneCategoryMetric) -> String {
    let mut parts = Vec::new();
    if let Some(count) = category.object_count {
        parts.push(format!("{count} planned"));
    }
    if let Some(count) = category.detection_count {
        parts.push(format!("{count} detected"));
    }
    if let Some(count) = category.asset_count {
        parts.push(format!("{count} assets"));
    }
    if let Some(count) = category.placement_count {
        parts.push(format!("{count} placed"));
    }
    if parts.is_empty() {
        "no counts".to_string()
    } else {
        parts.join(" / ")
    }
}
