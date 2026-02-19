//! This crate provides core functionality for the Bevy Engine Editor.
//!
//! NOTE: This module is an internalized copy of the standalone
//! `bevy_editor_core` crate while upstreaming/publishing is in progress.
//! Keep behavior aligned with upstream and switch back to the published crate
//! when available.

pub mod actions;
pub mod keybinding;
pub mod selection;
pub mod utils;

use bevy::prelude::*;

use self::{
    actions::ActionsPlugin, keybinding::KeybindingPlugin, selection::SelectionPlugin,
    utils::CoreUtilsPlugin,
};

/// Crate prelude.
pub mod prelude {
    pub use super::{
        actions::{ActionAppExt, ActionWorldExt},
        keybinding::{Keybinding, KeybindingAppExt},
        selection::EditorSelection,
    };
}

/// Core plugin for the editor.
#[derive(Default)]
pub struct EditorCorePlugin;

impl Plugin for EditorCorePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ActionsPlugin,
            KeybindingPlugin,
            SelectionPlugin,
            CoreUtilsPlugin,
        ));
    }
}
