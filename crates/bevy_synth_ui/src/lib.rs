/// Internalized editor-core module.
///
/// TODO: switch this module to the upstream published `bevy_editor_core` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_editor_core;
/// Internalized file-dialog module.
///
/// TODO: switch this module to the upstream published `bevy_file_dialog` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_file_dialog;
/// Internalized transform-gizmos module.
///
/// TODO: switch this module to the upstream published `bevy_transform_gizmos` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_transform_gizmos;

mod ui;

pub use ui::*;
