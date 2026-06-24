# burn_synth_scene

This crate is part of the bevy_synth workspace. It owns the formal scene-image
to object-image to asset-composition pipeline used by the MCP server.

## BSN Viewer Flow

Write a Bevy/MCP scene command envelope from a generated BSN scene:

```sh
cargo run -p burn_synth_scene -- write-commands \
  --bsn tmp/runs/<run_id>/scene.bsn \
  --assets-json tmp/runs/<run_id>/asset_bindings.json \
  --output tmp/runs/<run_id>/scene_commands.json
```

Launch the Bevy UI directly as a BSN scene viewer:

```sh
cargo run -p bevy_synth -- \
  --scene-bsn tmp/runs/<run_id>/scene.bsn \
  --scene-assets-json tmp/runs/<run_id>/asset_bindings.json
```
