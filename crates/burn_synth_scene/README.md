# burn_synth_scene

This crate is part of the bevy_synth workspace. It owns the formal scene-image
to object-image to asset-composition pipeline used by the MCP server.

## BSN Viewer Flow

Write a Bevy/MCP scene command envelope from a self-contained BSN scene:

```sh
cargo run -p burn_synth_scene -- write-commands \
  --bsn tmp/runs/<run_id>/scene.bsn \
  --output tmp/runs/<run_id>/scene_commands.json
```

Launch the Bevy UI directly as a BSN scene viewer:

```sh
cargo run -p bevy_synth -- \
  --scene-bsn tmp/runs/<run_id>/scene.bsn
```

BSN asset declarations are concrete when they use `cache:` or `path:`:

```bsn
synth_scene_v1 {
asset chair_asset = "cache:central-chair-cache-key";
asset table_asset = "path:/tmp/table.glb";
}
```

Use `--assets-json tmp/runs/<run_id>/asset_bindings.json` only for symbolic
`generated:` declarations from an in-progress scene-generation run, or when a
sidecar is needed to preserve richer metadata such as local bounds and
canonical-frame evidence.
