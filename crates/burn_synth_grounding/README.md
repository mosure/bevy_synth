# burn_synth_grounding

Scene-scale grounding providers for `burn_synth`.

This crate owns model-backed and provider-backed grounding sidecars used by
scene composition:

- DepthPro depth, intrinsics, floor-plane, and contact evidence.
- LocateAnything object count, box, point, and instance evidence.
- Provider-neutral adapters from detections/depth maps into
  `burn_synth_scene::SceneGroundingEvidence`.

`burn_synth_mcp` is only a transport and orchestration layer. It calls this
crate when exposing scene grounding through MCP or CLI commands.
