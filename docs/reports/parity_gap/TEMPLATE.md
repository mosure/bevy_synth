# Parity Gap Run Report Template

- `run_id`:
- `date_utc`:
- `git_ref`:
- `dirty_worktree`:

## Scope

- Workstream(s):
- Goal:
- Backend:
- Input(s):

## Command(s)

```bash
# command 1
# command 2
```

## Invariant Summary

- Canonical WGPU fail-fast only:
- Pre-extraction host readbacks:
- Decode dispatch presence:
- Runtime source identity:

## Timings (ms)

- preprocess_ms:
- runtime_setup_ms:
- sparse_ms:
- shape_slat_ms:
- tex_slat_ms:
- decode_ms:
- decode_shape_decoder_ms:
- decode_tex_decoder_ms:
- decode_attr_merge_ms:
- decode_mesh_ms:
- decode_pbr_ms:
- total_ms:

## Kernel / Telemetry Counters

- host_readback_count:
- host_readback_elements:
- decode_shape_wgpu_dispatches:
- decode_tex_wgpu_dispatches:
- neighbor_build_ms:
- neighbor_query_ms:
- sparse_conv_ms:

## Outcome

- Status: `pass` / `fail` / `blocked`
- Blocking issue(s):
- Next action:
