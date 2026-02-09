# Trellis2 Hook References

This folder stores Trellis2 reference hook captures used by `burn_trellis` correctness tests.

## Files

- `trellis2_preprocess_input.png`: synthetic RGBA input used for preprocess parity checks.
- `trellis2_preprocess_reference.safetensors`: reference tensor dump produced by upstream TRELLIS.2 preprocess logic.
- `trellis2_full_reference_alpha_512.safetensors`: canonical full-pipeline reference capture used for e2e parity checks.

## Regenerate

From repo root:

```powershell
python crates/burn_trellis/tool/trellis2_capture_preprocess_hook.py `
  --trellis-root F:\repos\TRELLIS.2 `
  --input-image crates/burn_trellis/assets/hooks/trellis2_preprocess_input.png `
  --output-hook crates/burn_trellis/assets/hooks/trellis2_preprocess_reference.safetensors
```

For strict e2e parity, capture a full-resolution hook without sampling/truncation:

```powershell
$env:HF_HOME = "E:\models\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:ATTN_BACKEND = "sdpa"

python crates/burn_trellis/tool/trellis2_capture_full_hook.py `
  --trellis-root F:\repos\TRELLIS.2 `
  --weights-root E:\models\huggingface\hub\models--microsoft--TRELLIS.2-4B\snapshots\af44b45f2e35a493886929c6d786e563ec68364d `
  --input-image crates/burn_trellis/assets/hooks/trellis2_preprocess_input.png `
  --output-hook crates/burn_trellis/assets/hooks/trellis2_full_reference_alpha_512.safetensors `
  --pipeline-type 512 `
  --seed 42 `
  --num-samples 1 `
  --sampler-snapshots 3 `
  --max-dense-elements 200000000 `
  --max-rows 2000000 `
  --capture-decode-inputs `
  --device cuda
```

Strict tests now require:
- no sampled/truncated metadata (`*.row_sampled`, `*.flat_sampled_from`, `*.list_truncated`)
- decoder input keys (`decode_shape_slat.input.*`, `decode_tex_slat.input.*`)
- full PBR hook keys.
- subdivision keys for levels `0..3` (`decode_shape_slat.subs.{level}.*`).
- no decoder row-capping in strict mode (`TRELLIS2_DECODER_TEST_MAX_ROWS` is rejected), because sparse conv neighborhoods require full coordinate context.

To run strict e2e parity validation (canonical profile):

```powershell
$env:TRELLIS2_E2E_STRICT = "1"
$env:TRELLIS2_E2E_DEVICE = "wgpu"
$env:TRELLIS2_E2E_SUBDIV_MAX = "1e-2"
cargo test -p burn_trellis --features runtime-model-wgpu trellis2_e2e_hook_alignment_against_reference -- --nocapture
```

Optional per-level subdivision gates:

```powershell
$env:TRELLIS2_E2E_SUBDIV_LEVEL0_MAX = "1e-2"
$env:TRELLIS2_E2E_SUBDIV_LEVEL1_MAX = "1e-2"
$env:TRELLIS2_E2E_SUBDIV_LEVEL2_MAX = "1e-2"
$env:TRELLIS2_E2E_SUBDIV_LEVEL3_MAX = "1e-2"
```
