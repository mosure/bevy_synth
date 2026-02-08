# Trellis2 Hook References

This folder stores Trellis2 reference hook captures used by `burn_trellis` correctness tests.

## Files

- `trellis2_preprocess_input.png`: synthetic RGBA input used for preprocess parity checks.
- `trellis2_preprocess_reference.safetensors`: reference tensor dump produced by upstream TRELLIS.2 preprocess logic.

## Regenerate

From repo root:

```powershell
python crates/burn_trellis/tool/trellis2_capture_preprocess_hook.py `
  --trellis-root F:\repos\TRELLIS.2 `
  --input-image crates/burn_trellis/assets/hooks/trellis2_preprocess_input.png `
  --output-hook crates/burn_trellis/assets/hooks/trellis2_preprocess_reference.safetensors
```
