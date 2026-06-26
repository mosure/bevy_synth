# burn_locate_anything

Burn-side runtime, reference tooling, and import scaffolding for NVIDIA LocateAnything visual
grounding.

`LocateAnythingRuntime` supports an explicit `python_reference` backend for model-backed
detections through the upstream Torch implementation. The WGPU `burn_native` backend is opt-in
with `allow_experimental_native_detect=true`: preprocessing, MoonViT/projector, multimodal
Qwen2.5 generation, top-p/repetition sampling, and Parallel Box Decoding are wired through the
same runtime surface and are gated by upstream hook/parity tests.

Validated local fixture:

```text
image: /media/mosure/dolos/demo/Cisco/reconstruction/045-LYS01-3-Galaxy.jpg
query: conference table
reference output: <ref>conference table</ref><box><386><519><659><1000></box><|im_end|>
Burn WGPU selected-logit parity: max_abs <= 4.10e-4, rms <= 5.87e-5 across 3 Qwen forwards
Burn WGPU full detect: bbox [0.386, 0.519, 0.659, 1.0]
single-query timing: cold ~32s, warm ~1.5s on RTX PRO 6000 Blackwell
multi-query timing: table + 8 chair boxes, cold 37.29s, warm 3.28s
Python CUDA f32 reference: load ~8.1s, single-query infer ~1.05s, two-query infer ~1.96s
```
