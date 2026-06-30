#!/usr/bin/env python3
"""Benchmark upstream TripoSplat flow sampling with saved stage tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_CODE = ROOT / "tmp/upstream/TripoSplat/main"
DEFAULT_CKPT_ROOT = ROOT / "tmp/upstream/TripoSplat/VAST-AI-TripoSplat"
DEFAULT_STAGE_TENSORS = (
    ROOT
    / "tmp/runs/20260604T120916Z_triposplat_cuda_reference_true_f32_no_tf32/stage_tensors_f32.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark upstream Python/Torch TripoSplat flow sampling."
    )
    parser.add_argument("--upstream-code", type=Path, default=DEFAULT_UPSTREAM_CODE)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--stage-tensors", type=Path, default=DEFAULT_STAGE_TENSORS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="f16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--cfg-mode", choices=["separate", "batched"], default="separate")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-step", action="store_true")
    parser.add_argument("--record-attention-shapes", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="Optional safetensors path for final sampled latent/camera tensors.",
    )
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-prefix-steps", type=int, default=0)
    parser.add_argument("--forward-stats-output", type=Path)
    parser.add_argument("--forward-trace-output", type=Path)
    parser.add_argument("--forward-trace-step", type=int, default=0)
    parser.add_argument("--forward-trace-only", action="store_true")
    parser.add_argument("--forward-trace-tokens", type=int, default=512)
    return parser.parse_args()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_triposplat_torch_flow_bench")


def torch_dtype(value: str) -> Any:
    import torch

    return {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }[value]


def shifted_t(index: int, steps: int, shift: float) -> tuple[float, float]:
    import numpy as np

    t_seq = shift * np.linspace(1, 0, steps + 1) / (
        1 + (shift - 1) * np.linspace(1, 0, steps + 1)
    )
    return float(t_seq[index]), float(t_seq[index + 1])


def tensor_summary(tensor: Any) -> dict[str, Any]:
    value = tensor.detach()
    cpu = value.float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
    }


def finite_tensor_stats(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach()
    cpu = value.float().cpu()
    finite = torch.isfinite(cpu)
    finite_count = int(finite.sum().item())
    total = int(cpu.numel())
    if finite_count:
        finite_values = cpu[finite]
        min_finite = float(finite_values.min().item())
        max_finite = float(finite_values.max().item())
        mean_finite = float(finite_values.mean().item())
        rms_finite = float(torch.sqrt((finite_values * finite_values).mean()).item())
        max_abs_finite = float(finite_values.abs().max().item())
    else:
        min_finite = None
        max_finite = None
        mean_finite = None
        rms_finite = None
        max_abs_finite = None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "nonfinite_count": total - finite_count,
        "min_finite": min_finite,
        "max_finite": max_finite,
        "mean_finite": mean_finite,
        "rms_finite": rms_finite,
        "max_abs_finite": max_abs_finite,
    }


ATTENTION_SHAPE_CALLS: list[dict[str, Any]] = []


def install_attention_shape_hook() -> None:
    import torch
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    def wrapped_scaled_dot_product_attention(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        index = len(ATTENTION_SHAPE_CALLS)
        start = torch.cuda.Event(enable_timing=True) if query.is_cuda else None
        end = torch.cuda.Event(enable_timing=True) if query.is_cuda else None
        if start is not None:
            start.record()
        out = original(query, key, value, *args, **kwargs)
        elapsed_ms = None
        if end is not None and start is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        ATTENTION_SHAPE_CALLS.append(
            {
                "index": index,
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "output_shape": list(out.shape),
                "dtype": str(query.dtype).replace("torch.", ""),
                "device": str(query.device),
                "is_causal": bool(kwargs.get("is_causal", False)),
                "scale": kwargs.get("scale"),
                "elapsed_ms": elapsed_ms,
            }
        )
        return out

    F.scaled_dot_product_attention = wrapped_scaled_dot_product_attention


def load_flow_inputs(stage_tensors: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import safetensors.torch
    import torch

    tensors = safetensors.torch.load_file(str(stage_tensors), device=device)
    cond = {
        "feature1": tensors["feature1"],
        "feature2": tensors["feature2"],
    }
    noise = {
        "latent": tensors.get("flow_noise_latent"),
        "camera": tensors.get("flow_noise_camera"),
    }
    if noise["latent"] is None:
        noise["latent"] = torch.randn(
            1,
            8192,
            16,
            device=device,
            generator=torch.Generator(device=device).manual_seed(42),
        )
    if noise["camera"] is None:
        noise["camera"] = torch.randn(
            1,
            1,
            5,
            device=device,
            generator=torch.Generator(device=device).manual_seed(43),
        )
    return cond, noise


def cfg_prediction_batched(model: Any, sample: dict[str, Any], t: float, cond: dict[str, Any], neg_cond: dict[str, Any], guidance_scale: float) -> dict[str, Any]:
    import torch

    batched_sample = {
        key: torch.cat([value, value], dim=0) for key, value in sample.items()
    }
    batched_cond = {
        key: torch.cat([cond[key], neg_cond[key]], dim=0) for key in cond
    }
    t_scaled = torch.tensor([1000.0 * t, 1000.0 * t], device=model.device, dtype=torch.float32)
    out = model(batched_sample, t_scaled, batched_cond)
    pred: dict[str, Any] = {}
    for key, value in out.items():
        pos, neg = value[:1], value[1:2]
        pred[key] = guidance_scale * pos - (guidance_scale - 1.0) * neg
    return pred


def cfg_prediction_separate(model: Any, sample: dict[str, Any], t: float, cond: dict[str, Any], neg_cond: dict[str, Any], guidance_scale: float) -> dict[str, Any]:
    import torch

    t_scaled = torch.tensor([1000.0 * t], device=model.device, dtype=torch.float32)
    pos = model(sample, t_scaled, cond)
    if guidance_scale <= 1.0:
        return pos
    neg = model(sample, t_scaled, neg_cond)
    return {
        key: guidance_scale * pos[key] - (guidance_scale - 1.0) * neg[key]
        for key in pos
    }


def record_batched_forward_stats(
    model: Any,
    noise: dict[str, Any],
    cond: dict[str, Any],
    steps: int,
    guidance_scale: float,
    shift: float,
    output: Path,
) -> None:
    import types
    import torch

    stats: dict[str, Any] = {}

    def record(label: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            stats[label] = finite_tensor_stats(value)

    def hook(label: str):
        def inner(_module: Any, _inputs: Any, output_value: Any) -> None:
            record(label, output_value)

        return inner

    handles = [
        model.noise_refiner[0].attn.register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_00.block.attn.out.out")
        ),
        model.noise_refiner[0].mlp.register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_00.block.mlp.out")
        ),
        model.noise_refiner[0].register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_00.out")
        ),
        model.noise_refiner[1].attn.register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_01.block.attn.out.out")
        ),
        model.noise_refiner[1].mlp.register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_01.block.mlp.out")
        ),
        model.noise_refiner[1].register_forward_hook(
            hook("cfg.batched.forward.noise_refiner_01.out")
        ),
        model.cam_refiner.register_forward_hook(
            hook("cfg.batched.forward.cam_refiner.out")
        ),
        model.blocks[0].register_forward_hook(hook("cfg.batched.forward.main_00.out")),
        model.out_layer.register_forward_hook(
            hook("cfg.batched.forward.output_projection.latent")
        ),
    ]
    if getattr(model, "cam_out_layer", None) is not None:
        handles.append(
            model.cam_out_layer.register_forward_hook(
                hook("cfg.batched.forward.output_projection.camera")
            )
        )

    traced_block = model.noise_refiner[1]
    original_block_forward = traced_block.forward

    def traced_noise_refiner_01_forward(self: Any, x: Any, mod: Any = None, rotary_emb: Any = None) -> Any:
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            x = x + self.mlp(self.norm2(x))
            return x
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        if hasattr(self, "shift_table") and self.shift_table is not None:
            mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        record("cfg.batched.forward.noise_refiner_01.block.norm1_mod.out", h)
        h = self.attn(h, rope_emb=rotary_emb)
        x = x + h * gate_msa.unsqueeze(1)
        record("cfg.batched.forward.noise_refiner_01.block.attn_residual.out", x)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        record("cfg.batched.forward.noise_refiner_01.block.norm2_mod.out", h)
        h = self.mlp(h)
        x = x + h * gate_mlp.unsqueeze(1)
        record("cfg.batched.forward.noise_refiner_01.block.mlp_residual.out", x)
        return x

    traced_block.forward = types.MethodType(traced_noise_refiner_01_forward, traced_block)

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    batched_sample = {key: torch.cat([value, value], dim=0) for key, value in noise.items()}
    batched_cond = {key: torch.cat([cond[key], neg_cond[key]], dim=0) for key in cond}
    t, _ = shifted_t(0, steps, shift)
    t_scaled = torch.tensor([1000.0 * t, 1000.0 * t], device=model.device, dtype=torch.float32)
    dtype = model.dtype
    z = batched_sample["latent"].to(dtype)
    model.pos_pe = model.pos_pe.to(z.device)
    h_x = model.input_layer(z)
    t_emb = model.t_embedder(t_scaled)
    t_mod = model.adaLN_modulation(t_emb) if model.share_mod else t_emb
    pos = model.pos_embedder(model.pos_pe).to(dtype)
    record("cfg.batched.forward.input_layer.out", h_x)
    record("cfg.batched.forward.t_embedder.out", t_emb)
    record("cfg.batched.forward.t_mod.out", t_mod)
    record("cfg.batched.forward.latent_position.out", pos)
    record("cfg.batched.forward.input_timestep_position.out", h_x + pos)

    try:
        _ = cfg_prediction_batched(
            model, noise, t, cond, neg_cond, guidance_scale
        )
        torch.cuda.synchronize()
    finally:
        traced_block.forward = original_block_forward
        for handle in handles:
            handle.remove()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_batched_forward_trace(
    model: Any,
    noise: dict[str, Any],
    cond: dict[str, Any],
    steps: int,
    step: int,
    guidance_scale: float,
    shift: float,
    output: Path,
    token_limit: int,
) -> None:
    import torch
    import torch.nn.functional as F
    import safetensors.torch

    token_limit = max(1, int(token_limit))
    trace: dict[str, Any] = {}

    def record(label: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            clipped = value[:, : min(token_limit, value.shape[1])].detach().float().cpu()
            trace[label] = clipped

    def traced_block_forward(
        block: Any,
        label: str,
        x: Any,
        mod: Any = None,
        rotary_emb: Any = None,
    ) -> Any:
        if not block.modulation:
            h = block.norm1(x)
            record(f"{label}.norm1.out", h)
            h = block.attn(h, rope_emb=rotary_emb)
            record(f"{label}.attn.out.out", h)
            x = x + h
            record(f"{label}.attn_residual.out", x)
            h = block.norm2(x)
            record(f"{label}.norm2.out", h)
            h = block.mlp(h)
            record(f"{label}.mlp.out", h)
            x = x + h
            record(f"{label}.mlp_residual.out", x)
            return x

        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = mod + block.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = block.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        record(f"{label}.norm1_mod.out", h)
        h = block.attn(h, rope_emb=rotary_emb)
        record(f"{label}.attn.out.out", h)
        x = x + h * gate_msa.unsqueeze(1)
        record(f"{label}.attn_residual.out", x)
        h = block.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        record(f"{label}.norm2_mod.out", h)
        h = block.mlp(h)
        record(f"{label}.mlp.out", h)
        x = x + h * gate_mlp.unsqueeze(1)
        record(f"{label}.mlp_residual.out", x)
        return x

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    batched_sample = {key: torch.cat([value, value], dim=0) for key, value in noise.items()}
    batched_cond = {key: torch.cat([cond[key], neg_cond[key]], dim=0) for key in cond}
    t, _ = shifted_t(step, steps, shift)
    t_scaled = torch.tensor([1000.0 * t, 1000.0 * t], device=model.device, dtype=torch.float32)
    dtype = model.dtype

    z = batched_sample["latent"].to(dtype)
    feat1 = batched_cond["feature1"].to(dtype)
    feat2 = batched_cond["feature2"].to(dtype) if model.cond_embedder2 is not None else None
    model.pos_pe = model.pos_pe.to(z.device)

    h_x = model.input_layer(z)
    h_cond = model.cond_embedder(feat1)
    if feat2 is not None:
        h_cond = h_cond + model.cond_embedder2(feat2)
    t_emb = model.t_embedder(t_scaled)
    t_mod = model.adaLN_modulation(t_emb) if model.share_mod else t_emb
    pos = model.pos_embedder(model.pos_pe).to(dtype)
    record("cfg.batched.forward.input_layer.out", h_x)
    record("cfg.batched.forward.latent_position.out", pos)
    h_x = h_x + pos
    record("cfg.batched.forward.input_timestep_position.out", h_x)

    for index, block in enumerate(model.noise_refiner):
        h_x = traced_block_forward(
            block,
            f"cfg.batched.forward.noise_refiner_{index:02d}.block",
            h_x,
            mod=t_mod,
            rotary_emb=model.noise_repo_layers[index](h_x),
        )
        record(f"cfg.batched.forward.noise_refiner_{index:02d}.out", h_x)

    for index, block in enumerate(model.context_refiner):
        h_cond = traced_block_forward(
            block,
            f"cfg.batched.forward.context_refiner_{index:02d}.block",
            h_cond,
            mod=None,
            rotary_emb=model.context_repo_layers[index](h_cond),
        )
    record("cfg.batched.forward.condition_context.out", h_cond)

    h_cam = None
    if model.cam_channels is not None:
        cam = batched_sample["camera"].to(dtype)
        h_cam = model.cam_refiner(cam)
        record("cfg.batched.forward.cam_refiner.out", h_cam)

    h = torch.cat([h_x, h_cond], dim=1)
    if h_cam is not None:
        h = torch.cat([h, h_cam], dim=1)
    record("cfg.batched.forward.concat_main_tokens.out", h)

    for index, block in enumerate(model.blocks):
        h = traced_block_forward(
            block,
            f"cfg.batched.forward.main_{index:02d}.block",
            h,
            mod=t_mod,
            rotary_emb=model.repo_layers[index](h),
        )
        record(f"cfg.batched.forward.main_{index:02d}.out", h)

    latent_tokens = z.shape[1]
    h_x = F.layer_norm(h[:, :latent_tokens].float(), h.shape[-1:]).type(dtype)
    record("cfg.batched.forward.output_norm.latent", h_x)
    if h_cam is not None:
        h_cam = F.layer_norm(h[:, -h_cam.shape[1] :].float(), h.shape[-1:]).type(dtype)
        record("cfg.batched.forward.output_norm.camera", h_cam)

    if model.use_shift_table:
        shift_signal, scale = (model.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        h_x = h_x * (1 + scale) + shift_signal
        record("cfg.batched.forward.output_shift.latent", h_x)
        if h_cam is not None:
            h_cam = h_cam * (1 + scale) + shift_signal
            record("cfg.batched.forward.output_shift.camera", h_cam)

    latent = model.out_layer(h_x)
    record("cfg.batched.forward.output_projection.latent", latent)
    camera = None
    if h_cam is not None:
        camera = model.cam_out_layer(h_cam)
        record("cfg.batched.forward.output_projection.camera", camera)

    pos_latent, neg_latent = latent[:1], latent[1:2]
    blend_latent = guidance_scale * pos_latent - (guidance_scale - 1.0) * neg_latent
    record("cfg.batched.blend.latent", blend_latent)
    if camera is not None:
        pos_camera, neg_camera = camera[:1], camera[1:2]
        blend_camera = guidance_scale * pos_camera - (guidance_scale - 1.0) * neg_camera
        record("cfg.batched.blend.camera", blend_camera)

    torch.cuda.synchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        trace,
        str(output),
        metadata={
            "format": "triposplat_torch_forward_trace_v1",
            "dtype": str(dtype).replace("torch.", ""),
            "cfg_mode": "batched",
            "step": str(step),
            "forward_trace_tokens": str(token_limit),
        },
    )


def advance_sample_prefix(
    model: Any,
    noise: dict[str, Any],
    cond: dict[str, Any],
    total_steps: int,
    prefix_steps: int,
    guidance_scale: float,
    shift: float,
    cfg_mode: str,
) -> dict[str, Any]:
    import torch

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    sample = {key: value.clone() for key, value in noise.items()}
    predictor = cfg_prediction_separate if cfg_mode == "separate" else cfg_prediction_batched
    for index in range(prefix_steps):
        t, t_prev = shifted_t(index, total_steps, shift)
        x_t = {key: value.clone() for key, value in sample.items()}
        pred = predictor(model, x_t, t, cond, neg_cond, guidance_scale)
        dt = t - t_prev
        for key in sample:
            sample[key] = sample[key] - pred[key] * dt
    return sample


def run_sampler(
    model: Any,
    noise: dict[str, Any],
    cond: dict[str, Any],
    steps: int,
    guidance_scale: float,
    shift: float,
    cfg_mode: str,
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    import torch

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    sample = {key: value.clone() for key, value in noise.items()}
    step_ms: list[float] = []
    trace_prefix_steps = max(0, int(getattr(run_sampler, "trace_prefix_steps", 0)))
    traces: dict[str, Any] = {}
    if trace_prefix_steps > 0:
        traces["flow_step_000_latent"] = sample["latent"].clone()
        if "camera" in sample:
            traces["flow_step_000_camera"] = sample["camera"].clone()
    predictor = cfg_prediction_separate if cfg_mode == "separate" else cfg_prediction_batched
    for index in range(steps):
        t, t_prev = shifted_t(index, steps, shift)
        x_t = {key: value.clone() for key, value in sample.items()}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pred = predictor(model, x_t, t, cond, neg_cond, guidance_scale)
        if trace_prefix_steps > 0 and index < trace_prefix_steps:
            traces[f"flow_pred_{index:03d}_latent"] = pred["latent"].clone()
            if "camera" in pred:
                traces[f"flow_pred_{index:03d}_camera"] = pred["camera"].clone()
        dt = t - t_prev
        for key in sample:
            sample[key] = sample[key] - pred[key] * dt
        end.record()
        torch.cuda.synchronize()
        step_ms.append(float(start.elapsed_time(end)))
        trace_index = index + 1
        if trace_index <= trace_prefix_steps:
            traces[f"flow_step_{trace_index:03d}_latent"] = sample["latent"].clone()
            if "camera" in sample:
                traces[f"flow_step_{trace_index:03d}_camera"] = sample["camera"].clone()
    return sample, step_ms, traces


def profile_one_step(
    model: Any,
    noise: dict[str, Any],
    cond: dict[str, Any],
    guidance_scale: float,
    shift: float,
    cfg_mode: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import torch

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    sample = {key: value.clone() for key, value in noise.items()}
    t, _ = shifted_t(0, 20, shift)
    predictor = cfg_prediction_separate if cfg_mode == "separate" else cfg_prediction_batched
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        _ = predictor(model, sample, t, cond, neg_cond, guidance_scale)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    (output_dir / "torch_profiler_top_cuda.txt").write_text(table, encoding="utf-8")
    events = []
    for item in prof.key_averages():
        name = item.key
        if "attention" in name.lower() or "flash" in name.lower() or "cudnn" in name.lower():
            device_time = float(
                getattr(
                    item,
                    "cuda_time_total",
                    getattr(item, "device_time_total", 0.0),
                )
            )
            events.append(
                {
                    "name": name,
                    "cuda_time_total_us": device_time,
                    "cuda_time_avg_us": float(device_time / max(1, item.count)),
                    "count": int(item.count),
                }
            )
    events.sort(key=lambda event: event["cuda_time_total_us"], reverse=True)
    return events


def main() -> int:
    args = parse_args()
    args.upstream_code = args.upstream_code.resolve()
    args.ckpt_root = args.ckpt_root.resolve()
    args.stage_tensors = args.stage_tensors.resolve()
    output_dir = (args.output_dir or ROOT / "tmp/runs" / run_id()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.upstream_code))

    import torch
    from triposplat import load_flow_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    torch.set_grad_enabled(False)
    if args.record_attention_shapes:
        install_attention_shape_hook()
    dtype = torch_dtype(args.dtype)
    flow_path = args.ckpt_root / "diffusion_models/triposplat_fp16.safetensors"
    load_start = time.perf_counter()
    model = load_flow_model(str(flow_path), device=args.device, dtype=dtype).eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    cond, noise = load_flow_inputs(args.stage_tensors, args.device)

    if args.forward_stats_output:
        if args.cfg_mode != "batched":
            raise SystemExit("--forward-stats-output currently expects --cfg-mode batched")
        record_batched_forward_stats(
            model,
            noise,
            cond,
            args.steps,
            args.guidance_scale,
            args.shift,
            args.forward_stats_output,
        )

    if args.forward_trace_output:
        if args.cfg_mode != "batched":
            raise SystemExit("--forward-trace-output currently expects --cfg-mode batched")
        if args.forward_trace_step < 0 or args.forward_trace_step >= args.steps:
            raise SystemExit("--forward-trace-step must be in [0, steps)")
        trace_noise = advance_sample_prefix(
            model,
            noise,
            cond,
            args.steps,
            args.forward_trace_step,
            args.guidance_scale,
            args.shift,
            args.cfg_mode,
        )
        record_batched_forward_trace(
            model,
            trace_noise,
            cond,
            args.steps,
            args.forward_trace_step,
            args.guidance_scale,
            args.shift,
            args.forward_trace_output,
            args.forward_trace_tokens,
        )
        if args.forward_trace_only:
            meta = {
                "run_id": output_dir.name,
                "upstream_code": str(args.upstream_code),
                "stage_tensors": str(args.stage_tensors),
                "device": args.device,
                "torch_version": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": args.dtype,
                "cfg_mode": args.cfg_mode,
                "steps": args.steps,
                "forward_trace_step": args.forward_trace_step,
                "guidance_scale": args.guidance_scale,
                "shift": args.shift,
                "load_ms": load_ms,
                "trace_output": str(args.forward_trace_output),
                "tf32": {
                    "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                },
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(output_dir)
            print(json.dumps(meta, indent=2, sort_keys=True))
            return 0

    if args.warmup_steps > 0:
        run_sampler(
            model,
            noise,
            cond,
            args.warmup_steps,
            args.guidance_scale,
            args.shift,
            args.cfg_mode,
        )

    run_sampler.trace_prefix_steps = args.trace_prefix_steps if args.trace_output else 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    profile_events = (
        profile_one_step(
            model,
            noise,
            cond,
            args.guidance_scale,
            args.shift,
            args.cfg_mode,
            output_dir,
        )
        if args.profile_step
        else []
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sample, step_ms, traces = run_sampler(
        model,
        noise,
        cond,
        args.steps,
        args.guidance_scale,
        args.shift,
        args.cfg_mode,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    torch.cuda.synchronize()
    if args.trace_output:
        import safetensors.torch

        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(
            {name: value.detach().float().cpu() for name, value in traces.items()},
            str(args.trace_output),
            metadata={
                "format": "triposplat_torch_flow_trace_v1",
                "dtype": args.dtype,
                "cfg_mode": args.cfg_mode,
            },
        )
    if args.sample_output:
        import safetensors.torch

        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(
            {name: value.detach().float().cpu() for name, value in sample.items()},
            str(args.sample_output),
            metadata={
                "format": "triposplat_torch_flow_sample_v1",
                "dtype": args.dtype,
                "cfg_mode": args.cfg_mode,
                "steps": str(args.steps),
                "guidance_scale": str(args.guidance_scale),
                "shift": str(args.shift),
                "tf32_disabled": str(bool(args.disable_tf32)).lower(),
            },
        )

    meta = {
        "run_id": output_dir.name,
        "upstream_code": str(args.upstream_code),
        "stage_tensors": str(args.stage_tensors),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "cfg_mode": args.cfg_mode,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "load_ms": load_ms,
        "sample_ms_wall": total_ms,
        "sample_ms_cuda_sum": sum(step_ms),
        "step_ms": step_ms,
        "step_ms_avg": sum(step_ms) / len(step_ms),
        "step_ms_min": min(step_ms),
        "step_ms_max": max(step_ms),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        if torch.cuda.is_available()
        else None,
        "sdpa": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        },
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "output": {
            "latent": tensor_summary(sample["latent"]),
            "camera": tensor_summary(sample["camera"]) if "camera" in sample else None,
        },
        "sample_output": str(args.sample_output) if args.sample_output else None,
        "profile_attention_events": profile_events,
        "attention_shape_calls": ATTENTION_SHAPE_CALLS,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
