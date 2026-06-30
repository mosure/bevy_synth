#!/usr/bin/env python3
"""Generate upstream TripoSplat reference outputs.

This script intentionally runs the official Python implementation from a local
checkout and writes machine-readable stage evidence under tmp/runs by default.
It is a reference-data generator, not part of the Rust inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_CODE = ROOT / "tmp/upstream/TripoSplat/main"
DEFAULT_CKPT_ROOT = ROOT / "tmp/upstream/TripoSplat/VAST-AI-TripoSplat"
DEFAULT_INPUT = ROOT / "docs/input_chair.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream TripoSplat and write reference artifacts."
    )
    parser.add_argument("--upstream-code", type=Path, default=DEFAULT_UPSTREAM_CODE)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument(
        "--gaussians",
        type=int,
        action="append",
        default=[],
        help="Gaussian count. May be repeated; defaults to 32768.",
    )
    parser.add_argument(
        "--save-stage-arrays",
        action="store_true",
        help="Write feature/latent numpy arrays for numerical parity tests.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate local upstream/code/model paths.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip Gaussian decode/output writing after stage tensor generation.",
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip flow sampling; useful for conditioning-only reference tensors.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Use --input directly as the prepared image instead of running RMBG/preprocess.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["default", "bf16", "f16", "f32"],
        default="default",
        help="Override DINOv3, Flux2 VAE, flow, and decoder dtype after loading.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable PyTorch CUDA/cuDNN TF32 kernels for stricter f32 reference tensors.",
    )
    parser.add_argument(
        "--save-flow-steps",
        type=int,
        default=1,
        help="Number of prefix Euler flow states to save when --save-stage-arrays is set.",
    )
    parser.add_argument(
        "--save-flux-trace",
        action="store_true",
        help="When saving stage arrays, also export Flux2 VAE encoder intermediate tensors.",
    )
    parser.add_argument(
        "--save-decoder-trace",
        action="store_true",
        help="When decoding, also export decoder points/log-probs/features for Gaussian parity replay.",
    )
    return parser.parse_args()


def torch_dtype_from_arg(value: str):
    import torch

    return {
        "default": None,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[value]


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_triposplat_torch_reference")


def default_output_dir() -> Path:
    return ROOT / "tmp/runs" / run_id()


def required_files(ckpt_root: Path) -> dict[str, Path]:
    return {
        "flow": ckpt_root / "diffusion_models/triposplat_fp16.safetensors",
        "decoder": ckpt_root / "vae/triposplat_vae_decoder_fp16.safetensors",
        "dinov3": ckpt_root / "clip_vision/dino_v3_vit_h.safetensors",
        "flux2_vae_encoder": ckpt_root / "vae/flux2-vae.safetensors",
        "rmbg": ckpt_root / "background_removal/birefnet.safetensors",
    }


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    if not (args.upstream_code / "triposplat.py").is_file():
        missing.append(args.upstream_code / "triposplat.py")
    if not (args.upstream_code / "model.py").is_file():
        missing.append(args.upstream_code / "model.py")
    if not args.input.is_file():
        missing.append(args.input)
    for path in required_files(args.ckpt_root).values():
        if not path.is_file():
            missing.append(path)
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing required TripoSplat reference inputs:\n{joined}")


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach()
    cpu = value.float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "sha256_f32_le": sha256_array(cpu.numpy()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
    }


def sha256_array(array: Any) -> str:
    contiguous = array.copy(order="C")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def encode_image_with_noise(
    image: Any,
    dinov3: Any,
    vae_encoder: Any,
    generator: Any,
    save_flux_trace: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from triposplat import _DINOV3_NORMALIZE

    torch.set_grad_enabled(False)
    device = next(dinov3.parameters()).device
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(np.transpose(image_array, (2, 0, 1)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    img_normed = _DINOV3_NORMALIZE(img_tensor)
    dinov3_dtype = next(dinov3.parameters()).dtype
    vae_dtype = next(vae_encoder.parameters()).dtype

    dinov3_raw = dinov3(pixel_values=img_normed.to(dinov3_dtype))
    dinov3_feat = F.layer_norm(dinov3_raw.float(), dinov3_raw.shape[-1:])

    flux_image = img_tensor.to(vae_dtype) * 2 - 1
    flux_trace: dict[str, Any] = {}
    if save_flux_trace:
        x = vae_encoder.encoder.conv_in(flux_image)
        flux_trace["flux2_conv_in"] = x
        for index, block in enumerate(vae_encoder.encoder.down_0_resnets):
            x = block(x)
            flux_trace[f"flux2_down_0_resnet_{index}"] = x
        x = vae_encoder.encoder.down_0_sampler(x)
        flux_trace["flux2_down_0_sampler"] = x
        for index, block in enumerate(vae_encoder.encoder.down_1_resnets):
            x = block(x)
            flux_trace[f"flux2_down_1_resnet_{index}"] = x
        x = vae_encoder.encoder.down_1_sampler(x)
        flux_trace["flux2_down_1_sampler"] = x
        for index, block in enumerate(vae_encoder.encoder.down_2_resnets):
            x = block(x)
            flux_trace[f"flux2_down_2_resnet_{index}"] = x
        x = vae_encoder.encoder.down_2_sampler(x)
        flux_trace["flux2_down_2_sampler"] = x
        for index, block in enumerate(vae_encoder.encoder.down_3_resnets):
            x = block(x)
            flux_trace[f"flux2_down_3_resnet_{index}"] = x
        x = vae_encoder.encoder.mid_resnets[0](x)
        flux_trace["flux2_mid_resnet_0"] = x
        x = vae_encoder.encoder.mid_attn(x)
        flux_trace["flux2_mid_attn"] = x
        x = vae_encoder.encoder.mid_resnets[1](x)
        flux_trace["flux2_mid_resnet_1"] = x
        encoder_out = vae_encoder.encoder.conv_out(
            F.silu(vae_encoder.encoder.conv_norm_out(x))
        )
        flux_trace["flux2_encoder_out"] = encoder_out
        moments = vae_encoder.quant_conv(encoder_out)
    else:
        moments = vae_encoder.quant_conv(vae_encoder.encoder(flux_image))
    flux_trace["flux2_moments"] = moments
    mean, logvar = moments.chunk(2, dim=1)
    vae_noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    latents = mean + torch.exp(0.5 * logvar) * vae_noise
    flux_trace["flux2_latents"] = latents
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(
        0, 1, 3, 5, 2, 4
    )
    latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
    flux_trace["flux2_unshuffled"] = latents
    bn_mean = vae_encoder.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device, latents.dtype
    )
    bn_std = torch.sqrt(
        vae_encoder.bn.running_var.view(1, -1, 1, 1) + vae_encoder.bn.eps
    ).to(latents.device, latents.dtype)
    normalized_latents = (latents - bn_mean) / bn_std
    flux_trace["flux2_normalized"] = normalized_latents
    vae_feat = normalized_latents.to(torch.float32).flatten(2).transpose(
        1, 2
    ).contiguous()
    flux_trace["flux2_tokens"] = vae_feat
    zero_reg = torch.zeros(
        vae_feat.shape[0],
        5,
        vae_feat.shape[2],
        dtype=vae_feat.dtype,
        device=vae_feat.device,
    )
    vae_feat = torch.cat([zero_reg, vae_feat], dim=1)
    return (
        {"dinov3_raw": dinov3_raw, "feature1": dinov3_feat, "feature2": vae_feat},
        {"vae_mean": mean, "vae_logvar": logvar, "vae_noise": vae_noise},
        flux_trace,
    )


def sample_latent_with_noise(
    flow_model: Any,
    cond: dict[str, Any],
    steps: int,
    guidance_scale: float,
    shift: float,
    generator: Any,
    callback: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from triposplat import FlowEulerCfgSampler

    torch.set_grad_enabled(False)
    device = flow_model.device
    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    noise = {
        "latent": torch.randn(
            1,
            flow_model.q_token_length,
            flow_model.in_channels,
            device=device,
            generator=generator,
        )
    }
    if flow_model.cam_channels is not None:
        noise["camera"] = torch.randn(
            1,
            1,
            flow_model.cam_channels,
            device=device,
            generator=generator,
        )
    initial_noise = {key: value.clone() for key, value in noise.items()}
    sampler = FlowEulerCfgSampler()
    sample = {key: value.clone() for key, value in noise.items()}
    traces: dict[str, Any] = {
        "flow_step_000_latent": sample["latent"].clone(),
    }
    if "camera" in sample:
        traces["flow_step_000_camera"] = sample["camera"].clone()
    t_seq = shift * np.linspace(1, 0, steps + 1) / (
        1 + (shift - 1) * np.linspace(1, 0, steps + 1)
    )
    for index, (t, t_prev) in enumerate(zip(t_seq[:-1], t_seq[1:])):
        x_t = {key: value.clone() for key, value in sample.items()}
        pred_v = sampler._cfg_prediction(
            flow_model, x_t, t, cond, neg_cond, guidance_scale
        )
        if index == 0:
            traces["flow_pred_000_latent"] = pred_v["latent"].clone()
            if "camera" in pred_v:
                traces["flow_pred_000_camera"] = pred_v["camera"].clone()
        dt = t - t_prev
        for key in sample:
            sample[key] = sample[key] - pred_v[key] * dt
        trace_index = index + 1
        if trace_index <= max(0, int(getattr(callback, "save_flow_steps", 0))):
            traces[f"flow_step_{trace_index:03d}_latent"] = sample["latent"].clone()
            if "camera" in sample:
                traces[f"flow_step_{trace_index:03d}_camera"] = sample["camera"].clone()
        if callback is not None:
            callback(index + 1, steps)
    return sample, initial_noise, traces


def decode_latent_with_trace(
    pipe: Any, latent: Any, num_gaussians: int
) -> tuple[Any, dict[str, Any]]:
    from model import OctreeProbabilityFixedlenDecoder
    from triposplat import _build_gaussians

    num_decoder_tokens = max(1, num_gaussians // pipe.decoder.gaussians_per_point)
    points_pred = OctreeProbabilityFixedlenDecoder.sample(
        pipe.decoder.octree,
        latent,
        num_points=num_decoder_tokens,
        level=pipe.decoder._MAX_VOXEL_LEVEL,
        temperature=1.0,
        algo="systematic",
    )
    pred = pipe.decoder.gs(x=points_pred, cond=latent)
    gaussian = _build_gaussians(pipe.decoder.gs, points_pred, pred)[0]
    return gaussian, {
        "latent": latent,
        "decoder_points": points_pred["points"],
        "decoder_log_probs": points_pred["log_probs"],
        "decoder_features": pred["features"],
    }


def main() -> int:
    args = parse_args()
    args.upstream_code = args.upstream_code.resolve()
    args.ckpt_root = args.ckpt_root.resolve()
    args.input = args.input.resolve()
    args.output_dir = (args.output_dir or default_output_dir()).resolve()
    counts = args.gaussians or [32768]
    validate_paths(args)
    if args.validate_only:
        print("TripoSplat reference inputs are present.")
        return 0

    sys.path.insert(0, str(args.upstream_code))

    import numpy as np
    import safetensors.torch
    import torch
    from triposplat import (
        TripoSplatPipeline,
        load_decoder,
        load_dinov3,
        load_flow_model,
        load_vae_encoder,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested for upstream TripoSplat reference generation, "
            "but torch.cuda.is_available() is false. Re-run with --device cpu "
            "for a slow diagnostic run or use a CUDA-enabled Python environment."
        )
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = required_files(args.ckpt_root)
    meta: dict[str, Any] = {
        "run_id": args.output_dir.name,
        "upstream_code": str(args.upstream_code),
        "ckpt_root": str(args.ckpt_root),
        "input": str(args.input),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "erode_radius": args.erode_radius,
        "gaussians": counts,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_precision": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "stages": {},
        "outputs": [],
    }

    t0 = time.perf_counter()
    pipe = TripoSplatPipeline(
        ckpt_path=str(files["flow"]),
        decoder_path=str(files["decoder"]),
        dinov3_path=str(files["dinov3"]),
        flux2_vae_encoder_path=str(files["flux2_vae_encoder"]),
        rmbg_path=str(files["rmbg"]),
        device=args.device,
    )
    model_dtype = torch_dtype_from_arg(args.model_dtype)
    if model_dtype is not None:
        pipe.dinov3 = load_dinov3(
            str(files["dinov3"]), device=pipe._device, dtype=model_dtype
        )
        pipe.vae_encoder = load_vae_encoder(
            str(files["flux2_vae_encoder"]), device=pipe._device, dtype=model_dtype
        )
        pipe.flow_model = load_flow_model(
            str(files["flow"]), device=pipe._device, dtype=model_dtype
        )
        pipe.decoder = load_decoder(
            str(files["decoder"]), device=pipe._device, dtype=model_dtype
        )
    meta["model_dtype"] = args.model_dtype
    meta["stages"]["load_ms"] = (time.perf_counter() - t0) * 1000.0

    generator = torch.Generator(device=pipe._device).manual_seed(args.seed)
    t0 = time.perf_counter()
    prepared = (
        Image.open(args.input).convert("RGBA")
        if args.skip_preprocess
        else pipe.preprocess_image(args.input, erode_radius=args.erode_radius)
    )
    prepared_path = args.output_dir / "prepared.png"
    prepared.save(prepared_path)
    meta["stages"]["preprocess"] = {
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "skipped": bool(args.skip_preprocess),
        "size": list(prepared.size),
        "sha256": sha256_file(prepared_path),
        "path": str(prepared_path),
    }

    t0 = time.perf_counter()
    cond, encode_noise, flux_trace = encode_image_with_noise(
        prepared,
        pipe.dinov3,
        pipe.vae_encoder,
        generator,
        save_flux_trace=args.save_flux_trace,
    )
    meta["stages"]["encode"] = {
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "feature1": tensor_summary(cond["feature1"]),
        "feature2": tensor_summary(cond["feature2"]),
        "vae_noise": tensor_summary(encode_noise["vae_noise"]),
    }

    sample_steps: list[dict[str, int]] = []
    latent_out: dict[str, Any] = {}
    flow_noise: dict[str, Any] = {}
    flow_traces: dict[str, Any] = {}

    def on_step(step: int, total: int) -> None:
        sample_steps.append({"step": int(step), "total": int(total)})

    if args.skip_flow:
        meta["stages"]["sample"] = {"skipped": True}
    else:
        on_step.save_flow_steps = args.save_flow_steps
        t0 = time.perf_counter()
        latent_out, flow_noise, flow_traces = sample_latent_with_noise(
            pipe.flow_model,
            cond,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            generator=generator,
            callback=on_step,
        )
        meta["stages"]["sample"] = {
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "latent": tensor_summary(latent_out["latent"]),
            "camera": tensor_summary(latent_out["camera"])
            if "camera" in latent_out
            else None,
            "flow_noise_latent": tensor_summary(flow_noise["latent"]),
            "flow_noise_camera": tensor_summary(flow_noise["camera"])
            if "camera" in flow_noise
            else None,
            "flow_traces": {
                name: tensor_summary(value)
                for name, value in flow_traces.items()
                if name.startswith("flow_step_") or name.startswith("flow_pred_")
            },
            "step_callbacks": sample_steps,
        }

    if args.save_stage_arrays:
        prepared_array = np.asarray(prepared.convert("RGB"), dtype=np.float32) / 255.0
        prepared_chw = np.transpose(prepared_array, (2, 0, 1))[None, :, :, :]
        arrays = {
            "image_rgb_0_1": prepared_chw,
            "dinov3_raw": cond["dinov3_raw"].detach().float().cpu().numpy(),
            "feature1": cond["feature1"].detach().float().cpu().numpy(),
            "feature2": cond["feature2"].detach().float().cpu().numpy(),
            "vae_mean": encode_noise["vae_mean"].detach().float().cpu().numpy(),
            "vae_logvar": encode_noise["vae_logvar"].detach().float().cpu().numpy(),
            "vae_noise": encode_noise["vae_noise"].detach().float().cpu().numpy(),
        }
        if args.save_flux_trace:
            arrays.update(
                {
                    name: value.detach().float().cpu().numpy()
                    for name, value in flux_trace.items()
                }
            )
        if "latent" in flow_noise:
            arrays["flow_noise_latent"] = flow_noise["latent"].detach().float().cpu().numpy()
        if "latent" in latent_out:
            arrays["latent"] = latent_out["latent"].detach().float().cpu().numpy()
        for name, value in flow_traces.items():
            arrays[name] = value.detach().float().cpu().numpy()
        if "camera" in flow_noise:
            arrays["flow_noise_camera"] = flow_noise["camera"].detach().float().cpu().numpy()
        if "camera" in latent_out:
            arrays["camera"] = latent_out["camera"].detach().float().cpu().numpy()
        stage_path = args.output_dir / "stage_arrays_f32.npz"
        np.savez(stage_path, **arrays)
        stage_tensors_path = args.output_dir / "stage_tensors_f32.safetensors"
        safetensors.torch.save_file(
            {
                name: torch.from_numpy(array).contiguous()
                for name, array in arrays.items()
            },
            str(stage_tensors_path),
            metadata={
                "format": "triposplat_stage_tensors_v2",
                "dtype": "f32",
                "seed": str(args.seed),
                "steps": str(args.steps),
                "guidance_scale": str(args.guidance_scale),
                "shift": str(args.shift),
                "erode_radius": str(args.erode_radius),
            },
        )
        meta["stage_arrays_npz"] = {
            "path": str(stage_path),
            "sha256": sha256_file(stage_path),
        }
        meta["stage_tensors"] = {
            "path": str(stage_tensors_path),
            "sha256": sha256_file(stage_tensors_path),
            "format": "safetensors",
            "dtype": "f32",
            "tensors": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in arrays.items()
            },
        }

    if args.skip_decode or args.skip_flow:
        write_json(args.output_dir / "metadata.json", meta)
        print(args.output_dir)
        return 0

    for count in counts:
        t0 = time.perf_counter()
        if args.save_decoder_trace:
            torch.manual_seed(args.seed)
            gaussian, decoder_trace = decode_latent_with_trace(
                pipe, latent_out["latent"], num_gaussians=count
            )
            decoder_trace_path = args.output_dir / f"decoder_trace_{count}_f32.safetensors"
            safetensors.torch.save_file(
                {
                    name: value.detach().float().cpu().contiguous()
                    for name, value in decoder_trace.items()
                },
                str(decoder_trace_path),
                metadata={
                    "format": "triposplat_decoder_trace_v1",
                    "dtype": "f32",
                    "seed": str(args.seed),
                    "num_gaussians": str(count),
                },
            )
        else:
            decoder_trace_path = None
            gaussian = pipe.decode_latent(latent_out["latent"], num_gaussians=count)
        ply_path = args.output_dir / f"reference_{count}.ply"
        splat_path = args.output_dir / f"reference_{count}.splat"
        gaussian.save_ply(str(ply_path))
        gaussian.save_splat(str(splat_path))
        meta["outputs"].append(
            {
                "gaussians": int(count),
                "decode_and_write_ms": (time.perf_counter() - t0) * 1000.0,
                "ply_path": str(ply_path),
                "ply_sha256": sha256_file(ply_path),
                "splat_path": str(splat_path),
                "splat_sha256": sha256_file(splat_path),
                "splat_bytes": splat_path.stat().st_size,
                "decoder_trace_path": str(decoder_trace_path)
                if decoder_trace_path is not None
                else None,
                "decoder_trace_sha256": sha256_file(decoder_trace_path)
                if decoder_trace_path is not None
                else None,
            }
        )

    meta_path = args.output_dir / "reference.json"
    write_json(meta_path, meta)
    print(f"TripoSplat upstream reference written: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
