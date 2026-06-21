#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TRELLIS.2 Python reference hook and optional GLB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input image for full pipeline reference generation.",
    )
    parser.add_argument(
        "--trellis-root",
        type=Path,
        default=Path("tmp/upstream/TRELLIS.2/main"),
        help="Local TRELLIS.2 upstream checkout.",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path("crates/burn_trellis/assets/models/TRELLIS.2-4B"),
    )
    parser.add_argument(
        "--local-dino",
        type=Path,
        default=Path(
            "tmp/runs/20260618T235837Z_trellis2_dinov3_bpk_extract/"
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument("--output-hook", type=Path)
    parser.add_argument("--output-glb", type=Path)
    parser.add_argument("--output-obj", type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument(
        "--replay-hook",
        type=Path,
        help=(
            "Optional safetensors hook containing decode_shape_slat.input.* and "
            "decode_tex_slat.input.* tensors. When set, skip preprocess/encode/sample "
            "and replay only the upstream Python decode/export stages."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", default="1024_cascade")
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument(
        "--flow-dtype",
        choices=["default", "float32", "float16", "bfloat16"],
        default="default",
        help="Optionally force sparse/shape/texture flow model dtype.",
    )
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--sparse-attn-backend", default="")
    parser.add_argument(
        "--sparse-sdpa-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch upstream sparse full attention to PyTorch SDPA when flash_attn/xformers are unavailable.",
    )
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.0)
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument("--extension-webp", action="store_true")
    parser.add_argument(
        "--skip-hook-capture",
        action="store_true",
        help="Run the pipeline without saving hook tensors. Use this for timing-only benchmarks.",
    )
    parser.add_argument(
        "--skip-row-noise-capture",
        action="store_true",
        help=(
            "When saving hooks, capture dense sparse-structure noise and conditioning "
            "but skip row-shaped shape/texture noise. This avoids injecting row-noise "
            "fixtures when sparse decoder parity is still under investigation."
        ),
    )
    return parser.parse_args()


class BiRefNetPassthroughFallback:
    def __init__(self, model_name: str = "ignored"):
        self.model_name = model_name

    def to(self, _device: Any):
        return self

    def cuda(self):
        return self

    def cpu(self):
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            return image
        return image.convert("RGBA")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_pipeline_type(value: str) -> str:
    return "512" if value == "512_base" else value


def final_resolution_for_pipeline(pipeline_type: str) -> int:
    if pipeline_type.startswith("512"):
        return 512
    if pipeline_type.startswith("1536"):
        return 1536
    return 1024


def sparse_resolution_for_pipeline(pipeline_type: str) -> int:
    return 64 if pipeline_type == "1024" else 32


def to_cpu_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu().contiguous()
    else:
        out = torch.as_tensor(value).detach().cpu().contiguous()
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out.contiguous()


def scalar_tensor(value: float | int) -> torch.Tensor:
    return torch.tensor([float(value)], dtype=torch.float32)


def tensor_rows(captures: dict[str, torch.Tensor], key: str) -> int:
    tensor = captures.get(key)
    if tensor is None or tensor.ndim == 0:
        return 0
    return int(tensor.shape[0])


def sparse_rows(value: Any) -> int:
    coords = getattr(value, "coords", None)
    if coords is not None and hasattr(coords, "shape") and len(coords.shape) > 0:
        return int(coords.shape[0])
    feats = getattr(value, "feats", None)
    if feats is not None and hasattr(feats, "shape") and len(feats.shape) > 0:
        return int(feats.shape[0])
    return 0


def cond_tokens(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    cond = value.get("cond")
    if not isinstance(cond, torch.Tensor) or cond.ndim < 2:
        return 0
    if cond.ndim >= 3:
        return int(cond.shape[1])
    return int(cond.shape[0])


def cond_tokens_for_resolution(resolution: int) -> int:
    return (max(1, resolution) // 16) ** 2 + 5


def list_tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.float32).contiguous()


def require_tensor(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in tensors:
        raise RuntimeError(f"missing required replay tensor: {key}")
    return tensors[key]


def capture_sparse(
    captures: dict[str, torch.Tensor],
    prefix: str,
    value: Any,
    *,
    include_shape: bool = True,
) -> None:
    if hasattr(value, "coords"):
        captures[f"{prefix}.coords"] = to_cpu_tensor(value.coords, torch.float32)
    if hasattr(value, "feats"):
        captures[f"{prefix}.feats"] = to_cpu_tensor(value.feats, torch.float32)
    if include_shape and hasattr(value, "shape"):
        try:
            captures[f"{prefix}.shape"] = list_tensor(value.shape)
        except TypeError:
            pass
    if include_shape and hasattr(value, "spatial_shape"):
        captures[f"{prefix}.spatial_shape"] = list_tensor(value.spatial_shape)


def capture_mesh(
    captures: dict[str, torch.Tensor],
    prefix: str,
    vertices: Any,
    faces: Any,
) -> None:
    vertices_t = to_cpu_tensor(vertices, torch.float32)
    faces_t = to_cpu_tensor(faces, torch.float32)
    captures[f"{prefix}.vertices"] = vertices_t
    captures[f"{prefix}.faces"] = faces_t
    captures[f"{prefix}.vertices_count"] = scalar_tensor(vertices_t.shape[0])
    captures[f"{prefix}.faces_count"] = scalar_tensor(faces_t.shape[0])


def capture_sparse_noise(
    captures: dict[str, torch.Tensor],
    prefix: str,
    coords: torch.Tensor,
    feats: torch.Tensor,
) -> None:
    captures[f"{prefix}.coords"] = to_cpu_tensor(coords, torch.float32)
    captures[f"{prefix}.feats"] = to_cpu_tensor(feats, torch.float32)


def sampler_config_tensor(sampler: Any, params: dict[str, Any]) -> torch.Tensor:
    interval = params.get("guidance_interval", (0.0, 1.0))
    return torch.tensor(
        [
            float(params.get("steps", 50)),
            float(params.get("rescale_t", 1.0)),
            float(params.get("guidance_strength", 3.0)),
            float(params.get("guidance_rescale", 0.0)),
            float(interval[0]),
            float(interval[1]),
            float(getattr(sampler, "sigma_min", 1e-5)),
        ],
        dtype=torch.float32,
    ).contiguous()


def capture_sampler_config(
    captures: dict[str, torch.Tensor],
    prefix: str,
    sampler: Any,
    params: dict[str, Any],
) -> None:
    captures[f"{prefix}.config"] = sampler_config_tensor(sampler, params)


def export_obj(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    with path.open("w", encoding="utf-8") as handle:
        for v in vertices_np:
            handle.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in faces_np:
            handle.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def sparse_tensor_from_hook(
    sparse_cls: Any,
    tensors: dict[str, torch.Tensor],
    prefix: str,
    device: torch.device,
) -> Any:
    coords = require_tensor(tensors, f"{prefix}.coords").round().to(dtype=torch.int32)
    feats = require_tensor(tensors, f"{prefix}.feats").to(dtype=torch.float32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise RuntimeError(f"{prefix}.coords has invalid shape {tuple(coords.shape)}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise RuntimeError(
            f"{prefix}.feats shape {tuple(feats.shape)} is incompatible with "
            f"coords shape {tuple(coords.shape)}"
        )
    return sparse_cls(
        feats=feats.contiguous().to(device=device),
        coords=coords.contiguous().to(device=device),
    )


def create_patched_weights_root(weights_root: Path, local_dino: Path, out_root: Path) -> Path:
    payload = json.loads((weights_root / "pipeline.json").read_text(encoding="utf-8"))
    args = payload["args"]
    args["image_cond_model"]["args"]["model_name"] = str(local_dino)
    args["rembg_model"] = {"name": "BiRefNetPassthroughFallback", "args": {}}

    patched_root = out_root / "patched_weights"
    if patched_root.exists():
        shutil.rmtree(patched_root)
    patched_root.mkdir(parents=True, exist_ok=True)

    ckpts_link = patched_root / "ckpts"
    ckpts_link.symlink_to(weights_root / "ckpts")
    manifest_src = weights_root / "trellis2_import_manifest.json"
    if manifest_src.exists():
        (patched_root / "trellis2_import_manifest.json").symlink_to(manifest_src)
    (patched_root / "pipeline.json").write_text(
        json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": args}, indent=2),
        encoding="utf-8",
    )
    return patched_root


def force_flow_dtype(pipe: Any, dtype_name: str) -> None:
    if dtype_name == "default":
        return
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    for name, model in pipe.models.items():
        if "flow_model" in name and hasattr(model, "convert_to"):
            model.convert_to(dtype)


def patch_dinov3_layer_alias(pipe: Any) -> None:
    extractor = getattr(pipe, "image_cond_model", None)
    model = getattr(extractor, "model", None)
    if model is None or getattr(model, "layer", None) is not None:
        return
    inner = getattr(model, "model", None)
    inner_layer = getattr(inner, "layer", None)
    if inner_layer is not None:
        model.layer = inner_layer


def patch_sparse_sdpa_fallback() -> None:
    import torch.nn.functional as torch_f
    from trellis2.modules.sparse import VarLenTensor  # type: ignore
    from trellis2.modules.sparse.attention import full_attn as sparse_full_attn  # type: ignore
    from trellis2.modules.sparse.attention import modules as sparse_attn_modules  # type: ignore

    def seqlens(value: Any) -> list[int]:
        return [value.layout[i].stop - value.layout[i].start for i in range(value.shape[0])]

    def flatten_dense(value: torch.Tensor) -> tuple[torch.Tensor, list[int], tuple[int, ...]]:
        shape = tuple(value.shape)
        n, length = int(shape[0]), int(shape[1])
        return value.reshape(n * length, *shape[2:]), [length] * n, shape

    def sdpa_segments(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: list[int],
        kv_lens: list[int],
    ) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        q_start = 0
        kv_start = 0
        for q_len, kv_len in zip(q_lens, kv_lens):
            qi = q[q_start : q_start + q_len].permute(1, 0, 2).unsqueeze(0)
            ki = k[kv_start : kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)
            vi = v[kv_start : kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)
            oi = torch_f.scaled_dot_product_attention(qi, ki, vi)
            outs.append(oi.squeeze(0).permute(1, 0, 2).contiguous())
            q_start += q_len
            kv_start += kv_len
        return torch.cat(outs, dim=0)

    def sparse_sdpa(*args, **kwargs):
        arg_names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
        total = len(args) + len(kwargs)
        if total not in arg_names:
            raise AssertionError(f"Invalid number of arguments: {total}")
        values = list(args)
        for key in arg_names[total][len(args) :]:
            values.append(kwargs[key])

        output_template = None
        output_shape: tuple[int, ...] | None = None
        if total == 1:
            qkv = values[0]
            if not isinstance(qkv, VarLenTensor):
                raise AssertionError(f"qkv must be VarLenTensor, got {type(qkv)}")
            output_template = qkv
            q_lens = seqlens(qkv)
            kv_lens = q_lens
            q, k, v = qkv.feats.unbind(dim=1)
        elif total == 2:
            q, kv = values
            if isinstance(q, VarLenTensor):
                output_template = q
                q_lens = seqlens(q)
                q_flat = q.feats
            else:
                q_flat, q_lens, output_shape = flatten_dense(q)
            if isinstance(kv, VarLenTensor):
                kv_lens = seqlens(kv)
                k, v = kv.feats.unbind(dim=1)
            else:
                kv_flat, kv_lens, _ = flatten_dense(kv)
                k, v = kv_flat.unbind(dim=1)
            q = q_flat
        else:
            q, k, v = values
            if isinstance(q, VarLenTensor):
                output_template = q
                q_lens = seqlens(q)
                q_flat = q.feats
            else:
                q_flat, q_lens, output_shape = flatten_dense(q)
            if isinstance(k, VarLenTensor):
                kv_lens = seqlens(k)
                k_flat = k.feats
                v_flat = v.feats
            else:
                k_flat, kv_lens, _ = flatten_dense(k)
                v_flat, _, _ = flatten_dense(v)
            q, k, v = q_flat, k_flat, v_flat

        out = sdpa_segments(q, k, v, q_lens, kv_lens)
        if output_template is not None:
            return output_template.replace(out)
        assert output_shape is not None
        return out.reshape(output_shape[0], output_shape[1], *out.shape[1:])

    sparse_full_attn.sparse_scaled_dot_product_attention = sparse_sdpa
    sparse_attn_modules.sparse_scaled_dot_product_attention = sparse_sdpa


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    if args.input is not None:
        args.input = (root / args.input).resolve() if not args.input.is_absolute() else args.input
    args.trellis_root = (
        (root / args.trellis_root).resolve()
        if not args.trellis_root.is_absolute()
        else args.trellis_root
    )
    args.weights_root = (
        (root / args.weights_root).resolve()
        if not args.weights_root.is_absolute()
        else args.weights_root
    )
    args.local_dino = (
        (root / args.local_dino).resolve()
        if not args.local_dino.is_absolute()
        else args.local_dino
    )
    if args.output_hook is not None and not args.output_hook.is_absolute():
        args.output_hook = (root / args.output_hook).resolve()
    args.artifacts_dir = (
        (root / args.artifacts_dir).resolve()
        if not args.artifacts_dir.is_absolute()
        else args.artifacts_dir
    )
    if args.replay_hook is not None and not args.replay_hook.is_absolute():
        args.replay_hook = (root / args.replay_hook).resolve()
    if args.output_glb is not None and not args.output_glb.is_absolute():
        args.output_glb = (root / args.output_glb).resolve()
    if args.output_obj is not None and not args.output_obj.is_absolute():
        args.output_obj = (root / args.output_obj).resolve()

    if args.replay_hook is None and args.input is None:
        raise ValueError("--input is required unless --replay-hook is set")
    if not args.skip_hook_capture and args.output_hook is None:
        raise ValueError("--output-hook is required unless --skip-hook-capture is set")
    validation_paths = [
        (args.trellis_root, "TRELLIS.2 root"),
        (args.weights_root, "weights root"),
        (args.local_dino, "local DINOv3 root"),
    ]
    if args.input is not None:
        validation_paths.append((args.input, "input image"))
    if args.replay_hook is not None:
        validation_paths.append((args.replay_hook, "replay hook"))
    for path, label in validation_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Python TRELLIS.2 reference path")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if args.output_hook is not None:
        args.output_hook.parent.mkdir(parents=True, exist_ok=True)
    if args.output_glb is not None:
        args.output_glb.parent.mkdir(parents=True, exist_ok=True)
    if args.output_obj is not None:
        args.output_obj.parent.mkdir(parents=True, exist_ok=True)

    os.environ["ATTN_BACKEND"] = args.attention_backend
    if args.sparse_attn_backend:
        os.environ["SPARSE_ATTN_BACKEND"] = args.sparse_attn_backend
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Avoid host cuDNN/runtime skew while keeping CUDA tensor kernels active.
    torch.backends.cudnn.enabled = False

    sys.path.insert(0, str(args.trellis_root))
    if args.sparse_sdpa_fallback:
        patch_sparse_sdpa_fallback()
    from trellis2.pipelines import rembg as rembg_mod  # type: ignore
    from trellis2.pipelines.trellis2_image_to_3d import (  # type: ignore
        Trellis2ImageTo3DPipeline,
    )

    rembg_mod.BiRefNetPassthroughFallback = BiRefNetPassthroughFallback
    patched_root = create_patched_weights_root(args.weights_root, args.local_dino, args.artifacts_dir)

    capture_enabled = not args.skip_hook_capture
    captures: dict[str, torch.Tensor] = {}
    runtime_shapes: dict[str, int] = {}
    stage_timings_s: dict[str, float] = {}
    pipeline_type = normalize_pipeline_type(args.pipeline_type)
    final_resolution = final_resolution_for_pipeline(pipeline_type)
    sparse_resolution = sparse_resolution_for_pipeline(pipeline_type)

    load_start = time.perf_counter()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(patched_root))
    patch_dinov3_layer_alias(pipe)
    force_flow_dtype(pipe, args.flow_dtype)
    pipe.cuda()
    load_s = time.perf_counter() - load_start

    orig_preprocess = pipe.preprocess_image
    orig_get_cond = pipe.get_cond
    orig_sample_sparse_structure = pipe.sample_sparse_structure
    orig_sample_shape_slat = pipe.sample_shape_slat
    orig_sample_shape_slat_cascade = pipe.sample_shape_slat_cascade
    orig_sample_tex_slat = pipe.sample_tex_slat
    orig_decode_shape = pipe.decode_shape_slat
    orig_decode_tex = pipe.decode_tex_slat
    orig_decode_latent = pipe.decode_latent

    def time_stage(name: str, fn: Any, *fn_args: Any, **fn_kwargs: Any) -> Any:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*fn_args, **fn_kwargs)
        torch.cuda.synchronize()
        stage_timings_s[name] = stage_timings_s.get(name, 0.0) + (
            time.perf_counter() - start
        )
        return out

    def preprocess_wrapper(image: Image.Image) -> Image.Image:
        out = time_stage("preprocess", orig_preprocess, image)
        if capture_enabled:
            arr = np.asarray(out, dtype=np.uint8).copy()
            tensor = torch.from_numpy(arr).contiguous()
            captures["preprocess_image.output"] = tensor
            captures["run.image"] = tensor.clone()
        return out

    def get_cond_wrapper(
        image: Any, resolution: int, include_neg_cond: bool = True
    ) -> dict:
        out = time_stage(
            f"get_cond_{resolution}",
            orig_get_cond,
            image,
            resolution,
            include_neg_cond,
        )
        runtime_shapes[f"cond_{resolution}_tokens"] = cond_tokens(out)
        if capture_enabled:
            cond = out.get("cond") if isinstance(out, dict) else None
            neg_cond = out.get("neg_cond") if isinstance(out, dict) else None
            if isinstance(cond, torch.Tensor):
                captures[f"get_cond_{resolution}.out.cond"] = to_cpu_tensor(
                    cond, torch.float32
                )
            if isinstance(neg_cond, torch.Tensor):
                captures[f"get_cond_{resolution}.out.neg_cond"] = to_cpu_tensor(
                    neg_cond, torch.float32
                )
        return out

    def sample_sparse_structure_wrapper(
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        if capture_enabled:
            flow_model = pipe.models["sparse_structure_flow_model"]
            reso = int(flow_model.resolution)
            channels = int(flow_model.in_channels)
            rng_state = torch.get_rng_state()
            noise = torch.randn(num_samples, channels, reso, reso, reso).contiguous()
            torch.set_rng_state(rng_state)
            captures["sample_sparse_structure.noise"] = noise
            params = {**pipe.sparse_structure_sampler_params, **sampler_params}
            capture_sampler_config(
                captures,
                "sample_sparse_structure.sampler",
                pipe.sparse_structure_sampler,
                params,
            )
            orig_sampler_sample = pipe.sparse_structure_sampler.sample

            def sparse_sampler_capture_wrapper(*sample_args: Any, **sample_kwargs: Any) -> Any:
                sampled = orig_sampler_sample(*sample_args, **sample_kwargs)
                samples = getattr(sampled, "samples", None)
                if isinstance(samples, torch.Tensor):
                    captures["sample_sparse_structure.latent"] = to_cpu_tensor(
                        samples, torch.float32
                    )
                pred_x_t = getattr(sampled, "pred_x_t", None)
                if isinstance(pred_x_t, list):
                    total_steps = len(pred_x_t)
                    for step_idx, step_value in enumerate(pred_x_t):
                        if isinstance(step_value, torch.Tensor):
                            captures[
                                f"sample_sparse_structure.sampler.step_{step_idx:03}_of_{total_steps:03}.x_t"
                            ] = to_cpu_tensor(step_value, torch.float32)
                return sampled

            pipe.sparse_structure_sampler.sample = sparse_sampler_capture_wrapper
        else:
            orig_sampler_sample = None
        try:
            out = time_stage(
                "sparse_structure",
                orig_sample_sparse_structure,
                cond,
                resolution,
                num_samples,
                sampler_params,
            )
        finally:
            if orig_sampler_sample is not None:
                pipe.sparse_structure_sampler.sample = orig_sampler_sample
        runtime_shapes["sparse_coords"] = int(out.shape[0]) if out.ndim > 0 else 0
        if capture_enabled:
            captures["sample_sparse_structure.coords"] = to_cpu_tensor(out, torch.float32)
        return out

    def sample_shape_slat_wrapper(
        cond: dict,
        flow_model: Any,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> Any:
        resolution = int(getattr(flow_model, "resolution", 0) or 0)
        name = f"shape_slat_{resolution}" if resolution > 0 else "shape_slat"
        if capture_enabled and not args.skip_row_noise_capture:
            rng_state = torch.get_rng_state()
            noise = torch.randn(coords.shape[0], flow_model.in_channels).contiguous()
            torch.set_rng_state(rng_state)
            capture_sparse_noise(captures, "sample_shape_slat.noise", coords, noise)
            params = {**pipe.shape_slat_sampler_params, **sampler_params}
            capture_sampler_config(
                captures,
                "sample_shape_slat.sampler",
                pipe.shape_slat_sampler,
                params,
            )
        out = time_stage(
            name,
            orig_sample_shape_slat,
            cond,
            flow_model,
            coords,
            sampler_params,
        )
        runtime_shapes["shape_slat_rows"] = sparse_rows(out)
        if capture_enabled:
            capture_sparse(captures, f"{name}.output", out)
        return out

    def sample_shape_slat_cascade_wrapper(
        lr_cond: dict,
        cond: dict,
        flow_model_lr: Any,
        flow_model: Any,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> Any:
        if capture_enabled and not args.skip_row_noise_capture:
            rng_state = torch.get_rng_state()
            noise = torch.randn(coords.shape[0], flow_model_lr.in_channels).contiguous()
            torch.set_rng_state(rng_state)
            capture_sparse_noise(captures, "sample_shape_slat_lr.noise", coords, noise)
            params = {**pipe.shape_slat_sampler_params, **sampler_params}
            capture_sampler_config(
                captures,
                "sample_shape_slat.sampler",
                pipe.shape_slat_sampler,
                params,
            )
        out = time_stage(
            "shape_slat_cascade",
            orig_sample_shape_slat_cascade,
            lr_cond,
            cond,
            flow_model_lr,
            flow_model,
            lr_resolution,
            resolution,
            coords,
            sampler_params,
            max_num_tokens,
        )
        slat = out[0] if isinstance(out, tuple) else out
        runtime_shapes["shape_slat_rows"] = sparse_rows(slat)
        if isinstance(out, tuple) and len(out) > 1:
            runtime_shapes["shape_final_resolution"] = int(out[1])
        if capture_enabled:
            capture_sparse(captures, "shape_slat_cascade.output", slat)
        return out

    def sample_tex_slat_wrapper(
        cond: dict,
        flow_model: Any,
        shape_slat: Any,
        sampler_params: dict = {},
    ) -> Any:
        if capture_enabled and not args.skip_row_noise_capture:
            in_channels = (
                flow_model.in_channels
                if hasattr(flow_model, "in_channels")
                else flow_model[0].in_channels
            )
            shape_channels = int(shape_slat.feats.shape[1])
            rng_state = torch.get_rng_state()
            noise = torch.randn(
                shape_slat.coords.shape[0], int(in_channels) - shape_channels
            ).contiguous()
            torch.set_rng_state(rng_state)
            capture_sparse_noise(captures, "sample_tex_slat.noise", shape_slat.coords, noise)
            params = {**pipe.tex_slat_sampler_params, **sampler_params}
            capture_sampler_config(
                captures,
                "sample_tex_slat.sampler",
                pipe.tex_slat_sampler,
                params,
            )
        out = time_stage(
            "tex_slat",
            orig_sample_tex_slat,
            cond,
            flow_model,
            shape_slat,
            sampler_params,
        )
        runtime_shapes["tex_slat_rows"] = sparse_rows(out)
        if capture_enabled:
            capture_sparse(captures, "tex_slat.output", out)
        return out

    def decode_shape_wrapper(slat: Any, resolution: int):
        runtime_shapes["decode_shape_input_rows"] = sparse_rows(slat)
        if capture_enabled:
            capture_sparse(captures, "decode_shape_slat.input", slat)
        meshes, subs = time_stage("decode_shape", orig_decode_shape, slat, resolution)
        if capture_enabled:
            for index, sub in enumerate(subs):
                capture_sparse(captures, f"decode_shape_slat.subs.{index}", sub)
            if meshes:
                mesh0 = meshes[0]
                capture_mesh(captures, "decode_shape_slat.meshes.0", mesh0.vertices, mesh0.faces)
        return meshes, subs

    def decode_tex_wrapper(slat: Any, subs: Any):
        runtime_shapes["decode_tex_input_rows"] = sparse_rows(slat)
        if capture_enabled:
            capture_sparse(captures, "decode_tex_slat.input", slat)
        voxels = time_stage("decode_tex", orig_decode_tex, slat, subs)
        runtime_shapes["decode_tex_voxel_rows"] = sparse_rows(voxels)
        if capture_enabled:
            capture_sparse(captures, "decode_tex_slat.voxels", voxels)
        return voxels

    def decode_latent_wrapper(shape_slat: Any, tex_slat: Any, resolution: int):
        return time_stage("decode_latent_total", orig_decode_latent, shape_slat, tex_slat, resolution)

    pipe.preprocess_image = preprocess_wrapper
    pipe.get_cond = get_cond_wrapper
    pipe.sample_sparse_structure = sample_sparse_structure_wrapper
    pipe.sample_shape_slat = sample_shape_slat_wrapper
    pipe.sample_shape_slat_cascade = sample_shape_slat_cascade_wrapper
    pipe.sample_tex_slat = sample_tex_slat_wrapper
    pipe.decode_shape_slat = decode_shape_wrapper
    pipe.decode_tex_slat = decode_tex_wrapper
    pipe.decode_latent = decode_latent_wrapper

    infer_start = time.perf_counter()
    if args.replay_hook is not None:
        from trellis2.modules.sparse import SparseTensor  # type: ignore

        replay_tensors = load_file(str(args.replay_hook), device="cpu")
        shape_slat = sparse_tensor_from_hook(
            SparseTensor,
            replay_tensors,
            "decode_shape_slat.input",
            torch.device("cuda"),
        )
        tex_slat = sparse_tensor_from_hook(
            SparseTensor,
            replay_tensors,
            "decode_tex_slat.input",
            torch.device("cuda"),
        )
        if "run.final_resolution" in replay_tensors:
            final_resolution = int(
                round(float(replay_tensors["run.final_resolution"].flatten()[0].item()))
            )
        meshes = pipe.decode_latent(shape_slat, tex_slat, final_resolution)
    else:
        assert args.input is not None
        image = Image.open(args.input)
        meshes = pipe.run(
            image=image,
            seed=args.seed,
            pipeline_type=pipeline_type,
            max_num_tokens=args.max_num_tokens,
        )
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - infer_start

    mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes
    runtime_shapes["mesh_vertices"] = int(mesh.vertices.shape[0])
    runtime_shapes["mesh_faces"] = int(mesh.faces.shape[0])
    runtime_shapes["mesh_voxel_rows"] = int(mesh.coords.shape[0])
    if capture_enabled:
        capture_mesh(captures, "decode_latent.mesh.0", mesh.vertices, mesh.faces)
        captures["decode_latent.mesh.0.voxel_coords"] = to_cpu_tensor(mesh.coords, torch.float32)
        captures["decode_latent.mesh.0.voxel_attrs"] = to_cpu_tensor(mesh.attrs, torch.float32)
        captures["decode_latent.mesh.0.voxel_count"] = scalar_tensor(mesh.coords.shape[0])
        captures["run.final_resolution"] = scalar_tensor(final_resolution)
        captures["run.sparse_structure_resolution"] = scalar_tensor(sparse_resolution)

    if args.output_hook is not None:
        save_file(captures, str(args.output_hook))

    if args.output_obj is not None:
        export_obj(args.output_obj, mesh.vertices, mesh.faces)

    post_s = 0.0
    if args.output_glb is not None:
        import o_voxel  # type: ignore

        post_start = time.perf_counter()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            voxel_size=mesh.voxel_size,
            decimation_target=max(1, args.decimation_target),
            texture_size=max(64, args.texture_size),
            remesh=not args.no_remesh,
            remesh_band=float(args.remesh_band),
            remesh_project=float(args.remesh_project),
            verbose=True,
        )
        glb.export(args.output_glb, extension_webp=args.extension_webp)
        torch.cuda.synchronize()
        post_s = time.perf_counter() - post_start

    summary = {
        "status": "ok",
        "input": str(args.input),
        "trellis_root": str(args.trellis_root),
        "weights_root": str(args.weights_root),
        "patched_root": str(patched_root),
        "local_dino": str(args.local_dino),
        "output_hook": str(args.output_hook) if args.output_hook else None,
        "output_glb": str(args.output_glb) if args.output_glb else None,
        "output_obj": str(args.output_obj) if args.output_obj else None,
        "mode": "decoder_replay" if args.replay_hook is not None else "full_pipeline",
        "replay_hook": str(args.replay_hook) if args.replay_hook else None,
        "seed": args.seed,
        "pipeline_type": pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "flow_dtype": args.flow_dtype,
        "attention_backend": os.environ.get("ATTN_BACKEND"),
        "sparse_attn_backend": os.environ.get("SPARSE_ATTN_BACKEND"),
        "mesh": {
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "coords": int(mesh.coords.shape[0]),
            "attrs_channels": int(mesh.attrs.shape[1]),
            "voxel_size": float(mesh.voxel_size),
        },
        "shapes": {
            "sparse_coords": int(runtime_shapes.get("sparse_coords", 0)),
            "shape_slat_rows": int(runtime_shapes.get("shape_slat_rows", 0)),
            "tex_slat_rows": int(runtime_shapes.get("tex_slat_rows", 0)),
            "decode_shape_input_rows": int(
                runtime_shapes.get(
                    "decode_shape_input_rows",
                    tensor_rows(captures, "decode_shape_slat.input.coords"),
                )
            ),
            "decode_tex_input_rows": int(
                runtime_shapes.get(
                    "decode_tex_input_rows",
                    tensor_rows(captures, "decode_tex_slat.input.coords"),
                )
            ),
            "decode_tex_voxel_rows": int(
                runtime_shapes.get(
                    "decode_tex_voxel_rows",
                    tensor_rows(captures, "decode_tex_slat.voxels.coords"),
                )
            ),
            "cond_512_tokens": int(
                runtime_shapes.get(
                    "cond_512_tokens",
                    cond_tokens_for_resolution(512) if final_resolution >= 512 else 0,
                )
            ),
            "cond_1024_tokens": int(
                runtime_shapes.get(
                    "cond_1024_tokens",
                    cond_tokens_for_resolution(1024) if final_resolution >= 1024 else 0,
                )
            ),
        },
        "stage_timings_seconds": stage_timings_s,
        "timing_seconds": {
            "load": load_s,
            "infer": infer_s,
            "postprocess": post_s,
            "total": load_s + infer_s + post_s,
        },
    }
    write_json(args.artifacts_dir / "python_reference_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
