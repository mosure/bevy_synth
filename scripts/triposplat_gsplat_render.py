#!/usr/bin/env python3
"""Render a TripoSplat `.splat` cloud with gsplat.

The script intentionally avoids PIL/imageio so it can run in the project Torch
venv with only numpy, torch, and gsplat installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import time
import zlib
from typing import Any


SPLAT_RECORD_BYTES = 32
DEFAULT_BACKGROUND = (0.05, 0.055, 0.065)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a TripoSplat .splat file with gsplat/CUDA."
    )
    parser.add_argument("--input-splat", type=pathlib.Path, required=True)
    parser.add_argument(
        "--frame-splat",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Additional .splat files included only when computing shared camera framing.",
    )
    parser.add_argument("--output-image", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frame-margin", type=float, default=1.35)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1000.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--camera-model", choices=("ortho", "pinhole"), default="ortho")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--background", default=",".join(str(v) for v in DEFAULT_BACKGROUND))
    parser.add_argument(
        "--cuda-home",
        help="Set CUDA_HOME/CUDA_PATH before importing torch/gsplat.",
    )
    parser.add_argument(
        "--torch-cuda-arch-list",
        help="Set TORCH_CUDA_ARCH_LIST before importing torch/gsplat.",
    )
    parser.add_argument(
        "--python-include-root",
        type=pathlib.Path,
        help="Optional extracted sysroot with usr/include/pythonX.Y headers.",
    )
    parser.add_argument(
        "--packed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use gsplat packed sparse projection mode.",
    )
    return parser.parse_args()


def apply_environment(args: argparse.Namespace) -> None:
    if args.cuda_home:
        os.environ["CUDA_HOME"] = args.cuda_home
        os.environ["CUDA_PATH"] = args.cuda_home
    if args.torch_cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    if args.python_include_root:
        include_root = args.python_include_root / "usr" / "include"
        if include_root.exists():
            paths = [
                str(path)
                for path in include_root.iterdir()
                if path.is_dir() and path.name.startswith("python")
            ]
            paths.append(str(include_root))
        else:
            paths = [str(args.python_include_root)]
        for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
            current = os.environ.get(key)
            os.environ[key] = os.pathsep.join(paths + ([current] if current else []))


def parse_background(text: str) -> tuple[float, float, float]:
    parts = [float(value.strip()) for value in text.split(",")]
    if len(parts) != 3 or any(not math.isfinite(value) for value in parts):
        raise ValueError("--background must contain three finite comma-separated floats")
    return (parts[0], parts[1], parts[2])


def read_splat(path: pathlib.Path):
    import numpy as np

    data = path.read_bytes()
    if len(data) % SPLAT_RECORD_BYTES:
        raise ValueError(
            f"{path} byte length {len(data)} is not divisible by {SPLAT_RECORD_BYTES}"
        )
    dtype = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    records = np.frombuffer(data, dtype=dtype)
    if records.size == 0:
        raise ValueError(f"{path} contains no splat records")
    return records


def frame_for_records(record_sets, margin: float) -> tuple[Any, float, Any, Any]:
    import numpy as np

    mins = np.full((3,), np.inf, dtype=np.float32)
    maxs = np.full((3,), -np.inf, dtype=np.float32)
    for records in record_sets:
        positions = records["position"].astype(np.float32)
        scales = records["scale"].astype(np.float32)
        extents = np.maximum(np.abs(scales), 1.0e-4) * 3.0
        mins = np.minimum(mins, np.min(positions - extents, axis=0))
        maxs = np.maximum(maxs, np.max(positions + extents, axis=0))
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        raise ValueError("cannot frame non-finite splat bounds")
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs[:2] - mins[:2]) * 0.5 * margin), 1.0e-4)
    return center, radius, mins, maxs


def splat_tensors(records, center, radius: float, device: str):
    import numpy as np
    import torch

    positions = records["position"].astype(np.float32).copy()
    scales = np.maximum(records["scale"].astype(np.float32).copy(), 1.0e-6)
    rgba = records["rgba"].astype(np.float32)
    rotations = records["rotation"].astype(np.float32)

    means = positions.copy()
    means[:, 0] = (means[:, 0] - center[0]) / radius
    means[:, 1] = (means[:, 1] - center[1]) / radius
    means[:, 2] = means[:, 2] - float(np.min(positions[:, 2])) + 1.0
    scales = scales / radius

    quats = (rotations - 128.0) / 128.0
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(quat_norm, 1.0e-8)

    return {
        "means": torch.as_tensor(means, device=device, dtype=torch.float32),
        "quats": torch.as_tensor(quats, device=device, dtype=torch.float32),
        "scales": torch.as_tensor(scales, device=device, dtype=torch.float32),
        "opacities": torch.as_tensor(
            np.clip(rgba[:, 3] / 255.0, 0.0, 1.0), device=device, dtype=torch.float32
        ),
        "colors": torch.as_tensor(
            np.clip(rgba[:, :3] / 255.0, 0.0, 1.0), device=device, dtype=torch.float32
        ),
    }


def make_camera(width: int, height: int, device: str):
    import torch

    focal = float(min(width, height)) * 0.5
    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    intrinsics = torch.tensor(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return viewmat, intrinsics


def render(args: argparse.Namespace, records, center, radius: float, background):
    import gsplat
    import torch

    tensors = splat_tensors(records, center, radius, args.device)
    viewmat, intrinsics = make_camera(args.width, args.height, args.device)
    bg = torch.tensor(background, device=args.device, dtype=torch.float32)

    timings_ms: list[float] = []
    last = None
    total = max(args.warmup + args.repeat, 1)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for iteration in range(total):
        started = time.perf_counter()
        colors, alphas, meta = gsplat.rasterization(
            tensors["means"],
            tensors["quats"],
            tensors["scales"],
            tensors["opacities"],
            tensors["colors"],
            viewmat,
            intrinsics,
            args.width,
            args.height,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            radius_clip=args.radius_clip,
            eps2d=args.eps2d,
            packed=args.packed,
            tile_size=args.tile_size,
            backgrounds=bg,
            camera_model=args.camera_model,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= args.warmup:
            timings_ms.append(elapsed_ms)
        last = (colors, alphas, meta)

    assert last is not None
    colors, alphas, meta = last
    rgba = torch.cat([colors[0].clamp(0.0, 1.0), alphas[0].clamp(0.0, 1.0)], dim=-1)
    image = (rgba * 255.0).round().to(torch.uint8).cpu().numpy()
    return image, meta, timings_ms


def write_png(path: pathlib.Path, rgba) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError(f"expected RGBA image, got {rgba.shape}")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))
    payload = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", payload)
        + chunk(b"IDAT", zlib.compress(rows, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(encoded)


def tensor_scalar(tensor, default: int = 0) -> int:
    try:
        return int(tensor.sum().item())
    except Exception:
        return default


def nvidia_smi() -> list[dict[str, str]]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                    "utilization_gpu_percent": parts[4],
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    apply_environment(args)
    background = parse_background(args.background)

    import gsplat
    import numpy as np
    import torch

    records = read_splat(args.input_splat)
    frame_records = [records] + [read_splat(path) for path in args.frame_splat]
    center, radius, bounds_min, bounds_max = frame_for_records(
        frame_records, args.frame_margin
    )
    image, meta, timings_ms = render(args, records, center, radius, background)
    write_png(args.output_image, image)

    peak_memory = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated())

    report = {
        "input_splat": str(args.input_splat),
        "frame_splats": [str(path) for path in args.frame_splat],
        "output_image": str(args.output_image),
        "records": int(records.size),
        "width": args.width,
        "height": args.height,
        "camera_model": args.camera_model,
        "background": list(background),
        "frame_margin": args.frame_margin,
        "frame_center": center.astype(float).tolist(),
        "frame_radius": float(radius),
        "bounds_min": bounds_min.astype(float).tolist(),
        "bounds_max": bounds_max.astype(float).tolist(),
        "visible_gaussians": tensor_scalar(meta.get("radii", np.array([])) > 0),
        "tiles_per_gauss_sum": tensor_scalar(meta.get("tiles_per_gauss", np.array([]))),
        "timings_ms": timings_ms,
        "mean_timing_ms": float(sum(timings_ms) / len(timings_ms)) if timings_ms else None,
        "peak_torch_memory_bytes": peak_memory,
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "gsplat_version": getattr(gsplat, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi(),
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
