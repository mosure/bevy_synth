#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import trimesh
except Exception as exc:  # pragma: no cover - exercised by operator env.
    raise SystemExit(f"trellis2_render_psnr.py requires trimesh: {exc}") from exc


@dataclass
class RenderCloud:
    vertices: np.ndarray
    colors: np.ndarray


VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "back": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "iso": ((0.55, 0.45, 0.70), (0.0, 1.0, 0.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless TRELLIS.2 GLB render parity via deterministic orthographic splat renders."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--views", default="front,back,left,right,top,iso")
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--fail-min-psnr", type=float, default=None)
    parser.add_argument("--fail-min-mask-iou", type=float, default=None)
    return parser.parse_args()


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector {value}")
    return value / norm


def load_cloud(path: Path) -> RenderCloud:
    if not path.exists():
        raise FileNotFoundError(path)
    scene = trimesh.load(path, force="scene")
    meshes = scene.dump(concatenate=False) if isinstance(scene, trimesh.Scene) else [scene]
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for mesh in meshes:
        if len(mesh.vertices) == 0:
            continue
        vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
        colors.append(mesh_vertex_colors(mesh))
    if not vertices:
        raise ValueError(f"GLB has no renderable vertices: {path}")
    return RenderCloud(
        vertices=np.concatenate(vertices, axis=0),
        colors=np.concatenate(colors, axis=0),
    )


def mesh_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    vertex_count = len(mesh.vertices)
    if visual is not None and getattr(visual, "kind", None) == "texture":
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        texture = first_material_texture(material)
        if uv is not None and texture is not None and len(uv) == vertex_count:
            return sample_texture(texture, np.asarray(uv, dtype=np.float32), material)
    if visual is not None and getattr(visual, "kind", None) == "vertex":
        vertex_colors = np.asarray(visual.vertex_colors, dtype=np.float32)
        if vertex_colors.shape[0] == vertex_count and vertex_colors.shape[1] >= 3:
            return vertex_colors[:, :3].clip(0.0, 255.0).astype(np.uint8)
    material = getattr(visual, "material", None) if visual is not None else None
    color = getattr(material, "main_color", None)
    if color is None:
        color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = np.array([200, 200, 200, 255], dtype=np.float32)
    color_arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if color_arr.size >= 3 and float(color_arr[:3].max(initial=0.0)) <= 1.0:
        color_arr[:3] *= 255.0
    return np.repeat(color_arr[:3].clip(0.0, 255.0).astype(np.uint8)[None, :], vertex_count, axis=0)


def first_material_texture(material: object) -> Image.Image | None:
    if material is None:
        return None
    for name in ("baseColorTexture", "image"):
        image = getattr(material, name, None)
        if image is not None:
            return image.convert("RGBA")
    return None


def sample_texture(image: Image.Image, uv: np.ndarray, material: object) -> np.ndarray:
    tex = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height, width = tex.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    x = np.clip(np.rint(u * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint((1.0 - v) * (height - 1)).astype(np.int64), 0, height - 1)
    colors = tex[y, x, :3]
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        factor_arr = np.asarray(factor, dtype=np.float32).reshape(-1)
        if factor_arr.size >= 3:
            colors *= factor_arr[:3]
    return colors.clip(0.0, 255.0).astype(np.uint8)


def render_cloud(
    cloud: RenderCloud,
    view_name: str,
    resolution: int,
    splat_radius: int,
    center: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    view_dir, up_hint = VIEWS[view_name]
    camera_axis = normalize(np.asarray(view_dir, dtype=np.float32))
    up_hint_arr = normalize(np.asarray(up_hint, dtype=np.float32))
    right = normalize(np.cross(up_hint_arr, camera_axis))
    up = normalize(np.cross(camera_axis, right))

    local = cloud.vertices - center[None, :]
    sx = np.dot(local, right)
    sy = np.dot(local, up)
    depth = np.dot(local, camera_axis)
    px = np.rint((sx / scale + 0.5) * (resolution - 1)).astype(np.int64)
    py = np.rint((0.5 - sy / scale) * (resolution - 1)).astype(np.int64)

    offsets = splat_offsets(splat_radius)
    pix_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for dx, dy in offsets:
        ox = px + dx
        oy = py + dy
        valid = (ox >= 0) & (ox < resolution) & (oy >= 0) & (oy < resolution)
        if not np.any(valid):
            continue
        pix_parts.append(oy[valid] * resolution + ox[valid])
        depth_parts.append(depth[valid])
        color_parts.append(cloud.colors[valid])

    image = np.zeros((resolution * resolution, 3), dtype=np.uint8)
    mask = np.zeros((resolution * resolution,), dtype=np.uint8)
    if pix_parts:
        pix = np.concatenate(pix_parts)
        z = np.concatenate(depth_parts)
        cols = np.concatenate(color_parts, axis=0)
        order = np.lexsort((-z, pix))
        sorted_pix = pix[order]
        keep = np.empty(sorted_pix.shape[0], dtype=bool)
        keep[0] = True
        keep[1:] = sorted_pix[1:] != sorted_pix[:-1]
        winners = order[keep]
        win_pix = pix[winners]
        image[win_pix] = cols[winners]
        mask[win_pix] = 255
    return image.reshape((resolution, resolution, 3)), mask.reshape((resolution, resolution))


def splat_offsets(radius: int) -> list[tuple[int, int]]:
    if radius <= 0:
        return [(0, 0)]
    out: list[tuple[int, int]] = []
    r2 = radius * radius
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if x * x + y * y <= r2:
                out.append((x, y))
    return out


def camera_frame(ref: RenderCloud, actual: RenderCloud, view_name: str, margin: float = 1.12) -> tuple[np.ndarray, float]:
    points = np.concatenate([ref.vertices, actual.vertices], axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    view_dir, up_hint = VIEWS[view_name]
    camera_axis = normalize(np.asarray(view_dir, dtype=np.float32))
    up_hint_arr = normalize(np.asarray(up_hint, dtype=np.float32))
    right = normalize(np.cross(up_hint_arr, camera_axis))
    up = normalize(np.cross(camera_axis, right))
    local = points - center[None, :]
    span_x = float(np.ptp(np.dot(local, right)))
    span_y = float(np.ptp(np.dot(local, up)))
    return center, max(span_x, span_y, 1.0e-6) * margin


def psnr(ref: np.ndarray, actual: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        selected = mask > 0
        if not np.any(selected):
            return float("inf")
        delta = ref.astype(np.float32)[selected] - actual.astype(np.float32)[selected]
    else:
        delta = ref.astype(np.float32) - actual.astype(np.float32)
    mse = float(np.mean(delta * delta))
    if mse <= 1.0e-12:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def mask_iou(ref_mask: np.ndarray, actual_mask: np.ndarray) -> float:
    ref = ref_mask > 0
    actual = actual_mask > 0
    union = np.logical_or(ref, actual)
    if not np.any(union):
        return 1.0
    return float(np.logical_and(ref, actual).sum() / union.sum())


def write_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def main() -> int:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown = sorted(set(views) - set(VIEWS))
    if unknown:
        raise ValueError(f"unknown views: {unknown}; valid={sorted(VIEWS)}")
    splat_radius = args.splat_radius if args.splat_radius > 0 else max(1, round(args.resolution / 256))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference = load_cloud(args.reference)
    actual = load_cloud(args.actual)
    per_view = []
    for view_name in views:
        center, scale = camera_frame(reference, actual, view_name)
        ref_rgb, ref_mask = render_cloud(reference, view_name, args.resolution, splat_radius, center, scale)
        actual_rgb, actual_mask = render_cloud(actual, view_name, args.resolution, splat_radius, center, scale)
        union_mask = np.where((ref_mask > 0) | (actual_mask > 0), 255, 0).astype(np.uint8)
        overlap_psnr = psnr(ref_rgb, actual_rgb, union_mask)
        full_psnr = psnr(ref_rgb, actual_rgb)
        iou = mask_iou(ref_mask, actual_mask)
        write_png(args.out_dir / f"{view_name}_reference.png", ref_rgb)
        write_png(args.out_dir / f"{view_name}_actual.png", actual_rgb)
        write_png(args.out_dir / f"{view_name}_reference_mask.png", ref_mask)
        write_png(args.out_dir / f"{view_name}_actual_mask.png", actual_mask)
        per_view.append(
            {
                "view": view_name,
                "psnr_rgb": overlap_psnr,
                "psnr_full_rgb": full_psnr,
                "mask_iou": iou,
                "reference_mask_pixels": int((ref_mask > 0).sum()),
                "actual_mask_pixels": int((actual_mask > 0).sum()),
                "scale": scale,
            }
        )

    psnrs = [float(row["psnr_rgb"]) for row in per_view if math.isfinite(float(row["psnr_rgb"]))]
    ious = [float(row["mask_iou"]) for row in per_view]
    summary = {
        "status": "ok",
        "reference": str(args.reference),
        "actual": str(args.actual),
        "resolution": args.resolution,
        "splat_radius": splat_radius,
        "views": per_view,
        "min_psnr_rgb": min(psnrs) if psnrs else float("inf"),
        "mean_psnr_rgb": float(np.mean(psnrs)) if psnrs else float("inf"),
        "min_mask_iou": min(ious) if ious else 1.0,
        "mean_mask_iou": float(np.mean(ious)) if ious else 1.0,
        "reference_vertices": int(reference.vertices.shape[0]),
        "actual_vertices": int(actual.vertices.shape[0]),
    }
    failures = []
    if args.fail_min_psnr is not None and summary["min_psnr_rgb"] < args.fail_min_psnr:
        failures.append(f"min_psnr_rgb {summary['min_psnr_rgb']:.4f} < {args.fail_min_psnr:.4f}")
    if args.fail_min_mask_iou is not None and summary["min_mask_iou"] < args.fail_min_mask_iou:
        failures.append(f"min_mask_iou {summary['min_mask_iou']:.4f} < {args.fail_min_mask_iou:.4f}")
    if failures:
        summary["status"] = "failed"
        summary["failures"] = failures
    (args.out_dir / "render_psnr.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
