from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


VIEWS = {
    "iso": (Vector((1.4, -1.6, 1.1)), 1.15),
    "front": (Vector((0.0, -2.2, 0.35)), 1.05),
    "side": (Vector((2.2, 0.0, 0.35)), 1.05),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render two GLBs in Blender and report RGB PSNR.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--fail-min-psnr", type=float, default=None)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh objects imported from {path}")
    return meshes


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (lo + hi) * 0.5
    extent = hi - lo
    return center, max(extent.length * 0.5, 0.01)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_render(resolution: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 64
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("world") if scene.world is None else scene.world
    scene.world.color = (0.03, 0.03, 0.035)


def add_camera_and_light(center: Vector, radius: float, view_dir: Vector, scale_mul: float) -> None:
    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = center + view_dir.normalized() * radius * 3.0
    look_at(cam, center)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(radius * 2.15 * scale_mul, 0.1)
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("key", "AREA")
    light = bpy.data.objects.new("key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((0.2, -1.0, 1.8)).normalized() * radius * 3.0
    light_data.energy = 450.0
    light_data.size = max(radius * 4.0, 1.0)


def remove_cameras_and_lights() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def render_asset(label: str, path: Path, out_dir: Path, resolution: int) -> list[Path]:
    clear_scene()
    setup_render(resolution)
    objects = import_glb(path)
    center, radius = bounds(objects)
    output_paths = []
    for view_name, (view_dir, scale_mul) in VIEWS.items():
        remove_cameras_and_lights()
        add_camera_and_light(center, radius, view_dir, scale_mul)
        out_path = out_dir / f"{label}_{view_name}.png"
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        output_paths.append(out_path)
    return output_paths


def load_rgb(path: Path) -> tuple[list[float], int, int]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = list(image.pixels)
        rgb = []
        for idx in range(0, len(pixels), 4):
            rgb.extend(pixels[idx : idx + 3])
        return rgb, width, height
    finally:
        bpy.data.images.remove(image)


def psnr(reference: Path, actual: Path) -> float:
    ref, ref_w, ref_h = load_rgb(reference)
    obs, obs_w, obs_h = load_rgb(actual)
    if (ref_w, ref_h) != (obs_w, obs_h):
        raise RuntimeError(f"image size mismatch: {reference}={ref_w}x{ref_h}, {actual}={obs_w}x{obs_h}")
    if len(ref) != len(obs):
        raise RuntimeError(f"image channel mismatch: {reference} vs {actual}")
    mse = sum((a - b) * (a - b) for a, b in zip(ref, obs)) / max(len(ref), 1)
    if mse <= 1.0e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_asset("reference", args.reference, args.out_dir, args.resolution)
    render_asset("actual", args.actual, args.out_dir, args.resolution)

    views = []
    for view_name in VIEWS:
        value = psnr(args.out_dir / f"reference_{view_name}.png", args.out_dir / f"actual_{view_name}.png")
        views.append({"view": view_name, "psnr_rgb": value})
    finite = [row["psnr_rgb"] for row in views if math.isfinite(row["psnr_rgb"])]
    summary = {
        "status": "ok",
        "reference": str(args.reference),
        "actual": str(args.actual),
        "resolution": args.resolution,
        "views": views,
        "min_psnr_rgb": min(finite) if finite else float("inf"),
        "mean_psnr_rgb": sum(finite) / len(finite) if finite else float("inf"),
    }
    if args.fail_min_psnr is not None and summary["min_psnr_rgb"] < args.fail_min_psnr:
        summary["status"] = "failed"
        summary["failures"] = [f"min_psnr_rgb {summary['min_psnr_rgb']:.4f} < {args.fail_min_psnr:.4f}"]
    (args.out_dir / "render_psnr.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if summary["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
