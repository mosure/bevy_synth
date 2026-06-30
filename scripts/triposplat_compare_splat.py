#!/usr/bin/env python3
"""Compare two TripoSplat .splat files with explicit numeric thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SPLAT_RECORD_BYTES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-position-abs", type=float, default=1.0e-3)
    parser.add_argument("--rms-position", type=float, default=1.0e-4)
    parser.add_argument("--max-scale-abs", type=float, default=1.0e-3)
    parser.add_argument("--rms-scale", type=float, default=1.0e-4)
    parser.add_argument("--max-rgba-abs", type=int, default=1)
    parser.add_argument("--max-rotation-u8-abs", type=int, default=1)
    return parser.parse_args()


def load_splat(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    if len(data) % SPLAT_RECORD_BYTES != 0:
        raise ValueError(
            f"{path} size {len(data)} is not a multiple of {SPLAT_RECORD_BYTES}"
        )
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, SPLAT_RECORD_BYTES)
    position = raw[:, 0:12].copy().view("<f4").reshape(-1, 3)
    scale = raw[:, 12:24].copy().view("<f4").reshape(-1, 3)
    rgba = raw[:, 24:28].astype(np.int16)
    rotation_u8 = raw[:, 28:32].astype(np.int16)
    return {
        "position": position,
        "scale": scale,
        "rgba": rgba,
        "rotation_u8": rotation_u8,
    }


def diff_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean() if abs_diff.size else 0.0),
        "rms": float(math.sqrt(float(np.mean(diff * diff))) if diff.size else 0.0),
    }


def main() -> int:
    args = parse_args()
    reference = load_splat(args.reference)
    candidate = load_splat(args.candidate)

    report: dict[str, Any] = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_count": int(reference["position"].shape[0]),
        "candidate_count": int(candidate["position"].shape[0]),
        "thresholds": {
            "max_position_abs": args.max_position_abs,
            "rms_position": args.rms_position,
            "max_scale_abs": args.max_scale_abs,
            "rms_scale": args.rms_scale,
            "max_rgba_abs": args.max_rgba_abs,
            "max_rotation_u8_abs": args.max_rotation_u8_abs,
        },
        "passed": False,
        "failures": [],
    }

    if report["reference_count"] != report["candidate_count"]:
        report["failures"].append(
            "splat count mismatch: "
            f"reference={report['reference_count']} candidate={report['candidate_count']}"
        )
    else:
        position = diff_stats(reference["position"], candidate["position"])
        scale = diff_stats(reference["scale"], candidate["scale"])
        rgba = diff_stats(reference["rgba"], candidate["rgba"])
        rotation = diff_stats(reference["rotation_u8"], candidate["rotation_u8"])
        report["diff"] = {
            "position": position,
            "scale": scale,
            "rgba": rgba,
            "rotation_u8": rotation,
        }
        if position["max_abs"] > args.max_position_abs:
            report["failures"].append(
                f"position max_abs {position['max_abs']} > {args.max_position_abs}"
            )
        if position["rms"] > args.rms_position:
            report["failures"].append(
                f"position rms {position['rms']} > {args.rms_position}"
            )
        if scale["max_abs"] > args.max_scale_abs:
            report["failures"].append(
                f"scale max_abs {scale['max_abs']} > {args.max_scale_abs}"
            )
        if scale["rms"] > args.rms_scale:
            report["failures"].append(f"scale rms {scale['rms']} > {args.rms_scale}")
        if rgba["max_abs"] > args.max_rgba_abs:
            report["failures"].append(
                f"rgba max_abs {rgba['max_abs']} > {args.max_rgba_abs}"
            )
        if rotation["max_abs"] > args.max_rotation_u8_abs:
            report["failures"].append(
                "rotation_u8 max_abs "
                f"{rotation['max_abs']} > {args.max_rotation_u8_abs}"
            )

    report["passed"] = not report["failures"]
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
