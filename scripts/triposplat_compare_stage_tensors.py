#!/usr/bin/env python3
"""Compare TripoSplat stage safetensors with explicit numeric thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


DEFAULT_TENSORS = ["image_rgb_0_1", "feature1", "feature2", "latent", "camera"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--tensor",
        action="append",
        default=[],
        help="Tensor name to compare. May be repeated; defaults to known TripoSplat stage tensors present in the reference file.",
    )
    parser.add_argument("--max-abs", type=float, default=1.0e-2)
    parser.add_argument("--mean-abs", type=float, default=1.0e-3)
    parser.add_argument("--rms", type=float, default=2.0e-3)
    return parser.parse_args()


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as tensors:
        return {name: tensors.get_tensor(name) for name in tensors.keys()}


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
    reference = load_tensors(args.reference)
    candidate = load_tensors(args.candidate)
    requested = args.tensor or [name for name in DEFAULT_TENSORS if name in reference]

    report: dict[str, Any] = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "thresholds": {
            "max_abs": args.max_abs,
            "mean_abs": args.mean_abs,
            "rms": args.rms,
        },
        "tensors": {},
        "passed": False,
        "failures": [],
    }

    if not requested:
        report["failures"].append("no tensors selected for comparison")

    for name in requested:
        if name not in reference:
            report["failures"].append(f"missing reference tensor {name}")
            continue
        if name not in candidate:
            report["failures"].append(f"missing candidate tensor {name}")
            continue
        lhs = reference[name]
        rhs = candidate[name]
        tensor_report: dict[str, Any] = {
            "reference_shape": list(lhs.shape),
            "candidate_shape": list(rhs.shape),
            "reference_dtype": str(lhs.dtype),
            "candidate_dtype": str(rhs.dtype),
        }
        if lhs.shape != rhs.shape:
            report["failures"].append(
                f"{name} shape mismatch: reference={lhs.shape} candidate={rhs.shape}"
            )
        else:
            stats = diff_stats(lhs, rhs)
            tensor_report["diff"] = stats
            if stats["max_abs"] > args.max_abs:
                report["failures"].append(
                    f"{name} max_abs {stats['max_abs']} > {args.max_abs}"
                )
            if stats["mean_abs"] > args.mean_abs:
                report["failures"].append(
                    f"{name} mean_abs {stats['mean_abs']} > {args.mean_abs}"
                )
            if stats["rms"] > args.rms:
                report["failures"].append(f"{name} rms {stats['rms']} > {args.rms}")
        report["tensors"][name] = tensor_report

    report["passed"] = not report["failures"]
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
