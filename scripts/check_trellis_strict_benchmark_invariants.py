#!/usr/bin/env python3
"""
Validate strict-benchmark invariants from a trellis2_run log.

This script is intentionally lightweight so it can run in CI or local loops:
- parses JSON lines emitted by `trellis2_run`
- enforces canonical invariants (status/dispatch/readback)
- optionally enforces absolute thresholds and baseline regression limits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _extract_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(obj.get("timings_ms"), dict):
        return {
            "status": obj.get("status"),
            "strict_benchmark": obj.get("strict_benchmark"),
            "timings_ms": obj["timings_ms"],
        }
    last = obj.get("last")
    if isinstance(last, dict) and isinstance(last.get("timings_ms"), dict):
        return {
            "status": obj.get("status"),
            "strict_benchmark": obj.get("strict_benchmark"),
            "timings_ms": last["timings_ms"],
        }
    return None


def _load_last_record(log_path: Path) -> tuple[dict[str, Any], int]:
    last_record: dict[str, Any] | None = None
    last_lineno = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            record = _extract_record(obj)
            if record is None:
                continue
            last_record = record
            last_lineno = lineno
    if last_record is None:
        raise ValueError(f"no benchmark JSON record found in '{log_path}'")
    return last_record, last_lineno


def _load_baseline_record(path: Path) -> dict[str, Any]:
    record, _ = _load_last_record(path)
    return record


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"timings_ms['{key}'] is not numeric: {value!r}")


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate trellis2 strict benchmark invariants from run logs."
    )
    parser.add_argument("log_path", type=Path, help="Path to trellis2_run log")
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Optional baseline log for regression checks",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=None,
        help="Allowed regression percent versus baseline for selected compare keys",
    )
    parser.add_argument(
        "--compare-keys",
        default="total,sparse,shape_slat,tex_slat,decode",
        help="Comma-separated timings_ms keys for baseline regression checks",
    )
    parser.add_argument("--max-total-ms", type=float, default=None)
    parser.add_argument("--max-sparse-ms", type=float, default=None)
    parser.add_argument("--max-shape-slat-ms", type=float, default=None)
    parser.add_argument("--max-tex-slat-ms", type=float, default=None)
    parser.add_argument("--max-decode-ms", type=float, default=None)
    parser.add_argument(
        "--min-shape-dispatches",
        type=int,
        default=1,
        help="Minimum decode_shape_wgpu_dispatches",
    )
    parser.add_argument(
        "--min-tex-dispatches",
        type=int,
        default=1,
        help="Minimum decode_tex_wgpu_dispatches",
    )
    args = parser.parse_args()

    record, lineno = _load_last_record(args.log_path)
    timings = record["timings_ms"]
    failures: list[str] = []

    _check(
        record.get("status") == "ok",
        f"status must be 'ok' (got {record.get('status')!r})",
        failures,
    )
    _check(
        record.get("strict_benchmark") is True,
        f"strict_benchmark must be true (got {record.get('strict_benchmark')!r})",
        failures,
    )

    host_readback = timings.get("host_readback_count")
    _check(
        host_readback == 0,
        f"host_readback_count must be 0 (got {host_readback!r})",
        failures,
    )

    shape_dispatches = timings.get("decode_shape_wgpu_dispatches")
    tex_dispatches = timings.get("decode_tex_wgpu_dispatches")
    decode_stage_fenced = timings.get("decode_stage_fenced")
    _check(
        isinstance(shape_dispatches, (int, float))
        and int(shape_dispatches) >= args.min_shape_dispatches,
        "decode_shape_wgpu_dispatches below minimum "
        f"({shape_dispatches!r} < {args.min_shape_dispatches})",
        failures,
    )
    _check(
        isinstance(tex_dispatches, (int, float))
        and int(tex_dispatches) >= args.min_tex_dispatches,
        "decode_tex_wgpu_dispatches below minimum "
        f"({tex_dispatches!r} < {args.min_tex_dispatches})",
        failures,
    )
    # Newer logs carry decode_stage_fenced; strict runs must keep per-stage
    # decode timing fenced so substage timings represent GPU completion.
    if decode_stage_fenced is not None:
        _check(
            decode_stage_fenced is True,
            f"decode_stage_fenced must be true in strict benchmark (got {decode_stage_fenced!r})",
            failures,
        )

    absolute_thresholds = [
        ("total", args.max_total_ms),
        ("sparse", args.max_sparse_ms),
        ("shape_slat", args.max_shape_slat_ms),
        ("tex_slat", args.max_tex_slat_ms),
        ("decode", args.max_decode_ms),
    ]
    for key, max_value in absolute_thresholds:
        if max_value is None:
            continue
        current = _as_float(timings.get(key), key)
        _check(
            current <= max_value,
            f"{key} exceeded threshold ({current:.3f} > {max_value:.3f})",
            failures,
        )

    if args.baseline_log is not None:
        if args.max_regression_pct is None:
            failures.append("--baseline-log requires --max-regression-pct")
        else:
            baseline = _load_baseline_record(args.baseline_log)
            baseline_timings = baseline["timings_ms"]
            keys = [key.strip() for key in args.compare_keys.split(",") if key.strip()]
            for key in keys:
                current = _as_float(timings.get(key), key)
                base = _as_float(baseline_timings.get(key), key)
                if base <= 0.0:
                    continue
                allowed = base * (1.0 + args.max_regression_pct / 100.0)
                _check(
                    current <= allowed,
                    f"{key} regressed beyond {args.max_regression_pct:.2f}% "
                    f"(current={current:.3f} baseline={base:.3f} allowed={allowed:.3f})",
                    failures,
                )

    print(f"validated record at {args.log_path}:{lineno}")
    print(
        "summary: "
        f"total={_as_float(timings.get('total'), 'total'):.3f} ms, "
        f"sparse={_as_float(timings.get('sparse'), 'sparse'):.3f} ms, "
        f"shape_slat={_as_float(timings.get('shape_slat'), 'shape_slat'):.3f} ms, "
        f"tex_slat={_as_float(timings.get('tex_slat'), 'tex_slat'):.3f} ms, "
        f"decode={_as_float(timings.get('decode'), 'decode'):.3f} ms, "
        f"decode_stage_fenced={decode_stage_fenced!r}"
    )

    if failures:
        print("invariant check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("all strict benchmark invariants passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
