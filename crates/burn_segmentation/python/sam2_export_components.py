#!/usr/bin/env python3
"""Export SAM2 checkpoint components to safetensors for Burn import/parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


COMPONENT_SPECS = {
    "image_encoder": {
        "prefixes": ("image_encoder.",),
        "keys": ("no_mem_embed",),
    },
    "prompt_encoder": {
        "prefixes": ("sam_prompt_encoder.",),
        "keys": (),
    },
    "mask_decoder": {
        "prefixes": ("sam_mask_decoder.",),
        "keys": (),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SAM2 .pt components.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    report = {
        "checkpoint": str(args.checkpoint),
        "components": {},
    }
    for component, spec in COMPONENT_SPECS.items():
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in state.items()
            if any(key.startswith(prefix) for prefix in spec["prefixes"])
            or key in spec["keys"]
        }
        if not tensors:
            raise RuntimeError(f"no tensors matched {component}")
        output = args.output_dir / f"{component}.safetensors"
        save_file(tensors, str(output))
        report["components"][component] = {
            "path": str(output),
            "tensor_count": len(tensors),
            "total_elements": int(sum(t.numel() for t in tensors.values())),
        }

    metadata_path = args.metadata or (args.output_dir / "sam2_component_export.json")
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
