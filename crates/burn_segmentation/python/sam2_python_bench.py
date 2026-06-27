#!/usr/bin/env python3
"""Warm SAM2 Python/CUDA benchmark matching burn_segmentation stage semantics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_box(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box must contain x0,y0,x1,y1")
    return np.asarray(parts, dtype=np.float32)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def encode_features(model, predictor, input_image):
    backbone_out = model.forward_image(input_image)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
    ][::-1]
    return {
        "image_embed": feats[-1],
        "high_res_feats": feats[:-1],
    }


def run_once(model, predictor, image, image_hw, box_normalized, device):
    sample = {}

    start = time.perf_counter()
    input_image = predictor._transforms(image)[None, ...].to(device)
    sync(device)
    sample["preprocess_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    features = encode_features(model, predictor, input_image)
    sync(device)
    sample["encode_ms"] = elapsed_ms(start)

    h, w = image_hw
    box_px = box_normalized.copy()
    box_px[[0, 2]] *= float(w)
    box_px[[1, 3]] *= float(h)

    start = time.perf_counter()
    box_tensor = torch.as_tensor(box_px, dtype=torch.float32, device=device).reshape(1, 4)
    box_coords = predictor._transforms.transform_boxes(
        box_tensor, normalize=True, orig_hw=(h, w)
    ).reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels),
        boxes=None,
        masks=None,
    )
    dense_pe = model.sam_prompt_encoder.get_dense_pe()
    sync(device)
    sample["prompt_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    low_res_masks, iou_predictions, _sam_tokens, _object_score_logits = model.sam_mask_decoder(
        image_embeddings=features["image_embed"],
        image_pe=dense_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=features["high_res_feats"],
    )
    sync(device)
    sample["decode_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    masks = predictor._transforms.postprocess_masks(low_res_masks, (h, w))
    sync(device)
    sample["postprocess_ms"] = elapsed_ms(start)
    sample["total_ms"] = sum(
        sample[key]
        for key in [
            "preprocess_ms",
            "encode_ms",
            "prompt_ms",
            "decode_ms",
            "postprocess_ms",
        ]
    )
    sample["score"] = float(iou_predictions.detach().cpu().reshape(-1)[0])
    sample["area_px"] = int((masks.detach().cpu().reshape(-1) > 0.0).sum().item())
    return sample


def average(samples):
    keys = [
        "preprocess_ms",
        "encode_ms",
        "prompt_ms",
        "decode_ms",
        "postprocess_ms",
        "total_ms",
    ]
    return {key: sum(sample[key] for sample in samples) / len(samples) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm upstream SAM2 Python benchmark.")
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--box", type=parse_box, default=parse_box("0.15,0.10,0.85,0.90"))
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    model = build_sam2(
        args.config,
        str(args.checkpoint),
        device=device,
        apply_postprocessing=False,
    )
    model.eval()
    predictor = SAM2ImagePredictor(model)
    sync(device)
    load_ms = elapsed_ms(load_start)

    image = np.asarray(Image.open(args.image).convert("RGB")).copy()
    image_hw = image.shape[:2]

    for _ in range(args.warmup_runs):
        run_once(model, predictor, image, image_hw, args.box, device)

    samples = [
        {"run": run, **run_once(model, predictor, image, image_hw, args.box, device)}
        for run in range(args.runs)
    ]
    report = {
        "upstream": "facebookresearch/sam2",
        "config": args.config,
        "checkpoint": args.checkpoint.name,
        "image": str(args.image),
        "image_size": [int(image_hw[0]), int(image_hw[1])],
        "box_normalized": args.box.tolist(),
        "device": str(device),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.runs,
        "load_ms": load_ms,
        "samples": samples,
        "average": average(samples),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
