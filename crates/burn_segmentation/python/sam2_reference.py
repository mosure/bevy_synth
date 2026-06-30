#!/usr/bin/env python3
"""Capture SAM2 still-image reference hooks for Burn parity tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


def parse_box(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box must contain x0,y0,x1,y1")
    return np.asarray(parts, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture upstream SAM2 image/prompt/mask reference tensors."
    )
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", default="configs/sam2.1/sam2.1_hiera_t.yaml", type=str
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--box",
        type=parse_box,
        default=parse_box("0.15,0.10,0.85,0.90"),
        help="Normalized source-image box x0,y0,x1,y1.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    load_start = time.perf_counter()
    model = build_sam2(
        args.config,
        str(args.checkpoint),
        device=device,
        apply_postprocessing=False,
    )
    model.eval()
    predictor = SAM2ImagePredictor(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    image = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    box_px = args.box.copy()
    box_px[[0, 2]] *= float(w)
    box_px[[1, 3]] *= float(h)

    input_image = predictor._transforms(image)[None, ...].to(device)
    with torch.no_grad():
        trunk_outputs = model.image_encoder.trunk(input_image)
        neck_features, neck_pos = model.image_encoder.neck(trunk_outputs)
        image_encoder_out = model.forward_image(input_image)
        _, hook_vision_feats, _, _ = model._prepare_backbone_features(image_encoder_out)
        hook_feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(hook_vision_feats[::-1], predictor._bb_feat_sizes[::-1])
        ][::-1]
        hook_image_embed = hook_feats[-1]
        if model.directly_add_no_mem_embed:
            hook_image_embed = hook_image_embed + model.no_mem_embed.reshape(1, -1, 1, 1)

    encode_start = time.perf_counter()
    predictor.set_image(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    box_tensor = torch.as_tensor(box_px, dtype=torch.float32, device=device).reshape(1, 4)
    box_coords = predictor._transforms.transform_boxes(
        box_tensor, normalize=True, orig_hw=(h, w)
    ).reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)

    prompt_start = time.perf_counter()
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels),
        boxes=None,
        masks=None,
    )
    dense_pe = model.sam_prompt_encoder.get_dense_pe()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prompt_ms = (time.perf_counter() - prompt_start) * 1000.0

    high_res_features = [feat[0].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
    decode_start = time.perf_counter()
    low_res_masks, iou_predictions, sam_tokens, object_score_logits = model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][0].unsqueeze(0),
        image_pe=dense_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    masks = predictor._transforms.postprocess_masks(low_res_masks, (h, w))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    decoder = model.sam_mask_decoder
    s = 1 if decoder.pred_obj_scores else 0
    if decoder.pred_obj_scores:
        output_tokens = torch.cat(
            [decoder.obj_score_token.weight, decoder.iou_token.weight, decoder.mask_tokens.weight],
            dim=0,
        )
    else:
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
    decoder_tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)
    decoder_src_input = predictor._features["image_embed"][0].unsqueeze(0) + dense_embeddings
    decoder_pos_src = dense_pe
    decoder_hs, decoder_src_tokens = decoder.transformer(
        decoder_src_input,
        decoder_pos_src,
        decoder_tokens,
    )
    decoder_iou_token_out = decoder_hs[:, s, :]
    decoder_mask_tokens_out = decoder_hs[:, s + 1 : (s + 1 + decoder.num_mask_tokens), :]
    decoder_src_nchw = decoder_src_tokens.transpose(1, 2).view(
        decoder_src_input.shape[0],
        decoder_src_input.shape[1],
        decoder_src_input.shape[2],
        decoder_src_input.shape[3],
    )
    dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
    decoder_upscaled_embedding = act1(ln1(dc1(decoder_src_nchw) + high_res_features[1]))
    decoder_upscaled_embedding = act2(dc2(decoder_upscaled_embedding) + high_res_features[0])
    decoder_hyper_in = torch.stack(
        [
            decoder.output_hypernetworks_mlps[i](decoder_mask_tokens_out[:, i, :])
            for i in range(decoder.num_mask_tokens)
        ],
        dim=1,
    )
    db, dc, dh, dw = decoder_upscaled_embedding.shape
    decoder_all_masks = (
        decoder_hyper_in @ decoder_upscaled_embedding.view(db, dc, dh * dw)
    ).view(db, -1, dh, dw)
    decoder_all_iou_predictions = decoder.iou_prediction_head(decoder_iou_token_out)
    decoder_all_object_score_logits = decoder.pred_obj_score_head(decoder_hs[:, 0, :])

    tensors = {
        "source_image_chw": (torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0),
        "image_encoder_input": input_image.detach().cpu(),
        "trunk_feat0": trunk_outputs[0].detach().cpu(),
        "trunk_feat1": trunk_outputs[1].detach().cpu(),
        "trunk_feat2": trunk_outputs[2].detach().cpu(),
        "trunk_feat3": trunk_outputs[3].detach().cpu(),
        "neck_feature0": neck_features[0].detach().cpu(),
        "neck_feature1": neck_features[1].detach().cpu(),
        "neck_feature2": neck_features[2].detach().cpu(),
        "neck_feature3": neck_features[3].detach().cpu(),
        "forward_backbone_fpn0": image_encoder_out["backbone_fpn"][0].detach().cpu(),
        "forward_backbone_fpn1": image_encoder_out["backbone_fpn"][1].detach().cpu(),
        "forward_backbone_fpn2": image_encoder_out["backbone_fpn"][2].detach().cpu(),
        "forward_vision_features": image_encoder_out["vision_features"].detach().cpu(),
        "hook_image_embed": hook_image_embed.detach().cpu(),
        "box_normalized": torch.from_numpy(args.box.astype(np.float32)),
        "box_pixels": torch.from_numpy(box_px.astype(np.float32)),
        "box_coords_transformed": box_coords.detach().cpu(),
        "sparse_prompt_embeddings": sparse_embeddings.detach().cpu(),
        "dense_prompt_embeddings": dense_embeddings.detach().cpu(),
        "dense_pe": dense_pe.detach().cpu(),
        "image_embed": predictor._features["image_embed"][0].detach().cpu(),
        "high_res_feat0": high_res_features[0].detach().cpu(),
        "high_res_feat1": high_res_features[1].detach().cpu(),
        "low_res_masks": low_res_masks.detach().cpu(),
        "iou_predictions": iou_predictions.detach().cpu(),
        "sam_tokens": sam_tokens.detach().cpu(),
        "object_score_logits": object_score_logits.detach().cpu(),
        "postprocessed_masks": masks.detach().cpu(),
        "decoder_tokens": decoder_tokens.detach().cpu(),
        "decoder_src_input": decoder_src_input.detach().cpu(),
        "decoder_hs": decoder_hs.detach().cpu(),
        "decoder_src_tokens": decoder_src_tokens.detach().cpu(),
        "decoder_iou_token_out": decoder_iou_token_out.detach().cpu(),
        "decoder_mask_tokens_out": decoder_mask_tokens_out.detach().cpu(),
        "decoder_upscaled_embedding": decoder_upscaled_embedding.detach().cpu(),
        "decoder_hyper_in": decoder_hyper_in.detach().cpu(),
        "decoder_all_masks": decoder_all_masks.detach().cpu(),
        "decoder_all_iou_predictions": decoder_all_iou_predictions.detach().cpu(),
        "decoder_all_object_score_logits": decoder_all_object_score_logits.detach().cpu(),
    }
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in tensors.items()
    }
    save_file(tensors, str(args.output))

    metadata = {
        "upstream": "facebookresearch/sam2",
        "config": args.config,
        "checkpoint": args.checkpoint.name,
        "image": str(args.image),
        "image_size": [int(h), int(w)],
        "box_normalized": args.box.tolist(),
        "box_pixels": box_px.tolist(),
        "device": str(device),
        "load_ms": load_ms,
        "encode_ms": encode_ms,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "pid": os.getpid(),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
