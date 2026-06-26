#!/usr/bin/env python3
"""Run the upstream LocateAnything reference model and emit JSON fixtures.

This is reference tooling for parity capture. It deliberately stays outside the
Rust runtime path and should be used to generate hook/timing evidence that Burn
implementations can compare against.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="assets/models/LocateAnything-3B")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocess-npz")
    parser.add_argument("--preprocess-safetensors")
    parser.add_argument("--hook-npz")
    parser.add_argument("--hook-safetensors")
    parser.add_argument("--hook-vision-layers", default="0,13,26")
    parser.add_argument("--hook-language-layers", default="0,17,35")
    parser.add_argument("--hook-max-language-forwards", type=int, default=4)
    parser.add_argument("--in-token-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "f16", "f32"], default="bf16")
    parser.add_argument("--generation-mode", choices=["fast", "slow", "hybrid"], default="hybrid")
    parser.add_argument("--attn", choices=["sdpa", "magi", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--batch-runtime", action="store_true")
    return parser.parse_args()


def parse_layer_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "f16":
        return torch.float16
    return torch.float32


def build_messages(image: Image.Image, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def detect_prompt(query: str) -> str:
    return f"Locate all the instances that matches the following description: {query}."


def normalized_coord(value: str) -> float:
    parsed = float(value)
    if parsed > 1.0:
        parsed /= 1000.0
    if not np.isfinite(parsed):
        return 0.0
    return float(min(1.0, max(0.0, parsed)))


def parse_detections(answer: str, source_query: str) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    ref_pattern = re.compile(r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)", re.DOTALL)
    box_pattern = re.compile(
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )
    point_pattern = re.compile(
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )
    for label, boxes in ref_pattern.findall(answer):
        if "<box>none</box>" in boxes:
            continue
        for match in box_pattern.findall(boxes):
            x1, y1, x2, y2 = [normalized_coord(value) for value in match]
            detections.append(
                {
                    "label": label.strip(),
                    "bbox": [x1, y1, x2, y2],
                    "source_query": source_query,
                }
            )
        if not box_pattern.search(boxes):
            for match in point_pattern.findall(boxes):
                x, y = [normalized_coord(value) for value in match]
                detections.append(
                    {
                        "label": label.strip(),
                        "bbox": [x, y, x, y],
                        "point": [x, y],
                        "source_query": source_query,
                    }
                )
    return detections


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        if tensors:
            if len(tensors) == 1:
                return tensors[0]
            try:
                return torch.cat(tensors, dim=0)
            except Exception:
                return tensors[0]
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if hasattr(value, "logits") and torch.is_tensor(value.logits):
        return value.logits
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    return None


def store_tensor(hooks: dict[str, np.ndarray], name: str, value: Any) -> None:
    tensor = first_tensor(value)
    if tensor is None:
        return
    hooks[name] = tensor.detach().float().cpu().numpy()


def locate_token_ids(model: Any) -> np.ndarray:
    token_ids = getattr(model, "token_ids", {})
    special = [
        token_ids.get("box_start_token_id"),
        token_ids.get("box_end_token_id"),
        token_ids.get("ref_start_token_id"),
        token_ids.get("ref_end_token_id"),
        token_ids.get("none_token_id"),
        token_ids.get("null_token_id"),
        token_ids.get("im_end_token_id"),
        token_ids.get("switch_token_id"),
        token_ids.get("default_mask_token_id"),
    ]
    coord_start = token_ids.get("coord_start_token_id", 151677)
    coord_end = token_ids.get("coord_end_token_id", 152677)
    ids = [value for value in special if value is not None]
    ids.extend(range(coord_start, coord_end + 1))
    return np.asarray(sorted(set(int(value) for value in ids)), dtype=np.int64)


def install_hooks(
    model: Any,
    vision_layers: list[int],
    language_layers: list[int],
    max_language_forwards: int,
) -> tuple[dict[str, np.ndarray], list[Any]]:
    hooks: dict[str, np.ndarray] = {}
    handles: list[Any] = []
    seen_once: set[str] = set()
    language_forward_count = {"count": 0}
    selected_ids = locate_token_ids(model)
    hooks["selected_token_ids"] = selected_ids
    selected_ids_tensor = torch.as_tensor(selected_ids, dtype=torch.long, device=model.device)

    def once(name: str):
        def hook(_module, _inputs, output):
            if name in seen_once:
                return
            seen_once.add(name)
            store_tensor(hooks, name, output)

        return hook

    def language_forward_hook(_module, _inputs, output):
        idx = language_forward_count["count"]
        language_forward_count["count"] += 1
        if idx >= max_language_forwards:
            return
        logits = getattr(output, "logits", None)
        if logits is None:
            return
        tail = logits[:, -min(6, logits.shape[1]) :, :]
        selected = tail.index_select(-1, selected_ids_tensor).detach().float().cpu().numpy()
        values, indices = torch.topk(tail, k=min(16, tail.shape[-1]), dim=-1)
        hooks[f"language_forward_{idx:02d}_tail_selected_logits"] = selected
        hooks[f"language_forward_{idx:02d}_tail_topk_values"] = (
            values.detach().float().cpu().numpy()
        )
        hooks[f"language_forward_{idx:02d}_tail_topk_ids"] = (
            indices.detach().cpu().numpy().astype(np.int64)
        )

    handles.append(model.vision_model.patch_embed.register_forward_hook(once("vision.patch_embed")))
    handles.append(model.vision_model.encoder.final_layernorm.register_forward_hook(once("vision.final_layernorm")))
    handles.append(model.vision_model.register_forward_hook(once("vision.merged_tokens")))
    handles.append(model.mlp1.register_forward_hook(once("projector.mlp1")))
    handles.append(model.language_model.model.embed_tokens.register_forward_hook(once("language.embed_tokens")))
    handles.append(model.language_model.model.norm.register_forward_hook(once("language.final_norm")))
    handles.append(model.language_model.register_forward_hook(language_forward_hook))

    blocks = model.vision_model.encoder.blocks
    for layer in vision_layers:
        if 0 <= layer < len(blocks):
            handles.append(blocks[layer].register_forward_hook(once(f"vision.block_{layer:02d}")))

    layers = model.language_model.model.layers
    for layer in language_layers:
        if 0 <= layer < len(layers):
            handles.append(layers[layer].register_forward_hook(once(f"language.layer_{layer:02d}")))

    return hooks, handles


def load_standard(model_root: str, device: str, dtype: torch.dtype, attn: str):
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_root, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_root, trust_remote_code=True)
    config._attn_implementation = attn
    config._attn_implementation_internal = attn
    config.text_config._attn_implementation = attn
    config.text_config._attn_implementation_internal = attn
    model = AutoModel.from_pretrained(
        model_root,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    ).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    return tokenizer, processor, model, time.perf_counter() - t0


def apply_processor_overrides(processor: Any, in_token_limit: int | None) -> None:
    if in_token_limit is None:
        return
    if in_token_limit <= 0:
        raise ValueError("--in-token-limit must be greater than zero")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor has no image_processor to configure")
    image_processor.in_token_limit = int(in_token_limit)


def run_standard(args: argparse.Namespace) -> dict[str, Any]:
    dtype = dtype_from_arg(args.dtype)
    tokenizer, processor, model, load_s = load_standard(args.model_root, args.device, dtype, args.attn)
    apply_processor_overrides(processor, args.in_token_limit)
    hook_capture = None
    hook_handles = []
    if args.hook_npz or args.hook_safetensors:
        hook_capture, hook_handles = install_hooks(
            model,
            parse_layer_list(args.hook_vision_layers),
            parse_layer_list(args.hook_language_layers),
            args.hook_max_language_forwards,
        )
    image = Image.open(args.image).convert("RGB")

    results = []
    preprocess_capture = None
    for query in args.query:
        prompt = detect_prompt(query)
        messages = build_messages(image, prompt)
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        t0 = time.perf_counter()
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(args.device)
        pixel_values = inputs["pixel_values"].to(dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)
        if preprocess_capture is None:
            preprocess_capture = {
                "pixel_values": pixel_values.detach().float().cpu().numpy(),
                "input_ids": input_ids.detach().cpu().numpy(),
                "attention_mask": inputs["attention_mask"].detach().cpu().numpy(),
                "image_grid_hws": None
                if image_grid_hws is None
                else to_numpy(image_grid_hws),
            }
        if args.device == "cuda":
            torch.cuda.synchronize()
        preprocess_s = time.perf_counter() - t0

        top_k_for_generate = None if args.top_k <= 0 else args.top_k
        t1 = time.perf_counter()
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            generation_mode=args.generation_mode,
            temperature=args.temperature,
            do_sample=args.temperature > 0.0,
            top_p=args.top_p,
            top_k=top_k_for_generate,
            repetition_penalty=args.repetition_penalty,
            verbose=True,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t1
        answer = response[0] if isinstance(response, tuple) else response
        if hook_capture is not None and "generated_token_ids" not in hook_capture:
            generated_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            hook_capture["generated_token_ids"] = np.asarray(generated_ids, dtype=np.int64)
        results.append(
            {
                "query": query,
                "prompt": prompt,
                "answer": answer,
                "detections": parse_detections(answer, query),
                "timings_ms": {
                    "preprocess": preprocess_s * 1000.0,
                    "infer": infer_s * 1000.0,
                },
            }
        )

    for handle in hook_handles:
        handle.remove()

    if args.preprocess_npz and preprocess_capture is not None:
        Path(args.preprocess_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.preprocess_npz, **preprocess_capture)
    if args.preprocess_safetensors and preprocess_capture is not None:
        from safetensors.numpy import save_file

        Path(args.preprocess_safetensors).parent.mkdir(parents=True, exist_ok=True)
        safe_preprocess = {
            name: np.ascontiguousarray(value)
            for name, value in preprocess_capture.items()
            if value is not None and value.dtype.kind in ("b", "i", "u", "f")
        }
        save_file(safe_preprocess, args.preprocess_safetensors)
    if args.hook_npz and hook_capture is not None:
        Path(args.hook_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.hook_npz, **hook_capture)
    if args.hook_safetensors and hook_capture is not None:
        from safetensors.numpy import save_file

        Path(args.hook_safetensors).parent.mkdir(parents=True, exist_ok=True)
        safe_hooks = {
            name: np.ascontiguousarray(value)
            for name, value in hook_capture.items()
            if value.dtype.kind in ("b", "i", "u", "f")
        }
        save_file(safe_hooks, args.hook_safetensors)

    return {
        "model_root": args.model_root,
        "image": args.image,
        "device": args.device,
        "dtype": args.dtype,
        "generation_mode": args.generation_mode,
        "attn": args.attn,
        "in_token_limit": args.in_token_limit,
        "load_ms": load_s * 1000.0,
        "results": results,
    }


def run_batch_runtime(args: argparse.Namespace) -> dict[str, Any]:
    dtype = dtype_from_arg(args.dtype)
    os.environ["LA_FLASH_MODEL"] = args.model_root
    os.environ["LA_FLASH_ATTN"] = "la_flash"
    os.environ["LA_FLASH_VISION_ATTN"] = "auto"
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"
    os.environ["LA_FLASH_HYBRID_GROUP_SIZE"] = "0"
    sys.path.insert(0, args.model_root)
    from batch_utils import generate_batch_hybrid, get_last_hybrid_stats, load

    t0 = time.perf_counter()
    tokenizer, processor, model = load()
    del tokenizer, processor, model
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    image = Image.open(args.image).convert("RGB")
    prompts = [(image, detect_prompt(query)) for query in args.query]
    t1 = time.perf_counter()
    answers = generate_batch_hybrid(
        prompts,
        temperature=args.temperature,
        top_p=None if args.top_p < 0 else args.top_p,
        top_k=None if args.top_k <= 0 else args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        scheduler="pipeline",
        group_size=0,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    infer_s = time.perf_counter() - t1
    stats = get_last_hybrid_stats()
    return {
        "model_root": args.model_root,
        "image": args.image,
        "device": args.device,
        "dtype": str(dtype),
        "generation_mode": "hybrid_batch_runtime",
        "load_ms": load_s * 1000.0,
        "batch_infer_ms": infer_s * 1000.0,
        "batch_stats": stats,
        "results": [
            {
                "query": query,
                "prompt": detect_prompt(query),
                "answer": answer,
                "detections": parse_detections(answer, query),
            }
            for query, answer in zip(args.query, answers)
        ],
    }


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = run_batch_runtime(args) if args.batch_runtime else run_standard(args)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
