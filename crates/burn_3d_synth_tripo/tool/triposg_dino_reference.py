import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import BitImageProcessor, Dinov2Model
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]


class HookRecorder:
    def __init__(self) -> None:
        self.tensors: Dict[str, torch.Tensor] = {}

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.tensors:
            raise ValueError(f"hook tensor `{name}` recorded more than once")
        self.tensors[name] = tensor.detach().cpu().contiguous().float().clone()


def resolve_weights_root(weights: Optional[str]) -> str:
    if weights:
        if os.path.isdir(weights):
            return weights
        if os.path.isfile(weights):
            parent = os.path.dirname(weights)
            if os.path.basename(parent) == "image_encoder_dinov2":
                return os.path.dirname(parent)
            return parent
    env_root = os.environ.get("TRIPOSG_WEIGHTS_ROOT")
    if env_root and os.path.exists(env_root):
        return env_root
    fallback = "E:\\repos\\TripoSG\\pretrained_weights\\TripoSG"
    if os.path.exists(fallback):
        return fallback
    return str(PROJECT_DIR / "assets/models/MIDI-3D")


def resolve_model_dir(weights_root: str, weights: Optional[str]) -> str:
    if weights:
        if os.path.isdir(weights):
            if os.path.exists(os.path.join(weights, "model.safetensors")):
                return weights
            candidate = os.path.join(weights, "image_encoder_dinov2")
            if os.path.exists(os.path.join(candidate, "model.safetensors")):
                return candidate
        if os.path.isfile(weights):
            return os.path.dirname(weights)
    return os.path.join(weights_root, "image_encoder_dinov2")


def load_image_size(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return 518
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return int(config.get("image_size", 518))


def load_image_processor(weights_root: str) -> BitImageProcessor:
    path = os.path.join(weights_root, "feature_extractor_dinov2")
    return BitImageProcessor.from_pretrained(path, local_files_only=True)


def generate_inputs(path: str, batch: int, image_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    image = torch.rand(batch, 3, image_size, image_size, dtype=torch.float32) * 255.0
    save_file({"input.image": image}, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TripoSG DINOv2 hook reference tensors"
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--inputs",
        default=str(PROJECT_DIR / "assets/hooks/triposg_dino_input.safetensors"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "assets/hooks/triposg_dino_reference.safetensors"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--regen-inputs", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.inputs), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    weights_root = resolve_weights_root(args.weights)
    model_dir = resolve_model_dir(weights_root, args.weights)

    image_size = args.image_size or load_image_size(model_dir)

    if args.regen_inputs or not os.path.exists(args.inputs):
        generate_inputs(args.inputs, args.batch, image_size, args.seed)

    inputs = load_file(args.inputs)
    image = inputs["input.image"].float()

    image_processor = load_image_processor(weights_root)
    image_np = image.permute(0, 2, 3, 1).byte().cpu().numpy()
    pil_images = [Image.fromarray(frame) for frame in image_np]
    processed = image_processor(pil_images, return_tensors="pt").pixel_values

    model = Dinov2Model.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    hooks = HookRecorder()
    hooks.record("input.image", image)
    hooks.record("image.preprocessed", processed)

    with torch.no_grad():
        outputs = model(processed)
        image_embeds = outputs.last_hidden_state
        hooks.record("output.image_embeds", image_embeds)
        hooks.record("output.cls_token", image_embeds[:, 0:1, :])
        hooks.record("output.patch_tokens", image_embeds[:, 1:, :])

    save_file(hooks.tensors, args.output)
    print(f"Saved reference hooks to {args.output}")


if __name__ == "__main__":
    main()
