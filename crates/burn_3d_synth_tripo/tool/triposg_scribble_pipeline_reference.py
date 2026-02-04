import argparse
import importlib.metadata as importlib_metadata
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import BitImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, Dinov2Model
from PIL import Image
import numpy as np


_real_version = importlib_metadata.version


def _safe_version(pkg: str) -> str:
    """Patch importlib.metadata.version to handle broken dist-info (e.g., filelock)."""
    try:
        version = _real_version(pkg)
    except Exception as exc:  # pragma: no cover - defensive for metadata edge cases
        try:
            module = __import__(pkg)
        except Exception:
            raise exc
        version = getattr(module, "__version__", None)
        if version is None:
            raise exc
        return version
    if version is None:
        try:
            module = __import__(pkg)
        except Exception as exc:
            raise exc
        version = getattr(module, "__version__", None)
        if version is None:
            raise ValueError(f"Unable to determine version for {pkg}")
    return version


importlib_metadata.version = _safe_version

PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_weights_root(weights: Optional[str]) -> str:
    if weights:
        if os.path.isdir(weights):
            return weights
        if os.path.isfile(weights):
            return os.path.dirname(os.path.dirname(weights))
    env_root = os.environ.get("TRIPOSG_SCRIBBLE_WEIGHTS_ROOT")
    if env_root and os.path.exists(env_root):
        return env_root
    fallback = "E:\\repos\\TripoSG\\pretrained_weights\\TripoSG-scribble"
    if os.path.exists(fallback):
        return fallback
    return str(PROJECT_DIR / "assets/models/TripoSG-scribble")


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_image_processor(weights_root: str) -> BitImageProcessor:
    path = os.path.join(weights_root, "feature_extractor_dinov2")
    return BitImageProcessor.from_pretrained(path, local_files_only=True)


def load_image_size(weights_root: str) -> int:
    config_path = os.path.join(weights_root, "image_encoder_dinov2", "config.json")
    if not os.path.exists(config_path):
        return 518
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return int(config.get("image_size", 518))


def generate_inputs(path: str, batch: int, height: int, width: int, num_tokens: int, seed: int) -> None:
    torch.manual_seed(seed)
    image = torch.rand(batch, 3, height, width, dtype=torch.float32) * 255.0
    latents = torch.randn(batch, num_tokens, 64, dtype=torch.float32)
    save_file({"input.image": image, "input.latents": latents}, path)


def generate_grid_coords(bounds: List[float], resolution: int) -> torch.Tensor:
    xs = torch.linspace(bounds[0], bounds[3], resolution)
    ys = torch.linspace(bounds[1], bounds[4], resolution)
    zs = torch.linspace(bounds[2], bounds[5], resolution)
    coords = []
    for z in zs:
        for y in ys:
            for x in xs:
                coords.append(torch.stack([x, y, z]))
    return torch.stack(coords, dim=0)


def load_tokenizer(weights_root: str) -> CLIPTokenizer:
    tokenizer_path = os.path.join(weights_root, "tokenizer")
    return CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def load_text_encoder(weights_root: str) -> CLIPTextModelWithProjection:
    text_encoder_path = os.path.join(weights_root, "text_encoder")
    return CLIPTextModelWithProjection.from_pretrained(text_encoder_path, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TripoSG scribble pipeline reference tensors")
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--inputs",
        default=str(PROJECT_DIR / "assets/hooks/triposg_scribble_pipeline_input.safetensors"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "assets/hooks/triposg_scribble_pipeline_reference.safetensors"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--bounds", type=float, nargs=6, default=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--prompt", default="a cat with wings")
    parser.add_argument("--regen-inputs", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--rmbg-root", default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.inputs), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    weights_root = resolve_weights_root(args.weights)
    project_root = str(Path(weights_root).parents[1])
    sys.path.insert(0, project_root)

    image_size = load_image_size(weights_root)
    height = args.height or image_size
    width = args.width or image_size

    if args.regen_inputs or not os.path.exists(args.inputs):
        generate_inputs(args.inputs, args.batch, height, width, args.num_tokens, args.seed)

    inputs = load_file(args.inputs)
    image = inputs["input.image"].float()
    latents = inputs["input.latents"].float()

    image = None
    if args.image:
        rmbg_root = args.rmbg_root or os.environ.get("RMBG_WEIGHTS_ROOT") or "E:\\repos\\TripoSG\\pretrained_weights\\RMBG-1.4"
        scripts_root = str(Path(weights_root).parents[1] / "scripts")
        sys.path.insert(0, scripts_root)
        from image_process import prepare_image
        from briarmbg import BriaRMBG

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rmbg_net = BriaRMBG.from_pretrained(rmbg_root).to(device)
        rmbg_net.eval()

        img_pil = prepare_image(args.image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        image = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).unsqueeze(0).float()

    if image is None:
        image = inputs["input.image"].float()
    image = image.contiguous()
    latents = latents.contiguous()

    image_processor = load_image_processor(weights_root)
    image_np = image.permute(0, 2, 3, 1).byte().cpu().numpy()
    pil_images = [Image.fromarray(frame) for frame in image_np]
    processed = image_processor(pil_images, return_tensors="pt").pixel_values

    image_encoder = Dinov2Model.from_pretrained(
        os.path.join(weights_root, "image_encoder_dinov2"), local_files_only=True
    )
    image_encoder.eval()

    tokenizer = load_tokenizer(weights_root)
    text_encoder = load_text_encoder(weights_root)
    text_encoder.eval()

    from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler
    from triposg.models.transformers import TripoSGDiTModel
    from triposg.models.autoencoders import TripoSGVAEModel

    scheduler_config = load_json(os.path.join(weights_root, "scheduler", "scheduler_config.json"))
    scheduler = RectifiedFlowScheduler(
        num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
        shift=scheduler_config.get("shift", 1.0),
        use_dynamic_shifting=scheduler_config.get("use_dynamic_shifting", False),
    )
    scheduler.set_timesteps(args.num_steps)

    dit_config = load_json(os.path.join(weights_root, "transformer", "config.json"))
    transformer = TripoSGDiTModel(
        num_attention_heads=dit_config.get("num_attention_heads", 16),
        width=dit_config.get("width", 2048),
        in_channels=dit_config.get("in_channels", 64),
        num_layers=dit_config.get("num_layers", 21),
        cross_attention_dim=dit_config.get("cross_attention_dim", 1024),
        use_cross_attention_2=dit_config.get("cross_attention_2_dim") is not None,
        cross_attention_2_dim=dit_config.get("cross_attention_2_dim"),
    )
    dit_weights = load_file(os.path.join(weights_root, "transformer", "diffusion_pytorch_model.safetensors"))
    transformer.load_state_dict(dit_weights, strict=False)
    transformer.eval()

    vae_config = load_json(os.path.join(weights_root, "vae", "config.json"))
    vae = TripoSGVAEModel(
        in_channels=vae_config.get("in_channels", 3),
        latent_channels=vae_config.get("latent_channels", 64),
        num_attention_heads=vae_config.get("num_attention_heads", 8),
        width_encoder=vae_config.get("width_encoder", 512),
        width_decoder=vae_config.get("width_decoder", 1024),
        num_layers_encoder=vae_config.get("num_layers_encoder", 8),
        num_layers_decoder=vae_config.get("num_layers_decoder", 16),
        embedding_type=vae_config.get("embedding_type", "frequency"),
        embed_frequency=vae_config.get("embed_frequency", 8),
        embed_include_pi=vae_config.get("embed_include_pi", False),
    )
    vae_weights = load_file(os.path.join(weights_root, "vae", "diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(vae_weights, strict=False)
    vae.eval()

    extras: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        text_input = tokenizer(
            [args.prompt],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_embeds = text_encoder(text_input["input_ids"]).last_hidden_state
        extras["input.text_embeds"] = text_embeds.clone()

        image_embeds = image_encoder(processed).last_hidden_state
        extras["input.image_embeds"] = image_embeds.clone()

        do_guidance = args.guidance_scale > 1.0
        if do_guidance:
            text_embeds = torch.cat([torch.zeros_like(text_embeds), text_embeds], dim=0)
            image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds], dim=0)
        extras["input.text_embeds.guided"] = text_embeds.clone()
        extras["input.image_embeds.guided"] = image_embeds.clone()

        latents_out = latents.clone()
        for step_idx, t in enumerate(scheduler.timesteps):
            latent_model_input = latents_out
            if do_guidance:
                latent_model_input = torch.cat([latents_out, latents_out], dim=0)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = transformer(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeds,
                encoder_hidden_states_2=image_embeds,
                return_dict=False,
            )[0]

            if do_guidance:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)

            extras[f"output.noise_pred.step{step_idx}"] = noise_pred.clone()
            latents_out = scheduler.step(noise_pred, t, latents_out, return_dict=False)[0]
            extras[f"output.latents.step{step_idx}"] = latents_out.clone()

        coords = generate_grid_coords(args.bounds, args.resolution)
        grid_values = []
        for start in range(0, coords.shape[0], args.chunk_size):
            chunk = coords[start : start + args.chunk_size]
            chunk = chunk.unsqueeze(0)
            decoded = vae.decode(latents_out, sampled_points=chunk).sample
            grid_values.append(decoded.squeeze(0).squeeze(-1))
        grid_logits = torch.cat(grid_values, dim=0)

    payload = {
        "input.image": image,
        "input.latents": latents,
        "output.latents": latents_out.clone(),
        "output.grid_logits": grid_logits,
        "meta.num_steps": torch.tensor([args.num_steps], dtype=torch.float32),
        "meta.num_tokens": torch.tensor([args.num_tokens], dtype=torch.float32),
        "meta.guidance_scale": torch.tensor([args.guidance_scale], dtype=torch.float32),
        "meta.resolution": torch.tensor([args.resolution], dtype=torch.float32),
        "meta.chunk_size": torch.tensor([args.chunk_size], dtype=torch.float32),
        "meta.bounds": torch.tensor(args.bounds, dtype=torch.float32),
    }
    payload.update(extras)

    save_file(payload, args.output)
    print(f"Saved reference to {args.output}")


if __name__ == "__main__":
    main()
