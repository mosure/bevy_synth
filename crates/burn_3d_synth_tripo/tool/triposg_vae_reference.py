import argparse
import os
import re
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


class HookRecorder:
    def __init__(self) -> None:
        self.tensors: Dict[str, torch.Tensor] = {}

PROJECT_DIR = Path(__file__).resolve().parents[1]

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.tensors:
            raise ValueError(f"hook tensor `{name}` recorded more than once")
        self.tensors[name] = tensor.detach().cpu().contiguous().float().clone()


def record_tensor(hook: HookRecorder, name: str, tensor: torch.Tensor) -> None:
    hook.record(name, tensor)


class FrequencyPositionalEmbedding(torch.nn.Module):
    def __init__(self, num_freq: int, include_pi: bool) -> None:
        super().__init__()
        self.num_freq = num_freq
        self.include_pi = include_pi

    def embed_dim(self, input_dim: int) -> int:
        return input_dim + input_dim * self.num_freq * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scale_pi = torch.pi if self.include_pi else 1.0
        freqs = torch.tensor(
            [scale_pi * (2.0 ** freq) for freq in range(self.num_freq)],
            dtype=coords.dtype,
            device=coords.device,
        )
        embed = (coords[..., None] * freqs).reshape(*coords.shape[:-1], -1)
        return torch.cat((coords, embed.sin(), embed.cos()), dim=-1)


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2.0).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.epsilon)
        return x * self.weight


class CrossAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        use_norm_cross: bool,
        qk_norm: bool,
        use_triposg_split: bool,
        is_cross_attention: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)
        self.norm_cross = (
            torch.nn.LayerNorm(context_dim) if use_norm_cross else None
        )
        self.norm_q = RmsNorm(self.head_dim) if qk_norm else None
        self.norm_k = RmsNorm(self.head_dim) if qk_norm else None
        self.use_triposg_split = use_triposg_split
        self.is_cross_attention = is_cross_attention

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        hook: HookRecorder,
        hook_prefix: str,
    ) -> torch.Tensor:
        b, n, c = x.shape
        m = context.shape[1]

        if self.norm_cross is not None:
            context = self.norm_cross(context)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        record_tensor(hook, f"{hook_prefix}.q", q)
        record_tensor(hook, f"{hook_prefix}.k", k)
        record_tensor(hook, f"{hook_prefix}.v", v)

        if self.use_triposg_split:
            if self.is_cross_attention or m != n:
                kv = torch.cat((k, v), dim=-1)
                split_size = kv.shape[-1] // self.num_heads // 2
                kv = kv.view(b, m, self.num_heads, split_size * 2)
                k, v = torch.split(kv, split_size, dim=-1)
            else:
                qkv = torch.cat((q, k, v), dim=-1)
                split_size = qkv.shape[-1] // self.num_heads // 3
                qkv = qkv.view(b, n, self.num_heads, split_size * 3)
                q, k, v = torch.split(qkv, split_size, dim=-1)

            q = q.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(b, m, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(b, m, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            q = q.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(b, m, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(b, m, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        record_tensor(hook, f"{hook_prefix}.attn", attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.to_out(out)
        record_tensor(hook, f"{hook_prefix}.out", out)
        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, hidden_dim, bias=True)
        self.out = torch.nn.Linear(hidden_dim, dim, bias=True)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.0)

    def forward(
        self, x: torch.Tensor, hook: HookRecorder, hook_prefix: str
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.dropout(x)
        record_tensor(hook, hook_prefix, x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, width: int, heads: int, use_cross: bool) -> None:
        super().__init__()
        self.use_cross = use_cross
        self.norm1 = torch.nn.LayerNorm(width)
        self.attn1 = CrossAttention(width, width, heads, False, False, True, False)
        self.norm2 = torch.nn.LayerNorm(width)
        self.attn2 = CrossAttention(width, width, heads, True, False, True, True)
        self.norm3 = torch.nn.LayerNorm(width)
        self.ff = FeedForward(width, width * 4)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        hook: HookRecorder,
        idx: int,
    ) -> torch.Tensor:
        prefix = f"encoder.blocks.{idx}"
        if self.use_cross:
            x_norm = self.norm2(x)
            record_tensor(hook, f"{prefix}.norm2", x_norm)
            attn = self.attn2(x_norm, context, hook, f"{prefix}.attn2")
        else:
            x_norm = self.norm1(x)
            record_tensor(hook, f"{prefix}.norm1", x_norm)
            attn = self.attn1(x_norm, x_norm, hook, f"{prefix}.attn1")
        x = x + attn
        record_tensor(hook, f"{prefix}.attn_out", x)
        x_norm = self.norm3(x)
        record_tensor(hook, f"{prefix}.norm3", x_norm)
        ff = self.ff(x_norm, hook, f"{prefix}.ff")
        x = x + ff
        record_tensor(hook, f"{prefix}.out", x)
        return x


class TripoSGEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.proj_in = torch.nn.Linear(in_dim, width, bias=True)
        blocks = [EncoderBlock(width, heads, True)]
        blocks += [EncoderBlock(width, heads, False) for _ in range(layers)]
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm_out = torch.nn.LayerNorm(width)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hook: HookRecorder,
    ) -> torch.Tensor:
        hidden = self.proj_in(x_q)
        kv = self.proj_in(x_kv)
        record_tensor(hook, "encoder.proj_in", hidden)
        record_tensor(hook, "encoder.proj_in.kv", kv)

        for idx, block in enumerate(self.blocks):
            context = kv if idx == 0 else hidden
            hidden = block(hidden, context, hook, idx)

        hidden = self.norm_out(hidden)
        record_tensor(hook, "encoder.norm_out", hidden)
        return hidden


class DecoderBlock(torch.nn.Module):
    def __init__(self, width: int, heads: int, use_cross: bool) -> None:
        super().__init__()
        self.use_cross = use_cross
        self.norm1 = torch.nn.LayerNorm(width)
        self.attn1 = CrossAttention(width, width, heads, False, False, True, False)
        self.norm2 = torch.nn.LayerNorm(width)
        self.attn2 = CrossAttention(width, width, heads, True, False, True, True)
        self.norm3 = torch.nn.LayerNorm(width)
        self.ff = FeedForward(width, width * 4)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        hook: HookRecorder,
        idx: int,
    ) -> torch.Tensor:
        prefix = f"decoder.blocks.{idx}"
        if self.use_cross:
            x_norm = self.norm2(x)
            record_tensor(hook, f"{prefix}.norm2", x_norm)
            attn = self.attn2(x_norm, context, hook, f"{prefix}.attn2")
        else:
            x_norm = self.norm1(x)
            record_tensor(hook, f"{prefix}.norm1", x_norm)
            attn = self.attn1(x_norm, x_norm, hook, f"{prefix}.attn1")
        x = x + attn
        record_tensor(hook, f"{prefix}.attn_out", x)
        x_norm = self.norm3(x)
        record_tensor(hook, f"{prefix}.norm3", x_norm)
        ff = self.ff(x_norm, hook, f"{prefix}.ff")
        x = x + ff
        record_tensor(hook, f"{prefix}.out", x)
        return x


class TripoSGDecoder(torch.nn.Module):
    def __init__(self, in_dim: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.proj_query = torch.nn.Linear(in_dim, width, bias=True)
        blocks = [DecoderBlock(width, heads, False) for _ in range(layers)]
        blocks.append(DecoderBlock(width, heads, True))
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm_out = torch.nn.LayerNorm(width)
        self.proj_out = torch.nn.Linear(width, 1, bias=True)

    def forward(
        self,
        sample: torch.Tensor,
        queries: torch.Tensor,
        hook: HookRecorder,
        kv_cache: torch.Tensor = None,
    ):
        if kv_cache is None:
            hidden = sample
            for idx, block in enumerate(self.blocks[:-1]):
                hidden = block(hidden, hidden, hook, idx)
            kv_cache = hidden
            record_tensor(hook, "decoder.kv_cache", kv_cache)

        cross_idx = len(self.blocks) - 1
        hidden = self.blocks[cross_idx](queries, kv_cache, hook, cross_idx)
        hidden = self.norm_out(hidden)
        record_tensor(hook, "decoder.norm_out", hidden)
        hidden = self.proj_out(hidden)
        record_tensor(hook, "decoder.proj_out", hidden)
        output = hidden * -1.0
        record_tensor(hook, "decoder.output", output)
        return output, kv_cache


class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean = mean
        self.logvar = logvar

    def sample(self) -> torch.Tensor:
        std = torch.exp(self.logvar * 0.5)
        noise = torch.randn_like(std)
        return self.mean + std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class TripoSGVae(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_attention_heads: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
        width_encoder: int,
        width_decoder: int,
        embed_frequency: int,
        embed_include_pi: bool,
    ) -> None:
        super().__init__()
        self.freq_embed = FrequencyPositionalEmbedding(
            num_freq=embed_frequency,
            include_pi=embed_include_pi,
        )
        embed_dim = self.freq_embed.embed_dim(3)
        encoder_in = embed_dim + in_channels
        decoder_in = embed_dim

        self.encoder = TripoSGEncoder(
            encoder_in, width_encoder, num_layers_encoder, num_attention_heads
        )
        self.decoder = TripoSGDecoder(
            decoder_in, width_decoder, num_layers_decoder, num_attention_heads
        )
        self.quant = torch.nn.Linear(width_encoder, latent_channels * 2, bias=True)
        self.post_quant = torch.nn.Linear(latent_channels, width_decoder, bias=True)
        self.latent_channels = latent_channels

    def encode(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        hook: HookRecorder,
    ):
        embedded = self.freq_embed(coords)
        x = torch.cat([embedded, features], dim=-1)
        record_tensor(hook, "encoder.input", x)
        record_tensor(hook, "encoder.input.kv", x)
        hidden = self.encoder(x, x, hook)
        record_tensor(hook, "encoder.hidden", hidden)
        stats = self.quant(hidden)
        record_tensor(hook, "encoder.quant", stats)
        mean, logvar = torch.split(stats, self.latent_channels, dim=-1)
        record_tensor(hook, "encoder.mean", mean)
        record_tensor(hook, "encoder.logvar", logvar)
        return mean, logvar

    def decode(
        self,
        query_coords: torch.Tensor,
        latents: torch.Tensor,
        hook: HookRecorder,
    ) -> torch.Tensor:
        query_embed = self.freq_embed(query_coords)
        query_tokens = self.decoder.proj_query(query_embed)
        record_tensor(hook, "decoder.query", query_tokens)
        latent_proj = self.post_quant(latents)
        record_tensor(hook, "decoder.post_quant", latent_proj)
        output, _kv_cache = self.decoder(latent_proj, query_tokens, hook)
        return output

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        query_coords: torch.Tensor,
        use_mean: bool,
        hook: HookRecorder,
    ):
        mean, logvar = self.encode(coords, features, hook)
        dist = DiagonalGaussianDistribution(mean, logvar)
        latent = dist.mode() if use_mean else dist.sample()
        record_tensor(hook, "latent.sample", latent)
        decoded = self.decode(query_coords, latent, hook)
        return decoded


def remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"\.to_out\.0\.", ".to_out.", key)
        new_key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.proj.", new_key)
        new_key = re.sub(r"\.ff\.net\.2\.", ".ff.out.", new_key)
        remapped[new_key] = value
    return remapped


def generate_inputs(
    path: str, batch: int, num_points: int, num_queries: int, seed: int
):
    torch.manual_seed(seed)
    coords = torch.randn(batch, num_points, 3, dtype=torch.float32)
    features = torch.randn(batch, num_points, 3, dtype=torch.float32)
    query_coords = torch.randn(batch, num_queries, 3, dtype=torch.float32)
    save_file(
        {
            "input.coords": coords,
            "input.features": features,
            "input.query_coords": query_coords,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TripoSG VAE hook reference tensors"
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--inputs",
        default=str(PROJECT_DIR / "assets/hooks/triposg_vae_input.safetensors"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "assets/hooks/triposg_vae_reference.safetensors"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--use-mean", action="store_true")
    parser.add_argument("--regen-inputs", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.inputs), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.regen_inputs or not os.path.exists(args.inputs):
        generate_inputs(
            args.inputs, args.batch, args.num_points, args.num_queries, args.seed
        )

    weights_path = args.weights
    if weights_path is None:
        weights_root = os.environ.get("TRIPOSG_WEIGHTS_ROOT")
        if weights_root:
            candidate = os.path.join(
                weights_root, "vae", "diffusion_pytorch_model.safetensors"
            )
            if os.path.exists(candidate):
                weights_path = candidate
        if weights_path is None:
            candidate = os.path.join(
                "E:\\repos\\TripoSG\\pretrained_weights\\TripoSG",
                "vae",
                "diffusion_pytorch_model.safetensors",
            )
            if os.path.exists(candidate):
                weights_path = candidate
        if weights_path is None:
            weights_path = str(
                PROJECT_DIR / "assets/models/MIDI-3D/vae/diffusion_pytorch_model.safetensors"
            )

    weights = load_file(weights_path)
    state = remap_state_dict(weights)

    model = TripoSGVae(
        in_channels=3,
        latent_channels=64,
        num_attention_heads=8,
        num_layers_encoder=8,
        num_layers_decoder=16,
        width_encoder=512,
        width_decoder=1024,
        embed_frequency=8,
        embed_include_pi=False,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    with torch.no_grad():
        inputs = load_file(args.inputs)
        coords = inputs["input.coords"]
        features = inputs["input.features"]
        query_coords = inputs["input.query_coords"]

        hooks = HookRecorder()
        hooks.record("input.coords", coords)
        hooks.record("input.features", features)
        hooks.record("input.query_coords", query_coords)

        _out = model(
            coords,
            features,
            query_coords,
            args.use_mean,
            hooks,
        )

    save_file(hooks.tensors, args.output)
    print(f"Saved reference hooks to {args.output}")


if __name__ == "__main__":
    main()
