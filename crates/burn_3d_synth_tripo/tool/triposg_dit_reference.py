import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

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
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_cross_attention = context_dim != dim
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)
        self.norm_q = RmsNorm(self.head_dim) if qk_norm else None
        self.norm_k = RmsNorm(self.head_dim) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        hook: HookRecorder,
        hook_prefix: str,
    ) -> torch.Tensor:
        b, n, c = x.shape
        m = context.shape[1]

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        record_tensor(hook, f"{hook_prefix}.q", q)
        record_tensor(hook, f"{hook_prefix}.k", k)
        record_tensor(hook, f"{hook_prefix}.v", v)

        if self.is_cross_attention:
            q = q.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = torch.cat((k, v), dim=-1)
            kv = kv.view(b, m, self.num_heads, self.head_dim * 2)
            k, v = torch.split(kv, self.head_dim, dim=-1)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        else:
            qkv = torch.cat((q, k, v), dim=-1)
            qkv = qkv.view(b, n, self.num_heads, self.head_dim * 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

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


class TimestepEmbedding(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        return self.linear_2(x)


def timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
) -> torch.Tensor:
    half = embedding_dim // 2
    exponent = torch.arange(half, device=timesteps.device, dtype=torch.float32)
    exponent = exponent.mul(-torch.log(torch.tensor(10000.0, device=timesteps.device)))
    exponent = exponent.div(half - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0) * scale
    sin = emb.sin()
    cos = emb.cos()
    out = torch.cat((cos, sin), dim=1) if flip_sin_to_cos else torch.cat((sin, cos), dim=1)
    if embedding_dim % 2 == 1:
        out = torch.cat((out, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)), dim=1)
    return out


class TripoSGDiTBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_attention_dim: int,
        cross_attention_2_dim: Optional[int],
        use_self_attention: bool,
        use_cross_attention: bool,
        use_cross_attention_2: bool,
        use_skip: bool,
        skip_concat_front: bool,
        skip_norm_last: bool,
    ) -> None:
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_cross_attention_2 = use_cross_attention_2
        self.use_skip = use_skip
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last

        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, num_heads, qk_norm=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, cross_attention_dim, num_heads, qk_norm=True)
        if use_cross_attention_2:
            if cross_attention_2_dim is None:
                raise ValueError("cross_attention_2_dim required when use_cross_attention_2 is True")
            self.norm2_2 = torch.nn.LayerNorm(dim)
            self.attn2_2 = CrossAttention(
                dim, cross_attention_2_dim, num_heads, qk_norm=True
            )
        else:
            self.norm2_2 = None
            self.attn2_2 = None
        self.norm3 = torch.nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * 4)

        self.skip_norm = torch.nn.LayerNorm(dim)
        self.skip_linear = torch.nn.Linear(dim * 2, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_2: Optional[torch.Tensor],
        skip: torch.Tensor,
        hook: HookRecorder,
        idx: int,
    ) -> torch.Tensor:
        prefix = f"dit.blocks.{idx}"
        hidden = hidden_states

        if self.use_skip:
            if skip is None:
                raise ValueError("skip tensor required for this block")
            cat = (
                torch.cat((skip, hidden), dim=-1)
                if self.skip_concat_front
                else torch.cat((hidden, skip), dim=-1)
            )
            if self.skip_norm_last:
                hidden = self.skip_norm(self.skip_linear(cat))
            else:
                hidden = self.skip_linear(self.skip_norm(cat))
            record_tensor(hook, f"{prefix}.skip", hidden)

        if self.use_self_attention:
            norm_hidden = self.norm1(hidden)
            record_tensor(hook, f"{prefix}.norm1", norm_hidden)
            attn = self.attn1(norm_hidden, norm_hidden, hook, f"{prefix}.attn1")
            hidden = hidden + attn
            record_tensor(hook, f"{prefix}.attn1_out", hidden)

        if self.use_cross_attention:
            if self.use_cross_attention_2:
                norm_hidden = self.norm2(hidden)
                record_tensor(hook, f"{prefix}.norm2", norm_hidden)
                attn2 = self.attn2(
                    norm_hidden,
                    encoder_hidden_states,
                    hook,
                    f"{prefix}.attn2",
                )
                if encoder_hidden_states_2 is None:
                    raise ValueError("encoder_hidden_states_2 required for cross_attention_2")
                norm_hidden = self.norm2_2(hidden)
                record_tensor(hook, f"{prefix}.norm2_2", norm_hidden)
                attn2_2 = self.attn2_2(
                    norm_hidden,
                    encoder_hidden_states_2,
                    hook,
                    f"{prefix}.attn2_2",
                )
                hidden = hidden + attn2 + attn2_2
                record_tensor(hook, f"{prefix}.attn2_out", hidden)
            else:
                norm_hidden = self.norm2(hidden)
                record_tensor(hook, f"{prefix}.norm2", norm_hidden)
                attn = self.attn2(
                    norm_hidden,
                    encoder_hidden_states,
                    hook,
                    f"{prefix}.attn2",
                )
                hidden = hidden + attn
                record_tensor(hook, f"{prefix}.attn2_out", hidden)

        norm_hidden = self.norm3(hidden)
        record_tensor(hook, f"{prefix}.norm3", norm_hidden)
        ff = self.ff(norm_hidden, hook, f"{prefix}.ff")
        hidden = hidden + ff
        record_tensor(hook, f"{prefix}.out", hidden)
        return hidden


class TripoSGDiT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        num_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        cross_attention_2_dim: Optional[int],
        use_cross_attention_2: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim = width
        time_embed_dim = width * 4
        self.time_proj = TimestepEmbedding(width, time_embed_dim, width)
        self.proj_in = torch.nn.Linear(in_channels, width, bias=True)
        blocks = []
        half = num_layers // 2
        for layer in range(num_layers):
            use_skip = layer > half
            blocks.append(
                TripoSGDiTBlock(
                    dim=width,
                    num_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_2_dim=cross_attention_2_dim,
                    use_self_attention=True,
                    use_cross_attention=True,
                    use_cross_attention_2=use_cross_attention_2,
                    use_skip=use_skip,
                    skip_concat_front=True,
                    skip_norm_last=True,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm_out = torch.nn.LayerNorm(width)
        self.proj_out = torch.nn.Linear(width, in_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_2: Optional[torch.Tensor],
        hook: HookRecorder,
    ) -> torch.Tensor:
        batch, n, _ = hidden_states.shape
        temb = timestep_embedding(timestep, self.inner_dim, False, 0.0, 1.0)
        record_tensor(hook, "dit.temb", temb)
        temb = self.time_proj(temb)
        record_tensor(hook, "dit.temb_proj", temb)
        temb = temb.unsqueeze(1)

        hidden = self.proj_in(hidden_states)
        record_tensor(hook, "dit.proj_in", hidden)
        hidden = torch.cat((temb, hidden), dim=1)
        record_tensor(hook, "dit.tokens", hidden)

        skips = []
        half = len(self.blocks) // 2
        for idx, block in enumerate(self.blocks):
            skip = skips.pop() if idx > half else None
            hidden = block(
                hidden,
                encoder_hidden_states,
                encoder_hidden_states_2,
                skip,
                hook,
                idx,
            )
            if idx < half:
                skips.append(hidden)

        hidden = self.norm_out(hidden)
        record_tensor(hook, "dit.norm_out", hidden)
        hidden = hidden[:, 1 : n + 1, :]
        hidden = self.proj_out(hidden)
        record_tensor(hook, "dit.proj_out", hidden)
        return hidden


def remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"\.to_out\.0\.", ".to_out.", key)
        new_key = re.sub(r"\.ff\.net\.0\.proj\.", ".ff.proj.", new_key)
        new_key = re.sub(r"\.ff\.net\.2\.", ".ff.out.", new_key)
        remapped[new_key] = value
    return remapped


def generate_inputs(
    path: str,
    batch: int,
    num_tokens: int,
    text_len: int,
    text2_len: int,
    seed: int,
    cross_attention_dim: int,
    cross_attention_2_dim: Optional[int],
):
    torch.manual_seed(seed)
    hidden_states = torch.randn(batch, num_tokens, 64, dtype=torch.float32)
    encoder_hidden_states = torch.randn(
        batch, text_len, cross_attention_dim, dtype=torch.float32
    )
    encoder_hidden_states_2_dim = (
        cross_attention_2_dim if cross_attention_2_dim is not None else cross_attention_dim
    )
    encoder_hidden_states_2 = torch.randn(
        batch, text2_len, encoder_hidden_states_2_dim, dtype=torch.float32
    )
    timestep = torch.randint(0, 1000, (batch,), dtype=torch.int64).to(torch.float32)
    save_file(
        {
            "input.hidden_states": hidden_states,
            "input.encoder_hidden_states": encoder_hidden_states,
            "input.encoder_hidden_states_2": encoder_hidden_states_2,
            "input.timestep": timestep,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TripoSG DiT hook reference tensors"
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--inputs",
        default=str(PROJECT_DIR / "assets/hooks/triposg_dit_input.safetensors"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "assets/hooks/triposg_dit_reference.safetensors"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--text-len", type=int, default=4)
    parser.add_argument("--text2-len", type=int, default=4)
    parser.add_argument("--regen-inputs", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.inputs), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    weights_path = args.weights
    if weights_path is None:
        weights_root = os.environ.get("TRIPOSG_WEIGHTS_ROOT")
        if weights_root:
            candidate = os.path.join(
                weights_root, "transformer", "diffusion_pytorch_model.safetensors"
            )
            if os.path.exists(candidate):
                weights_path = candidate
        if weights_path is None:
            candidate = os.path.join(
                "E:\\repos\\TripoSG\\pretrained_weights\\TripoSG",
                "transformer",
                "diffusion_pytorch_model.safetensors",
            )
            if os.path.exists(candidate):
                weights_path = candidate
        if weights_path is None:
            weights_path = str(
                PROJECT_DIR
                / "assets/models/MIDI-3D/transformer/diffusion_pytorch_model.safetensors"
            )

    weights = load_file(weights_path)
    state = remap_state_dict(weights)

    config_path = os.path.join(os.path.dirname(weights_path), "config.json")
    cross_attention_dim = 768
    cross_attention_2_dim = 1024
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        cross_attention_dim = int(config.get("cross_attention_dim", cross_attention_dim))
        cross_attention_2_dim = config.get("cross_attention_2_dim", cross_attention_2_dim)

    use_cross_attention_2 = any("attn2_2" in key for key in weights.keys())
    if not use_cross_attention_2:
        cross_attention_2_dim = None

    if args.regen_inputs or not os.path.exists(args.inputs):
        generate_inputs(
            args.inputs,
            args.batch,
            args.num_tokens,
            args.text_len,
            args.text2_len,
            args.seed,
            cross_attention_dim,
            cross_attention_2_dim,
        )

    model = TripoSGDiT(
        in_channels=64,
        width=2048,
        num_layers=21,
        num_attention_heads=16,
        cross_attention_dim=cross_attention_dim,
        cross_attention_2_dim=cross_attention_2_dim,
        use_cross_attention_2=use_cross_attention_2,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    with torch.no_grad():
        inputs = load_file(args.inputs)
        hidden_states = inputs["input.hidden_states"]
        encoder_hidden_states = inputs["input.encoder_hidden_states"]
        encoder_hidden_states_2 = inputs["input.encoder_hidden_states_2"]
        timestep = inputs["input.timestep"]

        hooks = HookRecorder()
        hooks.record("input.hidden_states", hidden_states)
        hooks.record("input.encoder_hidden_states", encoder_hidden_states)
        hooks.record("input.encoder_hidden_states_2", encoder_hidden_states_2)
        hooks.record("input.timestep", timestep)

        _out = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_2,
            hooks,
        )

    save_file(hooks.tensors, args.output)
    print(f"Saved reference hooks to {args.output}")


if __name__ == "__main__":
    main()
