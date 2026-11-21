"""Resselt-backed GateRV3 reference for the benchmark scripts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from resselt.archs.gaterv3.arch import Attention, GateRV3, SampleMods
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from umzi_native.exporter._loader import load_checkpoint


def _child_count(state_dict: Mapping[str, Tensor], prefix: str) -> int:
    """Return ``max(index) + 1`` for keys of the form ``prefix.<int>.*``."""
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(m.group(1)) for key in state_dict if (m := pattern.match(key))}
    return max(indices) + 1 if indices else 0


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters needed to instantiate :class:`GateRV3`."""

    in_ch: int
    dim: int
    enc_blocks: tuple[int, ...]
    dec_blocks: tuple[int, ...]
    num_latent: int
    sisr_blocks: int
    scale: int = 1
    attention: bool = False
    flash: bool = True
    upsample: str = "pixelshuffle"

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Tensor]) -> ModelConfig:
        in_to_dim = state_dict.get("in_to_dim.weight")
        if in_to_dim is None or in_to_dim.ndim != 4:
            msg = "Not a GateRV3 checkpoint: missing 4D 'in_to_dim.weight'."
            raise ValueError(msg)

        dim = in_to_dim.shape[0]
        in_ch = in_to_dim.shape[1]

        enc_levels = _child_count(state_dict, "gater_encode")
        enc_blocks = tuple(_child_count(state_dict, f"gater_encode.{i}.gated") for i in range(enc_levels))
        dec_levels = _child_count(state_dict, "decode")
        dec_blocks = tuple(_child_count(state_dict, f"decode.{i}.gated") for i in range(dec_levels))

        num_latent = _child_count(state_dict, "latent")
        sisr_blocks = _child_count(state_dict, "span_n_b")
        attention = "latent.0.token_mix.qkv.weight" in state_dict

        flash = True
        flash_buffer = state_dict.get("latent.0.token_mix.flash")
        if flash_buffer is not None:
            flash = bool(flash_buffer.item())

        scale = 1
        upsample = "pixelshuffle"
        if "dim_to_in.weight" not in state_dict:
            # Non-1x models use `UniUpsample`, which stores its own metadata.
            meta = state_dict.get("dim_to_in.MetaUpsample")
            if meta is None:
                msg = (
                    "Cannot infer the upsampling stage: neither 'dim_to_in.weight' "
                    "nor 'dim_to_in.MetaUpsample' is present in the checkpoint."
                )
                raise ValueError(msg)
            values = [int(v) for v in meta.flatten().tolist()]
            upsample = SampleMods.__args__[values[1]]  # type: ignore[attr-defined]
            scale = values[2]

        if enc_levels == 0 or dec_levels == 0 or num_latent == 0:
            msg = "Malformed GateRV3 checkpoint: encoder/decoder/latent stage is empty."
            raise ValueError(msg)

        return cls(
            in_ch=in_ch,
            dim=dim,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_latent=num_latent,
            sisr_blocks=sisr_blocks,
            scale=scale,
            attention=attention,
            flash=flash,
            upsample=upsample,
        )

    def describe(self) -> str:
        levels = len(self.enc_blocks)
        return (
            f"in_ch={self.in_ch} dim={self.dim} scale={self.scale}x "
            f"enc_blocks={self.enc_blocks} dec_blocks={self.dec_blocks} "
            f"latent={self.num_latent} sisr_blocks={self.sisr_blocks} "
            f"attention={'on' if self.attention else 'off'} "
            f"upsample={self.upsample} latent_dim={self.dim * 2**levels}"
        )


def _spatial_attention(layer: Attention, x: Tensor) -> Tensor:
    batch, channels, height, width = x.shape
    qkv = layer.qkv_dwconv(layer.qkv(x))
    q, k, v = (
        part.view(batch, layer.num_heads, layer.head_dim, height * width).transpose(2, 3)
        for part in qkv.chunk(3, dim=1)
    )
    result = scaled_dot_product_attention(q, k, v, is_causal=False)
    return layer.project_out(result.transpose(2, 3).contiguous().view(batch, channels, height, width))


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
) -> tuple[GateRV3, ModelConfig]:
    """Load the benchmark reference and preserve the original spatial attention."""
    model, state = load_checkpoint(Path(path))
    if not isinstance(model, GateRV3):
        raise ValueError(f"The GateRV3 benchmark requires a GateRV3 checkpoint: {path}")
    config = ModelConfig.from_state_dict(state)
    if config.attention and config.flash:
        for layer in model.modules():
            if isinstance(layer, Attention):
                layer.forward = MethodType(_spatial_attention, layer)

    # Conv3XC folds in CPU FP32 before moving to the requested device or dtype.
    model.eval().requires_grad_(False)
    return model.to(device=resolve_device(device), dtype=dtype), config


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``"auto"`` to cuda -> mps -> cpu."""
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
