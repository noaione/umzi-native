"""Checkpoint inspection and model construction for GateRV3.

The checkpoints shipped in ``models/`` are raw ``state_dict`` objects, so the
architecture has to be rebuilt before the weights can be loaded. Instead of
hardcoding hyper-parameters, ``ModelConfig.from_state_dict`` derives them from
the tensor shapes / key names, which keeps the loader working across model
variants (1x / 2x / 4x, with or without attention in the latent stage).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .gaterv3 import GateRV3, SampleMods


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
    upsample_mid_dim: int = 48
    groups: int = 4

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Tensor]) -> ModelConfig:
        in_to_dim = state_dict.get("in_to_dim.weight")
        if in_to_dim is None or in_to_dim.ndim != 4:
            msg = "Not a GateRV3 checkpoint: missing 4D 'in_to_dim.weight'."
            raise ValueError(msg)

        dim = in_to_dim.shape[0]
        in_ch = in_to_dim.shape[1]

        enc_levels = _child_count(state_dict, "gater_encode")
        enc_blocks = tuple(
            _child_count(state_dict, f"gater_encode.{i}.gated")
            for i in range(enc_levels)
        )
        dec_levels = _child_count(state_dict, "decode")
        dec_blocks = tuple(
            _child_count(state_dict, f"decode.{i}.gated") for i in range(dec_levels)
        )

        num_latent = _child_count(state_dict, "latent")
        sisr_blocks = _child_count(state_dict, "span_n_b")
        attention = "latent.0.token_mix.qkv.weight" in state_dict

        flash = True
        flash_buffer = state_dict.get("latent.0.token_mix.flash")
        if flash_buffer is not None:
            flash = bool(flash_buffer.item())

        scale = 1
        upsample = "pixelshuffle"
        upsample_mid_dim = 48
        groups = 4
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
            upsample_mid_dim = values[5]
            groups = values[6]

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
            upsample_mid_dim=upsample_mid_dim,
            groups=groups,
        )

    def build(self) -> GateRV3:
        """Instantiate a freshly initialised model with this configuration."""
        return GateRV3(
            in_ch=self.in_ch,
            dim=self.dim,
            enc_blocks=self.enc_blocks,
            dec_blocks=self.dec_blocks,
            num_latent=self.num_latent,
            scale=self.scale,
            upsample=self.upsample,  # type: ignore[arg-type]
            upsample_mid_dim=self.upsample_mid_dim,
            attention=self.attention,
            sisr_blocks=self.sisr_blocks,
            flash=self.flash,
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


def read_state_dict(path: str | Path) -> dict[str, Tensor]:
    """Load a checkpoint file and normalise common wrapper formats."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        msg = f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        raise TypeError(msg)

    for key in ("state_dict", "model", "params", "net"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            checkpoint = inner
            break

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }
    if not state_dict:
        msg = f"No tensors found in checkpoint {path}"
        raise ValueError(msg)
    return state_dict


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> tuple[GateRV3, ModelConfig]:
    """Rebuild ``GateRV3`` from ``path`` and load its weights.

    ``model.eval()`` is called on CPU *before* moving the module, because the
    fused ``Conv3XC.eval_conv`` weights are computed lazily on the first call
    to ``train(False)``.

    ``dtype`` casts the (already fused) weights, e.g. ``torch.float16`` for the
    fast path on MPS/CUDA.
    """
    state_dict = read_state_dict(path)
    config = ModelConfig.from_state_dict(state_dict)

    model = config.build()
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Some exports drop the non-trainable `flash` flag; it only selects the
    # attention kernel and defaults to the reference value.
    ignored = [key for key in incompatible.missing_keys if key.endswith(".flash")]
    missing = [key for key in incompatible.missing_keys if key not in ignored]
    if strict and (missing or incompatible.unexpected_keys):
        msg = (
            f"Checkpoint does not match the inferred architecture {config.describe()}\n"
            f"  missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected: {incompatible.unexpected_keys[:8]}"
        )
        raise RuntimeError(msg)

    # Fuses the re-parameterised Conv3XC layers; must happen before `.to(device)`.
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    resolved = resolve_device(device)
    if dtype is None:
        model.to(resolved)
    else:
        # Cast after fusion so the re-parameterisation stays in float32.
        model.to(device=resolved, dtype=dtype)
    return model, config


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
