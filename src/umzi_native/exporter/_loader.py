"""Read CPU state dictionaries and construct models through Resselt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from pickle import UnpicklingError  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

_ATTENTION_FLAG = re.compile(r"latent\.\d+\.token_mix\.flash")
_FORMATS = (".safetensors", ".pth", ".pt", ".ckpt")


def _read_checkpoint(path: Path) -> dict[str, Tensor]:
    import torch

    if path.suffix.lower() == ".safetensors":
        from safetensors import SafetensorError
        from safetensors.torch import load_file

        try:
            checkpoint = load_file(path, device="cpu")
        except SafetensorError as error:
            raise ValueError(f"Cannot read a SafeTensors checkpoint from {path}: {error}") from error
    else:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, UnpicklingError) as error:
            raise ValueError(f"Cannot read a PyTorch tensor state dictionary from {path}: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a tensor state dictionary in {path}")
    for key in ("state_dict", "model", "params", "net", "params_ema", "params-ema"):
        inner = checkpoint.get(key)
        if isinstance(inner, Mapping):
            checkpoint = inner
            break

    state = {}
    for key, value in checkpoint.items():
        if isinstance(key, str) and isinstance(value, torch.Tensor):
            name = key.removeprefix("module.").removeprefix("netG.")
            if name in state:
                raise ValueError(f"Duplicate checkpoint tensor {name!r} in {path}")
            state[name] = value
    if not state:
        raise ValueError(f"No tensors found in checkpoint {path}")
    return state


def attention_flags(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value for key, value in state.items() if _ATTENTION_FLAG.fullmatch(key)}


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Tensor]]:
    if path.suffix.lower() not in _FORMATS:
        raise ValueError(f"Unsupported checkpoint extension {path.suffix!r}. Expected {', '.join(_FORMATS)}")
    try:
        import resselt
    except ModuleNotFoundError as error:
        if error.name != "resselt":
            raise
        raise ImportError(
            'Model export requires Resselt. Install it with pip install "umzi-native[exporter]".'
        ) from error
    from resselt.registry import ArchitectureNotFound

    state = _read_checkpoint(path)
    flags = attention_flags(state)
    # Resselt does not register the old GateRV3 attention buffers. Keep them
    # for validation while giving Resselt only the tensors it can load.
    weights = {key: value for key, value in state.items() if key not in flags} if flags else state.copy()
    try:
        model = resselt.load_from_state_dict(weights)
    except ArchitectureNotFound as error:
        raise ValueError(f"Resselt did not recognize the checkpoint architecture in {path}") from error
    return model, state
