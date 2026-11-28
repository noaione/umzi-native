"""Export checkpoint weights for the native Umzi runtime."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from .._loader import PathArg
from .._model import ModelInfo, peek_model
from ._loader import load_checkpoint

# Match the Resselt metadata name after case folding. Use module names to keep
# packers and their dependencies out of the import path until export is needed.
_PACKERS = {
    "gaterv3": "._gaterv3",
    "gfisrv2": "._gfisrv2",
    "gaterv2": "._gaterv2",
    "moesr": "._moesr",
    "mosr": "._mosr",
    "smosr": "._smosr",
    "figsr": "._figsr",
}


def export_model(checkpoint: PathArg, output: PathArg, *, kernel: str | None = None) -> ModelInfo:
    """Export a SafeTensors or PyTorch checkpoint to an Umzi model package.

    Export runs on CPU. Install the ``exporter`` extra for Resselt and its
    dependencies. ``kernel`` can assert the detected native family.
    """
    source = Path(checkpoint).resolve()
    destination = Path(output).absolute()
    if destination.suffix.lower() != ".ummodel":
        raise ValueError("The output filename must end in .ummodel")
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    with source.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").digest()
    model, state = load_checkpoint(source)
    architecture = getattr(getattr(model, "parameters_info", None), "name", type(model).__name__)
    family = architecture.casefold()
    if family not in _PACKERS:
        raise ValueError(
            f"No native packer for Resselt architecture {architecture!r}. Available: {', '.join(_PACKERS)}"
        )
    if kernel is not None and kernel.casefold() != family:
        raise ValueError(f"Requested native kernel {kernel!r}, but detected {family!r}")

    packer = import_module(_PACKERS[family], __package__)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        weights = Path(directory) / "weights.bin"
        package = Path(directory) / "model.ummodel"
        schedule, alignment = packer.build_schedule(model, state, weights)
        schedule.write_model(package, weights, digest, alignment)
        info = peek_model(package)
        if not info.supported:
            raise ValueError(f"Exported package failed validation: {'; '.join(info.problems)}")
        package.replace(destination)
    return replace(info, path=destination.resolve())
