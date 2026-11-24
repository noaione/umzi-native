"""Pack the direct-shuffle FiGSR checkpoint for the native CUDA session."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
from resselt.archs.figsr.arch import FIGSR, FourierUnit, GatedCNNBlock, InceptionConv2d, RMSNorm
from torch import Tensor, nn
from torch.nn import functional

from .._model import (
    FIGSR_ARCHITECTURE,
    FIGSR_BLOCK_FILTERS,
    FIGSR_CONFIG,
    FIGSR_FAMILY,
    FIGSR_HEAD,
    FIGSR_TAIL,
    SCHEMAS,
    figsr_parameter_sizes,
)
from ._package import write_package


class Package:
    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: FIGSR_ARCHITECTURE.pack(*FIGSR_CONFIG), 3: weights, 4: checkpoint_sha256}
        write_package(path, FIGSR_FAMILY, SCHEMAS["figsr"], sections)


class Packer:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.sizes: list[int] = []

    def add(self, tensor: Tensor) -> None:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite FiGSR parameter after conversion")
        array = value.numpy()
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        self.file.write(bytes(-self.file.tell() % 256))
        self.file.write(data)
        self.sizes.append(len(data))

    def norm(self, norm: nn.Module, channels: int) -> None:
        if (
            not isinstance(norm, RMSNorm)
            or norm.scale.shape != (channels,)
            or norm.offset.shape != (channels,)
            or norm.eps.shape != (1,)
            or norm.rms.shape != (1,)
            or not bool((norm.eps > 0).all() and (norm.rms > 0).all())
        ):
            raise ValueError("Unsupported FiGSR RMS normalization")
        self.add(torch.cat((norm.scale, norm.offset, norm.eps, norm.rms)))

    def convolution(
        self,
        layer: nn.Module,
        packed: tuple[int, int, int, int, int],
        *,
        input_channels: int | None = None,
        output_channels: int | None = None,
    ) -> None:
        cin, cout, kh, kw, groups = packed
        if (
            not isinstance(layer, nn.Conv2d)
            or layer.in_channels != (input_channels if input_channels is not None else cin)
            or layer.out_channels != (output_channels if output_channels is not None else cout)
            or layer.groups != groups
            or layer.kernel_size != (kh, kw)
            or layer.padding != (kh // 2, kw // 2)
            or layer.padding_mode != "zeros"
            or layer.stride != (1, 1)
            or layer.dilation != (1, 1)
            or layer.bias is None
        ):
            raise ValueError(f"Unsupported FiGSR convolution for packed shape {packed}")
        weight = functional.pad(
            layer.weight, (0, 0, 0, 0, 0, (cin - layer.in_channels) // groups, 0, cout - layer.out_channels)
        )
        self.add(weight.permute(0, 2, 3, 1))
        self.add(functional.pad(layer.bias, (0, cout - layer.out_channels)))

    def block(self, block: nn.Module, index: int, dim: int, hidden: int, fourier: int) -> None:
        if (
            not isinstance(block, GatedCNNBlock)
            or not isinstance(block.act, nn.SiLU)
            or not isinstance(block.conv, InceptionConv2d)
            or not isinstance(block.conv.fu, FourierUnit)
            or tuple(block.split_indices) != (hidden, hidden - dim, fourier, fourier, fourier, fourier)
            or not isinstance(block.conv.fu.gelu, nn.GELU)
            or block.conv.fu.gelu.approximate != "none"
        ):
            raise ValueError(f"Unsupported FiGSR block {index}")
        unit = block.conv.fu
        self.norm(block.norm, dim)
        self.norm(unit.rn, 2 * fourier)
        self.norm(unit.post_norm, fourier)
        layers = (block.fc1, unit.fpe, unit.fdc, block.conv.convhw, block.conv.convw, block.conv.convh, block.fc2)
        for layer, packed in zip(layers, FIGSR_BLOCK_FILTERS, strict=True):
            self.convolution(layer, packed)

    def build(self, model: FIGSR) -> None:
        cin, cout, scale, _, alignment, dim, blocks, hidden, fourier, _, _ = FIGSR_CONFIG
        first, second = model.gfisr_body_half, model.gfisr_body_half_2
        if (
            len(first) != blocks
            or len(second) != blocks + 1
            or model.pad != alignment
            or model.scale != scale
            or len(model.upscale) != 2
            or not isinstance(model.upscale[1], nn.PixelShuffle)
            or model.upscale[1].upscale_factor != scale
            or model.shift.shape != (1, cin, 1, 1)
            or model.scale_norm.shape != model.shift.shape
            or not bool((model.scale_norm != 0).all())
        ):
            raise ValueError("Unsupported FiGSR topology")
        self.add(torch.cat((model.shift.flatten(), model.scale_norm.flatten())))
        self.convolution(model.in_to_dim, FIGSR_HEAD, input_channels=cin)
        for index, block in enumerate([*first, *second[:-1]]):
            self.block(block, index, dim, hidden, fourier)
        self.convolution(second[-1], FIGSR_TAIL[0])
        self.convolution(model.cat_to_dim, FIGSR_TAIL[1])
        self.convolution(model.upscale[0], FIGSR_TAIL[2], output_channels=cout * scale**2)


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    if not isinstance(model, FIGSR):
        raise ValueError("The FiGSR packer requires a Resselt FIGSR model")
    with torch.inference_mode(), weights.open("wb") as file:
        packer = Packer(file)
        packer.build(model.eval())
    if packer.sizes != figsr_parameter_sizes():
        raise ValueError("FiGSR parameters differ from the native schema")
    return Package(), FIGSR_CONFIG[4]
