"""Pack the direct-shuffle GFISRv2 topology without narrowing checkpoint weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
from resselt.archs.gfisrv2.arch import GFISRV2, FourierUnit, GatedCNNBlock, RMSNorm
from torch import Tensor, nn

from .._model import GFISRV2_ARCHITECTURE, GFISRV2_CONFIG, GFISRV2_FAMILY, SCHEMAS, gfisrv2_parameter_sizes
from ._package import write_package

NORM_EPS = 1e-6


class Packer:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.sizes: list[int] = []

    def add(self, tensor: Tensor) -> None:
        data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        self.file.write(bytes(-self.file.tell() % 256))
        self.file.write(data)
        self.sizes.append(len(data))

    def norm(self, layer: nn.Module, channels: int) -> None:
        if (
            not isinstance(layer, RMSNorm)
            or layer.eps != NORM_EPS
            or layer.scale.shape != (channels, 1, 1)
            or layer.offset.shape != layer.scale.shape
        ):
            raise ValueError("Unsupported GFISRv2 RMSNorm")
        self.add(layer.scale)
        self.add(layer.offset)

    def conv(self, layer: nn.Module, cin: int, cout: int, kernel: tuple[int, int] = (3, 3), *, groups: int = 1) -> None:
        if (
            not isinstance(layer, nn.Conv2d)
            or (layer.in_channels, layer.out_channels, layer.groups) != (cin, cout, groups)
            or layer.kernel_size != kernel
            or layer.padding != (kernel[0] // 2, kernel[1] // 2)
            or layer.padding_mode != "zeros"
            or layer.stride != (1, 1)
            or layer.dilation != (1, 1)
            or layer.bias is None
        ):
            raise ValueError("Unsupported GFISRv2 convolution")
        self.add(layer.weight)
        self.add(layer.bias)

    def block(self, layer: nn.Module, index: int, dim: int, hidden: int, fourier: int) -> None:
        if (
            not isinstance(layer, GatedCNNBlock)
            or not isinstance(layer.act, nn.SiLU)
            or tuple(layer.split_indices) != (hidden, hidden - dim, dim)
            or layer.gamma.shape != (1, dim, 1, 1)
        ):
            raise ValueError(f"Unsupported GFISRv2 block {index}")
        band = (dim - fourier) // 3
        branches = (layer.conv.pconv, layer.conv.dwconv_hw, layer.conv.dwconv_w, layer.conv.dwconv_h)
        shift = index % 4
        canonical = branches[-shift:] + branches[:-shift] if shift else branches
        unit, square, row, column = canonical
        widths = (fourier, band, band, band)
        if not isinstance(unit, FourierUnit) or tuple(layer.conv.split_indexes) != widths[shift:] + widths[:shift]:
            raise ValueError(f"Unsupported GFISRv2 inception rotation in block {index}")
        self.norm(layer.norm, dim)
        self.conv(layer.fc1, dim, 2 * hidden)
        packed = 2 * fourier
        self.norm(unit.rn, packed)
        self.conv(unit.fpe, packed, packed, groups=packed)
        self.conv(unit.fdc, packed, packed, (1, 1))
        self.norm(unit.post_norm, fourier)
        self.conv(square, band, band, groups=band)
        self.conv(row, band, band, (1, 11), groups=band)
        self.conv(column, band, band, (11, 1), groups=band)
        self.conv(layer.fc2, hidden, dim)
        self.add(layer.gamma)

    def build(self, model: GFISRV2) -> None:
        cin, cout, scale, dim, blocks, hidden, fourier = GFISRV2_CONFIG
        body = model.gfisr_body
        if (
            model.scale != scale
            or len(body) != blocks + 3
            or not isinstance(body[-2], nn.SiLU)
            or len(model.upscale) != 2
            or not isinstance(model.upscale[1], nn.PixelShuffle)
            or model.upscale[1].upscale_factor != scale
        ):
            raise ValueError("Unsupported GFISRv2 topology")
        self.conv(model.in_to_dim, cin, dim)
        for index, layer in enumerate(body[:-3]):
            self.block(layer, index, dim, hidden, fourier)
        self.conv(body[-3], dim, 2 * dim)
        self.conv(body[-1], 2 * dim, dim)
        self.conv(model.upscale[0], dim, cout * scale**2)


class Package:
    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: GFISRV2_ARCHITECTURE.pack(*GFISRV2_CONFIG), 3: weights, 4: checkpoint_sha256}
        write_package(path, GFISRV2_FAMILY, SCHEMAS["gfisrv2"], sections)


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    if not isinstance(model, GFISRV2):
        raise ValueError("The GFISRv2 packer requires a Resselt GFISRv2 model")
    with torch.inference_mode(), weights.open("wb") as file:
        packer = Packer(file)
        packer.build(model.eval())
    if packer.sizes != gfisrv2_parameter_sizes():
        raise ValueError("GFISRv2 parameters differ from the native schema")
    return Package(), 1
