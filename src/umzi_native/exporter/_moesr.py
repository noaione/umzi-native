"""Pack the direct-shuffle MoESR topology for the native CUDA session."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
from resselt.archs.moesr.arch import Blocks, GatedCNNBlock, LayerNorm, MoESR
from torch import Tensor, nn

from .._model import MOESR_ARCHITECTURE, MOESR_CONFIG, MOESR_FAMILY, SCHEMAS, moesr_parameter_sizes
from ._package import write_package

ALIGNMENT = MOESR_CONFIG[3]
NORM_EPS = 1e-6
LEAKY_SLOPE = 0.1


class Packer:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.sizes: list[int] = []

    def add(self, value: Tensor, dtype: torch.dtype = torch.float32) -> None:
        data = value.detach().to(device="cpu", dtype=dtype).contiguous().numpy().tobytes()
        self.file.write(bytes(-self.file.tell() % 256))
        self.file.write(data)
        self.sizes.append(len(data))

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
            raise ValueError(f"Unsupported MoESR convolution: {layer}")
        if groups == 1:
            self.add(layer.weight.permute(0, 2, 3, 1), torch.float16)
        else:
            self.add(layer.weight)
        self.add(layer.bias)

    def gated(self, layer: nn.Module, dim: int, hidden: int) -> None:
        if (
            not isinstance(layer, GatedCNNBlock)
            or not isinstance(layer.norm, LayerNorm)
            or layer.norm.eps != NORM_EPS
            or not isinstance(layer.act, nn.Mish)
            or tuple(layer.split_indices) != (hidden, hidden - dim, dim)
            or tuple(layer.conv.split_indexes) != (40, 8, 8, 8)
            or tuple(layer.gamma.shape) != (1, dim, 1, 1)
        ):
            raise ValueError("Unsupported MoESR gated block")
        self.add(layer.norm.weight)
        self.add(layer.norm.bias)
        self.conv(layer.fc1, dim, 2 * hidden)
        self.conv(layer.conv.dwconv_hw, 8, 8, (3, 3), groups=8)
        self.conv(layer.conv.dwconv_w, 8, 8, (1, 11), groups=8)
        self.conv(layer.conv.dwconv_h, 8, 8, (11, 1), groups=8)
        self.conv(layer.fc2, hidden, dim)
        self.add(layer.gamma)

    def build(self, model: MoESR) -> None:
        cin, cout, scale, _, dim, groups, full, reduced, full_hidden, reduced_hidden = MOESR_CONFIG
        if model.scale != scale or len(model.blocks) != groups or len(model.upscale) != 2:
            raise ValueError("Unsupported MoESR topology")
        self.conv(model.in_to_dim, cin, dim)
        for group in model.blocks:
            if not isinstance(group, Blocks) or len(group.blocks) != full or len(group.msg.gated) != reduced:
                raise ValueError("Unsupported MoESR group")
            for layer in group.blocks:
                self.gated(layer, dim, full_hidden)
            down, up = group.msg.down, group.msg.up
            if (
                len(down) != 3
                or len(up) != 3
                or not isinstance(down[1], nn.PixelUnshuffle)
                or down[1].downscale_factor != 2
                or not isinstance(up[1], nn.PixelShuffle)
                or up[1].upscale_factor != 2
                or any(
                    not isinstance(layer, nn.LeakyReLU) or layer.negative_slope != LEAKY_SLOPE
                    for layer in (down[2], up[2])
                )
            ):
                raise ValueError("Unsupported MoESR multiscale branch")
            self.conv(down[0], dim, dim // 4)
            for layer in group.msg.gated:
                self.gated(layer, dim, reduced_hidden)
            self.conv(up[0], dim, dim * 4)
        if not isinstance(model.upscale[1], nn.PixelShuffle) or model.upscale[1].upscale_factor != scale:
            raise ValueError("MoESR requires direct 4x pixel shuffle")
        self.conv(model.upscale[0], dim, cout * scale**2)


class Package:
    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: MOESR_ARCHITECTURE.pack(*MOESR_CONFIG), 3: weights, 4: checkpoint_sha256}
        write_package(path, MOESR_FAMILY, SCHEMAS["moesr"], sections)


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    if not isinstance(model, MoESR):
        raise ValueError("The MoESR packer requires a Resselt MoESR model")
    with torch.inference_mode(), weights.open("wb") as file:
        packer = Packer(file)
        packer.build(model.eval())
    if packer.sizes != moesr_parameter_sizes():
        raise ValueError("Packed MoESR parameters differ from the native schema")
    return Package(), ALIGNMENT
