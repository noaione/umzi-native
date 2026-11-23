"""Pack the eval-mode convolutions of the direct-shuffle 2x SMoSR model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from resselt.archs.smosr.arch import SMB, ConvNXC, DOConv2d, SMoSR
from torch import Tensor, nn
from torch.nn import functional

from .._model import SCHEMAS, SMOSR_ARCHITECTURE, SMOSR_CONFIG, SMOSR_FAMILY, SMOSR_FILTERS
from ._package import write_package


class Package:
    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: SMOSR_ARCHITECTURE.pack(*SMOSR_CONFIG), 3: weights, 4: checkpoint_sha256}
        write_package(path, SMOSR_FAMILY, SCHEMAS["smosr"], sections)


def convolution(layer: nn.Module, cin: int, cout: int, size: int) -> nn.Conv2d:
    if isinstance(layer, (ConvNXC, DOConv2d)):
        layer = layer.eval_conv
    if (
        not isinstance(layer, nn.Conv2d)
        or (layer.in_channels, layer.out_channels, layer.groups) != (cin, cout, 1)
        or layer.kernel_size != (size, size)
        or layer.padding != (size // 2, size // 2)
        or layer.padding_mode != "zeros"
        or layer.stride != (1, 1)
        or layer.dilation != (1, 1)
        or layer.bias is None
    ):
        raise ValueError(f"Unsupported SMoSR convolution. Expected {cin} to {cout}, {size}x{size}")
    return layer


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    cin, cout, scale, context, dim, first, middle, end = SMOSR_CONFIG
    if (
        not isinstance(model, SMoSR)
        or model.scale != context * scale
        or (len(model.blocks_1), len(model.blocks_2), len(model.end_block)) != (first, middle, end + 1)
        or len(model.upsampler) != 2
        or not isinstance(model.upsampler[1], nn.PixelShuffle)
        or model.upsampler[1].upscale_factor != scale
    ):
        raise ValueError("Unsupported SMoSR topology")
    rgb_skip = cin * scale**2
    blocks = [*model.blocks_1, *model.blocks_2, model.end_block[0]]
    filters = [convolution(model.short, cin, rgb_skip, 1)]
    for index, block in enumerate(blocks):
        if (
            not isinstance(block, SMB)
            or len(block.body) != 5
            or not isinstance(block.body[1], nn.SiLU)
            or not isinstance(block.body[3], nn.SiLU)
        ):
            raise ValueError(f"Unsupported SMoSR block {index}")
        if index == 0:
            filters.append(convolution(block.short, cin, dim, 1))
        elif not isinstance(block.short, nn.Identity):
            raise ValueError(f"Unsupported SMoSR shortcut in block {index}")
        filters += [
            convolution(block.body[0], cin if index == 0 else dim, dim, 3),
            convolution(block.body[2], dim, dim, 3),
            convolution(block.body[4], dim, 2 * dim, 3),
        ]
    filters += [
        convolution(model.end_block[-1], dim, dim, 3),
        convolution(model.upsampler[0], rgb_skip + dim, cout * scale**2, 3),
    ]
    with torch.inference_mode(), weights.open("wb") as file:
        model.float().eval()
        for layer, (inputs, outputs, _) in zip(filters, SMOSR_FILTERS, strict=True):
            bias = layer.bias
            if bias is None:
                raise ValueError("Missing SMoSR bias")
            weight = functional.pad(
                layer.weight, (0, 0, 0, 0, 0, inputs - layer.in_channels, 0, outputs - bias.numel())
            )
            for tensor in (weight.permute(0, 2, 3, 1).half(), functional.pad(bias, (0, outputs - bias.numel()))):
                if not torch.isfinite(tensor).all():
                    raise ValueError("Non-finite SMoSR parameter after conversion")
                file.write(bytes(-file.tell() % 256))
                file.write(tensor.contiguous().numpy().tobytes())
    return Package(), 1
