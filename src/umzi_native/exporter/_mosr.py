"""Pack the MoSR architecture for the native CUDA session."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
from resselt.archs.mosr.arch import GPS, ConvBlock, GatedCNNBlock, LayerNorm, mosr
from resselt.utilities import DySample
from torch import Tensor, nn

from .._model import MOSR_ARCHITECTURE, MOSR_FAMILY, SCHEMAS, mosr_parameter_sizes
from ._package import write_package

NORM_EPS = 1e-6


class Packer:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.sizes: list[int] = []
        self.config: tuple[int, ...] | None = None

    def add(self, tensor: Tensor) -> None:
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(cpu).all():
            raise ValueError("Non-finite MoSR parameter")
        data = cpu.numpy().tobytes()
        self.file.write(bytes(-self.file.tell() % 256))
        self.file.write(data)
        self.sizes.append(len(data))

    def conv(
        self,
        layer: nn.Module,
        cin: int,
        cout: int,
        kernel: tuple[int, int] = (3, 3),
        *,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        if (
            not isinstance(layer, nn.Conv2d)
            or (layer.in_channels, layer.out_channels, layer.groups) != (cin, cout, groups)
            or layer.kernel_size != kernel
            or layer.padding != (kernel[0] // 2, kernel[1] // 2)
            or layer.padding_mode != "zeros"
            or layer.stride != (1, 1)
            or layer.dilation != (1, 1)
            or (layer.bias is not None) != bias
        ):
            raise ValueError(f"Unsupported MoSR convolution: {layer}")
        self.add(layer.weight)
        if layer.bias is not None:
            self.add(layer.bias)

    def gated(self, layer: nn.Module, dim: int, hidden: int, kernel_size: int) -> None:
        if (
            not isinstance(layer, GatedCNNBlock)
            or not isinstance(layer.norm, LayerNorm)
            or layer.norm.eps != NORM_EPS
            or tuple(layer.norm.weight.shape) != (dim,)
            or tuple(layer.norm.bias.shape) != (dim,)
            or not isinstance(layer.act, nn.Mish)
            or tuple(layer.split_indices) != (hidden, hidden - dim, dim)
        ):
            raise ValueError("Unsupported MoSR gated block")
        self.add(layer.norm.weight)
        self.add(layer.norm.bias)
        self.conv(layer.fc1, dim, 2 * hidden)
        self.conv(layer.conv, dim, dim, (kernel_size, kernel_size), groups=dim)
        self.conv(layer.fc2, hidden, dim)

    def build(self, model: mosr) -> None:
        gblocks = model.gblocks
        if len(gblocks) < 7:
            raise ValueError("Unsupported MoSR body length")
        stem = gblocks[0]
        if not isinstance(stem, nn.Conv2d):
            raise ValueError("Unsupported MoSR stem")
        in_ch, dim = stem.in_channels, stem.out_channels
        if in_ch != 3 or dim % 8 or not 16 <= dim <= 256:
            raise ValueError(f"Unsupported MoSR stem shape: {in_ch} -> {dim}")
        self.conv(stem, in_ch, dim)

        body_blocks = list(gblocks[1:-5])
        if not body_blocks or len(body_blocks) > 64 or not isinstance(body_blocks[0], GatedCNNBlock):
            raise ValueError("Unsupported MoSR gated block sequence")
        first = body_blocks[0]
        if len(first.split_indices) != 3:
            raise ValueError("Unsupported MoSR split_indices")
        hidden = first.split_indices[0]
        kernel_size = first.conv.kernel_size[0]
        if not dim <= hidden <= 4 * dim or kernel_size % 2 != 1 or not 1 <= kernel_size <= 31:
            raise ValueError(f"Unsupported MoSR block config: dim={dim}, hidden={hidden}, kernel={kernel_size}")
        for block in body_blocks:
            self.gated(block, dim, hidden, kernel_size)

        tail = list(gblocks[-5:])
        if not isinstance(tail[1], nn.Mish) or not isinstance(tail[3], nn.Mish):
            raise ValueError("Unsupported MoSR body tail activations")
        self.conv(tail[0], dim, 2 * dim)
        self.conv(tail[2], 2 * dim, dim)
        self.conv(tail[4], dim, dim, (1, 1))

        shortcut = model.shortcut
        if (
            not isinstance(shortcut, ConvBlock)
            or len(shortcut.block) != 4
            or not isinstance(shortcut.block[1], nn.Mish)
            or not isinstance(shortcut.block[3], nn.Mish)
        ):
            raise ValueError("Unsupported MoSR ConvBlock shortcut")
        self.conv(shortcut.block[0], in_ch, dim)
        self.conv(shortcut.block[2], dim, dim)
        self.conv(shortcut.conv11, in_ch, dim, (1, 1))

        upsampler = model.upsampler
        out_ch = 3
        dysample_groups = 0
        if isinstance(upsampler, GPS):
            if not isinstance(upsampler.ps, nn.PixelShuffle):
                raise ValueError("Unsupported MoSR GPS upsampler")
            scale = upsampler.ps.upscale_factor
            upsampler_kind = 1
            self.conv(upsampler.in_to_k, dim, out_ch * scale * scale * 8)
        elif isinstance(upsampler, nn.Sequential):
            if len(upsampler) != 2 or not isinstance(upsampler[1], nn.PixelShuffle):
                raise ValueError("Unsupported MoSR pixel-shuffle upsampler")
            scale = upsampler[1].upscale_factor
            upsampler_kind = 0
            self.conv(upsampler[0], dim, out_ch * scale * scale)
        elif isinstance(upsampler, DySample):
            if not upsampler.end_convolution:
                raise ValueError("MoSR DySample requires end_convolution=True")
            scale = upsampler.scale
            dysample_groups = upsampler.groups
            if scale < 2 or dysample_groups <= 0 or dim % dysample_groups != 0:
                raise ValueError("Unsupported MoSR DySample parameters")
            upsampler_kind = 2
            offset_ch = 2 * dysample_groups * scale * scale
            self.conv(upsampler.offset, dim, offset_ch, (1, 1))
            self.conv(upsampler.scope, dim, offset_ch, (1, 1), bias=False)
            self.conv(upsampler.end_conv, dim, out_ch, (1, 1))
        else:
            raise ValueError(f"Unsupported MoSR upsampler: {type(upsampler)}")

        self.config = (
            in_ch,
            out_ch,
            scale,
            len(body_blocks),
            dim,
            hidden,
            kernel_size,
            upsampler_kind,
            dysample_groups,
            0,
        )


class Package:
    def __init__(self, config: tuple[int, ...]):
        self.config = config

    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: MOSR_ARCHITECTURE.pack(*self.config), 3: weights, 4: checkpoint_sha256}
        write_package(path, MOSR_FAMILY, SCHEMAS["mosr"], sections)


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    if not isinstance(model, mosr):
        raise ValueError("The MoSR packer requires a Resselt mosr model")
    with torch.inference_mode(), weights.open("wb") as file:
        packer = Packer(file)
        packer.build(model.float().eval())
    if packer.config is None:
        raise ValueError("Missing MoSR configuration")
    in_ch, out_ch, scale, blocks, dim, hidden, kernel_size, upsampler_kind, dysample_groups, _ = packer.config
    expected_sizes = mosr_parameter_sizes(
        in_ch, out_ch, scale, blocks, dim, hidden, kernel_size, upsampler_kind, dysample_groups
    )
    if packer.sizes != expected_sizes:
        raise ValueError("Packed MoSR parameters differ from the native schema")
    return Package(packer.config), 1
