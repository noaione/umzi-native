"""Pack GateRV2 FP32 parameters in the order the native session executes them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

import torch
from resselt.archs.gaterv2.arch import Attention, Block, GatedCNNBlock, GateRV2, InceptionDWConv2d, MetaGated, RMSNorm
from torch import Tensor, nn

from .._model import (
    GATERV2_ARCHITECTURE,
    GATERV2_FAMILY,
    GATERV2_MAX_LEVELS,
    GATERV2_PARAMETER,
    SCHEMAS,
    gaterv2_parameter_counts,
)
from ._package import write_package

SCHEMA_VERSION = SCHEMAS["gaterv2"]
RMS_EPS = 1e-6


def _conv(layer: nn.Module, kernel: tuple[int, int], *, groups: int = 1, bias: bool = True) -> nn.Conv2d:
    if (
        not isinstance(layer, nn.Conv2d)
        or layer.kernel_size != kernel
        or layer.groups != groups
        or layer.stride != (1, 1)
        or layer.dilation != (1, 1)
        or layer.padding != (kernel[0] // 2, kernel[1] // 2)
        or layer.padding_mode != "zeros"
        or (layer.bias is not None) != bias
    ):
        raise ValueError(f"Unsupported GateRV2 convolution: {layer}")
    return layer


class Packer:
    def __init__(self, weights: BinaryIO):
        self.weights = weights
        self.table: list[tuple[int, int]] = []

    def add(self, *values: Tensor | None) -> None:
        for value in values:
            if value is None:
                raise ValueError("A GateRV2 parameter is missing")
            data = value.detach().to(torch.float32).contiguous().numpy().tobytes()
            offset = (self.weights.tell() + 255) & ~255
            self.weights.write(bytes(offset - self.weights.tell()))
            self.weights.write(data)
            self.table.append((offset, len(data) // 4))

    def conv(self, layer: nn.Module, kernel: int = 1, *, groups: int = 1, bias: bool = True) -> None:
        conv = _conv(layer, (kernel, kernel), groups=groups, bias=bias)
        self.add(conv.weight, *((conv.bias,) if bias else ()))

    def norm(self, layer: nn.Module) -> None:
        if not isinstance(layer, RMSNorm) or layer.eps != RMS_EPS:
            raise ValueError(f"Unsupported GateRV2 RMSNorm: {layer}")
        self.add(layer.scale, layer.offset)

    def gated(self, block: GatedCNNBlock, dim: int, *, attention: bool) -> None:
        hidden = 3 * dim // 2
        if list(block.split_indices) != [hidden, hidden - dim, dim] or not isinstance(block.act, nn.Mish):
            raise ValueError("Unsupported GateRV2 gated block split or activation")
        self.norm(block.norm)
        self.conv(block.fc1)
        mix = block.token_mix
        if attention:
            if not isinstance(mix, Attention) or mix.query_conv.out_channels != dim // 16:
                raise ValueError("GateRV2 latent blocks require linear attention with 16x key reduction")
            layers = [_conv(conv, (1, 1)) for conv in (mix.query_conv, mix.key_conv, mix.value_conv)]
            self.add(
                torch.cat([layer.weight for layer in layers]),
                torch.cat([cast(Tensor, layer.bias) for layer in layers]),
            )
        else:
            branch = dim // 8
            if (
                not isinstance(mix, InceptionDWConv2d)
                or tuple(mix.split_indexes) != (dim - 3 * branch,) + (branch,) * 3
            ):
                raise ValueError("Unsupported GateRV2 inception token mixer")
            for layer, kernel in ((mix.dwconv_hw, (3, 3)), (mix.dwconv_w, (1, 11)), (mix.dwconv_h, (11, 1))):
                conv = _conv(layer, kernel, groups=branch)
                self.add(conv.weight, conv.bias)
        self.conv(block.fc2)

    def meta(self, block: MetaGated, dim: int) -> None:
        local, sca = list(block.local), list(block.sca)
        if len(local) != 4 or len(sca) != 2 or not isinstance(sca[0], nn.AdaptiveAvgPool2d):
            raise ValueError("Unsupported GateRV2 MetaGated layout")
        self.norm(local[0])
        self.conv(local[1])
        self.conv(local[2], 3, groups=dim)
        self.conv(sca[1])
        self.add(block.gamma0)
        self.gated(block.glob, dim, attention=False)
        self.add(block.gamma1)

    def build(self, model: GateRV2, base: int) -> None:
        self.conv(model.in_to_dim, 3)
        for level, module in enumerate(model.encode):
            layer = cast(Block, module)
            for block in layer.gated:
                self.meta(cast(MetaGated, block), base << level)
            self.conv(layer.scale[0], 3, bias=False)
        width = base << len(model.encode)
        for block in model.latent:
            self.gated(cast(GatedCNNBlock, block), width, attention=True)
        for index, module in enumerate(model.decode):
            layer = cast(Block, module)
            self.conv(layer.scale[0], 3, bias=False)
            self.conv(layer.shor)
            for block in layer.gated:
                self.meta(cast(MetaGated, block), (width >> index) // 2)
        self.conv(model.dim_to_in, 3)


class Package:
    def __init__(self, architecture: bytes, table: bytes):
        self.architecture = architecture
        self.table = table

    def write_model(self, path: Path, weights: Path, checkpoint_sha256: bytes, _alignment: int) -> None:
        sections = {1: self.architecture, 2: self.table, 3: weights, 4: checkpoint_sha256}
        write_package(path, GATERV2_FAMILY, SCHEMA_VERSION, sections)


def build_schedule(model: nn.Module, _state: Mapping[str, Tensor], weights: Path) -> tuple[Package, int]:
    if not isinstance(model, GateRV2):
        raise ValueError("The GateRV2 packer requires a Resselt GateRV2 model")
    encoder = tuple(len(cast(Block, block).gated) for block in model.encode)
    decoder = tuple(len(cast(Block, block).gated) for block in model.decode)
    levels = len(encoder)
    base = model.in_to_dim.out_channels
    mixers = [cast(GatedCNNBlock, block).token_mix for block in model.latent]
    if (
        model.scale != 1
        or not isinstance(model.dim_to_in, nn.Conv2d)
        or model.in_to_dim.in_channels != 3
        or model.dim_to_in.out_channels != 3
        or not 1 <= levels <= GATERV2_MAX_LEVELS
        or len(decoder) != levels
        or model.pad != 1 << levels
        or base % 16
        or not 16 <= base <= 64
        or not mixers
        or not all(isinstance(mix, Attention) for mix in mixers)
    ):
        raise ValueError(f"Unsupported GateRV2 topology: base={base}, encoder={encoder}, decoder={decoder}")
    epsilons = {float(cast(Attention, mix).eps) for mix in mixers}
    if len(epsilons) != 1:
        raise ValueError("GateRV2 latent attention blocks use different epsilons")
    padding = (0,) * (GATERV2_MAX_LEVELS - levels)
    architecture = GATERV2_ARCHITECTURE.pack(
        3, 3, 1, model.pad, base, levels, len(mixers), epsilons.pop(), *encoder, *padding, *decoder, *padding
    )
    with torch.inference_mode(), weights.open("wb") as file:
        model.to(device="cpu", dtype=torch.float32).eval()
        packer = Packer(file)
        packer.build(model, base)
    if [count for _, count in packer.table] != gaterv2_parameter_counts(base, encoder, decoder, len(mixers)):
        raise ValueError("GateRV2 parameter layout differs from the native schema")
    table = b"".join(GATERV2_PARAMETER.pack(*entry) for entry in packer.table)
    return Package(architecture, table), model.pad
