"""Private MoESR FP32 path for parity checks before native packaging."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from resselt.archs.moesr.arch import Blocks, GatedCNNBlock, MoESR
from torch import Tensor, nn

from reference.native_stages import NativeKernels, Trace

KERNELS = (
    "pad_rgb",
    "layer_norm",
    "mish_gate",
    "mish_residual",
    "unshuffle2_leaky",
    "shuffle2_leaky_add",
    "shuffle4_crop",
)


class NativeStages(NativeKernels):
    def __init__(self, nvrtc_root: Path | None):
        super().__init__("MoESR", KERNELS, nvrtc_root)

    def layer_norm(self, source: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
        batch, channels, height, width = source.shape
        pixels = height * width
        result = torch.empty_like(source)
        self.launch("layer_norm", batch * pixels, (result, source, weight, bias), channels, pixels, eps)
        return result

    def gate(self, projected: Tensor, square: Tensor, row: Tensor, column: Tensor) -> Tensor:
        batch, channels, height, width = projected.shape
        hidden = channels // 2
        result = projected.new_empty((batch, hidden, height, width))
        self.launch(
            "mish_gate",
            result.numel(),
            (result, projected, square, row, column),
            hidden,
            height * width,
            square.size(1),
        )
        return result

    def residual(self, projected: Tensor, shortcut: Tensor, gamma: Tensor) -> Tensor:
        _, channels, height, width = projected.shape
        result = torch.empty_like(projected)
        self.launch("mish_residual", result.numel(), (result, projected, shortcut, gamma), channels, height * width)
        return result

    def down(self, projected: Tensor) -> Tensor:
        batch, channels, height, width = projected.shape
        result = projected.new_empty((batch, 4 * channels, height // 2, width // 2))
        self.launch("unshuffle2_leaky", result.numel(), (result, projected), height // 2, width // 2)
        return result

    def up(self, projected: Tensor, shortcut: Tensor) -> Tensor:
        result = torch.empty_like(shortcut)
        self.launch("shuffle2_leaky_add", result.numel(), (result, projected, shortcut), *projected.shape[-2:])
        return result

    def output(self, projected: Tensor, height: int, width: int) -> Tensor:
        result = projected.new_empty((projected.size(0), 3, 4 * height, 4 * width))
        self.launch("shuffle4_crop", result.numel(), (result, projected), *result.shape[-2:], *projected.shape[-2:])
        return result


class NativeMoESR:
    """Compose the checkpoint convolutions with private CUDA pointwise and layout stages."""

    def __init__(self, model: MoESR, native: NativeStages):
        if (
            model.scale != 4
            or model.in_to_dim.in_channels != 3
            or len(model.upscale) != 2
            or not isinstance(model.upscale[0], nn.Conv2d)
            or model.upscale[0].out_channels != 48
            or not isinstance(model.upscale[1], nn.PixelShuffle)
            or model.upscale[1].upscale_factor != 4
        ):
            raise ValueError("The validation path requires a direct-shuffle RGB 4x MoESR checkpoint")
        self.model = model
        self.native = native

    def _gated(self, block: GatedCNNBlock, shortcut: Tensor, name: str, trace: Trace | None) -> Tensor:
        norm = block.norm
        x = self.native.layer_norm(shortcut, norm.weight, norm.bias, norm.eps)
        if trace is not None:
            trace(f"{name}.norm", x)
        projected = block.fc1(x)
        hidden, identity, channels = block.split_indices
        mix = block.conv
        _, square, row, column = projected.narrow(1, hidden + identity, channels).split(mix.split_indexes, dim=1)
        x = self.native.gate(projected, mix.dwconv_hw(square), mix.dwconv_w(row), mix.dwconv_h(column))
        x = self.native.residual(block.fc2(x), shortcut, block.gamma)
        if trace is not None:
            trace(name, x)
        return x

    def __call__(self, source: Tensor, trace: Trace | None = None) -> Tensor:
        if source.ndim != 4 or source.shape[0] < 1 or source.shape[1] != 3:
            raise ValueError("MoESR expects a nonempty NCHW RGB batch")
        if not source.is_cuda or source.dtype != torch.float32:
            raise ValueError("The validation path requires CUDA FP32 input")
        height, width = source.shape[-2:]
        padded = self.native.pad(source.contiguous(), 2)
        if trace is not None:
            trace("padded", padded)
        initial = self.model.in_to_dim(padded)
        if trace is not None:
            trace("in_to_dim", initial)
        x = initial
        for index, module in enumerate(self.model.blocks):
            group = cast(Blocks, module)
            name = f"blocks.{index}"
            for i, block in enumerate(group.blocks):
                x = self._gated(cast(GatedCNNBlock, block), x, f"{name}.blocks.{i}", trace)
            reduced = self.native.down(group.msg.down[0](x))
            if trace is not None:
                trace(f"{name}.msg.down", reduced)
            for i, block in enumerate(group.msg.gated):
                reduced = self._gated(cast(GatedCNNBlock, block), reduced, f"{name}.msg.gated.{i}", trace)
            x = self.native.up(group.msg.up[0](reduced), x)
            if trace is not None:
                trace(f"{name}.msg", x)
        x = x + initial
        if trace is not None:
            trace("outer_residual", x)
        logits = self.model.upscale[0](x)
        if trace is not None:
            trace("upscale.0", logits)
        result = self.native.output(logits, height, width)
        if trace is not None:
            trace("output", result)
        return result
