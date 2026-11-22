"""Staged GFISRv2 FP32 runner for validating the native Fourier path."""

from __future__ import annotations

from typing import Protocol, cast

import torch
from resselt.archs.gfisrv2.arch import GFISRV2, FourierUnit, GatedCNNBlock, RMSNorm
from torch import Tensor, nn
from torch.nn import functional

from .native_stages import Trace


class FourierTransforms(Protocol):
    def forward(self, source: Tensor) -> Tensor: ...

    def inverse(self, source: Tensor, height: int, width: int) -> Tensor: ...


class TorchTransforms:
    def forward(self, source: Tensor) -> Tensor:
        return torch.fft.rfft2(source, norm="ortho")

    def inverse(self, source: Tensor, height: int, width: int) -> Tensor:
        return torch.fft.irfft2(source, s=(height, width), norm="ortho")


def rms_norm(source: Tensor, params: RMSNorm) -> Tensor:
    rms = source.norm(2, dim=1, keepdim=True) * (source.size(1) ** -0.5)
    return params.scale * (source / (rms + params.eps)) + params.offset


def fourier_unit(
    unit: FourierUnit, source: Tensor, transforms: FourierTransforms, trace: Trace | None = None
) -> Tensor:
    height, width = source.shape[-2:]
    spectrum = transforms.forward(source)
    # The learned layers read real planes followed by imaginary planes.
    packed = torch.cat((spectrum.real, spectrum.imag), dim=1)
    norm = rms_norm(packed, unit.rn)
    position = unit.fpe(norm)
    with_position = position + norm
    projected = unit.fdc(with_position)
    activated = functional.gelu(projected)
    # The checkpoint reinterprets adjacent channels as complex pairs.
    inverse_spectrum = torch.complex(activated[:, 0::2], activated[:, 1::2])
    inverse = transforms.inverse(inverse_spectrum, height, width)
    output = rms_norm(inverse, unit.post_norm)
    if trace is not None:
        for stage, tensor in (
            ("forward_spectrum", spectrum),
            ("packed", packed),
            ("spectral_norm", norm),
            ("position", position),
            ("position_residual", with_position),
            ("projected", projected),
            ("gelu", activated),
            ("inverse_spectrum", inverse_spectrum),
            ("inverse", inverse),
            ("output", output),
        ):
            trace(stage, tensor)
    return output


class NativeGFISRV2:
    """Use direct FFTs with the checkpoint convolutions on Torch CUDA."""

    def __init__(self, model: GFISRV2, transforms: FourierTransforms):
        if (
            not isinstance(model.in_to_dim, nn.Conv2d)
            or model.in_to_dim.in_channels != 3
            or model.scale != 4
            or len(model.upscale) != 2
            or not isinstance(model.upscale[1], nn.PixelShuffle)
            or model.upscale[1].upscale_factor != 4
            or not isinstance(model.upscale[0], nn.Conv2d)
            or model.upscale[0].out_channels != 48
        ):
            raise ValueError("The parity path requires the direct-shuffle RGB 4x GFISRv2 checkpoint")
        self.model = model
        self.transforms = transforms
        self.blocks = tuple(cast(GatedCNNBlock, block) for block in model.gfisr_body[:-3])
        self.tail = model.gfisr_body[-3:]

    def _gated(self, block: GatedCNNBlock, source: Tensor) -> Tensor:
        gate, identity, spatial = block.fc1(rms_norm(source, block.norm)).split(block.split_indices, dim=1)
        inception = block.conv
        parts = spatial.split(inception.split_indexes, dim=1)
        layers = (inception.pconv, inception.dwconv_hw, inception.dwconv_w, inception.dwconv_h)
        mixed = [
            fourier_unit(layer, part, self.transforms) if isinstance(layer, FourierUnit) else layer(part)
            for layer, part in zip(layers, parts, strict=True)
        ]
        projected = block.fc2(block.act(gate) * torch.cat((identity, *mixed), dim=1))
        return block.act(projected) * block.gamma + source

    def __call__(self, source: Tensor, trace: Trace | None = None) -> Tensor:
        if source.ndim != 4 or min(source.shape) < 1 or source.size(1) != 3:
            raise ValueError("GFISRv2 expects a nonempty NCHW RGB batch")
        if source.dtype != torch.float32 or not source.is_cuda:
            raise ValueError("The parity path requires CUDA FP32 input")
        initial = self.model.in_to_dim(source)
        if trace is not None:
            trace("in_to_dim", initial)
        x = initial
        for index, block in enumerate(self.blocks):
            x = self._gated(block, x)
            if trace is not None:
                trace(f"gfisr_body.{index}", x)
        x = self.tail(x)
        if trace is not None:
            trace("tail", x)
        x = x + initial
        if trace is not None:
            trace("feature_residual", x)
        height, width = source.shape[-2:]
        result = self.model.upscale(x)[:, :, : 4 * height, : 4 * width]
        if trace is not None:
            trace("output", result)
        return result
