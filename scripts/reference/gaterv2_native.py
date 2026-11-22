"""Private GateRV2 FP32 path for checking native primitives before packaging."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from resselt.archs.gaterv2.arch import Attention, Block, GatedCNNBlock, GateRV2, MetaGated
from torch import Tensor

from reference.native_stages import NativeKernels, Trace

KERNELS = ("pad_rgb", "crop_rgb", "normalize_qk", "finish_attention")


class NativeStages(NativeKernels):
    def __init__(self, nvrtc_root: Path | None):
        super().__init__("GateRV2", KERNELS, nvrtc_root)

    def crop(self, source: Tensor, height: int, width: int) -> Tensor:
        batch, channels, padded_height, padded_width = source.shape
        if (height, width) == (padded_height, padded_width):
            return source
        target = source.new_empty((batch, channels, height, width))
        self.launch(
            "crop_rgb", target.numel(), (target, source.contiguous()), height, width, padded_height, padded_width
        )
        return target

    def attention(self, query: Tensor, key: Tensor, value: Tensor, eps: float) -> Tensor:
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        batch, channels, tokens = query.shape
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        self.launch("normalize_qk", batch * tokens, (query_norm, key_norm, query, key), channels, tokens)
        query_t = query_norm.transpose(1, 2)
        contraction = torch.bmm(query_t, (key_norm.sum(dim=-1) + eps).unsqueeze(-1))
        matrix = torch.bmm(key_norm, value.transpose(1, 2))
        product = torch.bmm(query_t, matrix)
        value_sum = value.sum(dim=-1)
        result = torch.empty_like(value)
        self.launch(
            "finish_attention", result.numel(), (result, product, value_sum, contraction), value.size(1), tokens
        )
        return result


class NativeGateRV2:
    """Compose the FP32 network using native CUDA stages and Torch CUDA primitives."""

    def __init__(self, model: GateRV2, native: NativeStages):
        if model.scale != 1:
            raise ValueError("The private validation path requires an RGB 1x GateRV2 checkpoint")
        self.model = model
        self.native = native

    def _gated(self, block: GatedCNNBlock, x: Tensor, name: str, trace: Trace | None) -> Tensor:
        gate, identity, channels = torch.split(block.fc1(block.norm(x)), block.split_indices, dim=1)
        mix = block.token_mix
        if isinstance(mix, Attention):
            query, key, value = (conv(channels).flatten(2) for conv in (mix.query_conv, mix.key_conv, mix.value_conv))
            channels = self.native.attention(query, key, value, mix.eps).reshape(channels.shape)
            if trace is not None:
                trace(f"{name}.token_mix", channels)
        else:
            channels = mix(channels)
        return block.act(block.fc2(block.act(gate) * torch.cat((identity, channels), dim=1)))

    def _local_global(self, block: MetaGated, value: Tensor) -> Tensor:
        local = block.local(value)
        local = local * block.sca(local)
        value = local * block.gamma0 + value
        return self._gated(block.glob, value, "", None) * block.gamma1 + value

    def __call__(self, source: Tensor, trace: Trace | None = None) -> Tensor:
        if source.ndim != 4 or source.shape[0] < 1 or source.shape[1] != 3:
            raise ValueError("GateRV2 expects a nonempty NCHW RGB batch")
        if source.dtype != torch.float32 or not source.is_cuda:
            raise ValueError("The private validation path requires a CUDA FP32 input")
        height, width = source.shape[-2:]
        padded = self.native.pad(source.contiguous(), self.model.pad)
        if trace is not None:
            trace("padded", padded)
        x = self.model.in_to_dim(padded)
        if trace is not None:
            trace("in_to_dim", x)
        skips = []
        for level, module in enumerate(self.model.encode):
            layer = cast(Block, module)
            for block in layer.gated:
                x = self._local_global(cast(MetaGated, block), x)
            if trace is not None:
                trace(f"encode.{level}.gated", x)
            skips.append(x)
            x = layer.scale(x)
            if trace is not None:
                trace(f"encode.{level}.scale", x)
        for index, block in enumerate(self.model.latent):
            name = f"latent.{index}"
            x = self._gated(cast(GatedCNNBlock, block), x, name, trace)
            if trace is not None:
                trace(name, x)
        for level, (module, skip) in enumerate(zip(self.model.decode, reversed(skips), strict=True)):
            layer = cast(Block, module)
            x = layer.shor(torch.cat((layer.scale(x), skip), dim=1))
            if trace is not None:
                trace(f"decode.{level}.shor", x)
            for block in layer.gated:
                x = self._local_global(cast(MetaGated, block), x)
            if trace is not None:
                trace(f"decode.{level}.gated", x)
        x = self.model.dim_to_in(x)
        if trace is not None:
            trace("dim_to_in", x)
        result = self.native.crop(x + padded, height, width)
        if trace is not None:
            trace("output", result)
        return result
