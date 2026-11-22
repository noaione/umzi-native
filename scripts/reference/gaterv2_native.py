"""Private GateRV2 FP32 path for checking native primitives before packaging."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from build_native import load_build_module
from cuda.bindings import driver
from resselt.archs.gaterv2.arch import Attention, Block, GatedCNNBlock, GateRV2, MetaGated
from torch import Tensor

KERNELS = ("pad_rgb", "crop_rgb", "normalize_qk", "finish_attention")
THREADS = 128
Trace = Callable[[str, Tensor], None]


def check_cuda(result: tuple[Any, ...]) -> Any:
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {status}")
    return values[0] if values else None


class NativeStages:
    def __init__(self, nvrtc_root: Path | None):
        torch.empty(0, device="cuda")
        major, minor = torch.cuda.get_device_capability()
        build = load_build_module()
        toolchain = build.find_toolchain(build.BuildSettings(nvrtc_root=nvrtc_root))
        image = build._compile_kernels(
            build._load_nvrtc(toolchain.nvrtc),
            toolchain.includes,
            f"sm_{major}{minor}",
            build.NATIVE / "kernels/gaterv2",
            fast_math=False,
        )
        self.module = check_cuda(driver.cuModuleLoadData(image))
        try:
            self.functions = {
                name: check_cuda(driver.cuModuleGetFunction(self.module, name.encode())) for name in KERNELS
            }
        except Exception:
            check_cuda(driver.cuModuleUnload(self.module))
            raise

    def __enter__(self) -> NativeStages:
        return self

    def __exit__(self, *_args: object) -> None:
        try:
            torch.cuda.synchronize()
        finally:
            check_cuda(driver.cuModuleUnload(self.module))

    def launch(self, name: str, total: int, tensors: tuple[Tensor, ...], *sizes: int) -> None:
        args = (*(tensor.data_ptr() for tensor in tensors), *sizes, total)
        if total > 2**31 - 1 or any(size > 2**31 - 1 for size in sizes):
            raise ValueError("The validation kernel requires 32-bit tensor dimensions")
        types = (ctypes.c_uint64,) * len(tensors) + (ctypes.c_int,) * (len(sizes) + 1)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        check_cuda(
            driver.cuLaunchKernel(
                self.functions[name],
                (total + THREADS - 1) // THREADS,
                1,
                1,
                THREADS,
                1,
                1,
                0,
                stream,
                (args, types),
                0,
            )
        )

    def pad(self, source: Tensor, align: int) -> Tensor:
        batch, channels, height, width = source.shape
        if min(height, width) < 1:
            raise ValueError("GateRV2 needs positive spatial dimensions")
        pad_height, pad_width = (-height) % align, (-width) % align
        if pad_height >= height or pad_width >= width:
            raise ValueError(f"GateRV2 reflection padding is invalid for {height}x{width}")
        if not (pad_height or pad_width):
            return source
        padded_height, padded_width = height + pad_height, width + pad_width
        target = source.new_empty((batch, channels, padded_height, padded_width))
        self.launch("pad_rgb", target.numel(), (target, source), height, width, padded_height, padded_width)
        return target

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
