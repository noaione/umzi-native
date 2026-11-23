"""Load private CUDA validation kernels and launch them on the Torch stream."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import torch
from build_native import load_build_module
from cuda.bindings import driver  # pyrefly: ignore[missing-import]  # CUDA-only validation dependency.
from torch import Tensor

THREADS = 128
Trace = Callable[[str, Tensor], None]


def check_cuda(result: tuple[Any, ...]) -> Any:
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {status}")
    return values[0] if values else None


class NativeKernels:
    def __init__(self, family: str, names: tuple[str, ...], nvrtc_root: Path | None):
        self.family = family
        torch.empty(0, device="cuda")
        major, minor = torch.cuda.get_device_capability()
        build = load_build_module()
        toolchain = build.find_toolchain(build.BuildSettings(nvrtc_root=nvrtc_root))
        image = build._compile_kernels(
            build._load_nvrtc(toolchain.nvrtc),
            toolchain.includes,
            f"sm_{major}{minor}",
            build.NATIVE / "kernels" / family.lower(),
            fast_math=False,
        )
        self.module = check_cuda(driver.cuModuleLoadData(image))
        try:
            self.functions = {
                name: check_cuda(driver.cuModuleGetFunction(self.module, name.encode())) for name in names
            }
        except Exception:
            check_cuda(driver.cuModuleUnload(self.module))
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        try:
            torch.cuda.synchronize()
        finally:
            check_cuda(driver.cuModuleUnload(self.module))

    def launch(self, name: str, total: int, tensors: tuple[Tensor, ...], *scalars: int | float) -> None:
        args = (*(tensor.data_ptr() for tensor in tensors), *scalars, total)
        if total > 2**31 - 1 or any(size > 2**31 - 1 for size in scalars if isinstance(size, int)):
            raise ValueError("The validation kernel requires 32-bit tensor dimensions")
        types = (
            (ctypes.c_uint64,) * len(tensors)
            + tuple(ctypes.c_float if isinstance(scalar, float) else ctypes.c_int for scalar in scalars)
            + (ctypes.c_int,)
        )
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
            raise ValueError(f"{self.family} needs positive spatial dimensions")
        pad_height, pad_width = (-height) % align, (-width) % align
        if pad_height >= height or pad_width >= width:
            raise ValueError(f"{self.family} reflection padding is invalid for {height}x{width}")
        if not (pad_height or pad_width):
            return source
        padded_height, padded_width = height + pad_height, width + pad_width
        target = source.new_empty((batch, channels, padded_height, padded_width))
        self.launch("pad_rgb", target.numel(), (target, source), height, width, padded_height, padded_width)
        return target
