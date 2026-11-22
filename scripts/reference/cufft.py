"""Direct FP32 cuFFT transforms for the private GFISRv2 validator."""

from __future__ import annotations

import ctypes as ct
import sys
from math import sqrt
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

INT_PTR = ct.POINTER(ct.c_int)
CUFFT_R2C = 0x2A
CUFFT_C2R = 0x2C


def _load_library(path: Path | None) -> ct.CDLL:
    windows = sys.platform == "win32"
    name = "cufft64_12.dll" if windows else "libcufft.so.12"
    relative = Path("bin/x86_64" if windows else "lib") / name
    installed = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13" / relative
    return (ct.WinDLL if windows else ct.CDLL)(str(path or (installed if installed.is_file() else name)))


class Cufft:
    """One pair of out-of-place FP32 plans, reused on the current Torch stream."""

    def __init__(self, library: Path | None):
        self.library = _load_library(library)
        self.stream = torch.cuda.current_stream()
        self.shape: tuple[int, ...] = ()
        self.handles: list[ct.c_int] = []
        self.workspace: list[Tensor] = []
        ptr = ct.c_void_p
        integer = ct.c_int
        for name, args in {
            "cufftGetVersion": [INT_PTR],
            "cufftCreate": [INT_PTR],
            "cufftSetAutoAllocation": [integer, integer],
            "cufftMakePlanMany": [
                integer,
                integer,
                INT_PTR,
                INT_PTR,
                integer,
                integer,
                INT_PTR,
                integer,
                integer,
                integer,
                integer,
                ct.POINTER(ct.c_size_t),
            ],
            "cufftSetWorkArea": [integer, ptr],
            "cufftSetStream": [integer, ptr],
            "cufftExecR2C": [integer, ptr, ptr],
            "cufftExecC2R": [integer, ptr, ptr],
            "cufftDestroy": [integer],
        }.items():
            symbol = getattr(self.library, name)
            symbol.argtypes = args
            symbol.restype = integer

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self._release()

    def _call(self, name: str, *args: object) -> None:
        status = getattr(self.library, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with cuFFT status {status}")

    def version(self) -> int:
        version = ct.c_int()
        self._call("cufftGetVersion", ct.byref(version))
        return version.value

    def _release(self) -> None:
        if self.handles:
            handles, self.handles = self.handles, []
            self.shape = ()
            try:
                self.stream.synchronize()
            finally:
                statuses = [self.library.cufftDestroy(handle) for handle in handles]
                self.workspace.clear()
                if any(statuses):
                    raise RuntimeError(f"cufftDestroy failed with cuFFT statuses {statuses}")

    def _prepare(self, shape: tuple[int, ...], source: Tensor) -> None:
        if source.device != self.stream.device or torch.cuda.current_stream() != self.stream:
            raise ValueError("The private cuFFT runner requires its original Torch CUDA stream")
        if self.shape == shape:
            return
        self._release()
        batch, channels, height, width = shape
        dimensions = (ct.c_int * 2)(height, width)
        try:
            for kind in (CUFFT_R2C, CUFFT_C2R):
                handle, size = ct.c_int(), ct.c_size_t()
                self._call("cufftCreate", ct.byref(handle))
                self.handles.append(handle)
                self._call("cufftSetAutoAllocation", handle, 0)
                self._call(
                    "cufftMakePlanMany",
                    handle,
                    2,
                    dimensions,
                    None,
                    1,
                    0,
                    None,
                    1,
                    0,
                    kind,
                    batch * channels,
                    ct.byref(size),
                )
                self._call("cufftSetStream", handle, self.stream.cuda_stream)
                if size.value:
                    workspace = torch.empty(size.value, dtype=torch.uint8, device=source.device)
                    self.workspace.append(workspace)
                    self._call("cufftSetWorkArea", handle, workspace.data_ptr())
        except Exception:
            self._release()
            raise
        self.shape = shape

    def forward(self, source: Tensor) -> Tensor:
        if source.ndim != 4 or source.dtype != torch.float32 or not source.is_cuda:
            raise ValueError("cuFFT forward expects an NCHW CUDA FP32 tensor")
        batch, channels, height, width = source.shape
        self._prepare(tuple(source.shape), source)
        source = source.contiguous()
        result = torch.empty((batch, channels, height, width // 2 + 1), dtype=torch.complex64, device=source.device)
        self._call("cufftExecR2C", self.handles[0], source.data_ptr(), result.data_ptr())
        torch.view_as_real(result).mul_(1 / sqrt(height * width))
        return result

    def inverse(self, source: Tensor, height: int, width: int) -> Tensor:
        if (
            source.ndim != 4
            or source.dtype != torch.complex64
            or not source.is_cuda
            or source.shape[-2:] != (height, width // 2 + 1)
        ):
            raise ValueError("cuFFT inverse expects a matching CUDA complex64 half-spectrum")
        shape = (*source.shape[:2], height, width)
        self._prepare(shape, source)
        # Out-of-place C2R writes into the complex input.
        spectrum = source.clone(memory_format=torch.contiguous_format)
        result = torch.empty(shape, dtype=torch.float32, device=source.device)
        self._call("cufftExecC2R", self.handles[1], spectrum.data_ptr(), result.data_ptr())
        return result.mul_(1 / sqrt(height * width))
