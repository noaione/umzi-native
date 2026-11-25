"""NumPy inference sessions for native Umzi models."""

from __future__ import annotations

import ctypes
import operator
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal, Self

import numpy as np

from ._abi import (
    IoABI,
    ModelABI,
    ShapeABI,
    TensorABI,
    TilingABI,
    check_status,
    open_session,
    read_session_info,
    session_setup,
)
from ._loader import PathArg, find_session_library, load_cuda_libraries, load_session_library
from ._model import FIGSR_FAMILY, GFISRV2_FAMILY, MAGIC, PREFIX

__all__ = ["InferenceSession", "SessionOptions", "TensorInfo"]

_DTYPE = {1: ("tensor(float)", np.dtype("float32")), 2: ("tensor(float16)", np.dtype("float16"))}


@dataclass
class SessionOptions:
    """Settings applied to every shape prepared by a session."""

    precision: str = "mixed"
    device_id: int = 0
    cache_dir: PathArg | None = None
    enable_cuda_graph: bool = True
    dll_path: PathArg | None = None
    dll_dirs: Sequence[PathArg] = ()
    kernel_options: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[str | int, ...]
    type: str


def _tensor_info(tensor: TensorABI) -> TensorInfo:
    shape = tensor.shape.dimensions_tuple()
    dynamic_names = ("batch", "channels", "height", "width") if len(shape) == 4 else ()
    return TensorInfo(
        tensor.name.decode(),
        tuple(dim or (dynamic_names[i] if dynamic_names else f"dim{i}") for i, dim in enumerate(shape)),
        _DTYPE[tensor.dtype][0],
    )


class InferenceSession:
    """Run an Umzi model on CUDA with host NumPy arrays."""

    def __init__(self, model_path: PathArg, sess_options: SessionOptions | None = None):
        options = sess_options if sess_options is not None else SessionOptions()
        if options.precision not in ("mixed", "fp16", "fp32"):
            raise ValueError("precision must be 'mixed', 'fp16', or 'fp32'")
        device_id = operator.index(options.device_id)
        if not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a non-negative CUDA device index")
        model = Path(model_path).resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Umzi model package not found: {model}")
        kernel_options = {}
        for name, value in options.kernel_options.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("A kernel option name must be a non-empty string without NUL bytes")
            number = operator.index(value)
            if not -(2**63) <= number < 2**63:
                raise ValueError(f"Kernel option {name} exceeds the int64 range")
            kernel_options[name] = number
        cache = str(Path(options.cache_dir).resolve()).encode("utf-8") if options.cache_dir else None
        self._library = find_session_library(model, options.dll_path)
        self._lock = Lock()
        if options.dll_dirs:
            with model.open("rb") as file:
                header = file.read(PREFIX.size)
            needs_cufft = (
                len(header) == PREFIX.size
                and header[:8] == MAGIC
                and PREFIX.unpack(header)[2] in (GFISRV2_FAMILY, FIGSR_FAMILY)
            )
            load_cuda_libraries(*options.dll_dirs, include_cufft=needs_cufft)
        self._dll = load_session_library(self._library)
        self._handle = open_session(
            self._dll, model, device_id, options.precision, options.enable_cuda_graph, cache, kernel_options
        )
        self._finalize = weakref.finalize(self, self._dll.umzi_session_close, self._handle)
        try:
            metadata = ModelABI(ctypes.sizeof(ModelABI))
            io = IoABI(ctypes.sizeof(IoABI))
            check_status(self._dll, self._dll.umzi_session_model(self._handle, ctypes.byref(metadata)))
            check_status(self._dll, self._dll.umzi_session_tensors(self._handle, ctypes.byref(io)))
            self._kernel = metadata.kernel.decode()
            self._scale_factor = int(metadata.scale)
            self._context = int(metadata.context)
            self._input = _tensor_info(io.input)
            self._output = _tensor_info(io.output)
            self._dtype = _DTYPE[io.input.dtype][1]
        except BaseException:
            self._finalize()
            raise

    def _ensure_open(self) -> None:
        if not self._finalize.alive:
            raise RuntimeError("The Umzi session is closed")

    def close(self) -> None:
        with self._lock:
            self._finalize()

    def __repr__(self) -> str:
        name = type(self).__name__
        # Construction can fail at any step, so a traceback may format the instance early.
        if not hasattr(self, "_kernel"):
            return f"<{name} (unopened)>"
        state = "closed" if self.closed else "open"
        return (
            f"<{name} {state} kernel={self._kernel!r} dtype={self._dtype.name} scale={self._scale_factor} "
            f"context={self._context} input={self._input.shape} output={self._output.shape}>"
        )

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def library_path(self) -> Path:
        return self._library

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def scale_factor(self) -> int:
        return self._scale_factor

    @property
    def context(self) -> int:
        return self._context

    @property
    def closed(self) -> bool:
        """Whether the session has been closed and released its device resources."""
        return not self._finalize.alive

    def get_inputs(self) -> list[TensorInfo]:
        return [self._input]

    def get_outputs(self) -> list[TensorInfo]:
        return [self._output]

    def prepare(self, shape: Sequence[int]) -> None:
        """Allocate and tune a shape ahead of the first inference call."""
        dims = ShapeABI.from_dimensions(shape)
        with self._lock:
            self._ensure_open()
            check_status(self._dll, self._dll.umzi_session_prepare(self._handle, ctypes.byref(dims)))

    def get_session_info(self) -> dict:
        """Return cache, timings, device-buffer information, and kernel diagnostics."""
        with self._lock:
            self._ensure_open()
            info = read_session_info(self._dll, self._handle)
            setup = session_setup(self._dll, self._handle, info)
            dims = info.prepared_input.dimensions_tuple()
            return {
                **setup,
                "kernel": self.kernel,
                "padded_shape": dims or None,
                "buffer_bytes": dict(
                    zip(("weights", "activations", "workspace", "rgb_io"), info.buffer_bytes, strict=True)
                ),
                "startup_device_used_bytes": dict(
                    zip(
                        ("context", "tuning_workspace", "tuned", "captured"),
                        info.startup_device_used_bytes,
                        strict=True,
                    )
                )
                if dims
                else None,
            }

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run synchronous inference and return the requested output."""
        output_name = self._output.name
        if isinstance(output_names, str) or (
            output_names is not None and list(output_names) not in ([], [output_name])
        ):
            raise ValueError(f"output_names must be None, [], or [{output_name!r}]")
        if set(input_feed) != {self._input.name}:
            raise ValueError(f"input_feed must contain one entry named {self._input.name!r}")
        sample = input_feed[self._input.name]
        if not isinstance(sample, np.ndarray) or sample.dtype != self._dtype:
            raise TypeError(f"input must be a NumPy ndarray with dtype {self._dtype}")
        shape = ShapeABI.from_dimensions(sample.shape)
        with self._lock:
            self._ensure_open()
            io = IoABI(ctypes.sizeof(IoABI))
            check_status(
                self._dll, self._dll.umzi_session_describe(self._handle, ctypes.byref(shape), ctypes.byref(io))
            )
            source = np.require(sample, requirements=("C", "A"))
            restored = np.empty(io.output.shape.dimensions_tuple(), dtype=_DTYPE[io.output.dtype][1])
            check_status(
                self._dll,
                self._dll.umzi_session_run(
                    self._handle,
                    ctypes.byref(shape),
                    source.ctypes.data,
                    source.nbytes,
                    restored.ctypes.data,
                    restored.nbytes,
                ),
            )
        return [restored]

    def run_tiled(
        self,
        image: np.ndarray,
        *,
        tile_size: int,
        batch_size: int,
        overlap: int = 0,
        tiling: Literal["fixed", "variable"] = "fixed",
        overlap_method: Literal["centered", "linear"] = "centered",
        background: str = "white",
        output_dtype: type[np.uint8] | type[np.float32] = np.uint8,
    ) -> np.ndarray:
        """Restore a contiguous HWC uint8 image with CUDA tiles.

        Overlap is the total shared width in source pixels. The default method
        blends the center 16 source pixels of each join at most. With fixed
        tiles, it also places the last tile flush and mirrors short axes.
        Variable tiling adapts edge shapes and uses replicated edge padding.
        Float32 output contains stitched values before clipping and quantization.
        """
        if tiling not in ("fixed", "variable"):
            raise ValueError("tiling must be 'fixed' or 'variable'")
        if overlap_method not in ("centered", "linear"):
            raise ValueError("overlap_method must be 'centered' or 'linear'")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError("The GPU tiler expects a uint8 NumPy image")
        if image.ndim != 3 or not image.flags.c_contiguous or image.shape[2] != self._input.shape[1]:
            raise ValueError("The GPU tiler expects a contiguous HWC image with the model's input channels")
        output_channels = self._output.shape[1]
        if not isinstance(output_channels, int) or output_channels < 1:
            raise ValueError("The GPU tiler requires a known output channel count")
        tile_size, batch_size, overlap = map(operator.index, (tile_size, batch_size, overlap))
        height, width = image.shape[:2]
        scale = self.scale_factor
        if (
            min(height, width, tile_size, batch_size, scale) < 1
            or max(height, width, tile_size) * scale >= 2**31
            or batch_size >= 2**31
            or not 0 <= overlap < (2**32 if overlap_method == "centered" else tile_size - 2 * self.context)
        ):
            raise ValueError("Invalid GPU tiling dimensions or overlap")
        if background not in ("white", "black"):
            raise ValueError("background must be 'white' or 'black'")
        dtype = np.dtype(output_dtype)
        if dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise ValueError("output_dtype must be uint8 or float32")
        options = TilingABI(
            ctypes.sizeof(TilingABI),
            tile_size,
            batch_size,
            overlap,
            255 if background == "white" else 0,
            int(dtype == np.dtype(np.float32)),
            int(overlap_method == "linear"),
        )

        with self._lock:
            self._ensure_open()
            symbol = "umzi_session_run_tiled" if tiling == "fixed" else f"umzi_session_run_tiled_{tiling}"
            tiled = getattr(self._dll, symbol, None)
            if tiled is None:
                feature = "GPU tiling" if tiling == "fixed" else f"{tiling} GPU tiling"
                raise RuntimeError(f"This Umzi native library does not support {feature}")
            if overlap > 0 and overlap_method == "centered" and self._dll.umzi_abi_version() < 2:
                raise RuntimeError(
                    "Centered overlap requires a newer Umzi library. Use overlap_method='linear' with this build."
                )
            restored = np.empty((height * scale, width * scale, output_channels), dtype=dtype)
            check_status(
                self._dll,
                tiled(
                    self._handle,
                    height,
                    width,
                    ctypes.byref(options),
                    image.ctypes.data,
                    image.nbytes,
                    restored.ctypes.data,
                    restored.nbytes,
                ),
            )
        return restored
