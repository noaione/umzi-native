"""NumPy inference sessions for packed GateRV3 models."""

from __future__ import annotations

import ctypes
import operator
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Self

import numpy as np

from ._native import (
    KERNEL_OPTIONS,
    IoABI,
    OptionsABI,
    PathArg,
    check_status,
    load_session_library,
    read_session_info,
    session_setup,
)

__all__ = ["InferenceSession", "SessionOptions", "TensorInfo"]


@dataclass
class SessionOptions:
    """Settings applied to every shape prepared by a session."""

    precision: str = "mixed"
    device_id: int = 0
    cache_dir: PathArg | None = None
    enable_cuda_graph: bool = True
    fuse_silu: bool = False
    tune_projections: bool = False
    fuse_gate_pool: bool = False
    fp16_qkv: bool = False
    dll_path: PathArg | None = None
    dll_dirs: Sequence[PathArg] = ()


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[str | int, ...]
    type: str


def _dimensions(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 4:
        raise ValueError("Expected a four-dimensional RGB NCHW shape")
    n, c, h, w = map(operator.index, shape)
    if c != 3 or any(size < 1 or size >= 2**32 for size in (n, h, w)):
        raise ValueError(
            "Expected three RGB channels and positive NCHW dimensions within uint32 range"
        )
    return n, h, w


class InferenceSession:
    """Run a packed GateRV3 model on CUDA with host NumPy arrays."""

    def __init__(self, model_path: PathArg, sess_options: SessionOptions | None = None):
        options = sess_options if sess_options is not None else SessionOptions()
        if options.precision not in ("mixed", "fp16"):
            raise ValueError("precision must be 'mixed' or 'fp16'")
        device_id = operator.index(options.device_id)
        if not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a non-negative CUDA device index")
        model = Path(model_path).resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Packed GateRV3 model not found: {model}")
        dll = (
            Path(options.dll_path).resolve()
            if options.dll_path is not None
            else model.with_name("gaterv3.dll")
        )
        self._dtype = np.dtype("float16" if options.precision == "fp16" else "float32")
        self._lock = Lock()
        flags = (
            int(options.enable_cuda_graph)
            | (2 if options.precision == "fp16" else 0)
            | (4 if options.fuse_silu else 0)
        )
        flags |= sum(
            value
            for name, value in KERNEL_OPTIONS.items()
            if getattr(options, name.replace("-", "_"))
        )
        cache = (
            str(Path(options.cache_dir).resolve()).encode("utf-8")
            if options.cache_dir
            else None
        )
        config = OptionsABI(ctypes.sizeof(OptionsABI), device_id, flags, 0, cache)
        self._dll, cleanup = load_session_library(dll, options.dll_dirs)
        try:
            self._handle = self._dll.gt_session_open(
                str(model).encode("utf-8"), ctypes.byref(config)
            )
            if not self._handle:
                check_status(self._dll, -1)
            cleanup.callback(self._dll.gt_session_close, self._handle)
            self._finalize = weakref.finalize(self, cleanup.close)
        except BaseException:
            cleanup.close()
            raise

    def _ensure_open(self) -> None:
        if not self._finalize.alive:
            raise RuntimeError("The GateRV3 session is closed")

    def close(self) -> None:
        with self._lock:
            self._finalize()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _tensor_info(self, name: str) -> TensorInfo:
        dtype = (
            "tensor(float16)" if self._dtype == np.dtype("float16") else "tensor(float)"
        )
        return TensorInfo(name, ("batch", 3, "height", "width"), dtype)

    def get_inputs(self) -> list[TensorInfo]:
        return [self._tensor_info("input")]

    def get_outputs(self) -> list[TensorInfo]:
        return [self._tensor_info("output")]

    def prepare(self, shape: Sequence[int]) -> None:
        """Allocate and tune a shape ahead of the first inference call."""
        dims = _dimensions(shape)
        with self._lock:
            self._ensure_open()
            check_status(self._dll, self._dll.gt_session_prepare(self._handle, *dims))

    def get_session_info(self) -> dict:
        """Return cache, timings, optimization and device-buffer information."""
        with self._lock:
            self._ensure_open()
            info = read_session_info(self._dll, self._handle)
            return {
                **session_setup(self._dll, self._handle, info),
                "padded_shape": (info.batch, 3, info.padded_height, info.padded_width)
                if info.batch
                else None,
                "buffer_bytes": dict(
                    zip(
                        ("weights", "activations", "workspace", "rgb_io"),
                        info.buffer_bytes,
                        strict=True,
                    )
                ),
                "startup_device_used_bytes": dict(
                    zip(
                        ("context", "tuning_workspace", "tuned", "captured"),
                        info.startup_device_used_bytes,
                        strict=True,
                    )
                )
                if info.batch
                else None,
            }

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run synchronous inference and return the requested RGB output."""
        if isinstance(output_names, str) or (
            output_names is not None and list(output_names) not in ([], ["output"])
        ):
            raise ValueError("output_names must be None, [], or ['output']")
        if set(input_feed) != {"input"}:
            raise ValueError("input_feed must contain one entry named 'input'")
        sample = input_feed["input"]
        if not isinstance(sample, np.ndarray) or sample.dtype != self._dtype:
            raise TypeError(f"input must be a NumPy ndarray with dtype {self._dtype}")
        dims = _dimensions(sample.shape)
        with self._lock:
            self._ensure_open()
            io = IoABI()
            io.struct_size = ctypes.sizeof(io)
            check_status(
                self._dll,
                self._dll.gt_session_describe(self._handle, *dims, ctypes.byref(io)),
            )
            source = np.require(sample, requirements=("C", "A"))
            restored = np.empty(
                (io.batch, io.channels, io.height, io.width), dtype=self._dtype
            )
            check_status(
                self._dll,
                self._dll.gt_session_run(
                    self._handle,
                    *dims,
                    source.ctypes.data,
                    source.nbytes,
                    restored.ctypes.data,
                    restored.nbytes,
                ),
            )
        return [restored]
