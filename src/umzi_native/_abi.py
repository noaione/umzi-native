"""ctypes bindings for the shared native Umzi ABI."""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path

CACHE_STATUSES = ("disabled", "miss", "hit", "rebuilt", "unavailable")


class OptionABI(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("value", ctypes.c_int64)]


class OptionsABI(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("precision", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("cache_dir", ctypes.c_char_p),
        ("kernel_options", ctypes.POINTER(OptionABI)),
        ("kernel_option_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class ShapeABI(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_uint32), ("dimensions", ctypes.c_uint32 * 4)]

    @classmethod
    def from_dimensions(cls, dimensions: Sequence[int]):
        dims = tuple(map(operator.index, dimensions))
        if not 1 <= len(dims) <= 4 or any(not 0 < dim < 2**32 for dim in dims):
            raise ValueError("Expected between one and four positive dimensions within uint32 range")
        return cls(len(dims), (ctypes.c_uint32 * 4)(*dims))

    def dimensions_tuple(self) -> tuple[int, ...]:
        return tuple(self.dimensions[: self.rank])


class TensorABI(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("shape", ShapeABI),
        ("dtype", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
    ]


class IoABI(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("input", TensorABI), ("output", TensorABI)]


class ModelABI(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("kernel", ctypes.c_char_p),
        ("scale", ctypes.c_uint32),
        ("context", ctypes.c_uint32),
    ]


class InfoABI(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("cache_status", ctypes.c_uint32),
        ("prepared_input", ShapeABI),
        ("buffer_bytes", ctypes.c_uint64 * 4),
        ("startup_device_used_bytes", ctypes.c_uint64 * 4),
        ("model_load_ms", ctypes.c_double),
        ("shape_setup_ms", ctypes.c_double * 4),
    ]


class DiagnosticABI(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("name", ctypes.c_char_p), ("value", ctypes.c_int64)]


class TilingABI(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "tile_size", "batch_size", "overlap", "background", "output_format")
    ]


def bind_session_api(dll: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    shape = ctypes.POINTER(ShapeABI)
    bindings = (
        ("umzi_abi_version", [], ctypes.c_uint32),
        ("umzi_error", [], ctypes.c_char_p),
        ("umzi_error_code", [], ctypes.c_int),
        ("umzi_session_open", [ctypes.c_char_p, ctypes.POINTER(OptionsABI)], pointer),
        ("umzi_session_close", [pointer], None),
        ("umzi_session_model", [pointer, ctypes.POINTER(ModelABI)], ctypes.c_int),
        ("umzi_session_tensors", [pointer, ctypes.POINTER(IoABI)], ctypes.c_int),
        ("umzi_session_describe", [pointer, shape, ctypes.POINTER(IoABI)], ctypes.c_int),
        ("umzi_session_prepare", [pointer, shape], ctypes.c_int),
        ("umzi_session_run", [pointer, shape, pointer, ctypes.c_uint64, pointer, ctypes.c_uint64], ctypes.c_int),
        ("umzi_session_enqueue", [pointer, shape, *([ctypes.c_uint64] * 4), pointer], ctypes.c_int),
        ("umzi_session_info", [pointer, ctypes.POINTER(InfoABI)], ctypes.c_int),
        ("umzi_session_diagnostic_count", [pointer], ctypes.c_uint32),
        ("umzi_session_diagnostic", [pointer, ctypes.c_uint32, ctypes.POINTER(DiagnosticABI)], ctypes.c_int),
        ("umzi_session_cache_message", [pointer], ctypes.c_char_p),
        ("umzi_session_kernel_message", [pointer], ctypes.c_char_p),
        ("umzi_session_profile", [pointer, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32], ctypes.c_int),
    )
    for name, args, result in bindings:
        function = getattr(dll, name)
        function.argtypes = args
        function.restype = result
    if dll.umzi_abi_version() != 1:
        raise OSError("Unsupported Umzi native ABI version")
    for name in ("umzi_session_run_tiled", "umzi_session_run_tiled_variable"):
        tiled = getattr(dll, name, None)
        if tiled is None:
            continue
        tiled.argtypes = [
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(TilingABI),
            pointer,
            ctypes.c_uint64,
            pointer,
            ctypes.c_uint64,
        ]
        tiled.restype = ctypes.c_int


def check_status(dll: ctypes.CDLL, status: int) -> None:
    if status:
        message = dll.umzi_error().decode("utf-8", errors="replace")
        if status == -2:
            raise MemoryError(message)
        raise RuntimeError(message)


def open_session(
    dll: ctypes.CDLL,
    path: Path,
    device: int,
    precision: str,
    graph: bool,
    cache_dir: bytes | None,
    kernel_options: Mapping[str, int],
) -> int:
    entries = (OptionABI * len(kernel_options))(
        *(OptionABI(name.encode("utf-8"), value) for name, value in kernel_options.items())
    )
    options = OptionsABI(
        ctypes.sizeof(OptionsABI),
        device,
        {"mixed": 0, "fp16": 1, "fp32": 2}[precision],
        int(graph),
        cache_dir,
        entries,
        len(entries),
        0,
    )
    handle = dll.umzi_session_open(str(path).encode("utf-8"), ctypes.byref(options))
    if not handle:
        check_status(dll, dll.umzi_error_code())
    return handle


def read_session_info(dll: ctypes.CDLL, handle: int | None) -> InfoABI:
    info = InfoABI(ctypes.sizeof(InfoABI))
    check_status(dll, dll.umzi_session_info(handle, ctypes.byref(info)))
    return info


def session_setup(dll: ctypes.CDLL, handle: int | None, info: InfoABI) -> dict:
    diagnostics = {}
    for index in range(dll.umzi_session_diagnostic_count(handle)):
        stat = DiagnosticABI(ctypes.sizeof(DiagnosticABI))
        check_status(dll, dll.umzi_session_diagnostic(handle, index, ctypes.byref(stat)))
        diagnostics[stat.name.decode()] = stat.value
    return {
        "stages_ms": dict(
            zip(
                ("model", "shape_allocation", "plans", "allocation_warmup", "capture"),
                (info.model_load_ms, *info.shape_setup_ms),
                strict=True,
            )
        ),
        "cache_status": CACHE_STATUSES[info.cache_status],
        "cache_message": dll.umzi_session_cache_message(handle).decode("utf-8", errors="replace"),
        "kernel_message": dll.umzi_session_kernel_message(handle).decode("utf-8", errors="replace"),
        "diagnostics": diagnostics,
    }
