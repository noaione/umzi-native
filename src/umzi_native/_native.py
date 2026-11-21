"""Windows DLL discovery and shared GateRV3 session bindings."""

from __future__ import annotations

import ctypes
import os
import site
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

PathArg = str | os.PathLike[str]
KERNEL_OPTIONS = {"tune-projections": 8, "fuse-gate-pool": 16, "fp16-qkv": 32}
CACHE_STATUSES = ("disabled", "miss", "hit", "rebuilt", "unavailable")
OPTIMIZATION_STATS = (
    "replay_tuned_shapes",
    "fused_gate_pool_sites",
    "fp16_qkv_projections",
    "split_gate_registers",
    "fused_gate_registers",
)
_load_lock = Lock()


class OptionsABI(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("cache_dir", ctypes.c_char_p),
    ]


class IoABI(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "rgb_type", "batch", "channels", "height", "width")
    ] + [("input_bytes", ctypes.c_uint64), ("output_bytes", ctypes.c_uint64)]


class InfoABI(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "cache_status",
            "batch",
            "padded_height",
            "padded_width",
            "reserved",
        )
    ] + [
        ("buffer_bytes", ctypes.c_uint64 * 4),
        ("startup_device_used_bytes", ctypes.c_uint64 * 4),
        ("model_load_ms", ctypes.c_double),
        ("shape_setup_ms", ctypes.c_double * 4),
        ("optimization_stats", ctypes.c_int32 * 5),
        ("fused_silu_count", ctypes.c_int32),
    ]


def _runtime_dirs(dll_path: Path, extra: Sequence[PathArg]) -> list[Path]:
    provided = [Path(path).resolve() for path in extra]
    for path in provided:
        if not path.is_dir():
            raise FileNotFoundError(f"CUDA DLL directory does not exist: {path}")

    sites = [*site.getsitepackages(), site.getusersitepackages()]
    wheels = [
        Path(base) / "nvidia" / suffix
        for base in sites
        for suffix in ("cu13/bin/x86_64", "cudnn/bin")
    ]
    search = [
        dll_path.parent,
        *provided,
        *wheels,
        *(Path(value) for value in os.get_exec_path() if value),
    ]
    directories = dict.fromkeys(provided)
    for name in ("cublasLt64_13.dll", "cudnn64_9.dll"):
        path = next(
            (candidate for candidate in search if (candidate / name).is_file()), None
        )
        if path is not None and path != dll_path.parent:
            directories[path.resolve()] = None
    return list(directories)


def _preload_runtime_components() -> None:
    # cuDNN uses LoadLibraryA for some components after the DLL is loaded.
    # Use the already selected installations without changing process PATH.
    kernel = ctypes.WinDLL("kernel32")
    loaded = kernel.GetModuleHandleW
    loaded.argtypes = [ctypes.c_wchar_p]
    loaded.restype = ctypes.c_void_p
    filename = kernel.GetModuleFileNameW
    filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    filename.restype = ctypes.c_uint32

    def directory(name: str) -> Path:
        handle = loaded(name)
        path = ctypes.create_unicode_buffer(32768)
        if not handle or not 0 < filename(handle, path, len(path)) < len(path):
            raise OSError(f"Could not locate the loaded library {name}")
        return Path(path.value).parent

    def preload(folder: Path, patterns: tuple[str, ...]) -> None:
        for pattern in patterns:
            for path in sorted(folder.glob(pattern)):
                if not loaded(path.name):
                    ctypes.WinDLL(str(path))

    cuda = directory("cublasLt64_13.dll")
    cudnn = directory("cudnn64_9.dll")
    graph = "cudnn_graph64_9.dll"
    if not loaded(graph) and not (cudnn / graph).is_file():
        raise OSError(f"Missing {graph} in the cuDNN installation at {cudnn}")
    preload(
        cuda,
        ("cublas64_13.dll", "cudart64_13.dll", "nvJitLink_13*.dll", "nvrtc*64_13*.dll"),
    )
    preload(cudnn, ("cudnn*64_9.dll",))


def load_session_library(
    dll_path: Path, extra: Sequence[PathArg] = ()
) -> tuple[ctypes.CDLL, ExitStack]:
    if sys.platform != "win32":
        raise OSError("GateRV3 inference currently requires the Windows CUDA DLL")
    if not dll_path.is_file():
        raise FileNotFoundError(f"GateRV3 DLL not found: {dll_path}")
    stack = ExitStack()
    try:
        with _load_lock:
            for path in _runtime_dirs(dll_path, extra):
                stack.enter_context(os.add_dll_directory(str(path)))
            try:
                dll = ctypes.CDLL(str(dll_path))
                _preload_runtime_components()
            except OSError as error:
                raise OSError(
                    f"Could not load the runtime for {dll_path}. Check the CUDA driver, cuBLASLt 13, "
                    "cuDNN 9 and the MSVC runtime. SessionOptions.dll_dirs can locate CUDA DLLs."
                ) from error
        try:
            _bind_session_api(dll)
        except AttributeError as error:
            raise OSError(
                f"{dll_path} does not expose the packed GateRV3 session API"
            ) from error
    except BaseException:
        stack.close()
        raise
    return dll, stack


def _bind_session_api(dll: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    dimensions = [pointer, *([ctypes.c_uint32] * 3)]
    bindings = (
        ("gt_error", [], ctypes.c_char_p),
        ("gt_session_open", [ctypes.c_char_p, ctypes.POINTER(OptionsABI)], pointer),
        ("gt_session_close", [pointer], None),
        ("gt_session_describe", [*dimensions, ctypes.POINTER(IoABI)], ctypes.c_int),
        ("gt_session_prepare", dimensions, ctypes.c_int),
        (
            "gt_session_run",
            [*dimensions, *([pointer, ctypes.c_uint64] * 2)],
            ctypes.c_int,
        ),
        (
            "gt_session_enqueue",
            [*dimensions, *([ctypes.c_uint64] * 4), pointer],
            ctypes.c_int,
        ),
        ("gt_session_info", [pointer, ctypes.POINTER(InfoABI)], ctypes.c_int),
        ("gt_session_cache_message", [pointer], ctypes.c_char_p),
        ("gt_session_fusion_message", [pointer], ctypes.c_char_p),
        (
            "gt_session_profile",
            [pointer, ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            ctypes.c_int,
        ),
    )
    for name, args, result in bindings:
        function = getattr(dll, name)
        function.argtypes = args
        function.restype = result


def check_status(dll: ctypes.CDLL, status: int) -> None:
    if status:
        message = dll.gt_error().decode("utf-8", errors="replace")
        if status == -2:
            raise MemoryError(message)
        raise RuntimeError(message)


def read_session_info(dll: ctypes.CDLL, handle: int | None) -> InfoABI:
    info = InfoABI()
    info.struct_size = ctypes.sizeof(info)
    check_status(dll, dll.gt_session_info(handle, ctypes.byref(info)))
    return info


def session_setup(dll: ctypes.CDLL, handle: int | None, info: InfoABI) -> dict:
    return {
        "stages_ms": dict(
            zip(
                ("model", "shape_allocation", "plans", "allocation_warmup", "capture"),
                (info.model_load_ms, *info.shape_setup_ms),
                strict=True,
            )
        ),
        "cache_status": CACHE_STATUSES[info.cache_status],
        "cache_message": dll.gt_session_cache_message(handle).decode(
            "utf-8", errors="replace"
        ),
        "fused_silu_count": info.fused_silu_count,
        "fusion_message": dll.gt_session_fusion_message(handle).decode(
            "utf-8", errors="replace"
        ),
        "optimization": dict(
            zip(OPTIMIZATION_STATS, info.optimization_stats, strict=True)
        ),
    }
