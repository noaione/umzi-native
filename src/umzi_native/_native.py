"""Native library loading and shared GateRV3 session bindings."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from threading import Lock

PathArg = str | os.PathLike[str]
WINDOWS = sys.platform == "win32"
LIBRARY_NAME = "gaterv3.dll" if WINDOWS else "libgaterv3.so"
PACKAGE_DIR = Path(__file__).resolve().parent
KERNEL_OPTIONS = {"tune-projections": 8, "fuse-gate-pool": 16, "fp16-qkv": 32}
CACHE_STATUSES = ("disabled", "miss", "hit", "rebuilt", "unavailable")
OPTIMIZATION_STATS = (
    "replay_tuned_shapes",
    "fused_gate_pool_sites",
    "fp16_qkv_projections",
    "split_gate_registers",
    "fused_gate_registers",
)
# The session library links these directly. cuDNN loads its remaining
# components by name at run time, so the explicit loader preloads them too.
CUDA_LIBRARIES = ("cublasLt64_13.dll", "cudnn64_9.dll") if WINDOWS else ("libcublasLt.so.13", "libcudnn.so.9")
CUDNN_COMPONENTS = "cudnn_*64_9.dll" if WINDOWS else "libcudnn_*.so.9"
_load_lock = Lock()
_session_libraries: dict[Path, ctypes.CDLL] = {}
_cuda_libraries: dict[Path, ctypes.CDLL] = {}


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
        (name, ctypes.c_uint32) for name in ("struct_size", "rgb_type", "batch", "channels", "height", "width")
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


def _load(path: Path) -> ctypes.CDLL:
    # Windows resolves a full-path load's dependencies among loaded modules,
    # beside the library, and in the system and add_dll_directory folders.
    # Linux matches NEEDED entries against loaded sonames and RUNPATH.
    return ctypes.CDLL(str(path))


def load_cuda_libraries(*directories: PathArg) -> dict[str, Path]:
    """Load cuBLASLt 13 and cuDNN 9 from explicit directories.

    Call this before opening a session when no other module, such as CUDA
    PyTorch, has already loaded these libraries into the process. Each
    library is taken from the first listed directory that contains it, and
    the cuDNN components beside ``cudnn`` are preloaded because cuDNN opens
    them by name later. Neither ``PATH`` nor ``LD_LIBRARY_PATH`` is changed.

    Returns the resolved path of each loaded library, keyed by file name.
    """
    folders = [Path(path).resolve() for path in directories]
    if not folders:
        raise ValueError("Supply at least one directory containing cuBLASLt 13 or cuDNN 9")
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"CUDA library directory does not exist: {folder}")
    required: list[Path] = []
    for name in CUDA_LIBRARIES:
        path = next((folder / name for folder in folders if (folder / name).is_file()), None)
        if path is None:
            searched = ", ".join(str(folder) for folder in folders)
            raise FileNotFoundError(f"{name} was not found in {searched}")
        required.append(path)
    cublas, cudnn = required
    # Components that cuBLASLt and cuDNN load by name when they are present.
    optional = sorted(
        path
        for pattern in (
            ("cublas64_13.dll", "cudart64_13.dll", "nvJitLink_13*.dll", "nvrtc*64_13*.dll")
            if WINDOWS
            else ("libcublas.so.13", "libcudart.so.13", "libnvJitLink.so.13", "libnvrtc*.so.13*")
        )
        for path in cublas.parent.glob(pattern)
    )
    components = sorted(cudnn.parent.glob(CUDNN_COMPONENTS))
    # cuDNN's engine libraries import the graph library, so load it first.
    components.sort(key=lambda path: "graph" not in path.name)
    loaded = {}
    with _load_lock:
        for path in (*optional, cublas, *components, cudnn):
            if path not in _cuda_libraries:
                try:
                    _cuda_libraries[path] = _load(path)
                except OSError as error:
                    raise OSError(f"Could not load {path}: {error}") from error
            loaded[path.name] = path
    return loaded


def find_session_library(model: Path, explicit: PathArg | None = None) -> Path:
    """Select the explicit library, the installed package copy, or the one beside the model."""
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GateRV3 library not found: {path}")
        return path
    for directory in (PACKAGE_DIR, model.parent):
        path = directory / LIBRARY_NAME
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"{LIBRARY_NAME} was not found in the installed package ({PACKAGE_DIR}) or beside "
        f"the model ({model.parent}). Install a platform wheel or set SessionOptions(dll_path=...)."
    )


def load_session_library(path: Path) -> ctypes.CDLL:
    """Load and bind a GateRV3 session library once per process."""
    if not path.is_file():
        raise FileNotFoundError(f"GateRV3 library not found: {path}")
    with _load_lock:
        library = _session_libraries.get(path)
        if library is not None:
            return library
        try:
            library = _load(path)
        except OSError as error:
            libraries = " and ".join(CUDA_LIBRARIES)
            raise OSError(
                f"Could not load {path} or one of its dependencies. It needs the CUDA driver, {libraries}. "
                "Import CUDA PyTorch first or call umzi_native.load_cuda_libraries() "
                "with the directories that contain them."
            ) from error
        try:
            _bind_session_api(library)
        except AttributeError as error:
            raise OSError(f"{path} does not expose the packed GateRV3 session API") from error
        _session_libraries[path] = library
    return library


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
        "cache_message": dll.gt_session_cache_message(handle).decode("utf-8", errors="replace"),
        "fused_silu_count": info.fused_silu_count,
        "fusion_message": dll.gt_session_fusion_message(handle).decode("utf-8", errors="replace"),
        "optimization": dict(zip(OPTIMIZATION_STATS, info.optimization_stats, strict=True)),
    }
