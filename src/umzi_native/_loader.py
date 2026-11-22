"""Umzi runtime discovery and explicit CUDA dependency loading."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from threading import Lock

from ._abi import bind_session_api

PathArg = str | os.PathLike[str]
WINDOWS = sys.platform == "win32"
LIBRARY_NAME = "umzi_native.dll" if WINDOWS else "libumzi_native.so"
PACKAGE_DIR = Path(__file__).resolve().parent
# The session library links these directly. cuDNN loads its remaining
# components by name at run time, so the explicit loader preloads them too.
CUDA_LIBRARIES = ("cublasLt64_13.dll", "cudnn64_9.dll") if WINDOWS else ("libcublasLt.so.13", "libcudnn.so.9")
CUDNN_COMPONENTS = "cudnn_*64_9.dll" if WINDOWS else "libcudnn_*.so.9"
CUFFT_LIBRARY = "cufft64_12.dll" if WINDOWS else "libcufft.so.12"
_load_lock = Lock()
_session_libraries: dict[Path, ctypes.CDLL] = {}
_cuda_libraries: dict[Path, ctypes.CDLL] = {}


def _load(path: Path) -> ctypes.CDLL:
    # Windows resolves a full-path load's dependencies among loaded modules,
    # beside the library, and in the system and add_dll_directory folders.
    # Linux matches NEEDED entries against loaded sonames and RUNPATH.
    return ctypes.CDLL(str(path))


def load_cuda_libraries(*directories: PathArg, include_cufft: bool = False) -> dict[str, Path]:
    """Load cuBLASLt 13 and cuDNN 9 from explicit directories.

    Call this before opening a session when no other module, such as CUDA
    PyTorch, has already loaded these libraries into the process. Each
    library is taken from the first listed directory that contains it, and
    the cuDNN components beside ``cudnn`` are preloaded because cuDNN opens
    them by name later. Neither ``PATH`` nor ``LD_LIBRARY_PATH`` is changed.

    With `include_cufft`, also preload cuFFT 12 if it is in the directories.
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
    fft = (
        next((folder / CUFFT_LIBRARY for folder in folders if (folder / CUFFT_LIBRARY).is_file()), None)
        if include_cufft
        else None
    )
    loaded = {}
    with _load_lock:
        for path in (*optional, cublas, *components, cudnn, *((fft,) if fft else ())):
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
            raise FileNotFoundError(f"Umzi library not found: {path}")
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
    """Load and bind a Umzi session library once per process."""
    if not path.is_file():
        raise FileNotFoundError(f"Umzi library not found: {path}")
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
            bind_session_api(library)
        except AttributeError as error:
            raise OSError(f"{path} does not expose the Umzi session API") from error
        _session_libraries[path] = library
    return library
