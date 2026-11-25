# Installation and CUDA setup

## Requirements

- **Python:** 3.12 or newer
- **GPU:** NVIDIA GPU with compute capability 8.6 or newer (Ampere / RTX 30 series and later)
- **Runtime libraries:** CUDA 13-compatible NVIDIA driver, cuBLASLt 13, and cuDNN 9 (plus cuFFT 12 for **GFISRv2** and **FiGSR**)
- **Windows:** Visual C++ Redistributable (usually already installed with Python or Visual Studio)

## Install

### Prebuilt wheels

Wheels are built for Windows x64 and Linux `manylinux_2_34_x86_64` (glibc 2.34+, OpenSSL 3) and published to the private package index:

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ umzi-native
```

To export `.safetensors` / `.pth` / `.pt` / `.ckpt` checkpoints into `.ummodel` files, install the `exporter` extra. This adds Resselt and PyTorch for CPU-side conversion; the base package only depends on NumPy.

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ "umzi-native[exporter]"
```

You can also download development wheels or standalone native SDK archives directly from [GitHub Actions](../.github/workflows/build.yml) and install them with `python -m pip install PATH_TO_WHEEL`.

## CUDA library loading

`umzi_native` loads the native runtime (`umzi_native.dll` on Windows, `libumzi_native.so` on Linux) in this order:

1. `SessionOptions(dll_path=...)`
2. The installed `umzi_native` package directory
3. The directory containing the `.ummodel` file

The wrapper does not search system CUDA folders or modify `PATH` / `LD_LIBRARY_PATH`. Before opening an `InferenceSession`, cuBLASLt 13 and cuDNN 9 (and cuFFT 12 for GFISRv2 and FiGSR) must be loaded into the process using either method below.

### Option 1: Import PyTorch first

If you have CUDA-enabled PyTorch installed (built for CUDA 13 / cuDNN 9), importing it before opening a session loads the bundled libraries from `torch/lib`:

```python
import torch
import umzi_native as umzi
```

### Option 2: Load standalone NVIDIA libraries

If you do not want PyTorch in your inference environment, point `SessionOptions(dll_dirs=...)` at the directories containing the NVIDIA shared libraries (for example, from `nvidia-cublas`, `nvidia-cudnn-cu13`, and `nvidia-cufft` wheels):

```python
from pathlib import Path
import umzi_native as umzi

options = umzi.SessionOptions(
    dll_dirs=[
        Path("path/to/nvidia/cu13/bin/x86_64"),   # Linux: .../cu13/lib
        Path("path/to/nvidia/cudnn/bin"),         # Linux: .../cudnn/lib
    ],
)
```

`SessionOptions(dll_dirs=...)` automatically preloads cuFFT 12 when opening a `gfisrv2` or `figsr` package. If you prefer to load libraries once upfront without passing `dll_dirs` to every session, call `load_cuda_libraries()` directly:

```python
umzi.load_cuda_libraries(
    "path/to/nvidia/cu13/bin/x86_64",
    "path/to/nvidia/cudnn/bin",
    include_cufft=True,  # Needed if you will open GFISRv2 or FiGSR models
)
```

If required libraries are missing, opening a session raises `OSError` naming the missing file.

## Building from source

Building a wheel or installing from a checkout compiles the native library via [Hatchling](https://hatch.pypa.io/latest/config/build/) and [hatch_build.py](../hatch_build.py). The build hook:

1. Compiles `kernels.cu` across all six kernel families (`gaterv3`, `gaterv2`, `moesr`, `gfisrv2`, `smosr`, `figsr`) using CUDA 13.0 NVRTC into `sm_86`, `sm_89`, and `sm_120` cubins plus a `compute_86` PTX fallback.
2. Compiles each family's `runner.cpp` and `native/runtime/` with MSVC on Windows or `c++` on Linux into `umzi_native.dll` / `libumzi_native.so`.
3. Tags the wheel `py3-none-<platform>` (`ctypes` ABI, no Python version dependency).

### Build prerequisites

- **Windows:** Visual Studio with MSVC x64 build tools (detected via `vswhere`, unless `cl.exe`, `INCLUDE`, and `LIB` are already active).
- **Linux:** A C++17 compiler (`CXX`, default `c++`), OpenSSL 3 headers, and `libcrypto.so.3`. No local CUDA driver is needed at link time (`libcuda.so.1` is referenced by soname).

CUDA 13 headers, CUDA 13.0 NVRTC, cuBLASLt, and cuDNN link targets are pulled into isolated builds automatically from the NVIDIA wheels listed in `[build-system]` in [pyproject.toml](../pyproject.toml).

```powershell
# Build and install from checkout
python -m pip install .

# Include the checkpoint exporter
python -m pip install ".[exporter]"

# Build a wheel without installing
python -m pip wheel . --no-deps -w dist
```

### Editable installs and standalone native builds

`pip install -e .` skips compiling the native library. For local development or standalone C/CLI bundles, run `scripts/build_native.py` directly (install the CUDA build wheels from `pyproject.toml` into that Python environment first) and pass the output library via `SessionOptions(dll_path=...)` or place it beside the `.ummodel` file:

```powershell
python scripts/build_native.py --output output/native
```

If your Python environment has a newer NVRTC than CUDA 13.0, install `nvidia-cuda-nvrtc>=13.0.88,<13.1` into a separate folder and pass `--nvrtc-root`:

```powershell
python -m pip install --no-deps --target output/cuda-13.0-nvrtc "nvidia-cuda-nvrtc>=13.0.88,<13.1"
python scripts/build_native.py --nvrtc-root output/cuda-13.0-nvrtc/nvidia/cu13 --output output/native
```

### Build environment variables

Wheel builds can be configured through environment variables:

| Variable | Effect |
| --- | --- |
| `UMZI_CUDA_ARCHES` | Space- or comma-separated cubin targets (for example `sm_86,sm_120`). Defaults to `sm_86 sm_89 sm_120`. |
| `UMZI_CUDA_ROOT`, `UMZI_CUDNN_ROOT` | Override the `nvidia/cu13` and `nvidia/cudnn` trees used instead of build-environment wheels. |
| `UMZI_NVRTC_ROOT` | Use CUDA 13.0 NVRTC from a separate `nvidia/cu13` tree. |
| `UMZI_CRT_INCLUDE`, `UMZI_CCCL_INCLUDE` | Override CRT and CCCL include directories. |
| `UMZI_OPENSSL_INCLUDE` | Linux OpenSSL include directories, separated by `os.pathsep`. |
| `UMZI_CUDA_DRIVER_LIB` | Path to Linux `libcuda.so.1` to link against. |
| `UMZI_NATIVE_BUILD_DIR` | Keep intermediate build artifacts and the `umzi_native` CLI binary in this directory. |
| `UMZI_NATIVE_LIBRARY` | Skip compilation and bundle a prebuilt `umzi_native.dll` / `libumzi_native.so`. |
| `UMZI_WHEEL_PLATFORM` | Override the wheel platform tag (defaults to `AUDITWHEEL_PLAT` or `sysconfig.get_platform()`). |
