# GateRV3 Python inference

The runtime source lives in `src/umzi_native/`. Each native kernel
family has its own directory under `native/`, starting with
`native/kernels/gaterv3/`. The Torch reference and checkpoint loader live under
`scripts/reference/` for export and validation.

`umzi_native` runs packed `.ummodel` files with NumPy through the native
Umzi runtime: `umzi_native.dll` on Windows or `libumzi_native.so` on Linux. It
uses the familiar ONNX Runtime pattern of opening a session, inspecting
named inputs and outputs, and calling `run` with an input dictionary. The
package depends only on NumPy. The library needs the CUDA driver,
cuBLASLt 13 and cuDNN 9, which the package does not ship. On Windows it
also needs the MSVC runtime.
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model)

## Install

Installing the package compiles the native library. `pip install .` from
the checkout, from an sdist, or from a Git URL builds a wheel with
[Hatchling](https://hatch.pypa.io/latest/config/build/), and Hatchling runs
the build hook in `hatch_build.py` at the repository root. The hook:

1. Compiles `native/kernels/gaterv3/kernels.cu` with CUDA 13.0 NVRTC into `sm_86`,
   `sm_89` and `sm_120` cubins plus `compute_86` PTX, and embeds them in a
   generated `cuda_images.h`. CUDA 13.0 emits PTX ISA 9.0 so the fallback
   can run on drivers that support the CUDA 13.0 baseline.
2. Compiles `native/kernels/gaterv3/runner.cpp` and links `umzi_native.dll` with MSVC on
   Windows, or `libumzi_native.so` with `c++` on Linux.
3. Adds the library to the wheel as `umzi_native/umzi_native.dll` or
   `umzi_native/libumzi_native.so`, and tags the wheel `py3-none-<platform>`,
   such as `win_amd64` or `linux_x86_64`. The library is loaded through
   ctypes, so the tag names no Python version or ABI.

The package does not use setuptools or `setup.py`. The CUDA 13 headers,
NVRTC, and the cuBLASLt and cuDNN link targets come from NVIDIA wheels in
`[build-system] requires`. pip installs them into the isolated build
environment, and they are not runtime dependencies. Hatchling cannot ask a
script-defined hook for its dependencies, so `pyproject.toml` repeats
`BUILD_REQUIREMENTS` from `hatch_build.py`. Keep the two lists in sync.

The build machine needs:

- Windows: Visual Studio with the MSVC x64 tools. The hook finds them
  with `vswhere`, unless `cl.exe`, `INCLUDE` and `LIB` are already set.
- Linux: a C++17 compiler (`CXX`, default `c++`), the OpenSSL 3 headers,
  and `libcrypto.so.3`. A CUDA driver is not required to link; the library
  references `libcuda.so.1` by name.

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip install .                 # builds the DLL, then installs
& $py -m pip wheel . --no-deps -w dist  # or keep the platform wheel
& $py scripts/export_native.py --output output/native/gaterv3.ummodel
```

On this machine, an isolated `pip wheel .` took about 30 seconds on Windows
and 95 seconds under WSL, most of it in NVRTC. The sdist contains `src/`,
`native/`, `hatch_build.py` and `pyproject.toml`, so installing an sdist
builds the same way. The wheel excludes any `.dll` or `.so` left in
`src/umzi_native/` and bundles only the library the hook just built.

Wheel builds read these optional environment variables, because build
frontends pass no command-line arguments to the hook:

| Variable | Effect |
| --- | --- |
| `UMZI_CUDA_ARCHES` | Cubin targets separated by spaces or commas, such as `sm_86,sm_120`. Defaults to all three. |
| `UMZI_CUDA_ROOT`, `UMZI_CUDNN_ROOT` | Use these `nvidia/cu13` and `nvidia/cudnn` trees instead of the build-environment wheels. |
| `UMZI_NVRTC_ROOT` | Use the CUDA 13.0 NVRTC wheel from this separate `nvidia/cu13` tree. |
| `UMZI_CRT_INCLUDE`, `UMZI_CCCL_INCLUDE` | Use these CRT and CCCL header roots. |
| `UMZI_OPENSSL_INCLUDE` | Linux OpenSSL include directories, separated by `os.pathsep`. |
| `UMZI_CUDA_DRIVER_LIB` | Linux `libcuda.so.1` to link against. |
| `UMZI_NATIVE_BUILD_DIR` | Build in this directory and keep it, including the `umzi_native` CLI linked from the same object. |
| `UMZI_NATIVE_LIBRARY` | Skip compilation and bundle this prebuilt library. |

Editable installs (`pip install -e .`) skip the native build. Build the
library with `scripts/build_native.py` and pass its path as
`SessionOptions(dll_path=...)`.

`scripts/build_native.py` runs the same build without packaging, for a
bundle directory or the benchmarks. It loads `hatch_build.py` and takes the
overrides as flags. Outside an isolated build it uses the NVIDIA wheels
installed in its own Python, so install the CUDA build wheels there first:

```powershell
& $py -m pip install "nvidia-cuda-runtime>=13.4,<13.5" "nvidia-cuda-nvrtc>=13.0.88,<13.1" `
  "nvidia-cuda-crt>=13.4,<13.5" "nvidia-cuda-cccl>=13.3,<14" "nvidia-cublas>=13.7,<14" "nvidia-cudnn-cu13>=9.26,<10"
& $py scripts/build_native.py --output output/umzi-native-windows
```

To keep a newer NVRTC in the system Python, install the CUDA 13.0 NVRTC
wheel in a separate directory and point the standalone build at it:

```powershell
& $py -m pip install --no-deps --target output/cuda-13.0-nvrtc "nvidia-cuda-nvrtc>=13.0.88,<13.1"
& $py scripts/build_native.py --nvrtc-root output/cuda-13.0-nvrtc/nvidia/cu13 --output output/umzi-native-windows
```

Two separate builds of the same sources produce identical cubins and PTX,
but MSVC produces different host objects, and their outputs can differ in
the last bits. Compare a library only with the CLI from the same build:
build the wheel with `UMZI_NATIVE_BUILD_DIR` set, or use the files that
`build_native.py` writes together.

### Library loading

A session loads the first Umzi runtime it finds in this order:

1. `SessionOptions(dll_path=...)`
2. The installed package directory
3. The packed model's directory

The wrapper does not search NVIDIA installations or change `PATH` or
`LD_LIBRARY_PATH`. Get cuBLASLt and cuDNN into the process one of two ways:

- Import CUDA PyTorch first. PyTorch 2.15 with CUDA 13.2 loads cuBLASLt 13
  and cuDNN 9.26 from `torch/lib`, and the session library binds to them.
- Call `umzi_native.load_cuda_libraries(cublas_dir, cudnn_dir)`, or pass
  the same directories as `SessionOptions(dll_dirs=[...])`. It loads
  cuBLASLt and its sibling CUDA libraries, then cuDNN and every cuDNN
  component, from the first listed directory that contains each one.

```python
import umzi_native as umzi

site = r"C:\Python314\Lib\site-packages\nvidia"
umzi.load_cuda_libraries(rf"{site}\cu13\bin\x86_64", rf"{site}\cudnn\bin")
```

Without either, opening a session raises `OSError` that names the missing
libraries and both remedies. `session.library_path` reports which GateRV3
library a session loaded.
[Native build and deployment](gaterv3-native-refactor.md#target-usage) ·
[NVIDIA cuDNN runtime dependencies](https://docs.nvidia.com/cudnn/installation/build-run-cudnn/)

## Open a session and run a batch

```python
from pathlib import Path

import numpy as np
import torch

import umzi_native as umzi

options = umzi.SessionOptions(cache_dir=Path("output/cache/gaterv3"))
tiles = np.zeros((3, 3, 192, 160), dtype=np.float32)

with umzi.InferenceSession("output/native/gaterv3.ummodel", sess_options=options) as session:
    input_name = session.get_inputs()[0].name
    (restored,) = session.run(None, {input_name: tiles})
    print(restored.shape, restored.dtype)
    print(session.get_session_info()["cache_status"])
```

The input is one NumPy array named `input` in RGB NCHW layout. Supply
RGB values on the model's 0–1 input scale. The 1× model returns an
array named `output` with the same shape and dtype. Use
`run(["output"], {"input": tiles})` to name the output explicitly.
Each call returns its own array. Strided inputs are copied into
contiguous storage when needed.

The default mixed session uses FP32 RGB arrays. For full FP16, set
`precision="fp16"` and pass a `np.float16` array. Both precisions use
the same packed model. Batch size, height and width can change between
calls. `session.prepare((batch, 3, height, width))` can prepare a shape
ahead of its first inference. The context manager closes the session
and releases its device resources.

## Session options

| Option | Effect |
| --- | --- |
| `precision` | `"mixed"` or `"fp16"`. |
| `device_id` | CUDA device index. Defaults to 0. |
| `cache_dir` | Tuned plans use `gaterv3_<hex-key>.umplan` in this directory. Defaults to `None`, which disables disk caching. |
| `enable_cuda_graph` | Enables replay for the prepared shape. Defaults to `True`. |
| `fuse_silu` | Tries convolution–SiLU fusion at eligible sites. |
| `tune_projections` | Runs the wider cuBLASLt candidate search. |
| `fuse_gate_pool` | Fuses eligible grouped gate and pooling sites. |
| `fp16_qkv` | Enables FP16 QKV accumulation in mixed precision. |
| `kernel_options` | Pass family-specific settings as named integer values. GateRV3 accepts its four optimization flags as 0 or 1. |
| `dll_path` | Load this Umzi runtime instead of the package or model-directory copy. |
| `dll_dirs` | Directories passed to `load_cuda_libraries()` before the session library loads. |

After preparing or running a shape, `get_session_info()` returns the
padded shape, cache status, setup times, runner buffer sizes and
kernel-specific counters in the `diagnostics` dictionary. An out-of-memory error during shape preparation
or inference raises `MemoryError`, and the same session can accept
a smaller batch. [Native optimization details](gaterv3-native-refactor.md#session-optimization-options)

The package header dispatches to GateRV3. `session.kernel`,
`session.scale_factor` and `session.context` expose model metadata.
[Runtime and format spec](umzi-native-runtime-plan.md)

## Historical verification

The reports and commands in this section describe the previous GateRV3
DLL and package names.


`scripts/check_python_session.py` needs only NumPy and the package under
test. For each precision it runs a 1×3×128×128 tile and a 3×3×65×79 batch
of odd tiles, then opens a second session on the same plan cache. Every
output must be finite and match the native CLI bit for bit. The first
session must report a cache miss, the second a hit with identical output.
The report records which GateRV3, cuBLASLt and cuDNN files the process
loaded. The model was copied to `output/python-loader-check/model dir 日本/`,
which has no library beside it.

Every wheel below was built by the Hatch hook in an isolated
`pip wheel . --no-deps`, with `UMZI_NATIVE_BUILD_DIR` set so the CLI for the
bitwise comparison came from the same build. Each wheel was installed with
`pip install --no-deps --target` into a scratch directory. A separate check
ran `pip install --no-cache-dir` on the sdist in a new virtual environment.
pip built the DLL from source, installed NumPy and the package, and did not
install any NVIDIA wheels. The session check passed there with `dll_dirs`.

| Platform | CUDA libraries | Result |
| --- | --- | --- |
| Windows, `import torch` first | `torch/lib` cuBLASLt and cuDNN | All checks passed |
| Windows, `dll_dirs` | `nvidia/cu13/bin/x86_64` and `nvidia/cudnn/bin` | All checks passed |
| Windows, sdist into a new venv, `dll_dirs` | Same as above | Session checks passed; no CLI comparison |
| Windows, neither | None loaded | `OSError` with both remedies |
| WSL Ubuntu 24.04, `dll_dirs` | Staged `cu13/lib` and `cudnn/lib` | All checks passed |
| WSL Ubuntu 24.04, neither | None loaded | `OSError` naming `libcublasLt.so.13` |

Both precisions padded the odd batch to 3×3×80×80. All runs loaded the
library from the installed package directory. The WSL run used Python
3.12 without PyTorch.
[Loader check reports](../output/python-loader-check/)

The session uses the best cubin for the device's compute capability.
It falls back to `compute_86` PTX on GPUs with compute capability 8.6 or
higher when a compatible cubin is unavailable. Devices below 8.6 fail
with a clear error. The fallback requires a CUDA 13-capable driver with
PTX JIT support. To exercise it on an RTX 3050 Laptop, build with
`--arch sm_89` so the device has no compatible cubin.
[CUDA code compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
· [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

On Windows with NVIDIA driver 610.62, the `sm_89`-only build emitted
PTX ISA 9.0 and passed mixed and FP16 inference through the fallback on
the RTX 3050 Laptop. Both precisions matched the CLI bit for bit for
128×128 input and a batch of three odd tiles, and hit the disk cache on
reopen. The default wheel passed the same checks with its `sm_86` cubin.
[Forced PTX report](../output/windows-ptx-fix/forced-ptx.json)
· [Default wheel report](../output/windows-ptx-fix/default-wheel.json)

An alternating batch-16 smoke test on the same GPU compared cubins built
with NVRTC 13.4 and 13.0 on a single synthetic 128×128 batch. Median CUDA
Graph replay time was 61.32 ms and 60.98 ms in mixed precision and 58.72 ms
and 56.23 ms in FP16, respectively.
[Batch-16 smoke timings](../output/windows-ptx-fix/batch16-smoke.json)

An earlier pass of the previous Windows wrapper also checked strided input
views, independent output arrays, concurrent calls on one session, and
recovery after a rejected oversized allocation. The session code for those
paths is unchanged.
[Previous Python session check](../output/python-runtime-validation/python-session-check.json)

Build and check the Linux wheel from WSL. pip downloads the CUDA build
wheels into its isolated environment. The OpenSSL headers come from the
staged `libssl-dev` package, and NumPy from the staged `python` directory:

```bash
B=/tmp/umzi-native-linux-build
export UMZI_OPENSSL_INCLUDE=$B/ssl/headers/usr/include:$B/ssl/headers/usr/include/x86_64-linux-gnu
export UMZI_NATIVE_BUILD_DIR=$PWD/output/umzi-native-linux-wheel
python3 -m pip wheel . --no-deps -w output/python-loader-check/wheels-linux
python3 -m pip install --no-deps --target /tmp/umzi-loader-site output/python-loader-check/wheels-linux/*.whl
PYTHONPATH=/tmp/umzi-loader-site:$B/python python3 scripts/check_python_session.py \
  --model output/native-refactor/gaterv3.gtmodel --cli output/umzi-native-linux-wheel/gaterv3 \
  --cuda-dir $B/cuda/nvidia/cu13/lib --cuda-dir $B/cuda/nvidia/cudnn/lib \
  --output output/python-loader-check/linux-explicit.json
```

## Open items

- The `sm_89` and `sm_120` cubins have not run on matching hardware.
- A full install of the `dev` extra into a clean environment has not
  been run.
- The Hatch build has not run on a Windows machine without Visual Studio,
  or on a Linux distribution other than WSL Ubuntu 24.04. A Linux build
  without the OpenSSL headers fails at compile time.
- The Linux path has not been checked with CUDA PyTorch imported first,
  because the WSL Python has no PyTorch.
