# GateRV3 Python inference

The runtime source lives in `src/umzi_native/`. Each native kernel
family has its own directory under `native/`, starting with
`native/gaterv3/`. The Torch reference and checkpoint loader live under
`scripts/reference/` for export and validation.

`umzi_native` wraps the Windows CUDA DLL for running packed `.gtmodel`
files with NumPy. It uses the familiar ONNX Runtime pattern of opening a
session, inspecting named inputs and outputs, and calling `run` with an
input dictionary. The Python package depends on NumPy. Inference uses
`gaterv3.dll`, its CUDA and MSVC runtime dependencies, and the packed model.
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model)

Install the wrapper from the checkout and build the DLL and model if
needed. The defaults below build for the RTX 3050 Laptop GPU.

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip install .
& $py scripts/build_native.py --output output/native-refactor
& $py scripts/export_native.py --output output/native-refactor/gaterv3.gtmodel
```

Place `gaterv3.dll` next to the packed model, or give its location with
`SessionOptions(dll_path=...)`. The loader finds cuBLASLt and cuDNN in
the system Python NVIDIA packages or on `PATH`. Set
`SessionOptions(dll_dirs=[...])` to supply their directories explicitly.
[Native build and deployment](gaterv3-native-refactor.md#target-usage) ·
[NVIDIA cuDNN runtime dependencies](https://docs.nvidia.com/cudnn/installation/build-run-cudnn/)

## Open a session and run a batch

```python
from pathlib import Path

import numpy as np
import umzi_native as umzi

options = umzi.SessionOptions(cache_dir=Path("output/cache/gaterv3"))
tiles = np.zeros((3, 3, 192, 160), dtype=np.float32)

with umzi.InferenceSession(
    "output/native-refactor/gaterv3.gtmodel", sess_options=options
) as session:
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
| `cache_dir` | Directory for tuned plans. Defaults to `None`, which disables disk caching. |
| `enable_cuda_graph` | Enables replay for the prepared shape. Defaults to `True`. |
| `fuse_silu` | Tries convolution–SiLU fusion at eligible sites. |
| `tune_projections` | Runs the wider cuBLASLt candidate search. |
| `fuse_gate_pool` | Fuses eligible grouped gate and pooling sites. |
| `fp16_qkv` | Enables FP16 QKV accumulation in mixed precision. |
| `dll_path`, `dll_dirs` | Override the runtime DLL path or supply CUDA dependency directories. |

After preparing or running a shape, `get_session_info()` returns the
padded shape, cache status, setup times, runner buffer sizes and
optimization counters. An out-of-memory error during shape preparation
or inference raises `MemoryError`, and the same session can accept
a smaller batch. [Native optimization details](gaterv3-native-refactor.md#session-optimization-options)

## Verification

The installed wheel loaded the model from a path containing spaces and
Japanese characters in a clean NumPy process. Both precisions matched
the native CLI bit for bit on a 128×128 tile and a batch of three odd
65×79 tiles. The check also exercised strided input views, independent
output arrays, concurrent calls on one session, a cache hit after a
shape change, and recovery after a rejected oversized allocation.
[Python session check](../output/python-runtime-validation/python-session-check.json)

## Restructure handoff

The install and loading examples above describe the previous Windows wrapper.
Update them after the new loader and wheel are verified.

### Implemented in the working tree

- GateRV3 CUDA and C++ sources moved from `src/native/` to
  `native/gaterv3/`. Future kernel families can have their own directories
  under `native/`. The Torch reference model and checkpoint loader moved to
  `scripts/reference/` for the benchmarks and exporter.
- The builder checks the CUDA headers and NVRTC for major version 13.
  `pip install ".[dev]"` declares the CUDA 13 development wheels, cuDNN,
  setuptools, Ruff, and Pyrefly. The regular Python install depends on NumPy.
  Resolution and installation of the complete extra still need verification.
- The native builder embeds `sm_86`, `sm_89`, and `sm_120` cubins and
  `compute_86` PTX. Packed sessions select a compatible cubin or fall back
  to PTX. The native cache has Windows and Linux implementations. The packed
  `GTR3PCK` version 1 layout is unchanged.
- Windows NVRTC 13.4 and MSVC produced
  `output/umzi-native-windows/gaterv3.dll` and `gaterv3.exe`. WSL Ubuntu
  24.04 with NVRTC 13.4 produced all four CUDA images and linked
  `output/umzi-native-linux/libgaterv3.so` and `gaterv3`. GPU inference
  with these artifacts is still pending.
- The initial checkout passed `ruff format --check scripts src`,
  `ruff check scripts src`, and `ruff check .`. Pyrefly had zero errors.
  The edited checkout still needs the full checks.

### Resume here

1. Replace automatic NVIDIA directory discovery and cuDNN preloading in
   `src/umzi_native/_native.py`. Have the session load the selected
   `gaterv3.dll` or `libgaterv3.so` from the installed package, beside the
   model, or via the existing explicit path. Keep the ONNX Runtime-style
   session API and expose an opt-in CUDA loader for explicit dependency paths.
   On this Windows checkout, importing CUDA PyTorch 13.2 brought the CUDA
   driver, cuBLASLt 13, and the checked cuDNN components into the process.
   Verify that `import torch` followed by `InferenceSession` runs inference
   without the wrapper discovering NVIDIA installations.
2. Package both platform libraries using
   `scripts/build_native.py --wheel`. The wheel staging and platform tag
   logic is in `scripts/build_native.py`, `setup.py`, and `pyproject.toml`.
   Package the Umzi library and leave NVIDIA runtime libraries to PyTorch or
   the explicit loader. Install each wheel into a temporary target and test
   lookup from the package directory with the model stored elsewhere.
3. Test mixed and FP16 inference with
   `output/native-refactor/gaterv3.gtmodel`, including an odd tile size,
   a batch larger than one, and a disk cache hit. The available GPU is an
   RTX 3050 Laptop (`sm_86`). To exercise PTX on this GPU, build separately
   with `--arch sm_89`, which omits the `sm_86` cubin. The `sm_89` and
   `sm_120` cubins still need execution on the respective hardware.
4. Update this guide and replace obsolete `src/native` links in
   `docs/gaterv3-native.md` and `docs/gaterv3-native-refactor.md`.
   Run `ruff format scripts src`, `ruff check scripts src`, `ruff check .`,
   and `pyrefly check` from the repo root.

Use `C:\Python314\python.exe` for the Windows build and checks.
The WSL CUDA 13.4 wheels are staged under `/tmp/umzi-native-linux-build/cuda`,
the Python build dependencies under `/tmp/umzi-native-linux-build/python`,
and the OpenSSL headers under `/tmp/umzi-native-linux-build/ssl/headers`.
The WSL driver library is `/usr/lib/wsl/lib/libcuda.so.1`. The Linux build
uses the existing WSL Python and these staged tools.
