# GateRV3 Python inference

The runtime source lives in `src/umzi_native/`. Each native kernel
family has its own directory under `native/`, starting with
`native/gaterv3/`. The Torch reference and checkpoint loader live under
`scripts/reference/` for export and validation.

`umzi_native` runs packed `.gtmodel` files with NumPy through the native
GateRV3 library: `gaterv3.dll` on Windows or `libgaterv3.so` on Linux. It
uses the familiar ONNX Runtime pattern of opening a session, inspecting
named inputs and outputs, and calling `run` with an input dictionary. The
package depends only on NumPy. The library needs the CUDA driver,
cuBLASLt 13 and cuDNN 9, which the package does not ship. On Windows it
also needs the MSVC runtime.
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model)

Build the platform wheel and the packed model, then install the wheel.
`--wheel` stages the newly built library in a `py3-none-<platform>` wheel
under `<output>/wheels`.

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip install ".[dev]"
& $py scripts/build_native.py --output output/umzi-native-windows --wheel
& $py scripts/export_native.py --output output/native-refactor/gaterv3.gtmodel
& $py -m pip install (Get-ChildItem output/umzi-native-windows/wheels/*.whl)
```

### Library loading

A session loads the first GateRV3 library it finds in this order:

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
| `dll_path` | Load this GateRV3 library instead of the package or model-directory copy. |
| `dll_dirs` | Directories passed to `load_cuda_libraries()` before the session library loads. |

After preparing or running a shape, `get_session_info()` returns the
padded shape, cache status, setup times, runner buffer sizes and
optimization counters. An out-of-memory error during shape preparation
or inference raises `MemoryError`, and the same session can accept
a smaller batch. [Native optimization details](gaterv3-native-refactor.md#session-optimization-options)

## Verification

`scripts/check_python_session.py` needs only NumPy and the package under
test. For each precision it runs a 1×3×128×128 tile and a 3×3×65×79 batch
of odd tiles, then opens a second session on the same plan cache. Every
output must be finite and match the native CLI bit for bit. The first
session must report a cache miss, the second a hit with identical output.
The report records which GateRV3, cuBLASLt and cuDNN files the process
loaded. Each wheel was installed with `pip install --no-deps --target` into
a scratch directory. The model was copied to
`output/python-loader-check/model dir 日本/`, which has no library beside it.

| Platform | CUDA libraries | Result |
| --- | --- | --- |
| Windows, `import torch` first | `torch/lib` cuBLASLt and cuDNN | All checks passed |
| Windows, `dll_dirs` | `nvidia/cu13/bin/x86_64` and `nvidia/cudnn/bin` | All checks passed |
| Windows, neither | None loaded | `OSError` with both remedies |
| WSL Ubuntu 24.04, `dll_dirs` | Staged `cu13/lib` and `cudnn/lib` | All checks passed |
| WSL Ubuntu 24.04, neither | None loaded | `OSError` naming `libcublasLt.so.13` |

Both precisions padded the odd batch to 3×3×80×80. All runs loaded the
library from the installed package directory. The WSL run used Python
3.12 without PyTorch. [Loader check reports](../output/python-loader-check/)

The PTX fallback runs when a build embeds no cubin for the device's major
architecture. An `--arch sm_89` build omits `sm_86`, so the RTX 3050 Laptop
loads its `compute_86` PTX. With NVRTC 13.4 that PTX declares ISA 9.4, and
the installed driver 610.62 (CUDA 13.3) rejected it:
`the provided PTX was compiled with an unsupported toolchain`. A probe
built with NVRTC 13.0.88 emitted ISA 9.0. Its PTX path passed every check
above in both precisions, including the CLI comparison.
[PTX probe report](../output/python-loader-check/windows-ptx-sm89-nvrtc130.json) ·
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

An earlier pass of the previous Windows wrapper also checked strided input
views, independent output arrays, concurrent calls on one session, and
recovery after a rejected oversized allocation. The session code for those
paths is unchanged.
[Previous Python session check](../output/python-runtime-validation/python-session-check.json)

Build and check the Linux wheel from WSL with the staged CUDA 13.4 wheels,
build-time Python packages, and OpenSSL headers:

```bash
B=/tmp/umzi-native-linux-build
PYTHONPATH=$B/python python3 scripts/build_native.py --output output/umzi-native-linux --wheel \
  --cuda-root $B/cuda/nvidia/cu13 --cudnn-root $B/cuda/nvidia/cudnn \
  --openssl-include $B/ssl/headers/usr/include \
  --openssl-include $B/ssl/headers/usr/include/x86_64-linux-gnu \
  --cuda-driver-lib /usr/lib/wsl/lib/libcuda.so.1
python3 -m pip install --no-deps --target /tmp/umzi-loader-site output/umzi-native-linux/wheels/*.whl
PYTHONPATH=/tmp/umzi-loader-site:$B/python python3 scripts/check_python_session.py \
  --model output/native-refactor/gaterv3.gtmodel --cli output/umzi-native-linux/gaterv3 \
  --cuda-dir $B/cuda/nvidia/cu13/lib --cuda-dir $B/cuda/nvidia/cudnn/lib \
  --output output/python-loader-check/linux-explicit.json
```

## Open items

- The PTX fallback needs a toolchain no newer than the oldest supported
  driver. NVRTC 13.4 PTX fails on driver 610.62. Either pin the PTX compile
  to an older NVRTC, or require a driver that supports CUDA 13.4.
- The `sm_89` and `sm_120` cubins have not run on matching hardware.
- `pip install --dry-run ".[dev]"` resolves on Windows to CUDA runtime,
  NVRTC and CRT 13.4.92, cuBLAS 13.10, CCCL 13.3.5, cuDNN 9.28.1, Ruff 0.17.6,
  Pyrefly 1.5.0 and setuptools 84.2.1. A full install into a clean
  environment has not been run.
- The Linux path has not been checked with CUDA PyTorch imported first,
  because the WSL Python has no PyTorch.
