# GateRV3 Python inference

The runtime source lives in `src/umzi_native/`. Each native kernel
family has its own directory under `native/`, starting with
`native/kernels/gaterv3/`. The optional exporter lives in
`src/umzi_native/exporter/`. Benchmark scripts use a Resselt-backed Torch
reference with GateRV3 spatial attention.

`umzi_native` runs packed `.ummodel` files with NumPy through the native
Umzi runtime: `umzi_native.dll` on Windows or `libumzi_native.so` on Linux. It
uses the familiar ONNX Runtime pattern of opening a session, inspecting
named inputs and outputs, and calling `run` with an input dictionary. The
package depends only on NumPy. The library needs the CUDA driver,
cuBLASLt 13 and cuDNN 9, which the package does not ship. On Windows it
also needs the MSVC runtime.
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html#load-and-run-a-model)

## Install

### GateRV3 checkpoints

| Checkpoint | Intended use | Padding alignment |
| --- | --- | ---: |
| `1x_umzi_digital_decompress_gaterv3_1.pth` | 1× Digital Decompression | 16 |
| `1x_umzi_idc_gaterv3.safetensors` | 1× Digital Decompression, fine-tuned for specific images | 8 |

Both export as `gaterv3` model packages. IDC uses three encoder levels and
Inception depthwise convolutions in its latent blocks. The package records
its required padding alignment.

Use the `exporter` extra to turn a checkpoint into a `.ummodel`. It installs
Resselt and its dependencies, including Torch. Export runs on the CPU and
accepts `.safetensors`, `.pth`, `.pt`, and `.ckpt` tensor state dictionaries.
From the checkout, install `.[exporter]` for the source build. For a published
wheel, install `umzi-native[exporter]` from your configured index.

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip install ".[exporter]"
& $py -m umzi_native export --help
& $py -m umzi_native export models/1x_umzi_idc_gaterv3.safetensors -o output/native/idc.ummodel
& $py -m umzi_native export models/1x_umzi_digital_decompress_gaterv3_1.pth -o output/native/gaterv3.ummodel
```

The Python equivalent is `umzi_native.exporter.export_model(checkpoint, output)`.
It returns the package's `ModelInfo`. Export detects the kernel family by
default. Pass `--kernel gaterv3` or `kernel="gaterv3"` to require that match.
The base install can read and run either package. Resselt is imported only
when you export.

Pass the resulting `.ummodel` path to `InferenceSession`. Use a runtime
built with 8-pixel alignment support for this package. The original
GateRV3 model still uses the same v1 payload and 16-pixel alignment.

On Windows, the IDC shape suite verified both precisions against the
matching Torch CUDA FP32 checkpoint. It covered batch 1 and 16, odd
rectangles, tiny tiles, CUDA Graph and direct dispatch, device buffers,
and cache recovery. A batch of seven 65×79 tiles padded to 72×80 and
scored 71.22 dB in mixed precision and 65.86 dB in FP16. The original
checkpoint passed the same checks with its own padding, and its export
was byte-identical to the previous package.
[IDC shape report](../output/gaterv3-idc-support/idc-shapes/shapes.json) ·
[Original checkpoint regression report](../output/gaterv3-idc-support/baseline-shapes/shapes.json)

The installed Windows wheel improved the original single-tile 65×79
probe from 35.44 to 71.49 dB in mixed precision and from 35.47 to 65.56 dB
in FP16. Both models kept their own alignment when open simultaneously,
and the installed IDC session matched the CLI bit for bit in both
precisions.
[Same-input comparison](../output/gaterv3-idc-support/same-input-probe.json) ·
[Installed wheel check](../output/gaterv3-idc-support/wheel-check.json)

Building from source compiles the native library. `pip install .` from
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

## Inspect a model package

`peek_model` reads a `.ummodel` file without loading the native library or
CUDA. It checks the container, checksum, kernel family, payload schema and
declared architecture against what this build accepts, then returns the
model information as a `ModelInfo`.

```python
import umzi_native as umzi

info = umzi.peek_model("output/native/gaterv3.ummodel")
info.supported  # True when a session can open the package
info.family_name  # "gaterv3"
info.payload.alignment  # required padding alignment
info.payload.operations  # 400
info.payload.op_kinds  # {"Pack": 1, "Conv": 142, ...}
```

`problems` explains why a well-formed package is unsupported, such as an
unknown kernel family, a newer payload schema or an unsupported
architecture. A file that is not a well-formed package, including a
truncated file or a checksum mismatch, raises `ValueError`. `sections` lists
the section table, and `payload.checkpoint` repeats the source checkpoint
digest when the exporter recorded one.

The same report is available from the command line. It exits non-zero when
any package is unsupported or unreadable.

```powershell
$py = 'C:\Python314\python.exe'
& $py -m umzi_native read --help
& $py -m umzi_native read output/native/gaterv3.ummodel output/native/idc.ummodel
& $py -m umzi_native read --json output/native/idc.ummodel
```

The older `python -m umzi_native MODEL` syntax also opens the package reader.

Peeking checks the operation table's structure but not every operation
against shapes. The native runtime performs the full validation when a
session opens the model.

## Run your first inference

Put the `.ummodel` at `output/native/gaterv3.ummodel` or use the path to your
package. Install Pillow for reading and saving the image in this example.
Import CUDA PyTorch before opening the session, or call the explicit CUDA
library loader shown above. The example processes a single 128×128 crop of a
repository sample so you can try it without running a whole manga page.

```python
from pathlib import Path

import numpy as np
import torch  # Loads the CUDA libraries used by the session.
from PIL import Image

import umzi_native as umzi

with Image.open("samples/input/page-001.jpg") as page:
    rgb = page.convert("RGB").crop((0, 0, 128, 128))
pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255)
tiles = pixels.transpose(2, 0, 1)[None, ...]  # RGB HWC to NCHW.

options = umzi.SessionOptions(cache_dir=Path("output/cache/gaterv3"))
with umzi.InferenceSession("output/native/gaterv3.ummodel", sess_options=options) as session:
    input_name = session.get_inputs()[0].name
    (restored,) = session.run(None, {input_name: tiles})
    print(restored.shape, restored.dtype, session.get_session_info()["cache_status"])

rgb_out = restored[0].transpose(1, 2, 0)
png = np.rint(np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)
Image.fromarray(png).save("output/restored-tile.png")
```

The input is one NumPy array named `input` in RGB NCHW layout. Supply
RGB values on the model's 0–1 input scale. The 1× model returns an
array named `output` with the same shape and dtype. Use
`run(["output"], {"input": tiles})` to name the output explicitly.
Each call returns its own array. Strided inputs are copied into
contiguous storage when needed.

The default mixed session uses FP32 RGB arrays. For full FP16, set
`precision="fp16"` and pass `tiles.astype(np.float16)`. Both precisions use
the same packed model. To batch several equal-size tiles, normalize each tile
and stack the CHW arrays with `np.stack(chw_tiles)`. `restored[i]` is the output
for tile `i`. The runtime pads each tile to the model's alignment and crops
the result back to the original size. Batch size, height, and width can
change between calls. `session.prepare(tiles.shape)` can allocate and tune
the shape ahead of the first `run()`. The context manager closes the session
and releases its device resources. For full-page tiling and stitching, see
the [manga tiling benchmark](gaterv3-manga-tiling.md).

## Session options

| `SessionOptions` field | Default | Effect |
| --- | --- | --- |
| `precision` | `"mixed"` | Mixed precision uses FP32 NumPy input and output and FP32 accumulation in the default convolution and projection paths. `"fp16"` uses FP16 arrays and enables the native FP16 compute path. |
| `device_id` | `0` | CUDA ordinal for the session. Use the ordinal visible to the process when CUDA devices are masked. |
| `cache_dir` | `None` | Set a writable directory to reuse tuned algorithms from `gaterv3_<hex-key>.umplan` on compatible opens. The key includes the model, padded shape and batch, precision, relevant options, and CUDA environment. With `None`, every fresh prepare tunes in memory. The CUDA Graph is captured in the process. |
| `enable_cuda_graph` | `True` | Capture and replay a CUDA Graph to reduce launch overhead for the prepared shape. `False` submits the operations directly. It still uses cuBLASLt and cuDNN plans. |
| `kernel_options` | `{}` | Pass the kernel-specific switches below as a mapping of names to integer values. |
| `dll_path` | `None` | Override the path to the Umzi DLL or shared library. By default the loader checks the installed package directory and then the model directory. |
| `dll_dirs` | `()` | Search these directories in order to explicitly load CUDA dependencies into the process before loading the Umzi library. An empty sequence uses libraries already available to the process. |

The session keeps one prepared padded shape and batch active at a time. If
either changes, the session prepares the new shape and can reuse matching
disk-cached algorithms. `cache_status` reports `disabled`, `miss`, `hit`,
`rebuilt`, or `unavailable`. Check `cache_message` for cache errors.

### GateRV3 kernel options

Every switch defaults to `0`. Set it to `1` to enable it. The kernel rejects
unknown names and values outside `0` and `1`.

| Key | Effect | Check in `get_session_info()` |
| --- | --- | --- |
| `fuse_silu` | Tries one cuDNN graph for compatible 3×3 convolutions followed by SiLU on 128×128, 32-channel feature maps. If there are no eligible sites or cuDNN cannot run or capture the fused plan, the runtime uses separate operations. | `diagnostics["fused_silu_count"]` and `kernel_message` |
| `tune_projections` | Tries more cuBLASLt candidates for eligible 1×1 projections, timing candidates with CUDA Graph replay during tuning. This can increase first-shape setup time. The switch is independent of CUDA Graph replay for inference. | `diagnostics["replay_tuned_shapes"]` |
| `fuse_gate_pool` | Combines the grouped spatial gate and the partial pooling reduction at eligible 128×128, 32-channel feature maps. Supports mixed and FP16. | `diagnostics["fused_gate_pool_sites"]` |
| `fp16_qkv` | In mixed precision, uses FP16 accumulation for the original GateRV3 model's folded QKV projections. The IDC model has no attention projections, and the full FP16 path already uses FP16 accumulation there. | `diagnostics["fp16_qkv_projections"]` |

For example, pass `SessionOptions(kernel_options={"tune_projections": 1,
"fuse_gate_pool": 1})` to enable replay-based tuning and gate/pool fusion.

After preparing or running a shape, `get_session_info()` returns the
padded shape, cache status, setup times, runner buffer sizes and
kernel-specific counters in the `diagnostics` dictionary. An out-of-memory error during shape preparation
or inference raises `MemoryError`, and the same session can accept
a smaller batch. [Native optimization details](gaterv3-native-refactor.md#session-optimization-options)

The package header dispatches to GateRV3. `session.kernel`,
`session.scale_factor` and `session.context` expose model metadata, and
`session.closed` reports whether the session still holds device resources.
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
