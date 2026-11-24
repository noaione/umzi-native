# Python inference quickstart

`umzi-native` runs packed `.ummodel` packages on NVIDIA GPUs through one
Python API (`InferenceSession`) across all supported model families:
**GateRV3**, **GateRV2**, **MoESR**, **GFISRv2**, **SMoSR**, and **FiGSR**.

**Prerequisites:**

- Python 3.12+ with `umzi-native` installed ([install & CUDA setup](gaterv3-python.md#install))
- A packed `.ummodel` file ([how to export a checkpoint](export.md))
- `Pillow` if you want to run the image example below (`python -m pip install Pillow`)

## 1. Run your first inference

Before opening a session, the CUDA runtime libraries (`cublasLt`, `cudnn`)
must be loaded in the process. The simplest way is to `import torch` first
(see [CUDA library loading](gaterv3-python.md#library-loading) if you don't
use PyTorch).

```python
from pathlib import Path

import numpy as np
import torch  # Makes PyTorch's bundled CUDA/cuDNN/cuBLASLt libraries available.
from PIL import Image

import umzi_native as umzi

# Load a 128x128 RGB crop and convert HWC uint8 [0, 255] -> NCHW float32 [0, 1].
with Image.open("samples/input/page-001.jpg") as page:
    rgb = page.convert("RGB").crop((0, 0, 128, 128))
pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255)
tiles = pixels.transpose(2, 0, 1)[None, ...]  # Shape: (1, 3, 128, 128)

# Point cache_dir at a writable folder so tuned GPU plans are reused across runs.
options = umzi.SessionOptions(cache_dir=Path("output/cache/gaterv3"))

with umzi.InferenceSession("output/native/gaterv3.ummodel", sess_options=options) as session:
    input_name = session.get_inputs()[0].name  # "input"
    (restored,) = session.run(None, {input_name: tiles})
    print(restored.shape, restored.dtype, session.get_session_info()["cache_status"])

# Convert NCHW float32 [0, 1] -> HWC uint8 [0, 255] and save.
rgb_out = restored[0].transpose(1, 2, 0)
png = np.rint(np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)
Image.fromarray(png).save("output/restored-tile.png")
```

## 2. Inputs, outputs, and batching

- **Tensor layout and range:** Pass a single NumPy array named `"input"` in
  RGB **NCHW** layout `(N, 3, H, W)` with values scaled to `[0, 1]`. Non-contiguous
  arrays are copied automatically when needed.
- **Output shape:** `session.run(None, {"input": tiles})` (or
  `session.run(["output"], {"input": tiles})`) returns a fresh NumPy array of
  shape `(N, 3, scale * H, scale * W)` with the same dtype as the input, where
  `scale` is `session.scale_factor`.
- **Precision and dtype must match:**
  - `precision="mixed"` (default) expects `np.float32` input and returns `np.float32`.
  - `precision="fp16"` expects `np.float16` input (`tiles.astype(np.float16)`) and returns `np.float16`.
  - `precision="fp32"` selects the GFISRv2, SMoSR, or FiGSR FP32 path and uses `np.float32` input and output.
  - The same `.ummodel` file works with all precision modes supported by its family.
- **Batching:** Stack multiple equal-size CHW tiles with `np.stack(chw_tiles)`
  to get `(N, 3, H, W)`; `restored[i]` corresponds to tile `i`.
- **Padding and dynamic shapes:** The runtime automatically pads each tile to
  the model's required alignment and crops the output back to `scale * H × scale * W`.
  You can change batch size, height, or width between calls—the session keeps
  one active prepared shape at a time and re-prepares when the padded shape
  changes.
- **Pre-warming a shape:** Call `session.prepare((N, 3, H, W))` ahead of time
  if you want allocation, algorithm tuning, and CUDA Graph capture to finish
  before your first `run()`.
- **Cleanup and OOM recovery:** Use `with umzi.InferenceSession(...) as session:`
  (or call `session.close()`) to free GPU memory. If a batch or tile size runs
  out of VRAM, `prepare()` or `run()` raises `MemoryError`, and you can
  immediately retry with a smaller batch on the same session.

For full-page tiling and stitching, see the
[manga tiling benchmark](gaterv3-manga-tiling.md).

## 3. Session options (`SessionOptions`)

Pass a `umzi.SessionOptions(...)` instance to `InferenceSession` to control
precision, GPU selection, disk plan caching, CUDA Graphs, library loading, and
model-specific kernel switches.

| Field | Default | What it does |
| --- | --- | --- |
| `precision` | `"mixed"` | All families support mixed precision with FP32 NumPy I/O and FP16 precision with FP16 NumPy I/O. GFISRv2, SMoSR, and FiGSR also support `"fp32"`. |
| `device_id` | `0` | CUDA GPU index to run on (respects `CUDA_VISIBLE_DEVICES`). |
| `cache_dir` | `None` | Writable directory for saving and reusing tuned GPU algorithm choices (`<family>_<key>.umplan`). **Strongly recommended:** first run tunes and writes the file; later runs skip tuning. Keys include the model, padded shape, precision, relevant options, GPU, driver, and CUDA/cuDNN versions. CUDA Graphs are still captured per process. |
| `enable_cuda_graph` | `True` | Captures the forward pass into a CUDA Graph and replays it to cut CPU launch overhead. Set `False` to submit kernels directly (still uses tuned cuBLASLt/cuDNN plans). |
| `kernel_options` | `{}` | Dictionary of `name: int` switches specific to the model family (see tables below). Unknown names or out-of-range values raise `RuntimeError`. |
| `dll_path` | `None` | Explicit path to `umzi_native.dll` / `libumzi_native.so`. By default the loader checks the installed package directory, then the `.ummodel` file's directory. |
| `dll_dirs` | `()` | Sequence of directories searched in order to explicitly load CUDA dependencies (`cublasLt`, `cudnn`) before loading `umzi_native`. Leave empty if PyTorch or the system PATH already loaded them. |

### Common configurations

```python
from pathlib import Path
import umzi_native as umzi

# 1. Fastest FP16 inference with persistent plan cache (works for GateRV3, GateRV2, MoESR)
fast_fp16 = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache"),
)

# 2. High-fidelity mixed precision on GPU 1, without CUDA Graphs
mixed_direct = umzi.SessionOptions(
    precision="mixed",
    device_id=1,
    cache_dir=Path("output/cache"),
    enable_cuda_graph=False,
)

# 3. Without PyTorch: point dll_dirs at standalone CUDA / cuDNN wheel directories
standalone = umzi.SessionOptions(
    cache_dir=Path("output/cache"),
    dll_dirs=[Path("C:/cuda-libs/cublas/bin"), Path("C:/cuda-libs/cudnn/bin")],
)
```

## 4. Model-specific `kernel_options`

The `.ummodel` header automatically selects the kernel family (`gaterv3`,
`gaterv2`, `moesr`, `gfisrv2`, or `smosr`). Families with kernel options accept
their own `kernel_options` keys as integers.

### GateRV3

All GateRV3 switches are off (`0`) by default; set to `1` to enable. Any other
key or value outside `{0, 1}` is rejected.

| Option | Default | What it does | Diagnostic in `get_session_info()` |
| --- | ---: | --- | --- |
| `fuse_silu` | `0` | Tries to fuse compatible bias-free 3×3 convolutions + SiLU via a cuDNN graph on 32-channel full-tile maps. Falls back to separate ops automatically if unsupported. | `diagnostics["fused_silu_count"]`, `kernel_message` |
| `tune_projections` | `0` | Benchmarks extra cuBLASLt candidates for eligible 1×1 projections using CUDA Graph replay during `prepare()`. Slower first-time setup, potentially faster steady-state. Independent of `enable_cuda_graph`. | `diagnostics["replay_tuned_shapes"]` |
| `fuse_gate_pool` | `0` | Fuses the grouped spatial gate and partial pooling on eligible 32-channel feature maps (mixed and FP16, including rectangular tiles; uses a specialized kernel at 128×128). | `diagnostics["fused_gate_pool_sites"]` |
| `fp16_qkv` | `0` | In `"mixed"` precision, allows FP16 accumulation on the original GateRV3 folded QKV attention projections. No effect on IDC models (no attention) or `"fp16"` mode (already FP16). | `diagnostics["fp16_qkv_projections"]` |

Changing `fuse_silu` or `fuse_gate_pool` reuses existing cached plans in
`cache_dir`; changing `tune_projections` or `fp16_qkv` uses a separate cache
key. Diagnostics in `get_session_info()["diagnostics"]` also include the common
plan metrics `timed_plans`, `plan_workspace_bytes`, and
`peak_plan_workspace_bytes`.

```python
options = umzi.SessionOptions(
    cache_dir=Path("output/cache/gaterv3"),
    kernel_options={"tune_projections": 1, "fuse_gate_pool": 1},
)
```

More detail: [GateRV3 optimization notes](gaterv3-native-refactor.md#session-optimization-options).

### GateRV2

| Option | Default | Valid values | What it does |
| --- | ---: | --- | --- |
| `tune_plans` | `1` | `0` or `1` | `1` times cuBLASLt and cuDNN candidates on first prepare for best throughput. Set `0` to pick the first heuristic plan immediately when startup latency matters more than peak speed. |
| `workspace_mib` | `64` | `0` to `1024` | Maximum temporary workspace (MiB) candidates may use during tuning. Prepare allocates only what the winning plans actually need after tuning. |

Both options are part of the `gaterv2_<key>.umplan` cache key. Diagnostics in
`get_session_info()["diagnostics"]`: `timed_plans`, `plan_workspace_bytes`,
`peak_plan_workspace_bytes`, `fp32_weight_bytes`, and `fp16_weight_bytes`.

```python
# Fast cold startup (skip candidate timing)
options = umzi.SessionOptions(
    cache_dir=Path("output/cache/gaterv2"),
    kernel_options={"tune_plans": 0},
)
```

More detail: [GateRV2 measurements](handoff-gaterv2.md).

### MoESR

MoESR requires `H >= 2` and `W >= 2` (for reflection padding).

| Option | Default | Valid values | What it does |
| --- | ---: | --- | --- |
| `fused_gate` | `1` | `0` or `1` | `1` (recommended) runs custom tensor-core kernels that fuse the 64→344 / 64→192 expansions with the Mish gate, a unified tiled depthwise branch kernel, and fused 176→64 / 96→64 projections with Mish, residual add, and next LayerNorm—faster in both FP16 and mixed, smaller activation buffers, only 4 cuDNN plans, and 0 B cuDNN workspace at batch 2. Set `0` to fall back to separate cuDNN convolutions (useful for A/B comparison). |
| `workspace_mib` | `512` | `0` to `4096` | Maximum cuDNN workspace (MiB) candidates may use during tuning. Prepare retains only what the selected plans need. Only matters noticeably when `fused_gate=0`, where caps below ~45 MiB per 256×256 tile force much slower convolution kernels at batch 2+. |

Both options are part of the `moesr_<key>.umplan` cache key. Diagnostics in
`get_session_info()["diagnostics"]`: `timed_plans`, `plan_workspace_bytes`,
`fused_gate`, and `peak_plan_workspace_bytes`.

```python
# Default MoESR FP16 (fused_gate=1 is already on by default)
options = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache/moesr"),
)

# Force zero cuDNN workspace (e.g. tight VRAM budget)
low_vram = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache/moesr"),
    kernel_options={"workspace_mib": 0},
)
```

More detail: [MoESR measurements](handoff-moesr.md).

### GFISRv2

GFISRv2 returns a 4× RGB result and requires cuFFT 12. If you load the
dependencies with `dll_dirs`, include the directory containing
`cufft64_12.dll` on Windows or `libcufft.so.12` on Linux. The GFISRv2 session
loads cuFFT when it opens.

- `"fp32"` uses FP32 throughout.
- `"fp16"` follows the eager FP16 arithmetic with half storage and FP32
  cuFFT transforms. Both NumPy tensors use `np.float16`.
- `"mixed"` uses half spatial convolutions with FP32 accumulation. The
  Fourier unit, residual stream, and NumPy input and output use FP32.

| Option | Default | Valid values | What it does |
| --- | ---: | --- | --- |
| `fused_conv` | `1` | `0` or `1` | `1` (recommended) runs custom tensor-core fused NHWC expansion and projection kernels in `"fp16"` and `"mixed"` modes. Set `0` to fall back to separate cuDNN backend convolutions. Ignored in `"fp32"`. |
| `workspace_mib` | `512` | `0` to `4096` | Maximum temporary cuDNN workspace (MiB) plan candidates may use during tuning in `"fp16"` and `"mixed"` modes. Prepare retains only what the selected plans need. |
| `unexact_erf` | `0` | `0` or `1` | `0` uses the PyTorch reference's erf rounding for Fourier GELU. `1` uses the build toolchain's CUDA `erff`. Pass via `--option unexact_erf=1` on the native CLI. |

`fused_conv`, `workspace_mib`, and `unexact_erf` are part of the
`gfisrv2_<key>.umplan` cache key in `"fp16"` and `"mixed"` modes. Diagnostics in
`get_session_info()["diagnostics"]`: `cufft_version`, `cufft_workspace_bytes`,
`unexact_erf`, `timed_plans`, `plan_workspace_bytes`, `fused_conv`, and
`peak_plan_workspace_bytes`.

```python
options = umzi.SessionOptions(precision="fp32", kernel_options={"unexact_erf": 1})
```

See the [GFISRv2 validation and gray renders](umzi-native-architecture-expansion.md#gfisrv2-native-fp32-package-and-session).

### SMoSR

SMoSR returns 2× RGB output for inputs at least three pixels high and wide.
Use `np.float32` arrays with `precision="fp32"` or `"mixed"` and `np.float16`
arrays with `precision="fp16"`. FP32 inference uses schema 2 packages.
Re-export the original checkpoint if you have a schema 1 package. Both
schemas support mixed and FP16. SMoSR accepts `enable_cuda_graph`.

For mixed and FP16, set `kernel_options={"tune_convolutions": 1}` to time
zero-workspace cuDNN plans for the 48→48 and 48→96 convolutions at shape
preparation. This option adds setup time and can change rounding. The default
uses the first supported plan. The native CLI accepts `--option tune_convolutions=1`.

See [SMoSR implementation and measurements](handoff-smosr.md).

### FiGSR

FiGSR returns 2× RGB output and accepts inputs at least six pixels high and
wide. The session handles reflection padding and even alignment internally.
Use `np.float32` arrays for `precision="fp32"` or `"mixed"` and `np.float16`
arrays for `precision="fp16"`. All three modes require cuFFT 12 for FP32
Fourier transforms.

Set `kernel_options={"tune_convolutions": 1}` to time alternate cuDNN
plans for the recurring block convolutions. Tuning adds shape setup time
and can change rounding. By default FiGSR uses its heuristic plans.
For FP32, `kernel_options={"fp32_nchw": 1}` uses NCHW for the expansion
and projection inside each block. It can be combined with
`tune_convolutions`.
For mixed precision, `kernel_options={"fp16_expand_accumulation": 1}`
speeds up the block expansion by accumulating in FP16. On the three
FiGSR benchmark images it reduced latency and lowered PSNR against the
Torch FP32 reference. It can be combined with `tune_convolutions`.
For mixed precision, `kernel_options={"fuse_row_gate": 1}` fuses the
1×17 convolution and its spatial gate with FP32 accumulation. It can be
combined with `tune_convolutions` and `fp16_expand_accumulation`.
For FP16 and mixed, `kernel_options={"fuse_expand_gate": 1}` fuses the
3×3 expansion, identity gate, and Fourier input packing. It uses the
selected expansion accumulation precision with a custom Tensor Core kernel.
See [FiGSR implementation and measurements](handoff-figsr.md).

## 5. Inspecting models and sessions

### Before opening a session (`peek_model`)

`umzi.peek_model(path)` inspects a `.ummodel` header without initializing CUDA:

```python
info = umzi.peek_model("output/native/moesr.ummodel")
print(info.family_name, info.supported, info.problems)
```

### On an open session

- **Metadata properties:** `session.kernel` (`"gaterv3"`, `"gaterv2"`, `"moesr"`, `"gfisrv2"`, or `"smosr"`),
  `session.scale_factor`, `session.context`,
  `session.library_path`, `session.closed`.
- **Runtime diagnostics (`session.get_session_info()`):** Call after
  `prepare()` or `run()` to inspect:
  - `padded_shape`: actual padded `(N, 3, H, W)` shape running on the GPU (`None` before first prepare).
  - `cache_status`:
    - `"disabled"` — `cache_dir` was not set.
    - `"miss"` — no saved plan file existed; algorithms were tuned and saved.
    - `"hit"` — reused saved plans from disk (`timed_plans == 0`).
    - `"rebuilt"` — saved file was corrupt or incompatible and was re-tuned and overwritten.
    - `"unavailable"` — `cache_dir` could not be read or written (see `cache_message`); inference still succeeded without disk caching.
  - `cache_message` / `kernel_message`: human-readable notes or fallback reasons when non-empty.
  - `stages_ms`: wall-clock setup breakdown (`model`, `shape_allocation`, `plans`, `allocation_warmup`, `capture`).
  - `buffer_bytes`: GPU bytes allocated for `weights`, `activations`, `workspace`, and `rgb_io`.
  - `startup_device_used_bytes`: total GPU memory in use across setup stages (`context`, `tuning_workspace`, `tuned`, `captured`).
  - `diagnostics`: family-specific integer counters listed in the tables above.

For the binary package format and C ABI specification, see the
[runtime and format spec](umzi-native-runtime-plan.md).
