# Python inference quickstart

`umzi-native` runs packed `.ummodel` models on NVIDIA GPUs through one Python API (`InferenceSession`) across all six supported families: **GateRV3**, **GateRV2**, **MoESR**, **GFISRv2**, **SMoSR**, and **FiGSR**.

**Prerequisites:**

- Python 3.12+ with `umzi-native` installed ([install & CUDA setup](setup.md))
- A packed `.ummodel` file ([how to export a checkpoint](export.md))
- `Pillow` if you want to run the image example below (`python -m pip install Pillow`)

## 1. Run your first inference

Before opening a session, the CUDA runtime libraries (`cublasLt` 13, `cudnn` 9, plus `cufft` 12 for GFISRv2 and FiGSR) must be loaded in the process. The simplest way is to `import torch` first; see [CUDA library loading](setup.md#cuda-library-loading) if you do not use PyTorch.

```python
from pathlib import Path

import numpy as np
import torch  # Loads PyTorch's bundled CUDA/cuDNN/cuBLASLt libraries.
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

- **Tensor layout and range:** Pass a single NumPy array named `"input"` in RGB **NCHW** layout `(N, 3, H, W)` with values in `[0, 1]`. Non-contiguous arrays are copied automatically when needed.
- **Output shape:** `session.run(None, {"input": tiles})` (or `session.run(["output"], {"input": tiles})`) returns a new NumPy array of shape `(N, 3, scale * H, scale * W)` with the same dtype as the input, where `scale` is `session.scale_factor`.
- **Precision and dtype must match:**
  - `precision="mixed"` (default) expects `np.float32` input and returns `np.float32`.
  - `precision="fp16"` expects `np.float16` input (`tiles.astype(np.float16)`) and returns `np.float16`.
  - `precision="fp32"` (available on GFISRv2, SMoSR, and FiGSR) expects `np.float32` input and returns `np.float32`.
  - One `.ummodel` file works with every precision mode supported by its family.
- **Batching:** Stack equal-size CHW tiles with `np.stack(chw_tiles)` to form `(N, 3, H, W)`; `restored[i]` corresponds to tile `i`.
- **Padding and dynamic shapes:** The runtime pads each tile to the model's required alignment internally and crops the output back to `scale * H × scale * W`. You can change batch size, height, or width between calls—the session keeps one active shape prepared at a time and re-prepares when the padded shape changes.
- **Pre-warming a shape:** Call `session.prepare((N, 3, H, W))` ahead of time if you want GPU allocation, algorithm tuning, and CUDA Graph capture to finish before the first `run()`.
- **Cleanup and OOM recovery:** Use `with umzi.InferenceSession(...) as session:` (or call `session.close()`) to release GPU memory. If a batch or tile size runs out of VRAM, `prepare()` or `run()` raises `MemoryError`, and you can immediately retry with a smaller batch on the same session.

### Full-page GPU tiling (`run_tiled`)

Instead of slicing tiles on the CPU, you can pass a full contiguous HWC `uint8` page directly to `session.run_tiled()`. The runtime extracts tiles, runs batched GPU inference, feathers overlaps (when `overlap > 0`), and returns the scaled `(scale * H, scale * W, 3)` page without extra CPU-GPU staging copies on the main runner.

```python
with Image.open("samples/input/page-001.jpg") as page:
    rgb_page = np.ascontiguousarray(page.convert("RGB"), dtype=np.uint8)

with umzi.InferenceSession("output/native/gaterv3.ummodel", sess_options=options) as session:
    restored = session.run_tiled(
        rgb_page,
        tile_size=128,
        batch_size=16,
        overlap=0,
        tiling="fixed",
    )

Image.fromarray(restored).save("output/restored-page.png")
```

| Parameter | Default | What it does |
| --- | --- | --- |
| `tile_size` | *(required)* | Nominal tile edge in source pixels (`>= 1`). For `"even"` and `"fewest-even"`, it caps both tile dimensions. |
| `batch_size` | *(required)* | Maximum tiles per GPU batch (`>= 1`). |
| `overlap` | `0` | Requested shared width in source pixels. `0` copies tiles directly. `"centered"` caps the request at half the nominal tile core. `"linear"` requires overlap below the nominal core size. |
| `tiling` | `"fixed"` | `"fixed"` keeps every batch at `(batch_size, 3, tile_size, tile_size)`. `"variable"` adapts edge tile shapes and final batches. `"even"` and `"fewest-even"` use equal rectangular tiles sized for the page and submit an exact final batch. |
| `overlap_method` | `"centered"` | `"centered"` blends across up to 16 source pixels at the center of each join. `"linear"` feathers across the full overlap. |
| `background` | `"white"` | Out-of-bounds pad color when using fixed shapes with zero overlap and zero model context (`"white"` = `255`, `"black"` = `0`). |
| `output_dtype` | `np.uint8` | `np.uint8` quantizes to RGB `[0, 255]` (using a ring-buffered FP32 strip accumulator when `overlap > 0`). `np.float32` returns raw unclipped FP32 stitched values. |

- **`"fixed"` vs. `"variable"`:** `"variable"` caches up to three edge runners in an LRU cache sharing model weights and immutable CUDA/cuDNN handles with the main session, evicting runners automatically if VRAM gets tight. On **GateRV3 Original**, nominal `tile_size <= 128` keeps the fixed square spatial shape (adapting only the tail batch size), while larger tiles (`> 128`) and other families can shrink rectangular edge dimensions too. Keep the `.ummodel` file readable while the session is open.
- **Centered overlap with fixed tiles:** The final tile sits flush with each far edge, and short axes are mirror-padded. Variable tiling uses its adaptive edge shapes and replicated padding with either overlap method. Both methods retain any context the model requires.
- **Even tiling:** The planner finds the fewest tiles along each axis using the size cap and effective overlap. It then shrinks the tile dimensions to cover the page plus that overlap and any required model context. With `"even"`, dimensions round up to multiples of 16 when they fit under the cap. Both layouts respect the model's minimum. GateRV3 Original keeps square tiles at caps up to 128 and a minimum edge of 128 at larger caps. With overlap, the first and last tiles sit flush and the intermediate starts are spaced evenly to source-pixel precision. Both blend methods use the actual overlap at each join. With zero overlap, cores abut and any extra pixels at the far edge are clipped. Out-of-bounds input uses replicated page pixels.
- For example, `tiling="even"`, `tile_size=512`, and `overlap=64` on a 900×1350 page use 2×3 tiles of 496×496 for a model with no extra context or spatial constraints. That uses about 6% fewer model-input pixels than six 512×512 tiles, excluding batch padding.
- **Older native libraries:** Set `overlap_method="linear"` to use nonzero overlap with an ABI 1 library.
- **`"fewest-even"`** uses the smallest equal rectangular tiles allowed by the page, overlap, and model constraints. For a 1733×2600 page with `tile_size=512` and `overlap=64`, it uses 24 tiles of 482×487 for a model with no extra context or spatial constraints.
- **Choosing between `"even"` and `"fewest-even"` for FFT models:** On **GFISRv2** (which runs cuFFT on the exact tile dimensions), prefer `"even"` when both modes emit the same tile count—rounding to multiples of 16 (such as 496×496 vs. 493×482 on a 900×1350 page at `tile_size=512`, `overlap=64`) keeps cuFFT on fast mixed-radix kernels and avoids Bluestein FFT overhead and extra FFT workspace.
- **Native library requirements:** `"even"` and `"fewest-even"` require the ABI 3 runtime. Model runners prepare the actual rectangular shape and final batch shape, with plan reuse where the model backend supports it.
- **Overlap memory:** With `overlap > 0` and `output_dtype=np.uint8`, the tiler keeps only a rolling strip of rows in FP32 instead of allocating a full-page FP32 accumulation buffer. Setting `output_dtype=np.float32` uses a full-page FP32 output buffer instead.

See the [GPU tiling plan and measurements](gpu-tiling-plan.md) and [manga tiling benchmark](gaterv3-manga-tiling.md) for workload results.

## 3. Session options (`SessionOptions`)

Pass a `umzi.SessionOptions(...)` instance to `InferenceSession` to configure precision, GPU selection, plan caching, CUDA Graphs, library paths, and model-specific options.

| Field | Default | What it does |
| --- | --- | --- |
| `precision` | `"mixed"` | `"mixed"` (FP32 NumPy I/O, half-precision weights/ops where supported), `"fp16"` (FP16 NumPy I/O), or `"fp32"` (GFISRv2, SMoSR, and FiGSR only). |
| `device_id` | `0` | CUDA GPU index to use (respects `CUDA_VISIBLE_DEVICES`). |
| `cache_dir` | `None` | Writable directory for saving tuned GPU algorithm choices (`.umplan` files) so subsequent runs skip tuning. Used by GateRV3, GateRV2, MoESR, MoSR, GFISRv2 (`mixed`/`fp16`), and SMoSR (`fuse_silu=1` or `tune_convolutions=1`). CUDA Graphs are still captured per process. |
| `enable_cuda_graph` | `True` | Captures the forward pass into a CUDA Graph and replays it to reduce CPU launch overhead. Set `False` to submit kernels directly. |
| `kernel_options` | `{}` | Dictionary of `name: int` switches specific to the model family (see below). Unknown names or invalid values raise `RuntimeError`. |
| `dll_path` | `None` | Explicit path to `umzi_native.dll` / `libumzi_native.so`. By default the loader checks the installed package directory, then the `.ummodel` file's directory. |
| `dll_dirs` | `()` | Directories searched in order to load CUDA dependencies (`cublasLt`, `cudnn`, and `cufft` for GFISRv2/FiGSR) before opening the session. Leave empty if PyTorch or `PATH` already loaded them. |

### Common configurations

```python
from pathlib import Path
import umzi_native as umzi

# 1. Fastest FP16 inference with persistent plan cache (all 6 families)
fast_fp16 = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache"),
)

# 2. Mixed precision on GPU 1, without CUDA Graphs
mixed_direct = umzi.SessionOptions(
    precision="mixed",
    device_id=1,
    cache_dir=Path("output/cache"),
    enable_cuda_graph=False,
)

# 3. Without PyTorch: point dll_dirs at standalone NVIDIA library directories
standalone = umzi.SessionOptions(
    cache_dir=Path("output/cache"),
    dll_dirs=[
        Path("path/to/nvidia/cu13/bin/x86_64"),
        Path("path/to/nvidia/cudnn/bin"),
    ],
)
```

## 4. Model-specific `kernel_options`

The `.ummodel` header selects the kernel family automatically (`gaterv3`, `gaterv2`, `moesr`, `gfisrv2`, `smosr`, `figsr`, or `mosr`). All `kernel_options` values are integers (`0` or `1` unless a range is noted).

### GateRV3

- **Scale & precisions:** 1× · `"mixed"`, `"fp16"`
- **Alignment:** 16 px (standard) or 8 px (IDC), padded automatically
- **Options:** All switches default to `0` (off); set to `1` to enable.

| Option | Default | Values | What it does |
| --- | ---: | --- | --- |
| `fuse_silu` | `0` | `0` or `1` | Fuses eligible 3×3 convolutions with SiLU via cuDNN when supported; falls back automatically otherwise. |
| `tune_projections` | `0` | `0` or `1` | Benchmarks extra 1×1 projection candidates using CUDA Graph replay during `prepare()`. Slower first setup, potentially faster steady-state. |
| `fuse_gate_pool` | `0` | `0` or `1` | Fuses the spatial gate and pooling step on eligible feature maps (uses a specialized kernel at 128×128). |
| `fp16_qkv` | `0` | `0` or `1` | In `"mixed"` mode, uses FP16 accumulation on QKV attention projections (faster, slightly lower precision). No effect on IDC or `"fp16"`. |

Changing `tune_projections` or `fp16_qkv` creates a separate `gaterv3_<key>.umplan` cache entry; toggling `fuse_silu` or `fuse_gate_pool` reuses existing cached plans.

- **Diagnostics (`get_session_info()["diagnostics"]`):** `replay_tuned_shapes`, `fused_gate_pool_sites`, `fp16_qkv_projections`, `split_gate_registers`, `fused_gate_registers`, `fused_silu_count`, `timed_plans`, `plan_workspace_bytes`, `peak_plan_workspace_bytes`.

```python
options = umzi.SessionOptions(
    cache_dir=Path("output/cache/gaterv3"),
    kernel_options={"tune_projections": 1, "fuse_gate_pool": 1},
)
```

More detail: [GateRV3 refactor & tuning notes](gaterv3-native-refactor.md#session-optimization-options).

### GateRV2

- **Scale & precisions:** 1× · `"mixed"`, `"fp16"`
- **Alignment:** 16 px (or $2^{\text{levels}}$), padded automatically

| Option | Default | Values | What it does |
| --- | ---: | --- | --- |
| `tune_plans` | `1` | `0` or `1` | `1` benchmarks cuBLASLt/cuDNN candidates on first `prepare()` for best speed. Set `0` to pick the first heuristic plan immediately for faster cold startup. |
| `workspace_mib` | `64` | `0` to `1024` | Max temporary workspace (MiB) candidates may use during tuning. After tuning, only what the winning plans need is kept. |

Both options are part of the `gaterv2_<key>.umplan` cache key.

- **Diagnostics (`get_session_info()["diagnostics"]`):** `timed_plans`, `plan_workspace_bytes`, `peak_plan_workspace_bytes`, `fp32_weight_bytes`, `fp16_weight_bytes`.

```python
# Fast cold startup (skip candidate benchmarking)
options = umzi.SessionOptions(
    cache_dir=Path("output/cache/gaterv2"),
    kernel_options={"tune_plans": 0},
)
```

More detail: [GateRV2 measurements](handoff-gaterv2.md).

### MoESR

- **Scale & precisions:** 4× · `"mixed"`, `"fp16"`
- **Input constraints:** Minimum `H >= 2`, `W >= 2` (2 px alignment handled internally)

| Option | Default | Values | What it does |
| --- | ---: | --- | --- |
| `fused_gate` | `1` | `0` or `1` | `1` (recommended) enables fused tensor-core and depthwise kernels—faster in both `"mixed"` and `"fp16"` with lower activation memory and zero retained cuDNN workspace. Set `0` to fall back to separate cuDNN convolutions for comparison. |
| `workspace_mib` | `512` | `0` to `4096` | Max temporary cuDNN workspace (MiB) used during tuning. Mainly affects `fused_gate=0`, where caps below ~45 MiB per 256×256 tile force slower kernels at batch 2+. |

Both options are part of the `moesr_<key>.umplan` cache key.

- **Diagnostics (`get_session_info()["diagnostics"]`):** `timed_plans`, `plan_workspace_bytes`, `fused_gate`, `peak_plan_workspace_bytes`.

```python
# Default MoESR FP16 (fused_gate=1 is on by default)
options = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache/moesr"),
)

# Tight VRAM budget: forbid cuDNN scratch workspace
low_vram = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache/moesr"),
    kernel_options={"workspace_mib": 0},
)
```

More detail: [MoESR measurements](handoff-moesr.md).

### GFISRv2

- **Scale & precisions:** 4× · `"fp32"`, `"mixed"`, `"fp16"`
- **Runtime dependency:** Requires cuFFT 12 (`cufft64_12.dll` / `libcufft.so.12`).
  - `"fp32"`: full FP32 weights, convolutions, and cuFFT.
  - `"mixed"`: FP16 spatial convolutions with FP32 accumulation; Fourier unit, residuals, and NumPy I/O stay FP32.
  - `"fp16"`: FP16 weights, activations, and NumPy I/O with FP32 accumulation on non-residual bookend convolutions (`first`, `tail`, `last`); Fourier transforms stay FP32 for stability.

| Option | Default | Values | Precision scope | What it does |
| --- | ---: | --- | --- | --- |
| `fused_conv` | `1` | `0` or `1` | `"mixed"`, `"fp16"` | `1` (recommended) runs custom fused tensor-core kernels. Set `0` to fall back to separate cuDNN convolutions. Ignored in `"fp32"`. |
| `fp32_project_accum` | `0` | `0` or `1` | `"fp16"` | `1` accumulates residual body projection convolutions in FP32 instead of FP16—higher PSNR/SSIMULACRA2 (matching or beating eager FP16) with modest latency overhead vs. default `"fp16"`. |
| `fp32_accum` | `0` | `0` or `1` | `"fp16"` | `1` accumulates both body expansion and projection convolutions in FP32 for highest `"fp16"` fidelity (slower than `fp32_project_accum=1`, still faster than `"mixed"`). |
| `workspace_mib` | `512` | `0` to `4096` | `"mixed"`, `"fp16"` | Max temporary cuDNN workspace (MiB) used during tuning. |
| `unexact_erf` | `0` | `0` or `1` | All | `0` matches PyTorch's exact `erf` rounding in the Fourier unit; `1` uses CUDA's `erff` (slightly faster, minor numeric difference). |

All five options are part of the `gfisrv2_<key>.umplan` cache key in `"mixed"` and `"fp16"` modes (`"fp32"` does not use plan caching).

- **Diagnostics (`get_session_info()["diagnostics"]`):** `cufft_version`, `cufft_workspace_bytes`, `unexact_erf`, `timed_plans`, `plan_workspace_bytes`, `fused_conv`, `peak_plan_workspace_bytes`, `fp32_accum`, `fp32_project_accum`.

```python
# Default balanced mixed precision
options = umzi.SessionOptions(
    precision="mixed",
    cache_dir=Path("output/cache/gfisrv2"),
)

# High-fidelity FP16 (FP32 accumulation on residual body projections + bookends)
fp16_proj32 = umzi.SessionOptions(
    precision="fp16",
    cache_dir=Path("output/cache/gfisrv2"),
    kernel_options={"fp32_project_accum": 1},
)
```

More detail: [GFISRv2 measurements](handoff-gfisrv2.md).

### SMoSR

- **Scale & precisions:** 2× · `"fp32"` (requires schema 2 export), `"mixed"`, `"fp16"`
- **Input constraints:** Minimum `H >= 3`, `W >= 3` (reflection padding handled internally)

| Option | Default | Values | Precision scope | What it does |
| --- | ---: | --- | --- | --- |
| `tune_convolutions` | `0` | `0` or `1` | `"mixed"`, `"fp16"` | `1` times zero-workspace cuDNN plans from heuristic A for body convolutions during `prepare()` instead of picking the first supported plan, and caches selections under `cache_dir` (`smosr-conv_<key>.umplan`). Adds cold setup time and can slightly alter rounding. |
| `fp16_accumulation` | `0` | `0` or `1` | `"mixed"` only | Allows FP16 accumulation in body convolutions instead of FP32 accumulation. |
| `fuse_silu` | `0` | `0` or `1` | `"mixed"`, `"fp16"` | Fuses eligible block convolutions with SiLU via cuDNN graphs and caches choices under `cache_dir` (`smosr-silu_<key>.umplan`). |

- **Diagnostics (`get_session_info()["diagnostics"]`):** `timed_candidates`, plus `fused_silu_shapes` when `fuse_silu=1`.

```python
options = umzi.SessionOptions(
    precision="mixed",
    cache_dir=Path("output/cache/smosr"),
    kernel_options={"tune_convolutions": 1, "fuse_silu": 1},
)
```

More detail: [SMoSR measurements](handoff-smosr.md).

### FiGSR

- **Scale & precisions:** 2× · `"fp32"`, `"mixed"`, `"fp16"`
- **Runtime dependency:** Requires cuFFT 12 for FP32 Fourier transforms in all three modes.
- **Input constraints:** Minimum `H >= 6`, `W >= 6` (reflection padding and even alignment handled internally)

| Option | Default | Values | Precision scope | What it does |
| --- | ---: | --- | --- | --- |
| `tune_convolutions` | `0` | `0` or `1` | All | `0` uses cuDNN heuristic plans; `1` times eligible plans during `prepare()`, adding setup time and potentially altering rounding. |
| `fp32_nchw` | `0` | `0` or `1` | `"fp32"` only | Uses NCHW layout instead of NHWC for block expansion and projection. |
| `fp16_expand_accumulation` | `0` | `0` or `1` | `"mixed"` only | Uses FP16 accumulation in block expansion (lower latency, lower PSNR vs. FP32). |
| `fuse_row_gate` | `0` | `0` or `1` | `"mixed"` only | Fuses the 1×17 convolution and spatial gate with FP32 accumulation. |
| `fuse_expand_gate` | `0` | `0` or `1` | `"mixed"`, `"fp16"` | Fuses the 3×3 expansion, identity gate, and Fourier input packing into one tensor-core kernel. |

- **Diagnostics (`get_session_info()["diagnostics"]`):** `fp32_accumulation_fallbacks`, `cufft_workspace_bytes`, plus `timed_convolution_candidates` and `changed_convolution_plans` when `tune_convolutions=1`.

```python
# Tuned mixed configuration
tuned_mixed = umzi.SessionOptions(
    precision="mixed",
    kernel_options={"tune_convolutions": 1, "fuse_row_gate": 1, "fuse_expand_gate": 1},
)
```

More detail: [FiGSR measurements](handoff-figsr.md).

### MoSR

- **Scale & precisions:** 1×–8× (4× `gps` / 2× `dys` tested) · `"fp32"`, `"mixed"`, `"fp16"`
- **Input constraints:** Any positive spatial dimensions (`H >= 1`, `W >= 1`)

| Option | Default | Values | Precision scope | What it does |
| --- | ---: | --- | --- | --- |
| `fused_gate` | `1` | `0` or `1` | `"mixed"`, `"fp16"` | `1` (recommended) enables fused tensor-core `expand_gate` / `project_res_norm` kernels and coalesced planar depthwise gating for `(dim=64, hidden=96)` or `(dim=48, hidden=72)` with `kernel_size=7` or `17`. Set `0` to fall back to separate cuDNN convolutions and NHWC 8-channel gating. Ignored in `"fp32"` or other block widths. |
| `dw_fp16` | `1` | `0` or `1` | `"fp16"` | `1` (recommended) uses packed `half2` arithmetic in planar depthwise gating when `fused_gate=1`. Set `0` for FP32 depthwise accumulation in `"fp16"`. |
| `fold_gps` | `0` | `0` or `1` | `"mixed"`, `"fp16"` | `1` pre-averages the `8×` GPS ensemble head on host and shrinks body scratch (saves ~44 MiB at `512×512` at a ~3 dB FP16 tail PSNR cost). Kept `0` by default to preserve full GPS ensemble precision. |
| `workspace_mib` | `512` | `0` to `4096` | All | Max temporary cuDNN workspace (MiB) used during tuning. |

`fused_gate`, `dw_fp16`, `fold_gps`, and `workspace_mib` are part of the `mosr_<key>.umplan` cache key.

- **Diagnostics (`get_session_info()["diagnostics"]`):** `timed_plans`, `plan_workspace_bytes`, `peak_plan_workspace_bytes`, `fp16_fallback_plans`.

More detail: [MoSR measurements](handoff-mosr.md).

## 5. Inspecting models and sessions

### Before opening a session (`peek_model`)

`umzi.peek_model(path)` reads and validates a `.ummodel` file without initializing CUDA (see [Inspect a package](export.md#inspect-a-package)):

```python
info = umzi.peek_model("output/native/moesr.ummodel")
print(info.family_name, info.supported, info.problems)
```

### On an open session

- **Properties:** `session.kernel` (`"gaterv3"`, `"gaterv2"`, `"moesr"`, `"gfisrv2"`, `"smosr"`, `"figsr"`, or `"mosr"`), `session.scale_factor`, `session.context`, `session.library_path`, `session.closed`.
- **Runtime diagnostics (`session.get_session_info()`):** Call after `prepare()` or `run()` to inspect:
  - `padded_shape`: actual padded `(N, 3, H, W)` shape running on the GPU (`None` before first prepare).
  - `cache_status`:
    - `"disabled"` — `cache_dir` was not set, or the active mode/family does not use disk plan caching.
    - `"miss"` — no saved plan file existed; algorithms were tuned and saved.
    - `"hit"` — reused saved plans from disk (`timed_plans == 0`).
    - `"rebuilt"` — saved file was corrupt or incompatible and was re-tuned and overwritten.
    - `"unavailable"` — `cache_dir` could not be read or written (see `cache_message`); inference still ran without disk caching.
  - `cache_message` / `kernel_message`: readable status notes or fallback reasons when non-empty.
  - `stages_ms`: wall-clock setup breakdown (`model`, `shape_allocation`, `plans`, `allocation_warmup`, `capture`).
  - `buffer_bytes`: GPU bytes allocated for `weights`, `activations`, `workspace`, and `rgb_io`.
  - `startup_device_used_bytes`: GPU memory in use across setup stages (`context`, `tuning_workspace`, `tuned`, `captured`).
  - `diagnostics`: family-specific integer counters listed in the tables above.

For the binary `.ummodel` format and C ABI specification, see the [runtime and format spec](umzi-native-runtime-plan.md).
