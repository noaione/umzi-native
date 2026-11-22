# Python inference quickstart

Install `umzi-native` with Python 3.12 or newer and have a `.ummodel`
package ready. To create one from a checkpoint, see the
[export guide](export.md). [Installation and CUDA library loading](gaterv3-python.md#install)

## Run your first inference

Put the `.ummodel` at `output/native/gaterv3.ummodel` or use the path to your
package. Run `python -m pip install Pillow` to read and save images in this example.
Import CUDA PyTorch before opening the session, or call the explicit CUDA
library loader. The example processes a single 128×128 crop of a
repository sample. [CUDA library loading](gaterv3-python.md#library-loading)

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
| `cache_dir` | `None` | Set a writable directory to reuse tuned algorithms from the family's `.umplan` files (GateRV3, GateRV2, and MoESR). The key includes the model, padded shape and batch, precision, plan options, and CUDA environment. Each process captures its own CUDA Graph. |
| `enable_cuda_graph` | `True` | Capture and replay a CUDA Graph to reduce launch overhead for the prepared shape. `False` submits the operations directly. It still uses cuBLASLt and cuDNN plans. |
| `kernel_options` | `{}` | Pass the kernel-specific switches below as a mapping of names to integer values. |
| `dll_path` | `None` | Override the path to the Umzi DLL or shared library. By default the loader checks the installed package directory and then the model directory. |
| `dll_dirs` | `()` | Search these directories in order to explicitly load CUDA dependencies into the process before loading the Umzi library. An empty sequence uses libraries already available to the process. |

The session keeps one prepared padded shape and batch active at a time. If
either changes, the session prepares the new shape and can reuse matching
disk-cached algorithms. `cache_status` reports `disabled`, `miss`, `hit`,
`rebuilt`, or `unavailable`. Check `cache_message` for cache errors.
Changing `fuse_silu` or `fuse_gate_pool` can reuse the same cached
convolution and projection algorithms.

### GateRV3 kernel options

Every switch defaults to `0`. Set it to `1` to enable it. The kernel rejects
unknown names and values outside `0` and `1`.

| Key | Effect | Check in `get_session_info()` |
| --- | --- | --- |
| `fuse_silu` | Tries a cuDNN graph for compatible bias-free 3×3 convolutions followed by SiLU on 32-channel feature maps at the padded tile size. If cuDNN cannot run or capture the fused plan for that shape and precision, the runtime uses separate operations. | `diagnostics["fused_silu_count"]` and `kernel_message` |
| `tune_projections` | Tries more cuBLASLt candidates for eligible 1×1 projections, timing candidates with CUDA Graph replay during tuning. This can increase first-shape setup time. The switch is independent of CUDA Graph replay for inference. | `diagnostics["replay_tuned_shapes"]` |
| `fuse_gate_pool` | Combines the grouped spatial gate and partial pooling on eligible 32-channel feature maps at the padded tile size. Supports mixed and FP16, including rectangular tiles. | `diagnostics["fused_gate_pool_sites"]` |
| `fp16_qkv` | In mixed precision, uses FP16 accumulation for the original GateRV3 model's folded QKV projections. The IDC model has no attention projections, and the full FP16 path already uses FP16 accumulation there. | `diagnostics["fp16_qkv_projections"]` |

For example, pass `SessionOptions(kernel_options={"tune_projections": 1,
"fuse_gate_pool": 1})` to enable replay-based tuning and gate/pool fusion.
The runtime uses a specialized gate/pool kernel when the padded tile is
128×128 and a shape-aware kernel for other sizes.

After preparing or running a shape, `get_session_info()` returns the
padded shape, cache status, setup times, runner buffer sizes and
kernel-specific counters in the `diagnostics` dictionary. An out-of-memory error during shape preparation
or inference raises `MemoryError`, and the same session can accept
a smaller batch. [Native optimization details](gaterv3-native-refactor.md#session-optimization-options)

The package header selects the kernel family. `session.kernel`,
`session.scale_factor` and `session.context` expose model metadata, and
`session.closed` reports whether the session still holds device resources.
[Runtime and format spec](umzi-native-runtime-plan.md)

### GateRV2 kernel options

| Key | Default | Effect |
| --- | ---: | --- |
| `tune_plans` | `1` | Time cuBLASLt and cuDNN candidates when preparing a shape. Set `0` to use the first heuristic plan and shorten prepare. |
| `workspace_mib` | `64` | Cap the candidate workspace in MiB. Accepts integers from `0` to `1024`. Prepare grows workspace as needed and retains the size required by the selected plans. |

For example, use `SessionOptions(kernel_options={"tune_plans": 0})` when
prepare latency matters on a new shape. Set `cache_dir` to keep the chosen
plans in `gaterv2_<hex-key>.umplan` files. With a compatible cache hit,
the session reuses those algorithms with `timed_plans == 0`. Inspect
`peak_plan_workspace_bytes` for the largest workspace allocation during
prepare and `plan_workspace_bytes` for the retained size. The same
`get_session_info()["diagnostics"]` dictionary reports `fp32_weight_bytes`
and `fp16_weight_bytes`. [GateRV2 measurements](handoff-gaterv2.md)

### MoESR kernel options

| Key | Default | Effect |
| --- | ---: | --- |
| `workspace_mib` | `512` | Cap the cuDNN candidate workspace in MiB. Accepts integers from `0` to `4096`. Prepare times the candidates that fit and keeps the size the selected plans need. Caps below about 45 MiB per 256×256 tile force slower plans at batch 2 and above. |

MoESR is 4×: an input of shape `(N, 3, H, W)` returns `(N, 3, 4H, 4W)` in
the session's I/O dtype. Both spatial dimensions must be at least two.
Set `cache_dir` to keep the chosen plans in `moesr_<hex-key>.umplan` files.
A compatible cache hit reports `timed_plans == 0`, and
`diagnostics["plan_workspace_bytes"]` reports the retained workspace.
[MoESR measurements](handoff-moesr.md)
