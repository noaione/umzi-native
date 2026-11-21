# GateRV3 native runtime refactor plan

Package the current 1× GateRV3 checkpoint as one model file loadable by
one Windows DLL. A session selects mixed precision or full FP16 and accepts
new tile and batch sizes on subsequent calls. Keep the verified 128×128
runner as the performance and fidelity control. This document describes
the proposed contract and implementation sequence.

## Implementation notes

The fixed bundle and measured report are frozen at
`output/native-projections` and
`output/native-projection-validation/verified-report.json`. The session
will own one uploaded copy of the weights and one live shape plan.
Repeated calls reuse that plan and its graph. A different padded shape
releases the old plan before allocating the new one. The caller-selected
disk cache can restore tactics when returning to an evicted shape.
Packed files start with the seven ASCII magic bytes `GTR3PCK` followed by
a zero byte in the eight-byte magic field. The format version is `1`.
Keep it at `1` as the layout evolves before the first public release.

Both benchmark scripts use the TensorRT-RTX workspace, CUDA Graph,
build-logging, and runtime-cache options from the sibling runtime.
[nao-manga-rls session setup](../../nao-manga-rls/nmanga/ml/onnx_session.py)
The fixed ONNX artifacts use matching minimum, optimum and maximum
profiles for batch 1 and batch 16 in separate sessions. Set
`--rtx-cache-dir` to choose the TensorRT-RTX cache directory.
[TensorRT-RTX provider options](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html#execution-provider-options)

## Target usage

Ship `gaterv3.dll`, its C header, and an MSVC import library for consumers
who link at build time. Export one file such as `gaterv3.gtmodel`. The
application opens that file once, selects a CUDA device and precision, and
runs repeated calls with contiguous RGB NCHW input. The DLL's generic CUDA
kernels are embedded in the DLL. The model file holds the execution
description and weights, so inference needs only the DLL, its CUDA and
MSVC runtime dependencies, and the exported file.

Command-line usage for three tiles, each 192 pixels high and
160 pixels wide. Raw tensor files require explicit dimensions.

```powershell
gaterv3 run --model .\gaterv3.gtmodel --precision mixed --batch 3 --height 192 --width 160 --input .\tiles.f32 --output .\restored.f32 --cache-dir .\cache\gaterv3
```

Build and export from the checkout root using system Python. The build
code lives in `hatch_build.py`, which also compiles the library for every
Python wheel; `scripts/build_native.py` runs it without packaging.
[Python install and build hook](gaterv3-python.md#install)

```powershell
$py = 'C:\Python314\python.exe'
& $py scripts/build_native.py --output output/native-refactor
& $py scripts/export_native.py --output output/native-refactor/gaterv3.gtmodel
```

The EXE and DLL load the built CUDA module from their own resources.
Use `--arch` at build time for the target GPU when it needs a different
CUDA architecture. Deployment also needs the CUDA driver, cuBLASLt,
cuDNN, and MSVC runtime. The raw CLI accepts `--direct` to skip CUDA
Graph capture, and `--device` to select a CUDA device.

Build the C session example with the MSVC developer shell and place the
DLL beside the resulting EXE or on the DLL search path.
[C session example](../examples/run_gaterv3.c)

```powershell
cl /std:c17 /O2 /MD /I native/gaterv3 examples/run_gaterv3.c output/native-refactor/gaterv3.lib /Fe:output/native-refactor/run_gaterv3.exe
& .\output\native-refactor\run_gaterv3.exe .\output\native-refactor\gaterv3.gtmodel
```

The C API should offer the same flow through one versioned session options
structure and one inference entry point for both precisions:

| Operation | Contract |
| --- | --- |
| Open model | Return an opaque session from a UTF-8 model path and options for device, precision, and optional cache directory. Load and upload weights once per session. |
| Describe I/O | Return RGB input/output types and required output shape and bytes for the requested batch, height, and width. The current model is 1×. |
| Run | Take caller-owned contiguous buffers and their byte lengths, plus batch, height, and width. The basic host call is synchronous. |
| Inspect | Report status and a readable diagnostic, including allocation size for memory failures, and expose buffer sizes and cache status for the active shape. |
| Close | Release the session and its owned GPU resources. |

Initialize `struct_size` in the [C declarations](../native/gaterv3/runner.h).
`gt_session_describe` reports the required byte lengths, and run and
enqueue validate them before dispatch. `gt_session_info` reports active
shape, setup times, cache state, and memory. A shape allocation failure
returns `GT_STATUS_OUT_OF_MEMORY` so the caller can retry with a smaller
request. The fixed-size entry points and positional CLI are available
with the legacy bundle, exported via `--legacy`. The benchmark uses the
packed API if the bundle directory contains `gaterv3.gtmodel` or a model
path is supplied with `--model-file`.

For GPU callers, `gt_session_enqueue` takes caller-owned device buffers
and a stream in the selected device's primary context. Synchronize the
stream before reusing or freeing the buffers. Serialize calls to a
session. A subsequent call waits for workspace reuse and shape changes.
Independent sessions own separate buffers and can run concurrently.

| Precision option | Feature tensors and accumulators | Host and device RGB I/O |
| --- | --- | --- |
| `mixed` (default) | FP16 features with the existing FP32 accumulation and final skip arithmetic | FP32 |
| `fp16` | Existing full-FP16 execution path | FP16 |

Precision is a session option. Both modes read the same model file and
select the appropriate kernel and RGB residual weights at runtime.

### Session optimization options

Set any combination of these opt-in flags when opening a session.
The flags apply to every requested shape. The packed CLI lists them
under `gaterv3 run --help`.

| CLI option | C session flag | Behavior |
| --- | --- | --- |
| `--fuse-silu` | `GT_CREATE_FUSE_SILU` | Tries cuDNN convolution/SiLU fusion at eligible 128×128 sites and uses split kernels if cuDNN provides no compatible plan. |
| `--tune-projections` | `GT_CREATE_TUNE_PROJECTIONS` | Benchmarks more cuBLASLt candidates for eligible 1×1 projections using CUDA Graph replay. The exporter folds the QKV weights into the projection for every packed model. |
| `--fuse-gate-pool` | `GT_CREATE_FUSE_GATE_POOL` | Fuses eligible 128×128, 32-channel grouped gate and pooling sites in both precisions. |
| `--fp16-qkv` | `GT_CREATE_FP16_QKV` | Uses FP16 accumulation for the folded QKV projections in mixed precision. Full FP16 already uses FP16 accumulation. |

```c
GtSessionOptions options = {0};
options.struct_size = sizeof(options);
options.flags = GT_CREATE_GRAPH | GT_CREATE_TUNE_PROJECTIONS | GT_CREATE_FUSE_GATE_POOL;
options.cache_dir = "cache/gaterv3";
GtSession* session = gt_session_open("gaterv3.gtmodel", &options);
```

After preparing a shape, `gt_session_info` reports the active fusion
and QKV counts. `GT_OPT_REPLAY_TUNED_SHAPES` counts only shapes tuned
during that preparation, so a disk-plan hit can report zero.
`gt_session_fusion_message` describes a convolution/SiLU fallback.
[C API declarations](../native/gaterv3/runner.h)

The combined flag check at 128×128 replay-tuned 13 projection shapes
and fused four gate–pool sites in each precision. Mixed precision fused
12 convolution/SiLU sites and selected FP16 accumulation for 12 QKV
projections. Full FP16 reported a cuDNN convolution/SiLU fallback.
The CLI and DLL produced identical outputs in both precisions.
[Session options check](../output/native-session-options/session-options-check.json)

## Single model file

- [x] Add a versioned little-endian container with a section table for the
  architecture, shape-independent operation list, packed weights, and
  checkpoint identity. Include the input channels, output scale, model
  padding multiple, attention parameters, and supported precision modes.
  Store both RGB skip precisions and share the common packed feature weights.
- [x] Lower the current checkpoint with the existing strict loader in system
  Python. Retain Conv3XC and QKV weight folding. Replace the fixed H/W tensor
  table and arena offsets with operations whose shapes can be resolved on
  load or on the first run for a new shape.
- [x] Validate the magic, format version, section bounds, required sections
  and integrity checks at load. Detect incompatible files before uploading
  weights and provide a clear error. Export atomically so an interrupted
  export leaves the prior file usable.
- [x] Demonstrate deployment from a fresh directory with the DLL and the
  one model file. Test the CLI and import-library consumer from a path with
  spaces and non-ASCII characters.

The format represents the current GateRV3 architecture. The weights,
metadata, and operation description are independent of batch size, tile
size, and the locally selected CUDA plans.

## Variable tile and batch sizes

- [x] Resolve each request from positive `N`, `H`, and `W` within the
  available memory and CUDA/kernel indexing limits. Accept rectangular and
  odd dimensions. Tiles within a dense NCHW batch share H and W. Callers can
  group differently sized tiles by shape and submit those groups separately.
- [x] Derive runtime tensor shapes, strides, convolution plans, temporary
  lifetimes, workspace and launch geometry from the input. Compute required
  bytes with checked arithmetic before device allocation. Process the last
  batch at its actual size. Define memory errors that let a caller retry with
  a smaller batch.
- [x] Pad right and bottom to the checkpoint's multiple of 16, then crop
  output back to original H and W. Match the Torch model's reflection
  padding where defined. For an axis so small that the required reflection
  pad is at least as long as that axis, use proposed edge replication on
  that axis and validate against a Torch reference adapter with the same
  preprocessing. Page-tiling edge replication remains a separate caller
  operation.
- [x] Generalize the shape-specialized CUDA code, including shuffle routing,
  packed depthwise kernels, reductions, and latent attention. The current
  latent kernel assumes an 8×8 grid and 64 tokens. The dynamic path must
  preserve global spatial attention across
  `(padded_height / 16) × (padded_width / 16)` tokens, with bounds for
  partial blocks. Evaluate tiled softmax to avoid materializing the full
  attention matrix at larger sizes. Keep the current 64-token fast path for
  the matching geometry.
- [x] Keep a direct dispatch path for new or one-off shapes. Cache CUDA
  Graphs in memory for repeated shapes with matching layouts, mode and
  stable buffer addresses. Bound resident shape graphs and arenas so input
  variety cannot grow VRAM indefinitely. Verify shape transitions including
  returning to an earlier shape after eviction.

## Configurable kernel plan cache

Use an optional `cache_dir` session option and a matching `--cache-dir`
CLI flag. An omitted or empty setting disables disk caching. A supplied
directory stores native planning artifacts and can be deleted and recreated
independently of the model. This follows the caller-selected-directory
pattern of the TensorRT-RTX execution provider.
[ONNX Runtime TensorRT-RTX cache documentation](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html#runtime-cache)

- [x] Persist selected cuBLASLt and cuDNN plans, compute and workspace
  requirements, and the metadata needed to validate them. Key entries by
  model content, native plan format and runtime/kernel build, precision,
  batch, padded H and W, layout, GPU identity/capability and CUDA driver,
  cuBLASLt and cuDNN versions. Include plan-affecting workspace limits when
  set. Keep captured CUDA Graph instances in live sessions and rebuild them
  from the cached plans and current buffer addresses.
- [x] Validate cached plans before use and fall back to tuning on a miss or
  incompatible entry. A corrupt entry or unwritable directory should
  produce a diagnostic while inference continues with an in-memory plan.
  Write cache entries atomically and tolerate concurrent readers and writers.
- [x] Report cache hit, miss, or rebuild status, with separate timings for
  model load, plan lookup or tuning, buffer allocation and graph capture.
  Test a fresh process against a warm disk cache and verify that changing
  the model, precision, shape or library version selects a compatible plan.

The cache contains derived planning data and the model file remains the
source of weights and the operation graph. cuBLASLt supports restoring its
serialized algorithm descriptor with the same library version.
[cuBLASLt algorithm descriptor documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgo-t)

## Implementation order and validation

1. Freeze the verified split bundle and reports as controls. Validate the
   [activation repack](gaterv3-native-optimization.md#1-repack-the-activation-arena)
   on the fixed path so its VRAM effect can be measured independently.
2. Add the container and versioned session API with the existing fixed
   execution path. Prove output parity through both the DLL and CLI using
   one exported file for both precision modes.
3. Add shape planning and variable kernel dispatch. Test mixed and full FP16
   at 128×128 batches 1 and 16, at non-power-of-two batches such as 3 and 7,
   and at batches above 16 where memory permits. Include rectangular tiles,
   odd dimensions, a tiny tile, and a larger latent grid. Exercise shape
   changes and a final partial batch from the same session.
4. Add the disk plan cache and shape graph reuse. Check cold and warm
   processes, cache invalidation, concurrent writes, capture versus direct
   output parity, and bounded VRAM after repeated shape changes.
5. Benchmark the convolution and SiLU candidate separately. Keep its effects
   distinguishable from the format, shape, and memory-planning changes.

Use untouched Torch CUDA FP32 as the reference for supported input shapes.
Record per-case raw-float PSNR, finite-value checks, maximum absolute error,
and SSIMULACRA2 for the rendered pages. Compare the new fixed cases against
the verified same-precision native control before promoting the refactor.
Keep eager FP16, TorchInductor FP16, and TensorRT-RTX comparisons at their
validated shapes, with RTX placement verified. Save every synchronized
batch-1 tile time, batch-16 per-page time and throughput, cold and cached
startup times, and isolated VRAM for both modes. Run the repo's Ruff and
Pyrefly checks when implementation changes Python.

## Measurements on the RTX 3050 Laptop GPU

### Variable shapes and cache recovery

One session per precision processed batches 1, 3, 7, 16, and 17, with
aligned and odd rectangles, a 1×7 tile, and a return to an earlier shape.
The graph, direct host path, and caller-supplied CUDA stream produced
bit-identical output in every case. The two batch-7 rectangles shared
their padded 80×80 graph, and returning to 128×128 hit the disk cache.
An oversized 60×512×512 request reported the required and available
bytes, and the same session then accepted a smaller request. Corrupt
and blocked cache locations were exercised with successful inference.

| Batch × tile | Mixed raw PSNR (dB) | Full FP16 raw PSNR (dB) |
| --- | ---: | ---: |
| 1 × 128×128 | 78.978 | 70.607 |
| 16 × 128×128 | 78.972 | 70.497 |
| 3 × 192×160 | 78.895 | 70.424 |
| 7 × 65×79 | 78.995 | 70.532 |
| 7 × 79×65 | 78.973 | 70.533 |
| 17 × 128×128 | 79.001 | 70.475 |
| 1 × 160×112 | 78.821 | 70.448 |
| 1 × 1×7 | 84.625 | 70.265 |

The 1×7 reference used the documented edge-replication adapter and
Torch's math attention backend. The fused Torch backend rejected the
singleton latent token's strides. The other rows used the regular
Torch CUDA FP32 forward. At batch 16 the packed and legacy runners
matched bit for bit on the seeded input in both precisions. At batch 1
the maximum differences were 0.000259 in mixed and 0.001587 in FP16.

The dynamic attention kernel combines blocks of 32 keys online and
keeps linear activation storage. The existing tensor-core kernel is
used when the latent grid contains 64 tokens.

### Deployment and cache keys

The CLI and import-library example ran from a fresh directory whose
path contained spaces and Japanese characters. The deployment used
the single packed model and the CUDA kernels embedded in the
runtime. For each precision, the CLI's direct output matched the
DLL's graph output exactly on a 63×79 tile. The loader rejected
invalid magic, version, section bounds, and checksum.

The new DLL also loaded the legacy bundle through the fixed API in
both precisions. On the seeded 128×128 sample, mixed output matched the
old DLL exactly and full FP16 differed by at most 0.000977.

Two processes wrote to a shared cache and a subsequent reader hit
the resulting entry. A change to the model content or the loaded
cuBLASLt build selected a different entry. Returning to 128×128
after four shape cycles consistently measured 86 MiB above the
context in mixed precision and 82 MiB in full FP16 during the
separate residency check.

### Fixed reference shapes

The packed and legacy paths used the same checkpoint and the same
128×128 tile geometry. Each quality row compares its runner with
untouched Torch CUDA FP32 before PNG quantization for PSNR and after
rendering for SSIMULACRA2. Each page cell shows PSNR in dB followed by
SSIMULACRA2.

| Runner | Aggregate PSNR (dB) | Mean SSIMULACRA2 | page-001 | page-005 | page-009 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Packed mixed | 85.547 | 92.352 | 83.800 / 94.469 | 88.843 / 91.111 | 85.391 / 91.475 |
| Packed full FP16 | 72.089 | 91.869 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 |
| ORT TensorRT-RTX FP16 | 72.133 | 91.899 | 73.818 / 94.070 | 71.707 / 90.548 | 71.275 / 91.080 |

Each batch-1 value averages 528 synchronized individual host calls
overall, with 176 calls per page. The reports retain each tile's
coordinates and elapsed time. The RTX profiling session verified
placement on the fixed batch-1 graph.

| Runner | Overall tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Packed mixed | 5.244 | 5.281 | 5.281 | 5.171 |
| Legacy mixed | 5.330 | 5.292 | 5.362 | 5.336 |
| Packed full FP16 | 4.774 | 4.845 | 4.770 | 4.708 |
| Torch eager FP16 | 46.242 | 49.693 | 45.645 | 43.390 |
| TorchInductor FP16 | 17.613 | 18.201 | 17.609 | 17.028 |
| ORT TensorRT-RTX FP16 | 6.072 | 6.166 | 6.127 | 5.922 |

Each batch-16 row times 33 host-I/O calls plus input batch assembly.
Torch and RTX controls use the same input pages and tile geometry.
The profiled TensorRT-RTX graph reported RTX placement for batch 16
as well. The runtime cache held entries for both fixed graphs.

| Runner | Total (s) | MPx/s | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Packed mixed | 2.145 | 3.927 | 0.713 / 3.937 | 0.715 / 3.927 | 0.717 / 3.918 |
| Legacy mixed | 2.143 | 3.931 | 0.713 / 3.936 | 0.710 / 3.958 | 0.720 / 3.901 |
| Packed full FP16 | 1.809 | 4.657 | 0.601 / 4.669 | 0.599 / 4.691 | 0.609 / 4.611 |
| Torch eager FP16 | 5.828 | 1.445 | 1.969 / 1.426 | 1.886 / 1.489 | 1.973 / 1.423 |
| TorchInductor FP16 | 7.725 | 1.090 | 2.563 / 1.096 | 2.648 / 1.060 | 2.514 / 1.117 |
| ORT TensorRT-RTX FP16 | 2.649 | 3.180 | 0.888 / 3.163 | 0.853 / 3.292 | 0.908 / 3.091 |

The separate full-FP16 legacy comparison measured 5.060 ms per
individual tile with the packed runner and 5.008 ms with the legacy
runner. Packed batch-16 page throughput varied between the two runs.
In the direct comparison it measured 4.172 MPx/s against 4.400 MPx/s
for the legacy runner. Its repeated batch-16 host median was
54.544 ms against 55.134 ms. The full per-page and individual tile
measurements are in the verified report.

The separate convolution/SiLU experiment fused 12 sites in mixed
precision. The repeated batch-16 host median was 59.591 ms with fusion
and 63.359 ms with the split kernels. In full FP16 the cuDNN probe found
no supported fused plan, so it used the existing split path. The
experimental flag stays opt-in.

The populated-cache measurements ran in a fresh Python process.
Warmed VRAM is the device-wide increase above the initial CUDA context.
Cold peak is the highest sampled startup value above that same baseline.

| Packed runner | Batch | Warmed VRAM (MiB) | Cold sampled peak (MiB) | First output cold / cached (ms) |
| --- | ---: | ---: | ---: | ---: |
| Mixed | 1 | 98 | 154 | 643 / 469 |
| Mixed | 16 | 210 | 274 | 1,053 / 339 |
| Full FP16 | 1 | 84 | 150 | 674 / 198 |
| Full FP16 | 16 | 204 | 268 | 5,710 / 243 |

The [verified report](../output/native-refactor-validation/verified-report.json)
links the source measurements, including all 528 per-tile timings
per runner, per-page throughput, raw quality, shape transitions,
deployment checks, and profiled RTX placement.
