# GateRV3 native runtime refactor plan

Package the current 1× GateRV3 checkpoint as one model file loadable by
one Windows DLL. A session selects mixed precision or full FP16 and accepts
new tile and batch sizes on subsequent calls. Keep the verified 128×128
runner as the performance and fidelity control. This document describes
the proposed contract and implementation sequence.

## Target usage

Ship `gaterv3.dll`, its C header, and an MSVC import library for consumers
who link at build time. Export one file such as `gaterv3.gtmodel`. The
application opens that file once, selects a CUDA device and precision, and
runs repeated calls with contiguous RGB NCHW input. The DLL's generic CUDA
kernels are embedded in the DLL. The model file holds the execution
description and weights, so inference needs only the DLL, its CUDA and
MSVC runtime dependencies, and the exported file.

Proposed command-line usage for three tiles, each 192 pixels high and
160 pixels wide. Raw tensor files require explicit dimensions.

```powershell
gaterv3 run --model .\gaterv3.gtmodel --precision mixed --batch 3 --height 192 --width 160 --input .\tiles.f32 --output .\restored.f32 --cache-dir .\cache\gaterv3
```

The C API should offer the same flow through one versioned session options
structure and one inference entry point for both precisions:

| Operation | Proposed contract |
| --- | --- |
| Open model | Return an opaque session from a UTF-8 model path and options for device, precision, and optional cache directory. Load and upload weights once per session. |
| Describe I/O | Return RGB input/output types and required output shape and bytes for the requested batch, height, and width. The current model is 1×. |
| Run | Take caller-owned contiguous buffers and their byte lengths, plus batch, height, and width. The basic host call is synchronous. |
| Inspect | Report status and a readable diagnostic, including allocation size for memory failures, and expose buffer sizes and cache status for the active shape. |
| Close | Release the session and its owned GPU resources. |

Use fixed-width types and a struct-size field for ABI extensions. Validate
buffer lengths before dispatch. Keep the existing fixed-size entry points
working during migration and route the CLI and Python benchmark through the
new API. A small C or C++ example should demonstrate the normal path.
For GPU callers, provide an advanced entry point with explicit device
pointers and stream, and document when buffers may be reused or freed.
Serialize calls within one session initially. Independent sessions can own
their own buffers and run concurrently.

| Precision option | Feature tensors and accumulators | Host and device RGB I/O |
| --- | --- | --- |
| `mixed` (default) | FP16 features with the existing FP32 accumulation and final skip arithmetic | FP32 |
| `fp16` | Existing full-FP16 execution path | FP16 |

Precision is a session option. Both modes read the same model file and
select the appropriate kernel and RGB residual weights at runtime.

## Single model file

- [ ] Add a versioned little-endian container with a section table for the
  architecture, shape-independent operation list, packed weights, and
  checkpoint identity. Include the input channels, output scale, model
  padding multiple, attention parameters, and supported precision modes.
  Store both RGB skip precisions and share the common packed feature weights.
- [ ] Lower the current checkpoint with the existing strict loader in system
  Python. Retain Conv3XC and QKV weight folding. Replace the fixed H/W tensor
  table and arena offsets with operations whose shapes can be resolved on
  load or on the first run for a new shape.
- [ ] Validate the magic, format version, section bounds, required sections
  and integrity checks at load. Detect incompatible files before uploading
  weights and provide a clear error. Export atomically so an interrupted
  export leaves the prior file usable.
- [ ] Demonstrate deployment from a fresh directory with the DLL and the
  one model file. Test the CLI and import-library consumer from a path with
  spaces and non-ASCII characters.

The format represents the current GateRV3 architecture. The weights,
metadata, and operation description are independent of batch size, tile
size, and the locally selected CUDA plans.

## Variable tile and batch sizes

- [ ] Resolve each request from positive `N`, `H`, and `W` within the
  available memory and CUDA/kernel indexing limits. Accept rectangular and
  odd dimensions. Tiles within a dense NCHW batch share H and W. Callers can
  group differently sized tiles by shape and submit those groups separately.
- [ ] Derive runtime tensor shapes, strides, convolution plans, temporary
  lifetimes, workspace and launch geometry from the input. Compute required
  bytes with checked arithmetic before device allocation. Process the last
  batch at its actual size. Define memory errors that let a caller retry with
  a smaller batch.
- [ ] Pad right and bottom to the checkpoint's multiple of 16, then crop
  output back to original H and W. Match the Torch model's reflection
  padding where defined. For an axis so small that the required reflection
  pad is at least as long as that axis, use proposed edge replication on
  that axis and validate against a Torch reference adapter with the same
  preprocessing. Page-tiling edge replication remains a separate caller
  operation.
- [ ] Generalize the shape-specialized CUDA code, including shuffle routing,
  packed depthwise kernels, reductions, and latent attention. The current
  latent kernel assumes an 8×8 grid and 64 tokens. The dynamic path must
  preserve global spatial attention across
  `(padded_height / 16) × (padded_width / 16)` tokens, with bounds for
  partial blocks. Evaluate tiled softmax to avoid materializing the full
  attention matrix at larger sizes. Keep the current 64-token fast path for
  the matching geometry.
- [ ] Keep a direct dispatch path for new or one-off shapes. Cache CUDA
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

- [ ] Persist selected cuBLASLt and cuDNN plans, compute and workspace
  requirements, and the metadata needed to validate them. Key entries by
  model content, native plan format and runtime/kernel build, precision,
  batch, padded H and W, layout, GPU identity/capability and CUDA driver,
  cuBLASLt and cuDNN versions. Include plan-affecting workspace limits when
  set. Keep captured CUDA Graph instances in live sessions and rebuild them
  from the cached plans and current buffer addresses.
- [ ] Validate cached plans before use and fall back to tuning on a miss or
  incompatible entry. A corrupt entry or unwritable directory should
  produce a diagnostic while inference continues with an in-memory plan.
  Write cache entries atomically and tolerate concurrent readers and writers.
- [ ] Report cache hit, miss, or rebuild status, with separate timings for
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
