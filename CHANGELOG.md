# Changelog

## Unreleased

Nothing yet!

## [0.0.6] - 2026-11-25

### Added

- GPU page tiling (`tiling="fixed"` and `tiling="variable"`) via
  `InferenceSession.run_tiled()`, `umzi_session_run_tiled()`, and
  `umzi_session_run_tiled_variable()`, with overlap feathering, a consolidated
  device arena, ring-buffered strip blending (`blend_tiles` / `finish_strip`),
  and shared immutable session state (`clone_shared()`) across variable-tiling
  edge shapes.
- Expanded GateRV3 tile-size support (including non-128 and dynamic spatial
  shapes) plus a `workspace_mib` kernel option (default `64` MiB).

### Changed

- Reduced VRAM usage and improved kernel throughput across native runners:
  - **GateRV3:** bounded activation arena packing, Level-0 in-place
    `Inception`, fused `Concat+Conv1x1`, and Tensor Core FP32-accumulated
    `Conv3x3+SiLU` / `Conv3x3+Span` kernels.
  - **GFISRv2 & MoESR:** aliased `freq_a`/`freq_b` inside `fft_buf` for reduced
    GFISRv2 precisions, and lean package-backed weight repacking on `fused_conv`
    / `fused_gate` fallback instead of retaining dual weight buffers.
  - **FiGSR:** compacted `Expanded`/`Values` staging with `fuse_expand_gate`.
  - **All kernels:** lazy host/device I/O staging and non-graph direct device
    execution.
- Unsupported precision/option combinations in **FiGSR** and **SMoSR** (e.g.,
  `fp32` with `fuse_silu`, `tune_convolutions`, or `fuse_expand_gate`) now log
  a warning via `kernel_message` and disable the option instead of throwing.

### Fixed

- Resolved native runtime and kernel audit issues covering CUDA context/handle
  lifetimes and per-clone cuDNN/cuBLASLt isolation, GPU tiler stream
  synchronization and single-runner OOM eviction, C ABI / diagnostic thread
  safety, Windows cache path handling, and edge-case shape/buffer bounds.

## [0.0.5] - 2026-11-25

### Added

- Native CUDA support (exporter, `peek_model()`, CUDA Graph and direct
  sessions) for two new 2× upscaling model families:
  - **SMoSR** (`fp32`/`mixed`/`fp16`): 2× direct pixel-shuffle network with
    internal reflection padding, schema 2 FP32 weight storage (preserving
    schema 1 compatibility for `mixed` and `fp16`), and `tune_convolutions`,
    `fp16_accumulation`, and `fuse_silu` (with `smosr-silu_*.umplan` caching)
    options.
  - **FiGSR** (`fp32`/`mixed`/`fp16`): 2× direct pixel-shuffle network using
    dynamic cuFFT 12 for FP32 Fourier transforms, fused tensor-core
    expansion/gate (`fuse_expand_gate`) and row gate (`fuse_row_gate`) kernels,
    plus `tune_convolutions`, `fp32_nchw`, and `fp16_expand_accumulation`
    options.

### Changed

- Shared `native/common/cufft_api.h` dynamic cuFFT 12 loader across GFISRv2 and
  FiGSR, and updated `SessionOptions(dll_dirs=...)` to preload cuFFT 12
  automatically for FiGSR packages as well.
- Extended `native/common/cudnn_conv_plan.h` with NCHW layout, grouped and
  rectangular convolutions, strict FP32 numerical filtering, and fused bias +
  SiLU graph support.

## [0.0.4] - 2026-11-23

### Changed

- Improved GateRV3 attention kernel fidelity scores while keeping the same
  throughput and memory usage.

## [0.0.3] - 2026-11-22

### Added

- Native CUDA support (exporter, `peek_model()`, CUDA Graph session, and
  `.umplan` plan cache) for three new model families:
  - **GateRV2** (`mixed`/`fp16`): fused NHWC kernels, FP32 linear
    attention, shared activation arena, `tune_plans` / `workspace_mib` options.
  - **MoESR** (`mixed`/`fp16`): custom tensor-core fused expand/project and
    tiled depthwise kernels with 0 B retained cuDNN workspace, `fused_gate` /
    `workspace_mib` options.
  - **GFISRv2** (`fp32`/`mixed`/`fp16`): dynamic cuFFT 12 loading (no cuFFT
    dependency for other families), bit-exact FP32 path, fused NHWC
    tensor-core/spectral reduced-precision path with 0 B retained cuDNN
    workspace, `fused_conv` / `workspace_mib` / `unexact_erf` options.
- Linux x64 (`manylinux_2_34_x86_64`) wheel and native SDK CI builds with ELF
  ABI/dependency verification, GPU-less `libcuda.so.1` stub linking in
  `hatch_build.py`, and multi-platform tag publishing.
- Quickstart, export guide, and consolidated final timing/fidelity tables in
  `docs/`.

### Changed

- Consolidated shared CUDA/cuDNN helpers, backend conv plans, and Ampere+ PTX
  tensor-core primitives into `native/common/`.
- Moved the `umzi_native` CLI into standalone `native/runtime/cli.cpp` (linked
  into the CLI binary only) with generic `--option name=value` kernel options.
- Aligned `startup_device_used_bytes`, common plan diagnostics (`timed_plans`,
  `plan_workspace_bytes`, `peak_plan_workspace_bytes`), and `kernel_message()`
  reporting across all four kernel families.

### Removed

- Removed `native/kernels/gaterv3/runner.h`, all 20 legacy `gt_*` DLL exports
  (leaving only the 17 base `umzi_*` ABI exports), the legacy fixed-bundle
  loader, and `legacy_cli`.

### Fixed

- Fixed missing/incomplete `startup_device_used_bytes` reporting in GateRV2,
  MoESR, and GFISRv2, and missing index bounds checking in MoESR's
  `umzi_session_diagnostic()`.
- Fixed GFISRv2 `mixed` TF32 drift on the spectral 1×1 projection by using a
  deterministic FP32 FMA kernel.

## [0.0.2] - 2026-11-22

- Extended the opt-in GateRV3 `fuse_gate_pool` and `fuse_silu` paths to
  eligible 32-channel feature maps at other padded tile sizes, including
  rectangles. Conv/SiLU uses a split fallback when cuDNN cannot fuse the shape.
- Reuse cached GateRV3 convolution and projection algorithms when toggling
  gate/pool or Conv/SiLU fusion.

## [0.0.1] - 2026-11-22

Initial release targeting `0.0.1`.

### Added

- Native CUDA inference for the original GateRV3 Digital Decompression model
  and the IDC fine-tune. The Python API accepts NumPy batches with variable
  batch size and tile dimensions in mixed or FP16 precision.
- `InferenceSession` and `SessionOptions` with CUDA Graph replay, explicit
  CUDA dependency loading, optional `.umplan` disk caching, and diagnostics.
- Version 1 `.ummodel` packages with the `UMZINPKG` magic bytes, per-kernel
  dispatch, and model inspection through `peek_model()` and
  `python -m umzi_native read`.
- Optional `exporter` extra with Resselt for CPU export from SafeTensors and
  PyTorch weight files. Export through `python -m umzi_native export` or
  `umzi_native.exporter.export_model()`.
- Windows x64 wheels with the native DLL, a separate native SDK artifact, and
  tag-triggered publishing to the private Python package index. Source builds
  use Python 3.12 or newer and CUDA 13.

### Fixed

- GateRV3 packages preserve the checkpoint's required padding alignment.
  The original model uses 16 pixels and IDC uses 8.
- The runtime falls back to embedded PTX when it cannot use a compatible cubin.
