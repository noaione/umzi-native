# Changelog

## Unreleased

Nothing yet!

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
