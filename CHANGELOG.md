# Changelog

## Unreleased

Nothing yet!

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
