# Repository instructions

Follow [Adding new model kernels](docs/adding-new-kernels.md) for new model
support and benchmarks. The GateRV3 settings below apply to that model.

## Environment and GateRV3 baseline

- Use `C:\Python314\python.exe` for Python commands in this checkout. The
  installed global environment contains CUDA PyTorch, ONNX, ONNX Runtime, and
  the TensorRT-RTX provider plugin. Do not switch to another Python environment
  without a direct request.
- The checkpoint is
  `models/1x_umzi_digital_decompress_gaterv3_1.pth`; input images are in
  `samples/input/`.
- The baseline configuration is 128×128-pixel tiles, batch size 16, and FP16
  candidates. Keep the untouched Torch CUDA FP32 model as the numeric reference.
- Compare eager FP16, TorchInductor FP16, and ONNX Runtime TensorRT-RTX FP16.
  Disable ORT telemetry, register TensorRT-RTX using its execution-provider
  device API, disable fallback, and verify the requested provider is active.
- Use `C:\Tools\imaging\ssimulacra2.exe` to compare rendered candidates with
  the matching FP32 reference images. Keep per-image scores as well as summary
  values.

## Benchmark measurements

- Report raw-float PSNR against the matching FP32 model output, with peak value
  1 and before PNG quantization. Report SSIMULACRA2 separately; it is not PSNR.
- Report batch-16 inference time and throughput independently, both overall
  and for each image.
- Measure actual per-tile latency by timing every tile in a synchronized,
  individual batch-1 inference call. Save each tile's elapsed time, image name,
  tile index, and coordinates, plus an average calculated from those individual
  samples. Do not estimate per-tile latency by dividing batch-16 runtime by the
  number of tiles. For TensorRT-RTX, use a separate fixed batch-1 graph/session
  for this latency measurement.
- Keep output names clear about whether a value is batch-16 throughput or
  individual batch-1 tile latency. Do not label latency as tiles per second.
- Keep the detailed baseline instructions and measured results in `docs/`.

## Python formatting and checking

After changing Python files, always run these commands from the repository
root:

```powershell
ruff format scripts src
ruff check scripts src
pyrefly check
```

Resolve reported issues before completing the change. Keep `pyrefly.toml`
aligned with the source and script layout.
