# Adding new model kernels

## Confirm the workload

Before starting any new model work or changing its benchmark workload, get
both workload details from the user.

- Batch size and input tile dimensions.
- Image files or sample directory to use for benchmarking and testing.

Ask for any missing choice and wait for the answer before starting. Use details
the user already supplied without asking again. Record the overlap used and
keep the checkpoint, images, and tiling identical across backends.

## Establish the baseline before optimizing

1. Verify that Resselt recognizes the checkpoint and can run a forward pass.
   Record the architecture, output scale, and any loading or execution blocker.
2. Run the original model in eager Torch CUDA FP32 as the reference. Benchmark
   TorchInductor FP16 and ONNX Runtime TensorRT-RTX FP16 on the same workload.
   For TensorRT-RTX, disable ORT telemetry and fallback, register the provider
   through the execution-provider device API, and verify that it runs the graph.
   Record the error and environment if a backend cannot complete the workload.
3. Reproduce the network in the native runtime and measure mixed precision and
   FP16. Document which operations and input/output tensors use each precision.
   Native FP32 is optional for reference and parity checks.
4. Record the first working measurements in `docs/` before tuning. For each
   later optimization, use the same workload to check speed and fidelity.
   Consider a kernel option if the change introduces a precision, memory, or
   hardware tradeoff that warrants keeping both paths available.

## Report timing, fidelity, and VRAM

- Report inference latency per image, the total, and the median across
  images, in milliseconds. You need to also calculate the MPx/s for the
  images. Identify the measured region and exclude model loading,
  compilation, warmup, and metric calculation from steady-state timing.
- Report ms/tile per image and the median of those per-image values. For
  batch-1 latency, time each tile in a synchronized batch-1 inference call and
  retain its elapsed time, image name, tile index, and coordinates. For a
  throughput run at a larger batch size, measure batch-1 latency separately
  and report batch inference time and throughput independently. Use a
  separate fixed batch-1 graph for TensorRT-RTX tile latency.
- Report per-image and summary PSNR against the matching raw-float Torch FP32
  output with peak value 1, before PNG quantization. Report SSIMULACRA2
  separately using the matching rendered reference and
  `C:\Tools\imaging\ssimulacra2.exe`.
- Report sampled peak device VRAM per image and the peak across images.
  Record the background before backend setup and distinguish device-wide VRAM
  from memory tracked by a backend's allocator.
- Record the checkpoint, input set, hardware, software versions, workload,
  commands, and artifact paths alongside the measurements.

## Keep a model handoff

Put each model's workload, architecture, package layout, precision choices,
verification commands, and results in a separate document under `docs/`.
Update it as decisions land and record failures, remaining checks, and where
to pick up the next optimization.
