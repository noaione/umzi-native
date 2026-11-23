# GateRV3 Vulkan on Apple M4

This report records the original Vulkan shader baseline. The [optimization report](gaterv3-vulkan-optimization.md) covers the current kernels and native FP16 mode.

The standalone Vulkan runtime reads the existing 1× RGB GateRV3 `.ummodel` packages for Original and IDC. It uses a Vulkan 1.2 compute queue, two storage-buffer bindings, 64-invocation workgroups, and 4 KiB of shared memory for tiled convolution. The loader enumerates portability devices and enables the portability subset on devices that advertise it. The macOS run used MoltenVK and KosmicKrisp through the LunarG SDK. [Vulkan portability enumeration](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_portability_enumeration.html) · [KosmicKrisp requirements](https://docs.mesa3d.org/drivers/kosmickrisp.html)

The Vulkan library exposes a versioned host C ABI and uses the existing GateRV3 schedule and weight validation. Weights stay on the GPU and the session replays one prepared shape's command buffer. Each call synchronously stages NCHW NumPy input and output. The CUDA library remains the default for Python sessions.

| Vulkan precision option | Checkpoint weights | Activation storage and arithmetic |
| --- | --- | --- |
| `mixed` | Existing packed FP16 weights | FP32 storage and accumulation with FP16 rounding at intermediate outputs |
| `fp32` | Existing packed FP16 weights | FP32 storage and arithmetic throughout |

Both reference modes accept and return NumPy `float32` arrays. The session reports Vulkan buffer allocation sizes and dispatch counts. It supports host I/O, in-memory pipeline caching, and one prepared shape per session.

## Build and run

These commands use the system Python on macOS, CMake 3.23 or later, Vulkan SDK 1.3.268 or later with `glslc` and `spirv-val`, and OpenSSL. The benchmark also uses Pillow, PyTorch, Resselt, ONNX, ONNX Runtime with its WebGPU plugin, and `ssimulacra2` on `PATH`.

```sh
source "$HOME/VulkanSDK/1.4.363.1/setup-env.sh"
cmake -S native/vulkan -B build/vulkan -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
cmake --build build/vulkan --parallel

mkdir -p output/vulkan
python3 scripts/export_native.py --output output/vulkan/original.ummodel
python3 scripts/export_native.py --model models/1x_umzi_idc_gaterv3.safetensors \
  --output output/vulkan/idc.ummodel

library=build/vulkan/libumzi_native_vulkan.dylib
python3 scripts/benchmark_vulkan.py --devices --library "$library"
python3 scripts/benchmark_vulkan.py --model output/vulkan/original.ummodel \
  --library "$library" --output output/vulkan/original-matrix \
  --vulkan-reference --vulkan-device 0 --vulkan-device 1 --ssimulacra2 /opt/homebrew/bin/ssimulacra2
python3 scripts/benchmark_vulkan.py --checkpoint models/1x_umzi_idc_gaterv3.safetensors \
  --model output/vulkan/idc.ummodel --library "$library" --output output/vulkan/idc-matrix \
  --vulkan-reference --vulkan-device 0 --vulkan-device 1 --ssimulacra2 /opt/homebrew/bin/ssimulacra2
```

Use the enumerated indices on your machine. The reference matrix includes eager PyTorch FP32 and FP16, ONNX Runtime WebGPU FP32 and FP16, and Vulkan FP32 and mixed. Use `--vulkan-precision` or `--webgpu-precision` to select one mode. Use `--torch-device cuda` for the CUDA Torch reference on an NVIDIA host. The existing Windows TensorRT-RTX and TorchInductor protocol is documented separately. [GateRV3 CUDA baseline](../gaterv3-baseline.md)

To run the shape, replay, and validation-layer checks for a package:

```sh
UMZI_VULKAN_VALIDATION=1 VK_LAYER_VALIDATE_SYNC=1 python3 scripts/validate_vulkan.py \
  --checkpoint models/1x_umzi_digital_decompress_gaterv3_1.pth \
  --model output/vulkan/original.ummodel --library "$library"
```

Use `PYTHONPATH=src` for imports from the source checkout. The library can be selected explicitly or discovered in the package or beside the model.

```python
import numpy as np
from umzi_native import InferenceSession, SessionOptions

options = SessionOptions(
    backend="vulkan",
    device_id=0,
    dll_path="build/vulkan/libumzi_native_vulkan.dylib",
    kernel_options={"reference": 1},
)
with InferenceSession("output/vulkan/original.ummodel", options) as session:
    shape = (1, 3, 128, 128)
    session.prepare(shape)
    (result,) = session.run(None, {"input": np.zeros(shape, dtype=np.float32)})
```

## Measurement protocol

The host-to-host timer starts with the CPU tile tensor already assembled and stops when synchronous inference returns CPU FP32 output. Transfers and dtype conversion are included. Loading, preparing, warming up, tile assembly, PNG encoding, and scoring are outside the timed interval. Each variant warms up twice for each shape. Results cover a single sequential pass per variant.

Each page is 1404×2000 and yields 176 nonoverlapping 128×128 tiles, with edge replication for incomplete tiles and cropping on assembly. Batch-16 inference time sums 11 calls per page. Throughput counts the original page pixels. Batch-1 page latency sums 176 separately timed synchronous calls. The reported ms/tile is the mean of those calls, with 528 samples across all pages per runner and checkpoint. Vulkan and WebGPU each use distinct batch-1 and batch-16 sessions.

The numeric reference for this Mac run is the repository's eager FP32 PyTorch model on MPS, including the spatial attention forward for the Original checkpoint. Raw PSNR uses peak 1 and the stitched float output before clipping and quantization. The overall PSNR pools squared error over RGB samples. SSIMULACRA2 separately compares the rendered PNGs, with an arithmetic mean across pages.

WebGPU uses separate fixed-shape FP32 and FP16 ONNX exports, the plugin execution-provider device API, and a GPU device with the NHWC layout preference. Telemetry, CPU EP fallback and Python session fallback are disabled. The runner requires all graph assignments to name the WebGPU EP before it times a session. [ONNX Runtime WebGPU execution provider](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)

GPU memory on the M4 is unified. The tables report the sampled peak increase in Metal GPU allocations above a process baseline. Each backend, precision, and batch uses a fresh process. The worker samples Metal's `currentAllocatedSize` through PyTorch before setup, after prepare and warmup, and every 5 ms during three real inference calls. The reported peak is an observed high-water mark for the warmed calls, which may miss shorter allocation spikes. It includes driver-managed and cached Metal resources. Memory probes run separately from timed inference. [Apple Metal allocation counter](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize?language=objc) · [PyTorch MPS driver memory counter](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html)

The complete local records are `output/vulkan/original-matrix/report.json` and `output/vulkan/idc-matrix/report.json`. They contain per-image timing and quality, all individual batch-1 timings with coordinates, GPU memory samples summarized by shape, device capabilities, Vulkan buffer sizes, and reference output hashes.

## Fixtures

| File | SHA-256 |
| --- | --- |
| `1x_umzi_digital_decompress_gaterv3_1.pth` | `3e2c315094669301fdf0860276cd15e43dce91ee5f9851534d83fd340d30fd91` |
| `1x_umzi_idc_gaterv3.safetensors` | `6856db3dc48c3aa793a7032d4799fb2dad6706573032d7b961a43bf49e04876d` |
| `original.ummodel` | `04acf7f9ebffba37b4a2413271d429eb7d091b93cb7bbab70a6c6ab353378b69` |
| `idc.ummodel` | `88d1b9c690ee1ac268ec4d8a6b7bff8ded477115df1b69fdfcdaf2053bb95086` |
| `page-001.jpg` | `3d1ed058cb9ca7899276d1f061831977119a288dfd37d04c70c3ff1154032670` |
| `page-005.jpg` | `ccdec45af63db7c17b735d05f3e7c23fc58b2b93daf3454f83e1ddae87a2dd2a` |
| `page-009.jpg` | `ef2f8521887b4b2047bd6c1038811657004644db440bce5038ecb5b754e840be` |

## Measured results

Measured on November 23, 2026 on an Apple M4 (Mac16,10, 16 GiB unified memory) running macOS 27.2.1.
The run used Vulkan SDK 1.4.363.1, MoltenVK 1.4.2, KosmicKrisp build 269a955c89, Python 3.13.3,
PyTorch 2.15.0, Resselt 1.4.1, ONNX Runtime 1.31.1, and WebGPU plugin 0.5.0.
Both Vulkan drivers reported subgroup size 32, advertised FP16 arithmetic and 16-bit storage, and reported
the cooperative matrix capability as unavailable.

### All three pages

| Checkpoint | Backend and precision | Batch-16 s | Batch-16 MPx/s | Batch-1 s | Batch-1 ms/tile | Peak GPU MiB batch 1 / 16 | Raw PSNR dB | SSIMULACRA2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | Torch MPS FP32 | 15.419 | 0.546 | 23.127 | 43.80 | 1191.5 / 2234.3 | Reference | Reference |
| Original | Torch MPS FP16 | 12.626 | 0.667 | 20.369 | 38.58 | 123.3 / 1140.8 | 72.39 | 91.98 |
| Original | ORT WebGPU FP32 | 18.108 | 0.465 | 20.205 | 38.27 | 246.7 / 1477.6 | 155.60 | 96.30 |
| Original | ORT WebGPU FP16 | 11.665 | 0.722 | 15.272 | 28.93 | 125.8 / 790.9 | 68.54 | 91.86 |
| Original | MoltenVK FP32 | 20.980 | 0.402 | 28.960 | 54.85 | 87.1 / 338.3 | 100.23 | 92.50 |
| Original | MoltenVK mixed | 20.988 | 0.401 | 29.598 | 56.06 | 87.1 / 338.3 | 85.78 | 92.31 |
| Original | KosmicKrisp FP32 | 28.770 | 0.293 | 37.331 | 70.70 | 89.6 / 340.9 | 100.23 | 92.49 |
| Original | KosmicKrisp mixed | 28.198 | 0.299 | 37.391 | 70.82 | 89.6 / 340.9 | 85.77 | 92.33 |
| IDC | Torch MPS FP32 | 12.738 | 0.661 | 16.022 | 30.34 | 107.3 / 2110.1 | Reference | Reference |
| IDC | Torch MPS FP16 | 10.415 | 0.809 | 14.783 | 28.00 | 67.2 / 1076.6 | 69.72 | 92.16 |
| IDC | ORT WebGPU FP32 | 15.128 | 0.557 | 15.232 | 28.85 | 109.4 / 1277.7 | 133.58 | 94.02 |
| IDC | ORT WebGPU FP16 | 9.514 | 0.885 | 10.963 | 20.76 | 57.1 / 690.9 | 56.26 | 90.97 |
| IDC | MoltenVK FP32 | 15.865 | 0.531 | 20.675 | 39.16 | 26.7 / 278.0 | 76.10 | 92.33 |
| IDC | MoltenVK mixed | 15.858 | 0.531 | 20.568 | 38.95 | 26.7 / 278.0 | 71.80 | 92.04 |
| IDC | KosmicKrisp FP32 | 21.104 | 0.399 | 25.703 | 48.68 | 28.5 / 279.8 | 76.10 | 92.27 |
| IDC | KosmicKrisp mixed | 21.035 | 0.400 | 25.547 | 48.39 | 28.5 / 279.8 | 71.79 | 92.05 |

### Original per image

| Image | Backend and precision | Batch-16 s | Batch-16 MPx/s | Batch-1 s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | Torch MPS FP32 | 5.073 | 0.554 | 7.668 | 43.57 | Reference | Reference |
| page-001.jpg | Torch MPS FP16 | 4.200 | 0.669 | 6.673 | 37.91 | 74.20 | 94.24 |
| page-001.jpg | ORT WebGPU FP32 | 6.064 | 0.463 | 6.726 | 38.22 | 155.35 | 96.53 |
| page-001.jpg | ORT WebGPU FP16 | 3.897 | 0.721 | 5.106 | 29.01 | 67.20 | 93.98 |
| page-001.jpg | MoltenVK FP32 | 6.991 | 0.402 | 9.683 | 55.02 | 98.23 | 94.51 |
| page-001.jpg | MoltenVK mixed | 6.989 | 0.402 | 9.810 | 55.74 | 83.96 | 94.40 |
| page-001.jpg | KosmicKrisp FP32 | 9.488 | 0.296 | 12.455 | 70.77 | 98.23 | 94.53 |
| page-001.jpg | KosmicKrisp mixed | 9.384 | 0.299 | 12.481 | 70.91 | 83.96 | 94.38 |
| page-005.jpg | Torch MPS FP32 | 5.075 | 0.553 | 7.610 | 43.24 | Reference | Reference |
| page-005.jpg | Torch MPS FP16 | 4.213 | 0.666 | 6.750 | 38.35 | 71.79 | 90.67 |
| page-005.jpg | ORT WebGPU FP32 | 6.065 | 0.463 | 6.752 | 38.37 | 155.86 | 97.14 |
| page-005.jpg | ORT WebGPU FP16 | 3.888 | 0.722 | 5.053 | 28.71 | 70.23 | 90.65 |
| page-005.jpg | MoltenVK FP32 | 6.997 | 0.401 | 9.639 | 54.77 | 102.52 | 91.29 |
| page-005.jpg | MoltenVK mixed | 7.003 | 0.401 | 9.894 | 56.21 | 88.98 | 91.04 |
| page-005.jpg | KosmicKrisp FP32 | 9.611 | 0.292 | 12.456 | 70.77 | 102.52 | 91.29 |
| page-005.jpg | KosmicKrisp mixed | 9.363 | 0.300 | 12.451 | 70.75 | 88.96 | 91.14 |
| page-009.jpg | Torch MPS FP32 | 5.272 | 0.533 | 7.849 | 44.60 | Reference | Reference |
| page-009.jpg | Torch MPS FP16 | 4.212 | 0.667 | 6.946 | 39.47 | 71.62 | 91.04 |
| page-009.jpg | ORT WebGPU FP32 | 5.979 | 0.470 | 6.727 | 38.22 | 155.63 | 95.23 |
| page-009.jpg | ORT WebGPU FP16 | 3.880 | 0.724 | 5.113 | 29.05 | 68.71 | 90.95 |
| page-009.jpg | MoltenVK FP32 | 6.992 | 0.402 | 9.638 | 54.76 | 101.06 | 91.70 |
| page-009.jpg | MoltenVK mixed | 6.996 | 0.401 | 9.895 | 56.22 | 85.76 | 91.50 |
| page-009.jpg | KosmicKrisp FP32 | 9.671 | 0.290 | 12.420 | 70.57 | 101.06 | 91.64 |
| page-009.jpg | KosmicKrisp mixed | 9.451 | 0.297 | 12.459 | 70.79 | 85.76 | 91.48 |

### IDC per image

| Image | Backend and precision | Batch-16 s | Batch-16 MPx/s | Batch-1 s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | Torch MPS FP32 | 4.328 | 0.649 | 5.382 | 30.58 | Reference | Reference |
| page-001.jpg | Torch MPS FP16 | 3.484 | 0.806 | 4.907 | 27.88 | 69.76 | 94.12 |
| page-001.jpg | ORT WebGPU FP32 | 5.034 | 0.558 | 5.081 | 28.87 | 135.91 | 95.62 |
| page-001.jpg | ORT WebGPU FP16 | 3.177 | 0.884 | 3.659 | 20.79 | 58.76 | 93.51 |
| page-001.jpg | MoltenVK FP32 | 5.286 | 0.531 | 6.889 | 39.14 | 75.21 | 94.43 |
| page-001.jpg | MoltenVK mixed | 5.284 | 0.531 | 6.831 | 38.81 | 71.24 | 94.08 |
| page-001.jpg | KosmicKrisp FP32 | 7.002 | 0.401 | 8.629 | 49.03 | 75.21 | 94.33 |
| page-001.jpg | KosmicKrisp mixed | 7.009 | 0.401 | 8.533 | 48.48 | 71.23 | 94.15 |
| page-005.jpg | Torch MPS FP32 | 4.204 | 0.668 | 5.316 | 30.21 | Reference | Reference |
| page-005.jpg | Torch MPS FP16 | 3.467 | 0.810 | 4.950 | 28.13 | 70.58 | 91.03 |
| page-005.jpg | ORT WebGPU FP32 | 5.018 | 0.560 | 5.076 | 28.84 | 132.60 | 93.49 |
| page-005.jpg | ORT WebGPU FP16 | 3.175 | 0.884 | 3.661 | 20.80 | 54.95 | 88.81 |
| page-005.jpg | MoltenVK FP32 | 5.277 | 0.532 | 6.872 | 39.04 | 79.27 | 91.07 |
| page-005.jpg | MoltenVK mixed | 5.286 | 0.531 | 6.870 | 39.03 | 73.63 | 90.88 |
| page-005.jpg | KosmicKrisp FP32 | 7.035 | 0.399 | 8.543 | 48.54 | 79.27 | 91.08 |
| page-005.jpg | KosmicKrisp mixed | 7.018 | 0.400 | 8.504 | 48.32 | 73.64 | 90.85 |
| page-009.jpg | Torch MPS FP32 | 4.206 | 0.668 | 5.324 | 30.25 | Reference | Reference |
| page-009.jpg | Torch MPS FP16 | 3.465 | 0.810 | 4.925 | 27.98 | 68.96 | 91.33 |
| page-009.jpg | ORT WebGPU FP32 | 5.075 | 0.553 | 5.076 | 28.84 | 132.91 | 92.95 |
| page-009.jpg | ORT WebGPU FP16 | 3.162 | 0.888 | 3.643 | 20.70 | 55.92 | 90.60 |
| page-009.jpg | MoltenVK FP32 | 5.302 | 0.530 | 6.914 | 39.28 | 74.99 | 91.49 |
| page-009.jpg | MoltenVK mixed | 5.288 | 0.531 | 6.868 | 39.02 | 70.97 | 91.17 |
| page-009.jpg | KosmicKrisp FP32 | 7.068 | 0.397 | 8.531 | 48.47 | 74.99 | 91.41 |
| page-009.jpg | KosmicKrisp mixed | 7.008 | 0.401 | 8.511 | 48.36 | 70.97 | 91.15 |

### Verified WebGPU graph placement

| Checkpoint | Precision | Batch-1 WebGPU nodes | Batch-16 WebGPU nodes |
| --- | --- | ---: | ---: |
| Original | FP32 | 1531 | 1531 |
| Original | FP16 | 1591 | 1591 |
| IDC | FP32 | 829 | 829 |
| IDC | FP16 | 829 | 829 |

### Vulkan validation

Both drivers completed the repeated prepare, replay, reopen, malformed-package, and invalid-input checks
with Vulkan synchronization validation enabled. The numerical check covered square and rectangular random
inputs, flat and checkerboard patterns, and a 1×3-pixel input. The lowest raw PSNR across both Vulkan
precisions and five test inputs was 78.09 dB for Original and 68.39 dB for IDC. The tests also verified
that mixed mode changes the result on the seeded square input on each driver.

Original recorded 424 compute dispatches per call. IDC recorded 253. Benchmark and validation runs in
this report cover the two drivers on Apple M4.
