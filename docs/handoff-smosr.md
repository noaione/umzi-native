# SMoSR implementation and baseline

## Workload

Use `models/2x_enhancr_da_smosr_v1.safetensors` with batch size 1 and 256×256
input tiles. Use the three images in `samples/2x-test` for benchmarking and
testing. The baseline uses zero overlap and replicate padding on incomplete
tiles. Rendered pages are cropped to twice the source width and height.

## Checkpoint and implementation

Resselt 1.4.1 recognized the checkpoint as `SMoSR`. A CUDA FP32 forward on
`(1, 3, 256, 256)` returned finite `(1, 3, 512, 512)` output. The model has 48
feature channels, six multiplicative residual blocks arranged as 2 + 3 + 1,
and a 2× direct pixel-shuffle tail.

The exporter calls Resselt's FP32 eval-mode weight folding and writes family
5, schema 1. The weight section stores 22 padded OHWI filters in execution
order with FP16 weights and FP32 biases, each starting on a 256-byte boundary.
The architecture and checkpoint SHA-256 are separate sections.

The native session in `native/kernels/smosr/` reflects two input pixels on each
edge, runs the RGB shortcut and six convolutional blocks, adds the skip over
the middle three blocks, concatenates the RGB shortcut with the final body
projection, then pixel-shuffles and crops. It uses the shared cuDNN graph
planner, selecting the first supported plan within its workspace cap. The
session supports host I/O, CUDA device enqueue, and optional CUDA graphs. The
reflection border is internal, so the session reports zero external tile
context and returns exactly `2H × 2W`.

Mixed mode uses FP32 RGB I/O, FP16 convolution weights and activation
storage, and FP32 convolution accumulation, biases, and pointwise
intermediates. The FP16 mode uses FP16 RGB I/O and convolution accumulation,
rounds biases and separate pointwise intermediates to FP16, and evaluates the
SiLU and tanh functions in FP32 before rounding their results. Reflection
padding requires input height and width of at least three.

## Reproduce the baseline

Run from the repository root. This build used the local CUDA 13.0 NVRTC wheel
already extracted under `output/nvrtc-13.0`. If CUDA 13.0 NVRTC is installed in
the build environment, the `--nvrtc-root` argument is unnecessary.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$smosrModel = 'output/smosr-session/smosr.ummodel'
$smosrLibrary = 'output/smosr-session/native-win-final/umzi_native.dll'
& $smosrPython -m umzi_native export models/2x_enhancr_da_smosr_v1.safetensors -o $smosrModel --kernel smosr
& $smosrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/smosr-session/native-win-final
& $smosrPython scripts/validate_smosr_session.py --model $smosrModel --library $smosrLibrary
foreach ($backend in @('fp32', 'inductor', 'rtx')) {
    & $smosrPython scripts/benchmark_architectures.py smosr $backend
}
foreach ($backend in @('native-mixed', 'native-fp16')) {
    & $smosrPython scripts/benchmark_architectures.py smosr $backend --native-model $smosrModel --native-library $smosrLibrary
}
```

The Windows sandbox could not read the file's inherited private tempfile
permissions after export. The benchmark used a byte-identical copy in the
same output directory named `smosr-benchmark.ummodel`. Both copies have
package digest `44dd41ad9af18d87774fc35ab4f6e140fac7ab538d320e3f16be5aebe8824f00`.

## Measurements

Reports and renders live in `output/architecture-baselines/smosr/batch-1/`.
Each report records checkpoint and image hashes and synchronized wall time
for every device-resident forward. Per-image latency is the sum of the tile
calls. Its ms/tile value is the mean of the individual calls. Each median is
across the three images. Host transfers, loading, compilation, warmup, and
rendering are outside the timed forward region. Native graph timing includes
its device-to-device staging copies.

PSNR uses raw RGB values before quantization with peak value 1. Pooled PSNR
uses the pixel-weighted mean squared error. SSIMULACRA2 compares the rendered
8-bit RGB PNGs. NVML samples device-wide usage at a requested 5 ms interval
during page inference. Its peak includes other GPU activity and can miss
shorter transients. WDDM did not expose per-process VRAM through NVIDIA SMI
on this machine. The JSON also records the background before backend setup,
PyTorch allocator peaks, and the native session's buffer sizes.

Measured on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), driver 610.62,
Python 3.14.5, Resselt 1.4.1, Torch 2.15.0+cu132 with cuDNN 9.26,
Triton-Windows 3.9.0.post30, ONNX 1.23.2, ONNX Runtime 1.31.1, and TensorRT-RTX
provider 0.5.0. The RTX run used a fixed
batch-1 graph. The ORT execution profile showed only
`NvTensorRTRTXExecutionProvider`, with CPU and Python fallback disabled.
The early `samples/input` reports are archived separately under
`output/architecture-baselines/smosr-input-preliminary/`.

### Summary

| Backend | Total ms | Median ms/image | Median ms/tile | Pooled PSNR (dB) | Mean SSIMULACRA2 | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | 6504.79 | 2092.00 | 26.96 | Reference | Reference | 1598.52 |
| TorchInductor FP16 | 2883.72 | 920.28 | 11.94 | 70.41 | 88.97 | 1575.58 |
| TensorRT-RTX FP16 | 1767.18 | 567.03 | 7.28 | 59.23 | 88.43 | 1603.45 |
| Native mixed | 2821.13 | 910.71 | 11.63 | 72.53 | 89.15 | 1520.06 |
| Native FP16 | 1879.52 | 599.82 | 7.77 | 61.80 | 88.69 | 1526.93 |

The native session reports 53.21 MiB of allocated inference buffers for mixed
precision and 51.34 MiB for FP16, including weights, activations, workspace,
and RGB staging buffers.

### Per image

| Image | Tiles | Backend | Inference ms | ms/tile | PSNR (dB) | SSIMULACRA2 | Peak VRAM (MiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | 77 | Torch CUDA FP32 | 2092.00 | 27.17 | Reference | Reference | 1596.33 |
| `9798855404661.jpg` | 77 | TorchInductor FP16 | 912.69 | 11.85 | 70.28 | 87.78 | 1575.58 |
| `9798855404661.jpg` | 77 | TensorRT-RTX FP16 | 560.29 | 7.28 | 58.65 | 87.34 | 1603.45 |
| `9798855404661.jpg` | 77 | Native mixed | 895.71 | 11.63 | 72.23 | 88.04 | 1520.06 |
| `9798855404661.jpg` | 77 | Native FP16 | 598.52 | 7.77 | 61.31 | 87.56 | 1526.93 |
| `9798855424195.jpg` | 77 | Torch CUDA FP32 | 2040.65 | 26.50 | Reference | Reference | 1596.33 |
| `9798855424195.jpg` | 77 | TorchInductor FP16 | 920.28 | 11.95 | 71.46 | 89.32 | 1523.70 |
| `9798855424195.jpg` | 77 | TensorRT-RTX FP16 | 567.03 | 7.36 | 59.51 | 88.69 | 1603.45 |
| `9798855424195.jpg` | 77 | Native mixed | 910.71 | 11.83 | 73.58 | 89.53 | 1520.06 |
| `9798855424195.jpg` | 77 | Native FP16 | 599.82 | 7.79 | 62.31 | 89.00 | 1524.43 |
| `9798855444216.jpg` | 88 | Torch CUDA FP32 | 2372.14 | 26.96 | Reference | Reference | 1598.52 |
| `9798855444216.jpg` | 88 | TorchInductor FP16 | 1050.76 | 11.94 | 69.71 | 89.81 | 1523.70 |
| `9798855444216.jpg` | 88 | TensorRT-RTX FP16 | 639.86 | 7.27 | 59.57 | 89.26 | 1602.86 |
| `9798855444216.jpg` | 88 | Native mixed | 1014.72 | 11.53 | 71.96 | 89.87 | 1520.06 |
| `9798855444216.jpg` | 88 | Native FP16 | 681.18 | 7.74 | 61.84 | 89.52 | 1524.43 |

The synthetic checkerboard had the lowest PSNR of the stress tests. It scored
60.06 dB with Inductor, 48.31 dB with TensorRT-RTX, 49.89 dB with native mixed
precision, and 37.44 dB with native FP16.

## Verification

The session check compares a 65×79 crop from `9798855404661.jpg` against
Torch CUDA FP32, then checks bitwise equality between host, device,
direct, captured, and alternate-stream execution. It measured 69.25 dB in
mixed precision and 59.80 dB in FP16 for that crop. The standalone native CLI
matched the Python host path bit for bit in both modes. The session check
also verifies rejection of an input too small for reflection padding.

The benchmark covers all three full images, their center tiles, rectangular
tensors, flat values, border impulses, noise, and a checkerboard. TensorRT-RTX
checks use its fixed 256×256 input shape. The same Windows DLL also ran
GateRV3 and GFISRv2 smoke tests through the shared session API.

## Pickup

Start from these mixed and FP16 reports. Compare any plan selection, bias
handling, or buffer reuse change against the same checkpoint, images, and
tiling. Recheck the checkerboard when changing accumulation or intermediate
rounding. Assess whether a measured precision, memory, or device-specific
tradeoff warrants a kernel option.
