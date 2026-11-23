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
and rounds biases and separate pointwise intermediates to FP16. The optimized
FP16 session precomputes the rounded SiLU and tanh outputs from FP32 evaluations.
Reflection padding requires input height and width of at least three.

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

## Original native call trace

`Runner.measure` moves the tile to CUDA before timing. `DeviceSession`
describes the output, allocates an output tensor, and sends the two device
pointers and current Torch stream through `umzi_session_enqueue`.
`Session::prepare` returns after waiting for any prior submission when the
shape is already prepared. The captured path copies the input into its
staging buffer, replays `Session::forward`, and copies the output back to the
caller's device buffer. Direct dispatch binds the caller's pointers and
launches `forward` on the same stream. The host API uses the same forward
with host-to-device and device-to-host transfers at its boundary.

The forward packs and reflects input to NHWC, runs two shortcut convolutions
and six three-convolution blocks, runs the end projection and tail
convolution, then shuffles and crops. Each tile runs 22 cuDNN convolutions,
12 bias-plus-SiLU kernels, six gated residual kernels, and one middle skip
copy. See [session code](../native/kernels/smosr/runner.cpp) and
[call adapter](../scripts/reference/native_session.py).

### VRAM attribution

The trace used a 256×256 center tile from each of the three `samples/2x-test`
images. Each precision and dispatch mode ran in a fresh process. The same
NVML sampler as the baseline measured device-wide peaks at a requested
5 ms interval over 48 synchronized forward calls. Windows' per-process
dedicated memory counter was read at setup and after warming the tiles,
matched to the CUDA adapter by LUID. See the [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
and [Windows GPU memory counter definitions](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).

| CUDA Graph process or device measurement | Native FP16 (MiB) | Native mixed (MiB) |
| --- | ---: | ---: |
| NVML device-wide before CUDA initialization | 1361.70 | 1359.82 |
| NVML device-wide peak during forwards | 1514.32 | 1522.32 |
| Process dedicated after CUDA initialization | 62.67 | 62.67 |
| Process dedicated after shape preparation | 132.67 | 138.67 |
| Process dedicated after warming the tiles | 154.67 | 162.67 |
| Native session buffers | 51.34 | 53.21 |
| PyTorch allocator reserved after warming the tiles | 22.00 | 24.00 |

Dedicated process snapshots matched between graph and direct dispatch in
each precision. The native session counters account for weights,
activations, workspace, and RGB staging. The startup and warmed snapshots
also include CUDA and cuDNN resources and the benchmark caller.

### Activation buffer reuse

A 256×256 tile becomes 260×260 internally. All eight NHWC activations use
half storage and occupy 48.48 MiB in both precision modes. The 96-channel
projection alone occupies 12.38 MiB. The selected cuDNN plans require zero
workspace here. The 256 MiB plan-search cap is an upper bound on candidate
workspace.

The live buffers during a middle block's last convolution are the RGB
shortcut (16 channels), saved skip (48), current state (48), convolution
input (48), and projection output (96). They total 256 channels or
33.01 MiB. The existing allocations total 376 channels.

A liveness check confirmed that one 256-channel arena can hold the same
forward schedule. Packed RGB input can use the second SiLU buffer before
it is needed. The first SiLU and gate projection can use the same storage
because the second convolution finishes reading the first SiLU before the
gate projection begins. The tail concat can span the adjacent saved skip
and state slots after the end convolution finishes reading them.

The resulting estimate is **15.47 MiB saved per prepared session** at this
shape, lowering total native buffers to about 37.74 MiB mixed and 35.86 MiB
FP16. This is an allocation-layout estimate. The first implementation can
keep the same arithmetic and launches, then check graph/direct, host/device,
odd shapes, and bitwise baseline fidelity. That makes buffer reuse suitable
as the default once those checks pass.

### Latency targets

Low-overhead wall timings were collected with the profiler disabled. A
separate run traced 12 forwards per mode using CUPTI through Torch Profiler.
The raw per-call timings, Windows memory snapshots, NVML peaks, summary
JSON, and Chrome traces are under `output/smosr-session/trace/`. The local
probe used `output/smosr-session/profile_smosr_trace.py` with `PYTHONPATH`
set to `src;scripts` and arguments `<mixed|fp16> <graph|direct>`.

| CUDA Graph stage | Native FP16 (ms/tile) | Native mixed (ms/tile) |
| --- | ---: | ---: |
| Median synchronized wall time | 7.82 | 11.49 |
| cuDNN convolutions in CUPTI trace | 5.80 | 11.13 |
| Bias and SiLU in CUPTI trace | 1.40 | 1.46 |
| Gated residual in CUPTI trace | 0.93 | 0.95 |
| Middle skip copy in CUPTI trace | 0.07 | 0.07 |
| Graph input and output staging in CUPTI trace | 0.02 | 0.05 |

CUPTI rows are averages from instrumented calls, collected separately
from the wall measurements. The 12 48→48 and six 48→96 3×3 convolutions
account for about 94–95% of cuDNN time in both precisions. Those shapes are
the first targets for timing alternate cuDNN plans under a workspace cap.
Bias and SiLU fusion is the next steady-state target, with checks for the
existing rounding and the checkerboard case. Staging and the skip copy
account for roughly 0.1 ms combined on this GPU.

Cold plan setup for the captured sessions took 1.73 s in FP16 and 2.36 s
in mixed precision. The shared planner gathers and deduplicates every
supported candidate even though SMoSR selects the first one. An early exit
after finding that plan is the first cold-start experiment. See [plan selection](../native/kernels/smosr/runner.cpp#L117)
and [candidate enumeration](../native/common/cudnn_conv_plan.h#L130).

## Optimization runs

Keep the original Windows DLL and Torch FP32 output as controls. Implement
activation reuse, remove the middle skip copy by swapping state buffers, and
stop at the first heuristic plan on the default path. Test packed half2
pointwise kernels against the original native outputs. Benchmark optional
plan tuning for the 48→48 and 48→96 convolutions with zero and bounded
workspace. Keep any measured precision or workspace tradeoff explicit.

The first tile screen retained the original output with the arena and
half2 kernels. Timing was 7.68 → 7.28 ms for FP16 and 11.25 → 10.96 ms for
mixed. Timing zero-workspace convolution candidates reduced mixed to
9.36 ms and changed its checkerboard PSNR from 49.89 to 49.78 dB. The same
search in FP16 took 7.54 ms and raised checkerboard PSNR to 44.25 dB.
Caps of 32, 64, 128, and 256 MiB increased tuning scratch without a further
speedup in this screen. The raw results are in
`output/smosr-session/screen-native-optimized.json`.

The second build fused the final middle skip into the gate and packed concat
in half2 pairs. It used FP16 SiLU and tanh tables generated with the original
device functions for all 65,536 half bit patterns. The tables cost 256 KiB per
FP16 session and keep half rounding at the same points. Both precisions
matched the original outputs on the three center tiles and checkerboard in
`output/smosr-session/screen-native-lut.json`. The final build explicitly
rounds the FP16 gated product before adding the middle skip.

The default planner returns its first supported plan before serializing or
timing candidates. `tune_convolutions=1` stays opt-in since it adds setup time
and can select a plan with different rounding. The shipped tuner searches
only zero-workspace engines for the 48→48 and 48→96 convolutions. See the
[cuDNN engine heuristics and autotuning documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html).

### Reproduce the optimized run

From the repository root, use the exported package and original Windows
DLL from the baseline. The benchmark keeps its matching Torch FP32 reference
and writes separate reports for each native tag.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$smosrModel = 'output/smosr-session/smosr-benchmark.ummodel'
$controlLibrary = 'output/smosr-session/native-win-final/umzi_native.dll'
$optimizedLibrary = 'output/smosr-session/native-optimized-final/umzi_native.dll'
& $smosrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/smosr-session/native-optimized-final
& $smosrPython scripts/validate_smosr_session.py --model $smosrModel --library $optimizedLibrary --baseline-library $controlLibrary
foreach ($backend in @('native-fp16', 'native-mixed')) {
    & $smosrPython scripts/benchmark_architectures.py smosr $backend --native-model $smosrModel --native-library $controlLibrary --native-tag control
    & $smosrPython scripts/benchmark_architectures.py smosr $backend --native-model $smosrModel --native-library $optimizedLibrary --native-tag optimized
}
& $smosrPython scripts/benchmark_architectures.py smosr native-mixed --native-model $smosrModel --native-library $optimizedLibrary --native-tag optimized-tuned --tune-convolutions
```

### Final full-page results

FP16 and mixed defaults matched the original native build byte for byte on
all 242 image tiles and all stress fixtures. Host, direct, captured, and
alternate-stream checks passed. Every benchmark used the same three
`samples/2x-test` images, batch 1, 256×256 tiles, and zero overlap.

| Variant | Total ms | Median ms/image | Median ms/tile | PSNR (dB) | Mean SSIMULACRA2 | First plans (ms) | Native buffers (MiB) | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 control | 1890.42 | 601.25 | 7.81 | 61.80 | 88.69 | 1776.95 | 51.34 | 1532.82 |
| FP16 optimized | 1804.30 | 578.09 | 7.44 | 61.80 | 88.69 | 109.15 | 36.11 | 1504.63 |
| Mixed control | 2828.26 | 904.06 | 11.70 | 72.53 | 89.15 | 2384.31 | 53.21 | 1533.19 |
| Mixed optimized | 2725.85 | 896.22 | 11.31 | 72.53 | 89.15 | 191.80 | 37.74 | 1511.94 |
| Mixed optimized + tuning | 2227.71 | 717.50 | 9.26 | 72.53 | 89.12 | 1452.63 | 37.74 | 1507.19 |

Default tile latency dropped 4.7% in FP16 and 3.4% in mixed.
The tuned mixed run cut tile latency 20.9% from the control and kept zero
cuDNN workspace. The activation arena is 33.01 MiB in both precisions.
The FP16 native buffer total includes the 256 KiB of activation tables.
NVML peaks represent device-wide usage during page inference.

The tuned mixed run used a different engine and changed the checkerboard
from 49.89 to 49.78 dB. It retained essentially the same page PSNR and
SSIMULACRA2 on this set. Tuning happens at shape preparation and adds cold
latency, so the default is the faster startup path.

| Image | Variant | Inference ms | ms/tile | PSNR (dB) | SSIMULACRA2 | NVML peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | FP16 control | 600.56 | 7.80 | 61.31 | 87.56 | 1532.82 |
| `9798855404661.jpg` | FP16 optimized | 571.58 | 7.42 | 61.31 | 87.56 | 1504.63 |
| `9798855404661.jpg` | Mixed control | 894.35 | 11.61 | 72.23 | 88.04 | 1533.19 |
| `9798855404661.jpg` | Mixed optimized | 834.62 | 10.84 | 72.23 | 88.04 | 1511.94 |
| `9798855404661.jpg` | Mixed optimized + tuning | 695.40 | 9.03 | 72.23 | 88.06 | 1507.19 |
| `9798855424195.jpg` | FP16 control | 601.25 | 7.81 | 62.31 | 89.00 | 1532.63 |
| `9798855424195.jpg` | FP16 optimized | 578.09 | 7.51 | 62.31 | 89.00 | 1504.63 |
| `9798855424195.jpg` | Mixed control | 904.06 | 11.74 | 73.58 | 89.53 | 1533.19 |
| `9798855424195.jpg` | Mixed optimized | 896.22 | 11.64 | 73.58 | 89.53 | 1505.19 |
| `9798855424195.jpg` | Mixed optimized + tuning | 717.50 | 9.32 | 73.58 | 89.47 | 1507.19 |
| `9798855444216.jpg` | FP16 control | 688.62 | 7.83 | 61.84 | 89.52 | 1532.63 |
| `9798855444216.jpg` | FP16 optimized | 654.63 | 7.44 | 61.84 | 89.52 | 1504.63 |
| `9798855444216.jpg` | Mixed control | 1029.86 | 11.70 | 71.96 | 89.87 | 1533.19 |
| `9798855444216.jpg` | Mixed optimized | 995.01 | 11.31 | 71.96 | 89.87 | 1505.19 |
| `9798855444216.jpg` | Mixed optimized + tuning | 814.81 | 9.26 | 71.96 | 89.83 | 1507.19 |

The full reports and rendered outputs are under
`output/architecture-baselines/smosr/batch-1/`. Each report includes
the original checkpoint and library hashes, the first shape preparation
times, and every synchronized tile timing and coordinate.

### Optimized FP16 trace

The final build's CUPTI rerun measured 1.02 ms per tile for SiLU and 0.10 ms
for concat, down from 1.40 ms and 0.20 ms in the original trace. The middle
skip copy and separate add are gone. The two dominant convolution shapes
still account for most GPU time. Windows dedicated memory for the warmed
trace process was 126.67 MiB, down from 154.67 MiB in the original trace.
The raw trace and summary are under `output/smosr-session/trace/` with the
prefix `fp16-graph-native-optimized-final`.

## Pickup

Use the optimized default as the bitwise control for further changes. The
largest remaining latency target is the FP16 48→48 and 48→96 3×3 convolution
pair. The engine sweep gave no FP16 speed gain. A convolution epilogue that
applies SiLU at the same rounding points is the next fusion experiment.
