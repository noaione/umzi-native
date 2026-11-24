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
5, schema 2. The weight section stores 22 padded OHWI filters and their biases
in FP32, each starting on a 256-byte boundary. The architecture and checkpoint
SHA-256 are separate sections. Legacy schema 1 packages stored FP16 weights
with FP32 biases and still run in mixed and FP16 sessions.

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

## Native FP32 and further optimization

Export the folded weights as FP32 in SMoSR schema 2. Preserve inference from
schema 1 in mixed and FP16, and convert schema 2 weights at session open for
those two modes. Use the same activation arena and forward schedule with
FP32 storage and cuDNN engines that preserve FP32 inputs and accumulation.
Measure this baseline before testing alternate engine selection and fusion.

The initial NHWC native FP32 baseline ran at 26.89 ms/tile, scored 141.90 dB
raw PSNR, and reserved 37.73 MiB of cuDNN workspace. Timing only zero-workspace
engines for the two hot convolution shapes slowed it to 51.83 ms/tile.
Those full runs are `native-fp32-strict` and `native-fp32-strict-tuned` under
`output/architecture-baselines/smosr/batch-1/`.

The NCHW layout, following the GFISRv2 FP32 path, reached 20.56 ms/tile
in the center-tile screen and used 0.44 MiB of workspace. Timing FP32 engines
within the heuristic workspace budget took 20.86 ms/tile. The final FP32
path uses the NCHW heuristic plans and rejects engines marked as Tensor Core,
input down-conversion, or reduced-precision reduction. See the [cuDNN engine
numerical notes](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html).

Bias and SiLU fusion found no zero-workspace FP32 plan. Mixed fusion ran at
11.43 ms/tile versus 10.92 ms unfused. With convolution tuning active, it
ran at 10.18 ms versus 9.26 ms unfused. The fused experiment and FP32 tuning
are recorded in `output/smosr-session/screen-native-fp32-nchw.json`.

### Reproduce the native FP32 run

Run from the repository root with the same Torch FP32 reference from the
original benchmark. The `--baseline-model` argument points at the original
schema 1 package for bytewise validation of mixed and FP16.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$model = 'output/smosr-session/smosr-schema2.ummodel'
$legacyModel = 'output/smosr-session/smosr-benchmark.ummodel'
$library = 'output/smosr-session/native-fp32-final/umzi_native.dll'
$legacyLibrary = 'output/smosr-session/native-optimized-final/umzi_native.dll'
& $smosrPython -m umzi_native export models/2x_enhancr_da_smosr_v1.safetensors -o $model --kernel smosr
& $smosrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/smosr-session/native-fp32-final
& $smosrPython scripts/validate_smosr_session.py --model $model --library $library --baseline-model $legacyModel --baseline-library $legacyLibrary
& $smosrPython scripts/benchmark_architectures.py smosr native-fp32 --native-model $model --native-library $library --native-tag nchw
```

In the Windows sandbox, testing used the byte-identical copy
`output/smosr-session/smosr-schema2-benchmark.ummodel`. The package digest is
`5719c5a1ebb6eb06752a51542655e98531f327c5eca60d0b7f8cd7700df90181`.

### Full-page FP32 results

All runs used the same three `samples/2x-test` images, batch 1, 256×256
input tiles, and zero overlap. The final native NCHW session used strictly
FP32 convolution and pointwise math. Its raw output had zero error on all
three pages and its rendered PNG files were byte-identical to the Torch
FP32 renders.

| Backend | Total ms | Median ms/image | Median ms/tile | PSNR (dB) | Mean SSIMULACRA2 | cuDNN workspace (MiB) | Native buffers (MiB) | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | 6504.79 | 2092.00 | 26.96 | Reference | Reference | — | — | 1598.52 |
| Native FP32 NHWC control | 6508.67 | 2083.34 | 26.89 | 141.90 | 90.91 | 37.73 | 109.45 | 1566.94 |
| Native FP32 NCHW | 4969.09 | 1589.23 | 20.52 | ∞ (zero error) | 100.00 | 0.44 | 72.16 | 1553.50 |

The NCHW path reduced median tile latency 23.7% from the NHWC control.
On the 8-bit PNGs, the NHWC control differed from Torch in 507–682 pixels
per image, each by at most one channel level. Raw PSNR and rendered
SSIMULACRA2 use separate representations.

| Image | NCHW inference ms | NCHW ms/tile | PSNR (dB) | SSIMULACRA2 | NVML peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | 1574.15 | 20.44 | ∞ (zero error) | 100.00 | 1553.50 |
| `9798855424195.jpg` | 1589.23 | 20.64 | ∞ (zero error) | 100.00 | 1553.50 |
| `9798855444216.jpg` | 1805.71 | 20.52 | ∞ (zero error) | 100.00 | 1553.50 |

The graph, direct, host, and alternate-stream paths agreed on the 65×79
crop. The 3×3 minimum input returned a 6×6 image. The three center tiles and
checkerboard matched Torch byte for byte with `NVIDIA_TF32_OVERRIDE` unset.
Mixed and FP16 from schema 2 matched the prior native schema 1
build bitwise on all 242 benchmark tiles and all stress fixtures.
The 128×160 FP32 stress tensor scored 131.95 dB. See
`output/architecture-baselines/smosr/batch-1/native-fp32-nchw/report.json`
for every tile timing, per-image scores, and plan setup measurements.

## Matched TensorRT-RTX comparison

Compare native and TensorRT-RTX FP32 and FP16 on the same three 2× test pages.
Use CUDA Graphs, synchronized device-resident tiles, and fresh cache paths
for the FP32 engine. Build the FP32 ONNX graph from the original weights with
`NVIDIA_TF32_OVERRIDE=0`, verify TensorRT-RTX node placement, and keep the
Torch FP32 raw arrays and PNGs as the fidelity reference.

The RTX FP32 graph has FLOAT inputs, outputs, and weights. The exported
graph was also checked for reduced-precision casts and constants. ONNX
Runtime profiled every executed node on `NvTensorRTRTXExecutionProvider`.
The FP32 comparison disables TF32 before creating either CUDA session.
See [NVIDIA's precision controls](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html).

Both backends reuse their device input and output. The timer includes
graph replay and synchronization. Native graph staging still copies to and
from its fixed CUDA graph addresses inside that interval.

### FP16 optimization

The CUPTI trace attributed about 4.9 ms per tile to cuDNN convolutions,
0.94 ms to the 12 SiLU kernels, and 0.92 ms to the six gated residuals.
The default FP16 path now uses 128-thread blocks and read-only lookup loads
for those two pointwise kernels. It retains the same lookup bits and half
arithmetic, so the change needs no kernel option.

128-bit vector loads made SiLU faster and the gate slower, saving roughly
0.01 ms/tile across the three pages. Approximate SiLU and tanh saved about
0.06 ms/tile across those pages and changed numeric output. With readback
over three passes, approximate activations measured 7.123 ms/tile by median
page average, versus 7.179 for the lookup build and 7.213 for TensorRT-RTX.
The speed difference was too small to add a precision option. The timing
and fidelity are in
`output/smosr-session/paired-native-fp16-vector/report.json` and
`output/smosr-session/paired-native-fast-activations-readback/report.json`.

An interleaved full-page run rotated the starting backend each tile. It
used the same prepared inputs for TensorRT-RTX, the new FP16 kernel, and
the previous native binary. Each cell shows summed inference time and the
mean of the independently synchronized tile calls.

| Image | Tiles | TensorRT-RTX FP16 ms / ms per tile | Previous native FP16 ms / ms per tile | New native FP16 ms / ms per tile |
| --- | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | 77 | 553.63 / 7.190 | 550.64 / 7.151 | 549.07 / 7.131 |
| `9798855424195.jpg` | 77 | 562.50 / 7.305 | 555.80 / 7.218 | 551.34 / 7.160 |
| `9798855444216.jpg` | 88 | 637.08 / 7.239 | 630.63 / 7.166 | 627.55 / 7.131 |
| Median | | 562.50 / 7.239 | 555.80 / 7.166 | 551.34 / 7.131 |

Across all 242 calls, median tile latency was 7.167 ms for TensorRT-RTX,
7.093 ms for the previous native kernel, and 7.085 ms for the new kernel.
The trace and every tile sample are under
`output/smosr-session/paired-native-fp16-lookup/`.

With output readback and the usual NVML sampler, three interleaved passes
were effectively tied. TensorRT-RTX and native measured 7.048 and 7.055
ms/tile by median page average, and 6.972 and 6.984 ms by median of all
726 tile calls per backend. This follow-up is under
`output/smosr-session/paired-native-fp16-lookup-readback/`.

A second vector experiment used 128-bit SiLU and 64-bit gate loads. In
three interleaved passes with readback, it took 7.109 ms/tile by median
page average, versus 7.107 for the simpler lookup build and 7.186 for
TensorRT-RTX. In separate processes, the vector build took 7.393 ms/tile
and RTX took 7.247. The simpler lookup build remains the default.
See `output/smosr-session/paired-native-fp16-wide-readback/report.json`
and the `native-fp16-wide-match` and `rtx-after-wide` reports.

### Full-page fidelity, latency, and GPU memory

These separate runs used an RTX 3050 Laptop GPU with driver 610.62 and
the ONNX Runtime TensorRT-RTX plugin 0.5.0. Each image contributes its 77
or 88 independently timed batch-1 calls. The image median covers each
page's summed inference time. The tile median covers all 242 samples.
The NVML peak is device-wide. The peak delta uses the CUDA-initialized
snapshot taken just before setting up each backend.

| Backend | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR (dB) | Mean SSIMULACRA2 | NVML peak MiB | Peak delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorRT-RTX FP32 | 9655.43 | 3091.76 | 39.55 | 40.12 | 100.64 | 89.39 | 1512.50 | 202 |
| Native FP32 | 4998.44 | 1602.65 | 20.37 | 20.68 | ∞ (zero error) | 100.00 | 1403.13 | 102 |
| TensorRT-RTX FP16 | 1708.47 | 550.32 | 6.75 | 7.03 | 59.23 | 88.43 | 1471.25 | 160 |
| Native FP16 | 1768.78 | 565.05 | 7.00 | 7.31 | 61.80 | 88.69 | 1374.82 | 72 |

| Image | Backend | Inference ms | Mean ms/tile | Median ms/tile | PSNR (dB) | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | TensorRT-RTX FP32 | 3032.74 | 39.39 | 39.26 | 99.73 | 88.30 | 1512.50 |
| `9798855404661.jpg` | Native FP32 | 1575.85 | 20.47 | 20.29 | ∞ | 100.00 | 1403.13 |
| `9798855404661.jpg` | TensorRT-RTX FP16 | 541.19 | 7.03 | 6.77 | 58.65 | 87.34 | 1471.25 |
| `9798855404661.jpg` | Native FP16 | 562.87 | 7.31 | 6.98 | 61.31 | 87.56 | 1374.82 |
| `9798855424195.jpg` | TensorRT-RTX FP32 | 3091.76 | 40.15 | 39.65 | 100.91 | 89.74 | 1509.88 |
| `9798855424195.jpg` | Native FP32 | 1602.65 | 20.81 | 20.41 | ∞ | 100.00 | 1403.13 |
| `9798855424195.jpg` | TensorRT-RTX FP16 | 550.32 | 7.15 | 6.73 | 59.51 | 88.69 | 1462.82 |
| `9798855424195.jpg` | Native FP16 | 565.05 | 7.34 | 7.08 | 62.31 | 89.00 | 1374.82 |
| `9798855444216.jpg` | TensorRT-RTX FP32 | 3530.93 | 40.12 | 39.65 | 101.41 | 90.13 | 1509.88 |
| `9798855444216.jpg` | Native FP32 | 1819.94 | 20.68 | 20.38 | ∞ | 100.00 | 1401.38 |
| `9798855444216.jpg` | TensorRT-RTX FP16 | 616.97 | 7.01 | 6.74 | 59.57 | 89.26 | 1462.82 |
| `9798855444216.jpg` | Native FP16 | 640.86 | 7.28 | 7.00 | 61.84 | 89.52 | 1373.13 |

Native FP32 reduced median page-average tile latency by 48.5%. The
separate FP16 run favored TensorRT-RTX by 0.28 ms/tile. Native FP16 matched
the previous native DLL byte for byte on all 242 tiles and stress inputs.
The FP16 advantage varied by run order and timing protocol.

### Reproduce the separate runs

Use the same schema 2 package and untouched Torch FP32 results as above.
The RTX runs each export a fixed-shape ONNX graph. The FP32 engine uses a
separate cache from the FP16 engine. `--tag` saves each run separately.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$model = 'output/smosr-session/smosr-schema2-benchmark.ummodel'
$library = 'output/smosr-session/native-fp16-lookup/umzi_native.dll'
& $smosrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/smosr-session/native-fp16-lookup
& $smosrPython scripts/benchmark_architectures.py smosr rtx-fp32
& $smosrPython scripts/benchmark_architectures.py smosr rtx --tag matched
& $smosrPython scripts/benchmark_architectures.py smosr native-fp32 --native-model $model --native-library $library --tag matched
& $smosrPython scripts/benchmark_architectures.py smosr native-fp16 --native-model $model --native-library $library --tag matched
```

## 512 and 1024 tile sweep

Use batch 1, zero overlap, and the same three `samples/2x-test` images.
Keep the 256-pixel runs as the comparison point. Generate a Torch CUDA FP32
reference at each new size before benchmarking TorchInductor FP16,
TensorRT-RTX FP16 and FP32, and the native FP32, FP16, and mixed modes.
Include the existing opt-in mixed convolution tuner. Use the same native
library and schema 2 package as in the 256-pixel comparison.

Fidelity uses the same-size Torch FP32 output. Compare the three Torch
references separately to see how the tile grid affects the rendered page.
Record synchronized forward latency, native buffer totals, and NVML peak
above the pre-setup snapshot. Save reports and graphs in separate
`tile-512` and `tile-1024` directories under `batch-1`.

The two 1733×2600 images and the 1812×2600 image used 77, 77, and 88 tiles
at 256. They each use 24 tiles at 512 and six at 1024. Both larger sizes
submit 18.87 million input pixels across the set, 19.0% more than the 256
grid's 15.86 million, because the benchmark pads the right and bottom tiles
to the fixed input size.

Native FP16 finished at both sizes. The median page took 668 ms at 512
and 670 ms at 1024, versus 565 ms at 256. Native buffers grew from
36.11 MiB at 256 to 138.74 MiB at 512 and 547.24 MiB at 1024. The 1024
native FP32 run fit in the GPU's memory and produced small raw differences
from same-size Torch FP32. See the fidelity scores below.

### Results

GeForce RTX 3050 Laptop GPU, 4 GiB, driver 610.62. The timer measures
synchronized forward calls. The page median uses summed inference time
from each image, and the tile median uses all 72 or 18 individual calls.
NVML reports the device-wide page peak and its increase above the
CUDA-initialized reading before backend setup. Native buffers are the
sum of weights, activations, workspace, and RGB staging from the session.

| Tile | Backend | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR (dB) | Mean SSIMULACRA2 | NVML peak MiB | NVML delta MiB | Native buffer MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | Torch CUDA FP32 | 7114.59 | 2372.26 | 98.27 | 98.84 | Ref. | Ref. | 1676.04 | 452 | — |
| 512 | TorchInductor FP16 | 3163.25 | 1053.39 | 42.26 | 43.89 | 70.41 | 88.97 | 1425.97 | 204 | — |
| 512 | TensorRT-RTX FP16 | 1751.98 | 586.09 | 23.90 | 24.42 | 59.26 | 88.51 | 1518.16 | 296 | — |
| 512 | TensorRT-RTX FP32 | 11124.29 | 3718.85 | 152.91 | 154.95 | 100.64 | 89.45 | 1695.97 | 474 | — |
| 512 | Native FP16 | 1996.92 | 668.02 | 27.26 | 27.83 | 61.79 | 88.72 | 1398.16 | 176 | 138.74 |
| 512 | Native mixed | 3060.27 | 1026.07 | 41.63 | 42.75 | 72.52 | 89.10 | 1402.16 | 180 | 145.99 |
| 512 | Native mixed, tuned | 2549.84 | 849.25 | 34.69 | 35.39 | 72.52 | 89.04 | 1404.16 | 182 | 145.99 |
| 512 | Native FP32 | 5593.80 | 1861.90 | 77.25 | 77.58 | ∞ | 100.00 | 1528.16 | 306 | 278.49 |
| 1024 | Torch CUDA FP32 | 7051.37 | 2348.09 | 390.10 | 391.35 | Ref. | Ref. | 2907.97 | 1686 | — |
| 1024 | TorchInductor FP16 | 3058.43 | 1030.46 | 165.71 | 171.74 | 70.41 | 88.98 | 2378.39 | 798 | — |
| 1024 | TensorRT-RTX FP16 | OOM | — | — | — | — | — | — | — | — |
| 1024 | TensorRT-RTX FP32 | OOM | — | — | — | — | — | — | — | — |
| 1024 | Native FP16 | 2011.53 | 670.11 | 110.56 | 111.69 | 61.93 | 88.72 | 1877.97 | 656 | 547.24 |
| 1024 | Native mixed | 2994.35 | 1004.74 | 162.66 | 167.46 | 72.52 | 89.08 | 1871.97 | 650 | 576.99 |
| 1024 | Native mixed, tuned | 2577.62 | 863.14 | 139.94 | 143.86 | 72.52 | 89.05 | 2204.02 | 704.93 | 576.99 |
| 1024 | Native FP32 | 5575.96 | 1862.98 | 307.97 | 310.50 | 129.68 | 90.21 | 2387.72 | 1165.62 | 1100.02 |

Each completed run contains per-image timing, fidelity, and VRAM, plus
each tile's duration and coordinates, in its `report.json`.
The native 1024-pixel FP32 maximum raw error was 2.26e-6. In the PNGs,
1,627 to 3,001 pixels per page differed by at most one channel level.

Against the 256-pixel Torch reference renders, the 512-pixel Torch outputs
scored 52.80–60.31 dB and 85.78–88.10 SSIMULACRA2. The 1024-pixel Torch
outputs scored 50.79–58.77 dB and 85.12–87.64. These scores measure the
effect of changing the grid. The table compares each backend against the
Torch reference at the same size.

Both TensorRT-RTX 1024-pixel engines failed during creation with CUDA
`OutOfMemory` and provider `EPFail`. Fresh runs with 128 and 32 MiB
workspace caps also failed. A standalone 32 MiB builder that reused the ONNX
graphs before initializing Torch CUDA failed for both precisions. See the
`rtx*-workspace-*` reports and `rtx*-standalone-32.log` files in `tile-1024`.
The 512-pixel profiles confirmed exclusive TensorRT-RTX node placement.

### Reproduce the sweep

Use the schema 2 package and DLL from the matched 256-pixel run.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$model = 'output/smosr-session/smosr-schema2-benchmark.ummodel'
$library = 'output/smosr-session/native-fp16-lookup/umzi_native.dll'
foreach ($tile in @(512, 1024)) {
    & $smosrPython scripts/benchmark_architectures.py smosr fp32 --tile $tile
    foreach ($backend in @('native-fp16', 'native-mixed', 'native-fp32')) {
        & $smosrPython scripts/benchmark_architectures.py smosr $backend --tile $tile --native-model $model --native-library $library
    }
    & $smosrPython scripts/benchmark_architectures.py smosr native-mixed --tile $tile --native-model $model --native-library $library --tag tuned --tune-convolutions
    foreach ($backend in @('rtx', 'rtx-fp32', 'inductor')) {
        & $smosrPython scripts/benchmark_architectures.py smosr $backend --tile $tile
    }
}
```

## Further mixed and FP16 tuning

Use the original batch-1 256×256 workload and the matched native DLL as
the control. Keep Torch FP32 untouched. The six 48→96 projections and
the twelve 48→48 convolutions account for most of the forward time.

A screen of additional cuDNN heuristic B configurations returned the same
74 mixed and 83 FP16 zero-workspace candidates. The tuned mixed medians
were 8.889 and 8.886 ms/tile before and after the extra heuristic. FP16 was
7.124 ms/tile by default and 7.284 ms/tile after expanded tuning. Approximate
mixed SiLU saved at most 0.04 ms/tile and changed numeric output. Both
experiments were dropped. See `output/smosr-session/screen-plan-b-silu.json`
and [cuDNN engine heuristics](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html).

Test whether the existing 48→48 plan can compute the value and gate in
separate outputs faster than the 48→96 plan. This keeps the same weights,
projection buffer size, and elementwise arithmetic.

The split was slower in the interleaved screen. Mixed went from 11.133
to 11.198 ms/tile by default and from 8.892 to 9.508 ms/tile with tuning.
FP16 went from 7.019 to 7.449 ms/tile, and strict FP32 went from 20.347
to 22.388 ms/tile. See `output/smosr-session/screen-split-projection.json`.

The next experiment keeps the original projection and tests up to nine
combinations of the heuristic plan and the fastest zero-workspace plans
found by isolated convolution timing. Score them by replaying the complete
captured CUDA Graph at the same shape, then benchmark the winner on images.

Whole-graph tuning recovered the FP16 heuristic plan's 7.12 ms/tile and
added setup work. Mixed measured 8.971 ms/tile with graph ranking and
8.864 with the existing tuner. See `output/smosr-session/screen-graph-tune.json`.
The existing tuner remains the control.

The accumulation screen keeps each mode's existing I/O and pointwise math.
Within the hot 3×3 layers it flips accumulation precision for the 48→48
group, the 48→96 group, or both. Screen each with and without the existing
zero-workspace tuner. Changing accumulation is a numeric trade-off and
belongs behind an option if it improves the page runs.

The interleaved accumulation screen returned 7.065 ms/tile for mixed
with both groups accumulating in FP16, versus 9.169 ms/tile for the
existing tuned mixed control. The three center tiles scored 67.18–71.96 dB
against Torch FP32, compared with 60.56–62.20 dB for native FP16. The
one-group variants were slower. Applying FP32 accumulation to either
group in native FP16 also slowed it down. See
`output/smosr-session/screen-hybrid-accumulation.json` for each trial.

Use `SessionOptions(precision="mixed", kernel_options={"fp16_accumulation": 1})`
to select FP16 accumulation for 48-input 3×3 convolutions. This includes
the 48→48 and 48→96 plans. Mixed I/O, bias, pointwise math, and the other
convolutions retain their original precision. The original mixed mode is
the default. `tune_convolutions=1` can be used with either mixed mode.

### Full-page results

All five runs use batch 1, 256×256 tiles, zero overlap, and the same 242
tiles on the RTX 3050 Laptop GPU. Fidelity compares to Torch FP32.
The tile median uses all individual synchronized calls. Image times sum
the calls for each page. Additional VRAM is the NVML peak over the reading
before backend setup.

| Mode | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR (dB) | Mean SSIMULACRA2 | NVML peak MiB | Additional MiB | Native buffers MiB | First plans ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed | 2680.82 | 853.66 | 10.72 | 11.09 | 72.53 | 89.15 | 1462.79 | 72.00 | 37.74 | 195.91 |
| Mixed, tuned | 2242.77 | 709.33 | 8.98 | 9.21 | 72.53 | 89.12 | 1464.79 | 80.94 | 37.74 | 1427.48 |
| Mixed with FP16 accumulation | 1806.08 | 585.34 | 7.17 | 7.47 | 69.58 | 89.05 | 1456.10 | 72.00 | 37.74 | 198.31 |
| Mixed with FP16 accumulation, tuned | 1773.30 | 572.35 | 7.13 | 7.31 | 69.58 | 89.05 | 1464.79 | 74.00 | 37.74 | 1207.70 |
| FP16 | 1777.50 | 572.17 | 7.03 | 7.33 | 61.80 | 88.69 | 1461.04 | 70.25 | 36.11 | 108.08 |

The opt-in mode reduced median tile latency by 20.2% and median page
latency by 17.5% against the tuned mixed control. Adding convolution
tuning to the new mode saved about 13 ms per median page in this run and
added roughly one second to first plan setup. Its measured page latency
was within 0.2 ms of native FP16 and its PSNR was 7.78 dB higher.

The defaults matched the previous native DLL byte for byte on all 242
tiles and the stress fixtures in all three precisions. The new option's
host, direct, captured, and alternate-stream outputs agreed for schema 1
and schema 2 on the rectangular sample. Invalid option values and
selecting the option in FP16 or FP32 were rejected. The new option was
benchmarked at 256×256.

Report directories under `output/architecture-baselines/smosr/batch-1/`
are `native-mixed-accum-control`, `native-mixed-accum-control-tuned`,
`native-mixed-fp16-accum`, `native-mixed-fp16-accum-tuned`, and
`native-fp16-accum-control`. Their `report.json` files contain per-image
PSNR, SSIMULACRA2, observed NVML peak, and all tile timings and coordinates.

### Reproduce the opt-in run

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$model = 'output/smosr-session/smosr-schema2-benchmark.ummodel'
$library = 'output/smosr-session/native-mixed-fp16-accumulation/umzi_native.dll'
& $smosrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/smosr-session/native-mixed-fp16-accumulation
& $smosrPython scripts/benchmark_architectures.py smosr native-mixed --native-model $model --native-library $library --tag fp16-accum --mixed-fp16-accumulation
& $smosrPython scripts/benchmark_architectures.py smosr native-mixed --native-model $model --native-library $library --tag fp16-accum-tuned --mixed-fp16-accumulation --tune-convolutions
```

## Final backend comparison

Run the seven backends sequentially on the same three `samples/2x-test`
images, with batch 1, 256×256 input tiles, and zero overlap. Rebuild the
Torch eager FP32 reference and export both TensorRT-RTX graphs in a fresh
report directory. Use the validated DLL and package from the mixed tuning
run. Select native mixed with `fp16_accumulation=1` and
`tune_convolutions=1`, native FP16 with the default cuDNN plans, and strict
native FP32 with the NCHW path.

All seven runs completed on the RTX 3050 Laptop GPU with driver 610.62,
Torch 2.15.0+cu132, cuDNN 9.26, ORT 1.31.1, and TensorRT-RTX EP plugin
0.5.0. TF32 was disabled. Both TensorRT-RTX placement profiles showed
exclusive RTX provider execution. The timed sessions enabled CUDA Graph
replay. The exported FP16 and FP32 graphs used their corresponding input
and output dtypes.

| Backend | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR (dB) | Mean SSIMULACRA2 | NVML peak MiB | Additional MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch eager FP32 | 6442.50 | 2112.88 | 26.25 | 26.29 | Reference | Reference | 1564.50 | 170.44 |
| TorchInductor FP16 | 2863.20 | 909.00 | 11.41 | 11.81 | 70.41 | 88.97 | 1488.94 | 84.75 |
| TensorRT-RTX FP16 | 1736.75 | 563.94 | 6.78 | 7.14 | 59.23 | 88.43 | 1568.82 | 169.97 |
| TensorRT-RTX FP32 | 9683.66 | 3092.33 | 39.60 | 40.16 | 100.64 | 89.39 | 1609.25 | 200.44 |
| Native FP32 | 4942.23 | 1576.96 | 20.14 | 20.48 | ∞ | 100.00 | 1509.25 | 102.00 |
| Native FP16 | 1768.09 | 573.10 | 7.01 | 7.27 | 61.80 | 88.69 | 1479.25 | 72.00 |
| Native mixed, FP16 accumulation and tuning | 1791.71 | 589.22 | 7.15 | 7.38 | 69.58 | 89.05 | 1481.25 | 74.00 |

Tile medians use the 242 individual synchronized calls. Image times sum
the calls within each image. Additional VRAM uses the sampled NVML peak
minus the reading before backend setup. Native buffers total 72.16 MiB
for FP32, 36.11 MiB for FP16, and 37.74 MiB for the selected mixed mode.
Native FP32 matched the Torch reference at every page pixel in raw float.

| Backend | `9798855404661.jpg` | `9798855424195.jpg` | `9798855444216.jpg` |
| --- | ---: | ---: | ---: |
| Torch eager FP32 | 2112.88 | 2024.51 | 2305.12 |
| TorchInductor FP16 | 908.85 | 909.00 | 1045.35 |
| TensorRT-RTX FP16 | 544.85 | 563.94 | 627.97 |
| TensorRT-RTX FP32 | 3049.39 | 3092.33 | 3541.94 |
| Native FP32 | 1562.40 | 1576.96 | 1802.87 |
| Native FP16 | 573.10 | 559.90 | 635.08 |
| Native mixed, FP16 accumulation and tuning | 553.09 | 589.22 | 649.41 |

The per-image table shows inference time in milliseconds. Every tile
duration and coordinate, per-image fidelity and VRAM, and backend setup
details are in `output/smosr-final-matrix/smosr/batch-1/<backend>/report.json`.

To repeat the comparison, use a fresh output directory.

```powershell
$env:PYTHONPATH = 'src'
$smosrPython = 'C:/Python314/python.exe'
$model = 'output/smosr-session/smosr-schema2-benchmark.ummodel'
$library = 'output/smosr-session/native-mixed-fp16-accumulation/umzi_native.dll'
$reportRoot = 'output/smosr-final-matrix-repeat'
foreach ($backend in @('fp32', 'inductor', 'rtx', 'rtx-fp32', 'native-fp32', 'native-fp16', 'native-mixed')) {
    $runArgs = @('scripts/benchmark_architectures.py', 'smosr', $backend, '--batch', '1', '--tile', '256', '--overlap', '0', '--input', 'samples/2x-test', '--output', $reportRoot)
    if ($backend -like 'native-*') { $runArgs += @('--native-model', $model, '--native-library', $library) }
    if ($backend -eq 'native-mixed') { $runArgs += @('--mixed-fp16-accumulation', '--tune-convolutions') }
    & $smosrPython @runArgs
    if ($LASTEXITCODE) { throw "Benchmark failed for $backend" }
}
```
