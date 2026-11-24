# FiGSR implementation and baseline

## Workload

Use `models/2x_enhancr_da_figsr.safetensors`, batch 1, 256×256 input tiles,
and all three `samples/2x-test` images. Use zero overlap and replicate the
edges of incomplete tiles. Crop the stitched result to 2× the input size.
Use the NVIDIA RTX 3050 Laptop GPU and `C:/Python314/python.exe`.

Verify eager CUDA FP32 in Resselt first. Use those renders as the reference
for TorchInductor FP32 and FP16 and the native FP32, mixed, and FP16 ports. Omit
TensorRT-RTX from the FiGSR matrix because the model uses DFT. Measure
synchronized batch-1 forward time per tile and image, NVML
peak and increase from before backend setup, raw PSNR, and SSIMULACRA2.
Record the first native implementation before changing its kernels for speed.

## Checkpoint and native plan

Resselt 1.4.1 recognized `FIGSR`. An eager FP32 forward on NVIDIA CUDA
mapped `(1, 3, 256, 256)` to `(1, 3, 512, 512)`. The model has 2,005,794
parameters, a 32-channel stem, and two groups of 12 gated residual blocks.
Each block splits 80 value channels into a 48-channel identity, an
eight-channel Fourier branch, and three eight-channel dense spatial
convolutions with 13×13, 1×17, and 17×1 kernels. The stem, first group,
and final feature maps feed a 1×1 projection and direct 2× pixel shuffle.

The port preserves the checkpoint's RGB normalization, four-pixel
reflection border with even-size alignment, RMS normalization, and spectral
channel order. Use the shared cuDNN convolution planner and cuFFT with
FP32 transforms. Establish all three native baselines before trying any
optional kernels.

## Package and precision

Family 6, schema 1 contains the fixed 11-integer topology, the weight
section, and the source checkpoint SHA-256. The weight section uses
256-byte parameter alignment. It stores RGB shift and scale, the stem,
24 blocks, the end convolution, the concatenation projection, and the
shuffle head. Each block includes three RMS parameter vectors, the gated
expansion and projection, the spectral depthwise and 1×1 convolutions,
and the three dense spatial convolutions. Filters are FP32 OHWI with
within-group input lanes. The stem pads RGB to eight lanes and the
shuffle head pads its 12 output lanes to 16.

- **FP32** uses FP32 filters, activations, I/O, and convolution accumulation.
  cuDNN plans with input down-conversion, reduced-precision reduction, or
  Tensor Core numerical notes are excluded in this mode.
- **Mixed** uses FP32 I/O and pointwise math with FP16 filters and
  activation storage. Convolutions accumulate in FP32.
- **FP16** uses FP16 I/O, filters, and activations. It rounds bias and
  pointwise intermediates at the half-precision boundaries. It requests
  FP16 convolution accumulation and uses FP32 accumulation when no
  supported half-accumulation plan is available for a convolution shape.

All modes use orthonormal FP32 real-to-complex and complex-to-real cuFFT
transforms. The incoming spectrum has separate real and imaginary channel
groups. The learned projection produces adjacent real/imaginary pairs for
the inverse. The C++ runner handles reflection padding internally and
accepts input dimensions of at least six pixels. It reports zero external
tile context and supports prepared CUDA Graphs and direct execution.

## Baseline measurements

The machine used the RTX 3050 Laptop GPU with driver 610.62, Python 3.14.5,
Torch 2.15.0+cu132, cuDNN 9.26, Resselt 1.4.1, and Triton 3.9.0.post30.
TF32 was disabled. Measurements cover 242 synchronized device-resident
forwards across three pages. VRAM is the NVML device-wide peak and its
increase over each process's reading before backend setup. Timings exclude
shape preparation, compilation, image transfers, and metric calculation.

| Backend | Total ms | Median image ms | Median tile ms | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch eager FP32 | 45038.40 | 14643.33 | 184.186 | 185.010 | Reference | Reference | 1376.39 | 410.74 |
| TorchInductor FP32 | 53423.90 | 16979.64 | 220.513 | 220.515 | 99.80 | 90.68 | 1847.18 | 676.88 |
| TorchInductor FP16 | 15963.04 | 5163.47 | 65.362 | 65.772 | 66.49 | 88.50 | 1185.29 | 85.74 |
| Native FP32 | 55789.91 | 17728.30 | 229.190 | 230.238 | 100.27 | 90.68 | 1256.55 | 317.95 |
| Native mixed | 19537.09 | 6232.96 | 80.455 | 80.816 | 68.06 | 88.76 | 1134.55 | 142.00 |
| Native FP16 | 16063.16 | 5114.77 | 66.195 | 66.390 | 62.86 | 88.31 | 1134.55 | 142.00 |

Both Inductor modes completed the full-graph benchmark. On the artificial
checkerboard tile, FP32 and FP16 Inductor scored 26.17 and 24.98 dB,
respectively, against eager FP32. The table aggregates the three pages.
Stress scores are recorded separately. TensorRT-RTX is outside the
requested matrix. The initial legacy export had mismatched I/O type and
DFT attributes, and the probe produced no inference measurement.

The native FP32 checkerboard scored 26.15 dB. Its page PSNR ranged from
95.67 to 121.81 dB. The half-storage modes used zero convolution workspace
for the benchmark shape. Native FP32 used 51.71 MiB of convolution
workspace. cuFFT reported zero explicit workspace for this shape in all
three modes.

### Per-image results

Tile time is the mean of the synchronized calls for each image. Raw PSNR
uses peak 1 before quantization, and SSIMULACRA2 compares rendered PNGs.

| Backend | Image | Inference ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager FP32 | `9798855404661.jpg` | 14643.33 | 190.173 | Reference | Reference | 1193.02 |
| Torch eager FP32 | `9798855424195.jpg` | 14245.77 | 185.010 | Reference | Reference | 1354.01 |
| Torch eager FP32 | `9798855444216.jpg` | 16149.30 | 183.515 | Reference | Reference | 1376.39 |
| TorchInductor FP32 | `9798855404661.jpg` | 16935.13 | 219.937 | 117.89 | 89.68 | 1745.93 |
| TorchInductor FP32 | `9798855424195.jpg` | 16979.64 | 220.515 | 121.38 | 91.07 | 1795.80 |
| TorchInductor FP32 | `9798855444216.jpg` | 19509.13 | 221.695 | 95.19 | 91.27 | 1847.18 |
| TorchInductor FP16 | `9798855404661.jpg` | 5011.61 | 65.086 | 66.31 | 87.35 | 1173.97 |
| TorchInductor FP16 | `9798855424195.jpg` | 5163.47 | 67.058 | 66.88 | 89.21 | 1185.29 |
| TorchInductor FP16 | `9798855444216.jpg` | 5787.96 | 65.772 | 66.31 | 88.94 | 1185.29 |
| Native FP32 | `9798855404661.jpg` | 17657.51 | 229.318 | 117.06 | 89.73 | 1202.60 |
| Native FP32 | `9798855424195.jpg` | 17728.30 | 230.238 | 121.81 | 91.09 | 1247.10 |
| Native FP32 | `9798855444216.jpg` | 20404.10 | 231.865 | 95.67 | 91.22 | 1256.55 |
| Native mixed | `9798855404661.jpg` | 6192.35 | 80.420 | 67.36 | 87.56 | 1134.55 |
| Native mixed | `9798855424195.jpg` | 6232.96 | 80.948 | 68.54 | 89.46 | 1134.55 |
| Native mixed | `9798855444216.jpg` | 7111.77 | 80.816 | 68.35 | 89.26 | 1134.55 |
| Native FP16 | `9798855404661.jpg` | 5106.03 | 66.312 | 62.28 | 87.06 | 1134.55 |
| Native FP16 | `9798855424195.jpg` | 5114.77 | 66.426 | 63.26 | 89.04 | 1134.55 |
| Native FP16 | `9798855444216.jpg` | 5842.36 | 66.390 | 63.10 | 88.84 | 1134.55 |

## Runtime verification

The exported package is `output/figsr-session/figsr.ummodel`. The
byte-identical `figsr-benchmark.ummodel` copy was used for validation
and timing because the workspace sandbox could not read the exported
file after the temporary package was moved into place. The package
digest is `db16357648ab9ef63707dd3662c5e5997104572e3cd65e948bc2e54e3102d9ed`.

All three native modes matched bitwise between CUDA Graph replay, direct
device dispatch, host execution, and an alternate CUDA stream on the
first center tile. The center tile scored 142.52, 75.41, and 66.23 dB for
FP32, mixed, and FP16 respectively. Shape checks covered a minimum 6×6
input, rejection of 5×6, and independent tiles in an odd-sized batch.
The FP16 plan diagnostic reported zero FP32 fallbacks at 256×256.
See `output/figsr-session/validation.log`.

A GFISRv2 mixed-precision smoke test also produced finite 4× output
through the shared cuDNN planner and cuFFT loader in the rebuilt library.

## Reproduce

Run from the repository root with the checkout's CUDA and cuFFT libraries.

```powershell
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$figsrPython = 'C:/Python314/python.exe'
$root = 'output/figsr-session'
$model = "$root/figsr.ummodel"
$library = "$root/native/umzi_native.dll"
& $figsrPython -m umzi_native export models/2x_enhancr_da_figsr.safetensors -o $model --kernel figsr
& $figsrPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output "$root/native"
foreach ($backend in 'fp32', 'inductor-fp32', 'inductor') {
    & $figsrPython scripts/benchmark_architectures.py figsr $backend
}
foreach ($backend in 'native-fp32', 'native-mixed', 'native-fp16') {
    & $figsrPython scripts/benchmark_architectures.py figsr $backend --native-model $model --native-library $library
}
```

Reports are under `output/architecture-baselines/figsr/batch-1/`. Each
`report.json` contains the individual image metrics, every synchronized
tile time with coordinates, validation fixtures, and setup measurements.
