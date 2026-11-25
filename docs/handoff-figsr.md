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
  activation storage. Convolutions accumulate in FP32 by default.
  `fp16_expand_accumulation=1` allows FP16 accumulation in the block
  expansion only.
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

## Optimization passes

Keep the baseline package and DLL as controls. Repeat the same batch-1,
256×256, zero-overlap workload. Use paired runs for latency decisions
and separate processes for NVML and fidelity against the eager FP32 renders.

### Pass 1. Fuse bias, gating, and normalization with their consumers

The control CUPTI trace is in `output/figsr-session/optimization-control-profile.json`.
FP16 spent 14.76 ms/tile in bias and residual kernels, 7.21 in the gate,
6.87 in channel copies, and 3.04 in cuFFT. The corresponding mixed times
were 14.01, 7.09, 6.77, and 2.90 ms. GFISRv2's fused data movement applies
here, while the FiGSR spatial branches use dense convolutions.

Apply the expansion bias when consuming each slice. Fuse the identity
branch copy and the spatial bias and output copies with the gate. Apply
the same gate after inverse Fourier normalization. Share the residual
update with the next block's RMS read, preserving the activation storage
rounding between them. Use 16-lane and eight-lane reductions in the
spectral normalization kernels.

| Precision | Control median ms/tile | Pass 1 median ms/tile | Control median ms/image | Pass 1 median ms/image |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 63.747 | 45.193 | 4916.21 | 3479.05 |
| Mixed | 77.670 | 59.373 | 5986.05 | 4584.88 |
| FP32 | 221.178 | 197.469 | 17115.21 | 15292.71 |

All 242 tiles in each precision and all stress fixtures matched the
control bitwise. FP16 custom kernel time fell from 41.02 to 18.09 ms in
the separate CUPTI run, and mixed fell from 39.40 to 17.76 ms. Fusion is
the default for all three precisions. See `paired-native-pass1.json` and
`optimization-pass1-profile.json` under `output/figsr-session/`.

**Next pass**. Time alternative cuDNN plans for the recurrent expansion,
projection, and three dense spatial convolutions. Keep the first plan as
the control and retain the existing workspace limit.

### Pass 2. Screen alternate cuDNN plans

Use the shared cuDNN plan timer from GFISRv2. Time the five recurrent
shapes with the precision and workspace allocated by the heuristic path.
Replace a plan only when its measured convolution time drops by at least
3%. Screen the effect on the full captured forward and compare fidelity.
The screen uses the opt-in `tune_convolutions=1` option because it adds
shape setup work and may change convolution accumulation order.

| Precision | Pass 1 ms/tile | Tuned ms/tile | Candidates timed | Plans changed | Tuned plan stage ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 47.004 | 44.036 | 187 | 3 | 2235.76 |
| Mixed | 61.050 | 54.682 | 179 | 3 | 3163.34 |
| FP32 | 196.970 | 184.196 | 36 | 1 | 841.00 |

These are medians of the three paired center-tile medians. All modes
gained speed and changed raw output on stress fixtures. Plan selection
stays opt-in. It keeps each precision's original convolution math type
and uses the previously allocated workspace. The screen is in
`output/figsr-session/paired-pass1-pass2-tuned.json`.

**Next pass**. Reuse the Fourier staging for the spatial branches and
the expansion buffer for the final concatenation after their last reads.

### Pass 3. Reuse inactive activation buffers

The spatial branches run after inverse Fourier normalization and gating.
At that point the spectral and planar real buffers can hold their input
and output respectively. The final three-way concatenation fits in the
larger expansion buffer after the last block. The reuse depends on
single-stream ordering and leaves the convolution workspace and rounding
points intact.

The activation allocation fell from 70.22 to 55.33 MiB in FP16 and mixed,
and from 136.17 to 106.40 MiB in FP32. The paired center-tile medians
were within 0.36 ms of Pass 1 in each precision. Stress outputs matched
Pass 1 bitwise. The final default renders and raw page MSE values matched
the original baseline on all three pages in each precision. Graph, direct,
host, and alternate-stream calls passed again. See
`output/figsr-session/paired-pass1-optimized.json` and
`output/figsr-session/optimization-validation.log`.

**Next pass**. Screen an NCHW layout for the two recurrent strict FP32
3×3 convolutions. The Pass 1 profile attributed 23.87 ms/tile to cuDNN's
layout conversions. Test the layout with strict FP32 math and compare both
page fidelity and the checkerboard stress tile before enabling it.

## Final optimized metrics

The default uses the fusions and buffer reuse. The tuned rows additionally
set `kernel_options={"tune_convolutions": 1}`. These are separate runs with
the same checkpoint, tiles, zero overlap, and FP32 reference as the
baseline. NVML reports device-wide VRAM and its increase over the reading
before backend setup.

| Native mode | Total ms | Median image ms | Median tile ms | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 default | 10875.47 | 3460.16 | 44.771 | 44.937 | 62.86 | 88.31 | 1003.73 | 124.00 |
| FP16 tuned | 10323.91 | 3287.16 | 42.523 | 42.690 | 62.82 | 88.33 | 1005.73 | 126.00 |
| Mixed default | 14187.28 | 4532.59 | 58.370 | 58.665 | 68.06 | 88.76 | 1003.73 | 124.00 |
| Mixed tuned | 12923.33 | 4113.82 | 53.220 | 53.426 | 68.08 | 88.69 | 1005.73 | 126.00 |
| FP32 default | 47388.22 | 15191.07 | 195.388 | 195.437 | 100.27 | 90.68 | 1107.73 | 228.00 |
| FP32 tuned | 44249.00 | 14109.48 | 182.613 | 182.773 | 100.27 | 90.69 | 1107.73 | 228.00 |

The final tuned plan stage took 2.15 s in FP16, 2.78 s in mixed, and
0.79 s in FP32. The heuristic plan stage took 0.12 s in each mode.
FiGSR reruns the tuner when the session prepares a new shape.

### Per-image tuned metrics

| Precision | Image | Inference ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | `9798855404661.jpg` | 3272.58 | 42.501 | 62.32 | 87.19 | 1005.73 |
| FP16 | `9798855424195.jpg` | 3287.16 | 42.690 | 63.29 | 89.01 | 1005.73 |
| FP16 | `9798855444216.jpg` | 3764.18 | 42.775 | 62.89 | 88.80 | 1005.73 |
| Mixed | `9798855404661.jpg` | 4094.57 | 53.176 | 67.38 | 87.44 | 1005.73 |
| Mixed | `9798855424195.jpg` | 4113.82 | 53.426 | 68.54 | 89.33 | 1005.73 |
| Mixed | `9798855444216.jpg` | 4714.93 | 53.579 | 68.41 | 89.28 | 1005.73 |
| FP32 | `9798855404661.jpg` | 14073.54 | 182.773 | 116.84 | 89.68 | 1107.73 |
| FP32 | `9798855424195.jpg` | 14109.48 | 183.240 | 121.94 | 91.18 | 1107.73 |
| FP32 | `9798855444216.jpg` | 16065.98 | 182.568 | 95.67 | 91.21 | 1107.73 |

The `native-{fp16,mixed,fp32}-optimized` and `-optimized-tuned` directories
under `output/architecture-baselines/figsr/batch-1/` preserve the full
per-image and per-tile records. The benchmark accepts
`--tag optimized-tuned --tune-convolutions` for the tuned run.

### Pass 4. Screen NCHW for the recurrent strict FP32 convolutions

Compare the tuned NHWC control with an NCHW expansion and projection at
batch 1, 256×256, zero overlap, and the same three `samples/2x-test` pages.
Pack the two filters in OIHW order and let the existing bias, gate, and
normalization kernels read or write those layouts directly. Leave the
spatial and spectral branch convolutions in NHWC. Compare runtime, NVML,
page fidelity, and the checkerboard stress output before choosing a default.

The paired center screen gave 199.87 ms/tile with tuned NHWC and 172.51
ms/tile with tuned NCHW, by median of the three tile medians. Every stress
output in that comparison matched bitwise. The cuDNN convolution
workspace fell from 51.71 to 17.15 MiB with the same activation allocation.
The NCHW CUPTI trace shows 2.13 ms/tile in cuDNN layout conversions,
compared with 23.87 ms in the previous post-fusion trace. The NCHW path
uses strict FP32 plans and preserves the same reduction tree in RMS
normalization. See `paired-optimized-fp32-layout-depthwise-nchw.json`
and `pass4-nchw-profile.json` under `output/figsr-session/`.

The first prototype shared layout branches with the half-storage paths.
It preserved their stress output and added about 1 ms/tile in mixed mode
in the center screen. Dedicated NCHW CUDA entry points restore the original
NHWC kernels. The retained option is `fp32_nchw=1` and can be used with
`tune_convolutions=1`.

In the final build, tuned NCHW lowered the paired median from 192.556 to
165.351 ms/tile and from 14971.48 to 12869.78 ms/image across the three
pages. The stress fixtures matched the tuned control bitwise in that
process. The paired page phase prepared the fixed shape again after two
rectangular stress cases, and the two tuned outputs differed on all 242
tiles. Use the separate raw PSNR and SSIMULACRA2 runs for page fidelity.
FP16 and mixed matched their controls bitwise on all stress fixtures with
center medians within 1 ms. The NCHW path passed graph, direct, host, and
alternate-stream inference, odd-size batching, and minimum-size checks.
See `paired-optimized-pass4-nchw-final-final-pages.json`,
`paired-optimized-pass4-nchw-final-half-controls.json`, and
`pass4-final-validation.log` under `output/figsr-session/`.

**Next pass**. Try replacing cuDNN's FP32 grouped spectral convolution
with a direct 16-channel depthwise kernel, as in GFISRv2. Fuse its bias and
residual and keep the transform and RMS stages fixed.

### Pass 5. Fuse the FP32 spectral depthwise convolution and residual

The previous FP32 trace attributes 19.76 ms/tile to cuDNN's grouped
convolution. Screen a direct 3×3 depthwise implementation that applies
the bias and residual before writing the spectrum. Measure it against
the same tuned control independently of the NCHW experiment.

The direct kernel ran in 0.89 ms/tile in the combined CUPTI trace. On
NHWC it moved the paired center median from 205.216 to 204.878 ms/tile.
With NCHW already enabled, the paired page run moved the median from
171.228 to 170.489 ms/tile and from 13201.11 to 13137.58 ms/image. All
242 outputs in that comparison matched bitwise. The 0.74 ms/tile gain
did not warrant an additional option, so the retained NCHW build uses
the cuDNN depthwise plan. See
`paired-fp32-layout-depthwise-fp32-layout-depthwise-pages-isolate-dw.json`
and `pass4-nchw-dw-profile.json` under `output/figsr-session/`.

**Next pass**. Time the three dense spatial convolutions individually in
the NCHW body path. They now account for much of the remaining strict FP32
time. Start with the 13×13 branch and compare strict cuDNN plans with a
specialized implementation before changing the arithmetic or workspace.

### Passes 4 and 5 final metrics

The separate FP32 runs used `tune_convolutions=1` in both processes and
`fp32_nchw=1` in the candidate. Every page had the same raw MSE against
Torch FP32 and the same rendered PNG bytes in the two runs. Both scored
26.188 dB on checkerboard. NVML values are device-wide, and the increase
uses each process's pre-backend reading.

| Tuned FP32 | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NHWC control | 45757.34 | 14590.62 | 188.739 | 189.285 | 100.27 | 90.69 | 1674.93 | 233.06 |
| NCHW expansion and projection | 39507.58 | 12593.63 | 163.035 | 163.554 | 100.27 | 90.69 | 1641.05 | 194.13 |

| Image | Control ms | NCHW ms | Control mean ms/tile | NCHW mean ms/tile | PSNR dB | SSIMULACRA2 | NCHW NVML peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | 14509.62 | 12518.39 | 188.437 | 162.577 | 116.84 | 89.68 | 1640.93 |
| `9798855424195.jpg` | 14590.62 | 12593.63 | 189.489 | 163.554 | 121.94 | 91.18 | 1640.93 |
| `9798855444216.jpg` | 16657.10 | 14395.56 | 189.285 | 163.586 | 95.67 | 91.21 | 1641.05 |

The fixed-shape plan stage took 807 ms in the control and 581 ms with
NCHW. The per-image device peaks and individual tile timings are in
`native-fp32-pass4-nhwc-control/report.json` and
`native-fp32-pass4-nchw/report.json` below
`output/architecture-baselines/figsr/batch-1/`.

Rebuild with `scripts/build_native.py` as above. For the optimized run,
pass `--tune-convolutions --figsr-fp32-nchw --tag pass4-nchw` to
`scripts/benchmark_architectures.py figsr native-fp32` with the package and
library paths. Python callers use
`SessionOptions(precision="fp32", kernel_options={"tune_convolutions": 1, "fp32_nchw": 1})`.

### Pass 6. Screen FP16 accumulation in mixed expansion

Use the tuned mixed control at batch 1, 256×256, zero overlap, and the same
three pages. The earlier mixed profile spent 17.65 ms/tile in the most
expensive recurring convolution kernel. Screen FP16 accumulation in the
32→160 expansion with the mixed projection, gates, I/O, and normalization
unchanged. Keep it opt-in if it improves latency because it changes the
numeric policy.

The center test moved from 54.347 to 50.109 ms/tile. The paired page test
lowered median tile latency from 55.140 to 50.671 ms and median image
latency from 4249.14 to 3915.51 ms across the three pages. The CUDA plan
diagnostic reported zero FP32 fallbacks for the requested expansion.
Keep `fp16_expand_accumulation=1` opt-in and combine it with
`tune_convolutions=1` for that configuration. The reports are
`paired-half-candidates-half-candidates-mixed-expand*.json`.

Separate warmed runs measured raw PSNR and rendered SSIMULACRA2 against
the Torch eager FP32 reference. The candidate lowered median tile
latency by 4.9% in those runs and by 8.1% with paired timing. It cost
3.44 dB PSNR and 0.15 mean SSIMULACRA2. NVML peak is device-wide. The
increase is measured from the device reading before the backend starts.

| Tuned mixed | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 expansion accumulation | 13296.07 | 4225.55 | 54.742 | 54.877 | 68.08 | 88.69 | 1448.22 | 126 |
| FP16 expansion accumulation | 12634.62 | 4021.70 | 52.072 | 52.230 | 64.64 | 88.54 | 1450.22 | 128 |

| Image | Expansion accumulation | Image ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `9798855404661.jpg` | FP32 | 4206.39 | 54.628 | 67.38 | 87.44 | 1448.22 |
| `9798855424195.jpg` | FP32 | 4225.55 | 54.877 | 68.54 | 89.33 | 1448.22 |
| `9798855444216.jpg` | FP32 | 4864.13 | 55.274 | 68.41 | 89.28 | 1448.22 |
| `9798855404661.jpg` | FP16 | 4007.35 | 52.044 | 63.74 | 87.36 | 1450.22 |
| `9798855424195.jpg` | FP16 | 4021.70 | 52.230 | 65.08 | 89.37 | 1450.22 |
| `9798855444216.jpg` | FP16 | 4605.57 | 52.336 | 65.24 | 88.87 | 1450.22 |

See `native-mixed-pass6-mixed-control/report.json` and
`native-mixed-pass6-mixed-expand/report.json` under
`output/architecture-baselines/figsr/batch-1/`. To reproduce the candidate,
run `scripts/benchmark_architectures.py figsr native-mixed` with
`--tune-convolutions --figsr-mixed-fp16-expand --tag pass6-mixed-expand`
and the model and library paths. Python callers use
`SessionOptions(precision="mixed", kernel_options={"tune_convolutions": 1, "fp16_expand_accumulation": 1})`.

**Next pass**. Check whether the FP16 gate can save time without changing
its rounding, then screen additional convolution plans if needed.

### Pass 7. Screen an exact SiLU lookup in FP16

Use the tuned FP16 control on the same workload. Before the gate's SiLU
the activation has already rounded to FP16. Precompute the original
rounded SiLU for every half bit pattern, then look it up in the three
FP16 gate kernels. Compare stress outputs bitwise and measure whether the
lookup avoids enough activation math to pay for the table reads.

The lookup matched the heuristic control bitwise on all stress inputs.
The heuristic center medians were 46.244 and 46.815 ms/tile for the control
and lookup. The tuned center medians were 43.598 and 44.650 ms/tile.
Drop the lookup. See the `fp16-lut-fixed` and `fp16-lut-tuned` paired
reports below `output/figsr-session/`.

**Next pass**. Test whether the current zero-workspace cap excludes faster
FP16 convolution plans.

### Pass 8. Screen a bounded FP16 tuning workspace

Allow up to 32 MiB of workspace in the FP16 plan screen for the expansion,
projection, and three dense spatial convolutions. Keep the steady-state
allocation at the maximum required by the selected plans. Compare warmed
time and fidelity with the existing tuned FP16 control.

The wider cap timed 194 candidates instead of 187. The selected plans
still used zero workspace, and the paired center median was 43.811
ms/tile compared with 43.734 ms for the control. Drop the workspace
option. See `paired-half-workspace-half-workspace-fp16-32mib.json`.

**Next pass**. Use packed FP16 bias and multiplication in the identity and
spatial gates, keeping the existing FP32 SiLU computation and FP16
rounding.

### Pass 9. Pack two channels per thread in the FP16 gates

Test `half2` loads, bias addition, and output multiplication for the two
spatially local gates. Convert both lanes to float for the current SiLU
expression and round back to FP16 before multiplication. The inverse
Fourier gate still uses the existing per-pixel RMS reduction.

All heuristic stress fixtures matched bitwise. The tuned page mean
improved by only 0.16–0.19 ms/tile on each image, and the independently
tuned sessions differed on all 242 outputs. Drop the packed
gates. See `fp16-packed-heuristic` and `fp16-packed-tuned` in the paired
reports.

**Next pass**. Load all eight Fourier input channels for a pixel together.
The planar extraction in the FP16 trace takes 2.79 ms/tile across the 24
blocks, with each channel currently issuing its own strided reads.

### Pass 10. Vectorize extraction of the Fourier input

For FP16 and mixed, read the eight adjacent half channels in one aligned
16-byte load per pixel. Add the bias and round at the same points as the
scalar path before writing the FP32 cuFFT input in planar order. Keep the
FFT plan and convolutions fixed for the parity screen.

The scalar and vectorized paths produced bitwise-identical outputs on
all 12 heuristic stress inputs in both modes, including the two
rectangular shapes. With tuning enabled in both FP16 sessions, the
paired 242-tile median fell from 44.203 to 42.322 ms (4.3%) and median
image time from 3419.99 to 3285.25 ms. The independently tuned FP16
sessions selected different plans and differed on all 242 outputs.
With tuning and FP16 expansion accumulation enabled in both mixed
sessions, the median fell from 50.861 to 48.790 ms/tile (4.1%) and from
3919.08 to 3761.71 ms/image. All 242 mixed outputs matched bitwise.

Retain the vectorized read for FP16 and mixed. It uses the same activation
buffer, FFT input format, and rounding as the scalar path. See the
`half-extract-heuristic`, `fp16-extract-tuned`, and `mixed-extract-tuned`
paired reports under `output/figsr-session/`.

Separate CUPTI traces put the new extraction at 0.63 ms/tile for FP16
and 0.71 ms/tile for mixed with FP16 expansion, summed across 24 blocks.
Convolutions remain the largest category at 25.17 ms and 31.50 ms per
tile, respectively. See `pass10-fp16-profile.json` and
`pass10-mixed-profile.json` under `output/figsr-session/`.

#### Per-operation follow-up trace

Attribute CUDA kernel time in stream order to the runner's layer calls.
Each column is the mean across six direct forwards, two per image's
center tile, at batch 1 and 256×256 with `tune_convolutions=1`. Times sum the
operation across all blocks. See `pass10-operation-profile.json` in
`output/figsr-session/`.

| Operation | FP16 ms/tile | Mixed ms/tile | Mixed with FP16 expansion ms/tile |
| --- | ---: | ---: | ---: |
| 3×3 expansion, 32→160 | 9.50 | 13.00 | 8.41 |
| Identity gate | 3.33 | 3.20 | 3.24 |
| 3×3 projection, 80→32 | 6.57 | 7.07 | 7.41 |
| Projection residual and next RMSNorm | 3.16 | 2.77 | 2.72 |
| 13×13 spatial convolution | 5.45 | 6.90 | 6.73 |
| 1×17 spatial convolution | 0.89 | 4.71 | 4.73 |
| 17×1 spatial convolution | 1.05 | 1.19 | 1.07 |
| cuFFT total | 3.24 | 3.06 | 3.09 |

**Next pass**. For FP16 and mixed, screen fusing the 3×3 expansion and
identity gate using the GFISRv2 expansion epilogue as a starting point.
Keep the existing rounding between convolution, bias, and gate. For a
smaller mixed-only experiment, screen a different FP32-accumulating plan
or a dedicated kernel for the 1×17 convolution. Both mixed modes show
the same slowdown on that branch. Then screen fusing the 3×3 projection
with its residual and next RMSNorm. Put precision changes behind kernel
options and measure fidelity against the existing reference.

### Pass 10 final metrics

All four runs used `tune_convolutions=1`, batch 1, and fixed 256×256 tiles
with zero overlap. Each run processed the same 242 tiles. Against the
FP32 Torch reference, FP16 stayed within 0.02 dB PSNR and 0.03 mean
SSIMULACRA2 of its scalar control. The two mixed runs reproduced the
per-image PSNR and SSIMULACRA2 of their respective scalar runs in pass 6.

| Tuned mode | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 scalar control | 10799.62 | 3443.82 | 44.462 | 44.610 | 62.81 | 88.36 | 1424.69 | 126 |
| FP16 vectorized | 10226.57 | 3254.48 | 42.050 | 42.266 | 62.82 | 88.33 | 1424.07 | 126 |
| Mixed vectorized | 12913.53 | 4120.33 | 53.186 | 53.421 | 68.08 | 88.69 | 1424.07 | 126 |
| Mixed vectorized, FP16 expansion | 11729.52 | 3729.79 | 48.335 | 48.439 | 64.64 | 88.54 | 1426.07 | 128 |

| Tuned mode | Image | Image ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 scalar control | `9798855404661.jpg` | 3430.08 | 44.546 | 62.19 | 87.13 | 1424.69 |
| FP16 scalar control | `9798855424195.jpg` | 3443.82 | 44.725 | 63.26 | 89.11 | 1424.69 |
| FP16 scalar control | `9798855444216.jpg` | 3925.72 | 44.610 | 63.03 | 88.85 | 1424.07 |
| FP16 vectorized | `9798855404661.jpg` | 3243.49 | 42.123 | 62.32 | 87.19 | 1424.07 |
| FP16 vectorized | `9798855424195.jpg` | 3254.48 | 42.266 | 63.29 | 89.01 | 1424.07 |
| FP16 vectorized | `9798855444216.jpg` | 3728.59 | 42.370 | 62.89 | 88.80 | 1424.07 |
| Mixed vectorized | `9798855404661.jpg` | 4092.16 | 53.145 | 67.38 | 87.44 | 1424.07 |
| Mixed vectorized | `9798855424195.jpg` | 4120.33 | 53.511 | 68.54 | 89.33 | 1424.07 |
| Mixed vectorized | `9798855444216.jpg` | 4701.03 | 53.421 | 68.41 | 89.28 | 1424.07 |
| Mixed vectorized, FP16 expansion | `9798855404661.jpg` | 3728.05 | 48.416 | 63.74 | 87.36 | 1426.07 |
| Mixed vectorized, FP16 expansion | `9798855424195.jpg` | 3729.79 | 48.439 | 65.08 | 89.37 | 1426.07 |
| Mixed vectorized, FP16 expansion | `9798855444216.jpg` | 4271.67 | 48.542 | 65.24 | 88.87 | 1426.07 |

Reports live under `output/architecture-baselines/figsr/batch-1/`.
Each `report.json` contains individual tile timings and the pre-setup
NVML reading. Use `--tune-convolutions` for all rows and
`--figsr-mixed-fp16-expand` for the last row. The vectorized path is
selected automatically by a rebuilt native library for FP16 and mixed.

- `native-fp16-pass10-fp16-control/report.json`
- `native-fp16-pass10-fp16-vector/report.json`
- `native-mixed-pass10-mixed-vector/report.json`
- `native-mixed-pass10-mixed-vector-expand/report.json`

The runtime smoke checks covered graph capture, direct CUDA execution,
host I/O, alternate streams, odd-size batches, and minimum dimensions
for all three modes and mixed with the new option. The benchmark CLI
rejected the mixed-only flag with the FP16 backend.

### Pass 11. Fuse the mixed 1×17 convolution and spatial gate

Keep FP32 accumulation. Stage a horizontal input window in shared memory,
apply the dense eight-channel filter on Tensor Cores, and consume the
rounded convolution output directly in the spatial gate. Compare the
opt-in kernel with the tuned cuDNN control before changing other branches.

In paired page runs, `fuse_row_gate=1` reduced the median from 53.095
to 48.795 ms/tile and from 4096.57 to 3764.87 ms/image with standard mixed
accumulation. With `fp16_expand_accumulation=1` enabled in both sessions,
the median fell from 50.088 to 45.505 ms/tile and from 3867.90 to 3504.68
ms/image. All 242 tiles matched bitwise in each comparison, and all 12
heuristic stress fixtures had zero difference. Keep the fused path
opt-in. See `paired-pass11-row-pass11-row-row-gate-*.json` under
`output/figsr-session/`.

The fused convolution and gate took 1.39 ms/tile in a separate trace.
See `pass11-row-profile.json`.

**Next pass**. Screen the shared expansion and identity gate with a
separate option to isolate its latency and rounding changes.

### Pass 12. Fuse the 3×3 expansion and identity gate

Use shared-memory tiles and the same Tensor Core instructions as GFISRv2
for the 32→160 expansion. Apply the identity gate to the paired channels
inside the kernel, at the original half-rounding points. The remaining
32 gate/value pairs still go to the existing Fourier and spatial paths.
Screen FP16 and mixed independently against the same tuned plan policy.

The first center screen moved the median from 42.402 to 41.755 ms/tile
in FP16, 52.477 to 49.874 ms/tile in mixed, and 45.427 to 44.120 ms/tile
in mixed with both `fuse_row_gate` and `fp16_expand_accumulation` enabled.
See `paired-pass12-expand-pass12-expand-expand-gate-*.json`.

**Next pass**. Have the same expansion kernel pack the first eight branch
values directly into the planar FP32 cuFFT input. Screen the combined
epilogue before retaining it as an option.

### Pass 13. Pack the FFT input in the fused expansion

Write the eight Fourier value channels directly to the existing planar
FP32 cuFFT input. Apply the same bias and half rounding as the standalone
extraction. Keep the corresponding gate channels for the inverse Fourier
path. Compare against each precision mode's tuned path, retaining the
fused row gate in both mixed sessions.

With `fuse_expand_gate=1`, the paired FP16 page median fell from 42.232
to 40.690 ms/tile and 3271.49 to 3159.09 ms/image. The independently
tuned FP16 sessions picked different plans and differed on all 242
tile outputs. Standard mixed fell from 48.920 to 46.128 ms/tile and
3771.65 to 3563.66 ms/image, with all 242 tile outputs matching bitwise.
With FP16 expansion accumulation enabled in both mixed sessions, the
median fell from 44.679 to 43.040 ms/tile and 3501.02 to 3373.87
ms/image. All 242 tiles matched bitwise there as well. Each comparison
matched 11 of 12 stress outputs exactly. See the `fused-expansion-*`
paired reports under `output/figsr-session/`.

Retain `fuse_expand_gate=1` as an opt-in for FP16 and mixed. It removes
two launches per block and uses the original FP32 cuFFT input buffer.

**Next pass**. Use the GFISRv2 fused projection and RMSNorm epilogue to
screen FiGSR's 80→32 convolution with the residual and next RMSNorm.
Retain the intermediate half rounding and check both accumulation modes
before enabling it on the measured workload.

### Passes 11–13 final metrics

All runs used `tune_convolutions=1`, batch 1, fixed 256×256 tiles, zero
overlap, and the same 242 tiles. NVML peak is device-wide and peak
increase uses each run's pre-setup device reading. Both fused mixed
variants reproduced their respective Pass 10 per-image PSNR and
SSIMULACRA2. The FP16 control and fused run were within 0.02 dB PSNR
and 0.03 mean SSIMULACRA2 of each other.

| Tuned mode | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 control | 10189.40 | 3239.16 | 41.977 | 42.067 | 62.81 | 88.36 | 1500.29 | 126.13 |
| FP16 fused | 9817.51 | 3130.33 | 40.363 | 40.654 | 62.82 | 88.33 | 1498.29 | 124.00 |
| Mixed fused | 11132.76 | 3561.13 | 45.846 | 46.106 | 68.08 | 88.69 | 1496.54 | 124.00 |
| Mixed fused with FP16 expansion | 10331.17 | 3311.28 | 42.553 | 42.616 | 64.64 | 88.54 | 1496.54 | 124.00 |

| Tuned mode | Image | Image ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 control | `9798855404661.jpg` | 3239.16 | 42.067 | 62.19 | 87.13 | 1500.29 |
| FP16 control | `9798855424195.jpg` | 3232.64 | 41.982 | 63.26 | 89.11 | 1500.29 |
| FP16 control | `9798855444216.jpg` | 3717.60 | 42.245 | 63.03 | 88.85 | 1500.29 |
| FP16 fused | `9798855404661.jpg` | 3106.70 | 40.347 | 62.32 | 87.19 | 1498.29 |
| FP16 fused | `9798855424195.jpg` | 3130.33 | 40.654 | 63.29 | 89.01 | 1498.29 |
| FP16 fused | `9798855444216.jpg` | 3580.48 | 40.687 | 62.89 | 88.80 | 1498.29 |
| Mixed fused | `9798855404661.jpg` | 3514.31 | 45.640 | 67.38 | 87.44 | 1496.54 |
| Mixed fused | `9798855424195.jpg` | 3561.13 | 46.248 | 68.54 | 89.33 | 1496.54 |
| Mixed fused | `9798855444216.jpg` | 4057.32 | 46.106 | 68.41 | 89.28 | 1496.54 |
| Mixed fused with FP16 expansion | `9798855404661.jpg` | 3269.65 | 42.463 | 63.74 | 87.36 | 1496.54 |
| Mixed fused with FP16 expansion | `9798855424195.jpg` | 3311.28 | 43.004 | 65.08 | 89.37 | 1496.54 |
| Mixed fused with FP16 expansion | `9798855444216.jpg` | 3750.24 | 42.616 | 65.24 | 88.87 | 1496.54 |

Use `--figsr-fuse-expand-gate` for the fused runs, `--figsr-fuse-row-gate`
for both fused mixed runs, and `--figsr-mixed-fp16-expand` for the last
row. Rebuild the native library and use the package and library paths
as in Pass 10. Reports are under
`output/architecture-baselines/figsr/batch-1/`.

- `native-fp16-pass13-fp16-control/report.json`
- `native-fp16-pass13-fp16-fused/report.json`
- `native-mixed-pass13-mixed-fused/report.json`
- `native-mixed-pass13-mixed-fast/report.json`

Separate CUPTI traces put the fused expansion, gate, and FFT packing at
12.20 ms/tile in FP16 and 15.90 ms/tile in standard mixed. The mixed
row kernel took 1.39 ms/tile with both fusions enabled. See
`pass13-fp16-profile.json` and `pass13-mixed-profile.json`.

Runtime checks covered graph, direct execution, host I/O, alternate
streams, odd-size batch 2, and the minimum 6×6 input. Unsupported
precision and kernel-option combinations were rejected at session open.

## Final Torch and native comparison

The rerun covered three Torch backends and four tuned native configurations in
separate processes on the RTX 3050 Laptop GPU. Use batch 1, fixed 256×256
tiles, zero overlap, and all three `samples/2x-test` images. The fresh
eager FP32 outputs are the common reference for raw PSNR and rendered
SSIMULACRA2. Reports and renders for this comparison are under
`output/figsr-final-rerun/figsr/batch-1/`.

Use the commands in [Reproduce](#reproduce) with
`--output output/figsr-final-rerun` for each backend. For the native
runs, this comparison used `output/figsr-session/figsr-benchmark.ummodel`
and `output/figsr-session/pass12-expand-fft/umzi_native.dll`. Pass
`--tune-convolutions` with these options.

| Backend | Additional flags |
| --- | --- |
| `native-fp32` | `--figsr-fp32-nchw` |
| `native-fp16` | `--figsr-fuse-expand-gate` |
| `native-mixed` | `--figsr-fuse-row-gate --figsr-fuse-expand-gate` |
| `native-mixed` with FP16 expansion | `--figsr-fuse-row-gate --figsr-fuse-expand-gate --figsr-mixed-fp16-expand --tag fp16-expand` |

All seven reports completed and matched the checkpoint hash, image
hashes, tile coordinates, input sizes, and device. Each configuration
timed 242 synchronized forwards. The totals below sum the three image
latencies. NVML is the device-wide reading, and the increase is relative
to each process's pre-backend reading. Each native half-storage mode
peaked at 124 MiB above its own starting reading.

| Backend | Total ms | Median ms/image | Median ms/tile | Median page mean ms/tile | PSNR dB | Mean SSIMULACRA2 | NVML peak MiB | Peak increase MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch eager FP32 | 44749.80 | 14240.18 | 184.680 | 184.937 | Reference | Reference | 1667.59 | 207.56 |
| TorchInductor FP32 | 51331.50 | 16453.90 | 210.420 | 211.611 | 99.80 | 90.68 | 1747.47 | 288.13 |
| TorchInductor FP16 | 16489.09 | 5417.74 | 66.729 | 67.264 | 66.49 | 88.50 | 1602.91 | 156.00 |
| Native FP32 | 39393.94 | 12531.69 | 162.426 | 162.749 | 100.27 | 90.69 | 1622.11 | 196.00 |
| Native FP16 | 9897.19 | 3157.23 | 40.706 | 40.900 | 62.82 | 88.33 | 1550.11 | 124.00 |
| Native mixed | 11151.03 | 3562.42 | 45.891 | 46.265 | 68.08 | 88.69 | 1549.79 | 124.00 |
| Native mixed with FP16 expansion | 10357.25 | 3294.05 | 42.673 | 42.780 | 64.64 | 88.54 | 1549.48 | 124.00 |

### Per-image final results

| Backend | Image | Inference ms | Mean ms/tile | PSNR dB | SSIMULACRA2 | NVML peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager FP32 | `9798855404661.jpg` | 14172.45 | 184.058 | Reference | Reference | 1667.59 |
| Torch eager FP32 | `9798855424195.jpg` | 14240.18 | 184.937 | Reference | Reference | 1667.59 |
| Torch eager FP32 | `9798855444216.jpg` | 16337.17 | 185.650 | Reference | Reference | 1667.34 |
| TorchInductor FP32 | `9798855404661.jpg` | 16453.90 | 213.687 | 117.89 | 89.68 | 1747.47 |
| TorchInductor FP32 | `9798855424195.jpg` | 16294.06 | 211.611 | 121.38 | 91.07 | 1746.91 |
| TorchInductor FP32 | `9798855444216.jpg` | 18583.54 | 211.177 | 95.19 | 91.27 | 1746.91 |
| TorchInductor FP16 | `9798855404661.jpg` | 5152.16 | 66.911 | 66.31 | 87.35 | 1602.91 |
| TorchInductor FP16 | `9798855424195.jpg` | 5417.74 | 70.360 | 66.88 | 89.21 | 1602.91 |
| TorchInductor FP16 | `9798855444216.jpg` | 5919.20 | 67.264 | 66.31 | 88.94 | 1602.91 |
| Native FP32 | `9798855404661.jpg` | 12519.53 | 162.591 | 116.84 | 89.68 | 1622.11 |
| Native FP32 | `9798855424195.jpg` | 12531.69 | 162.749 | 121.94 | 91.18 | 1622.11 |
| Native FP32 | `9798855444216.jpg` | 14342.72 | 162.985 | 95.67 | 91.21 | 1622.11 |
| Native FP16 | `9798855404661.jpg` | 3140.77 | 40.789 | 62.32 | 87.19 | 1550.11 |
| Native FP16 | `9798855424195.jpg` | 3157.23 | 41.003 | 63.29 | 89.01 | 1550.11 |
| Native FP16 | `9798855444216.jpg` | 3599.20 | 40.900 | 62.89 | 88.80 | 1550.11 |
| Native mixed | `9798855404661.jpg` | 3517.02 | 45.676 | 67.38 | 87.44 | 1549.79 |
| Native mixed | `9798855424195.jpg` | 3562.42 | 46.265 | 68.54 | 89.33 | 1549.79 |
| Native mixed | `9798855444216.jpg` | 4071.59 | 46.268 | 68.41 | 89.28 | 1549.48 |
| Native mixed with FP16 expansion | `9798855404661.jpg` | 3272.61 | 42.501 | 63.74 | 87.36 | 1549.48 |
| Native mixed with FP16 expansion | `9798855424195.jpg` | 3294.05 | 42.780 | 65.08 | 89.37 | 1549.48 |
| Native mixed with FP16 expansion | `9798855444216.jpg` | 3790.59 | 43.075 | 65.24 | 88.87 | 1549.48 |

Each backend directory contains `report.json`, including setup timings
and every tile's synchronized latency. Directory names are
`fp32`, `inductor-fp32`, `inductor`, `native-fp32`,
`native-fp16`, `native-mixed`, and
`native-mixed-fp16-expand`.

The first final comparison is archived under
`output/figsr-final-comparison/figsr/batch-1/`. Its original native FP16
run peaked at 1634.38 MiB on `9798855424195.jpg`, 213.56 MiB above its
starting usage. The exact repeat and the full matrix rerun both peaked
at 124 MiB above their respective starting readings. FiGSR's own
session buffers totaled 61.11 MiB for FP16 and 62.99 MiB for mixed.
Both reported zero convolution workspace.

### Native CUDA Graph timing check

Prepare all three native half-storage configurations with their final
options and rotate their execution order on each tile. The three sessions
use CUDA Graphs and the same 242-tile grid as the separate backend runs.

| Image | FP16 mean ms/tile | Mixed mean ms/tile | Mixed with FP16 expansion mean ms/tile |
| --- | ---: | ---: | ---: |
| `9798855404661.jpg` | 40.314 | 46.931 | 43.024 |
| `9798855424195.jpg` | 40.359 | 47.183 | 42.978 |
| `9798855444216.jpg` | 40.397 | 47.112 | 43.167 |
| Median across all tiles | 40.255 | 46.947 | 42.990 |

FP16 was faster on all 242 tiles against standard mixed and on 240 of
242 against mixed with FP16 expansion. The original separate runs
showed 45.298 ms/tile for FP16 and 42.390 ms/tile for mixed with FP16
expansion on `9798855404661.jpg`. The separate matrix rerun moved those
page means to 40.789 and 42.501 ms/tile, respectively. Individual samples
from the rotating-order check are in `native-interleaved.json`.
