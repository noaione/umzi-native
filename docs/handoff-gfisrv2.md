# GFISRv2 native tuning handoff

Pickup on `more-archs` at `951db54`. GFISRv2 FP32, FP16, and mixed are
implemented and validated on the RTX 3050 Laptop GPU. The next task is to
measure the current native runner on the matched 128×128 full-page workload,
then optimize the measured bottleneck. Preserve existing worktree changes.
Never auto-commit.

## Target and current evidence

Beat **4.8052 s/page median** from the captured portable ONNX Runtime TensorRT
10 preset with strict NHWC8 and native depthwise convolutions. Its four-round
range was 4.7665–4.9099 s/page, with 292.49 MiB steady peak above idle,
60.4422 dB raw page PSNR against eager FP32, and 87.2950 grayscale
SSIMULACRA2. Record the memory and fidelity tradeoffs next to speed.
[Source matrix](../../umzi-bench/docs/gfisrv2-trt-experiments.md#four-round-matrix-results)

## Test modes

| Mode | Input | Batch | Input tile | Input overlap | Timing scope |
| --- | --- | ---: | --- | ---: | --- |
| Native tuning | `samples/mscale/lb-page-00{3,4,5}.png` | 2 | 256×256 | 64 px | Synchronized batch-2 throughput and separately measured batch-1 tile latency |
| Matched TRT comparison | `samples/mscale/lb-page-003.png` | 1 | 128×128 | 32 px | Full-page median including tiling, transfers, inference and stitching |

Use 64 px as the working overlap for native tuning. It provides a 256 px
blend region at 4× and a 192 px input stride. On each 900×1350 page, the
native grid increases from 24 real tiles at 32 px overlap to 35 at 64 px,
with 36 tiles submitted after filling the last batch. That increases batch
calls from 12 to 18. The best overlap is unmeasured. Compare 32 and 64 px
once on page 003 with the same runner and precision. Inspect seams, text
edges and halftones at fixed page coordinates and compare full-page time.
Use each overlap's own eager FP32 render for precision comparisons. If
the wider overlap gives no visible seam benefit, 32 px is the cheaper
candidate to revisit.

### Archived 256×256 results

The table uses batch 2 and **32 px overlap**. The 64 px native tuning
baseline and the matched TRT comparison are pending.

| Native precision | Batch-2 time, all 3 pages (s) | Input MPx/s | Mean batch-1 tile (ms) | Warmed batch-2 VRAM delta (MiB) | Raw PSNR (dB) | Mean SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 14.364 | 0.2538 | 202.595 | 362 | ∞ | 100.000 |
| FP16 | 10.785 | 0.3380 | 148.899 | 270 | 60.166 | 86.385 |
| Mixed | 10.950 | 0.3329 | 151.537 | 342 | 63.667 | 86.633 |

- FP32 matched all three untouched eager FP32 pages bit for bit.
- FP16 matched eager FP16 on all 72 raw batch-2 page tiles and all three
  rendered pages. Of 18 stress cases, only 1×7, 2×2, and 3×5 differed. The
  worst absolute difference was 0.00977. Use these as the known tiny-shape
  results when checking regressions.
- Mixed keeps FP32 Fourier processing and residuals and uses half storage
  for spatial convolutions. It gained 3.50 dB over FP16 on the three pages
  at similar throughput. Both reduced modes used 76.2 MiB of batch-2
  workspace. These figures justify starting with convolution plans and
  workspace measurement.
- Keep `unexact_erf=0` for controls. Gray is especially sensitive to rounding.
  Check gray renders for blobs and texture alongside raw comparisons.
  Replicate padding removed the visible blobs on one flat tile while
  crosshatching persisted. The native model uses original zero padding.
  Evaluate padding changes independently on seams, text, and halftones.
  [Gray diagnostics](umzi-native-architecture-expansion.md#flat-gray-padding-diagnostic)

[Per-page results and validation details](umzi-native-architecture-expansion.md#native-fp16-and-mixed-precision)
· [Raw FP16 tile parity](../output/gfisrv2-session/precision-final-fp16/eager-fp16-page-tiles.json)

## 1. Run a short correctness check

Use `C:\Python314\python.exe`. Build into a fresh directory to preserve the
prior DLL, package and reports. The Windows commands use the locally
available CUDA 13.0 NVRTC under `output/nvrtc-13.0/nvidia/cu13`.
The checkpoint is `models/4x_wtp_MangaScale_GfisrV2.pth`. The validator exports
the family-2/schema-1 package and checks its deterministic layout and bad
inputs, FP32/FP16 references, graph/direct parity, host/device dispatch and
CLI output.

```powershell
git status --short --branch
$py = 'C:\Python314\python.exe'
$build = 'output/gfisrv2-session/pickup-win'
& $py scripts/build_native.py --output $build --nvrtc-root output/nvrtc-13.0/nvidia/cu13
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
foreach ($mode in 'fp32', 'fp16', 'mixed') {
    & $py scripts/validate_gfisrv2_session.py --library "$build/umzi_native.dll" --cli "$build/umzi_native.exe" --precision $mode --output "output/gfisrv2-session/pickup-smoke-$mode"
    if ($LASTEXITCODE -ne 0) { throw "GFISRv2 $mode validation failed" }
}
```

Add an explicit overlap setting to `benchmark_architectures.py` and
`validate_gfisrv2_session.py` before running the new three-page suite.
Use it for both eager FP32 reference generation and native validation.
Save the 64 px reference and candidate reports in new directories, and
verify the overlap in the reference report before scoring. The existing
`--pages` command uses 32 px and reads the archived reference in
`output/architecture-baselines/gfisrv2/batch-2/fp32/`.

## 2. Measure the matched full-page baseline

Add a small page benchmark that adapts the native host session to
`umzi-bench`'s `infer_tensor` contract. Use the sibling harness as the
reference for the grid and feather weights.
[Original page harness](../../umzi-bench/scripts/evaluate_umzi_unified_matrix.py)
· [Grid and stitcher](../../umzi-bench/src/janai/inference.py)

1. Use `samples/mscale/lb-page-003.png` (900×1350), batch 1, 128×128 tiles,
   32 input pixels overlap, and 4× output. Use two CPU threads and disable
   TF32. Follow the source harness's row-major 96 px grid, replicate edge
   padding, crop and normalized feather stitch. Expect 150 tile calls and a
   3600×5400 output. Compare at this same tiling against fresh untouched eager
   FP32. Generate the 256×256 reference for the tuning suite with 64 px overlap.
2. Benchmark native FP16 and mixed with CUDA Graph replay. Open the model and
   prepare the shape before timing. In each fresh worker, warm one full
   page. Time the next page from the decoded CPU tensor through input
   preparation, transfers, inference, and CPU stitching. Keep setup, file
   decoding, PNG encoding, profiling and scoring outside the interval.
3. Run four paired rounds with forward/reverse runner order and one GPU worker
   at a time. Save every sample, median, range, run order, and output hash.
   If the portable runner and frozen artifacts are available, pair it with
   the native candidate. At pickup, the sibling checkout has the documented
   result and source harness, and its local matrix summary is absent.
   The 4.8052 s/page target is historical until paired again. When ranges
   overlap, repeat the paired run before claiming a reliable win.
4. Measure VRAM in separate workers with NVML as steady peak above idle.
   Keep native session buffers and warmed free-memory deltas as diagnostics.
   Save RGB float outputs before quantization, raw PSNR at peak 1 and max
   error. Use the sibling scorer's grayscale rendering for SSIMULACRA2.
   If reporting batch-1 tile latency, measure each synchronized call
   separately and save image, index and coordinates.
   [Memory sampler](../../umzi-bench/scripts/umzi_vram.py)
   · [Quality scorer](../../umzi-bench/scripts/umzi_quality.py)

Record the Git revision, DLL and package hashes, input hash, precision,
kernel options, GPU and CUDA/cuDNN/cuFFT versions with the reports. The
expected page SHA-256 is
`528addf2f7cbe2de28fe1ae8f17135779c96ac7b62e5ed4322c697370c6f50da`.

## 3. Test one optimization at a time

| Order | Experiment | How to check |
| --- | --- | --- |
| 1 | Profile FP16 and mixed at batch 1 and 128×128. Separate cuDNN, FFT, custom kernels, transfers and stitching. Record the selected algorithm and workspace by convolution shape. | Run profiling separately, then reproduce the uninstrumented page timing. Use the profile to choose the next experiment. |
| 2 | Test cuDNN plan selection and workspace limits on the dominant convolutions. Start with expansion and projection if the trace confirms they dominate. | Compare one plan policy at a time using representative tiles. Record workspace, raw parity and warmed wall time. Carry the best candidate to the matched page benchmark. |
| 3 | If the trace shows layout conversions, depthwise, or gate preparation dominating, test a single layout or fusion change. The sibling strict NHWC8/native-depthwise result is a lead. | Check flat gray, checkerboard, noise, real crops, odd shapes and graph/direct parity first. Then repeat the matched page and three-page runs against the unchanged control. |

For a candidate that beats the target, retain the measured binary, the package
and the raw reports. Run GateRV2, GateRV3 (both checkpoints), and MoESR
session checks if changing shared dispatch, loading, or build code. After
Python edits run `ruff format scripts src`, `ruff check scripts src` and
`pyrefly check`. Run `git diff --check` before handing off.

## Code and artifacts

| Path | Use |
| --- | --- |
| [Native runner](../native/kernels/gfisrv2/runner.cpp) | Precision policies, cuDNN plans, cuFFT and CUDA Graph replay |
| [CUDA kernels](../native/kernels/gfisrv2/kernels.cu) | Norms, depthwise, Fourier packing, GELU, gates and casts |
| [Package exporter](../src/umzi_native/exporter/_gfisrv2.py) | Deterministic FP32 weights for all three modes |
| [Session validator](../scripts/validate_gfisrv2_session.py) | Stress tests, host/device/CLI checks and 256×256 page reports |
| [Device adapter](../scripts/reference/native_session.py) | Resident native buffers for profiling or tile checks |
| [Prior TRT handoff](../../umzi-bench/docs/handoff-gfisrv2-trt-performance.md) | Earlier arithmetic and layout experiments |

Local builds and reports are ignored by Git. The current Windows DLL is in
`output/gfisrv2-session/precision-final-win/`. Results are in the sibling
directories `precision-final-fp32/`, `precision-final-fp16/`, and
`precision-final-mixed/` under `output/gfisrv2-session/`.
