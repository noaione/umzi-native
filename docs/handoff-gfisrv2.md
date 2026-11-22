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

### Archived 256×256 results (32 px overlap)

The table below uses batch 2 and **32 px overlap** (archived prior to the 64 px
tuning baseline and matched 128×128 full-page baseline).

| Native precision | Batch-2 time, all 3 pages (s) | Input MPx/s | Mean batch-1 tile (ms) | Warmed batch-2 VRAM delta (MiB) | Raw PSNR (dB) | Mean SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 14.364 | 0.2538 | 202.595 | 362 | ∞ | 100.000 |
| FP16 | 10.785 | 0.3380 | 148.899 | 270 | 60.166 | 86.385 |
| Mixed | 10.950 | 0.3329 | 151.537 | 342 | 63.667 | 86.633 |

### Optimization baselines (2026-11-22, pickup build `658fcc9`)

Rebuilt cleanly into `output/gfisrv2-session/pickup-win/` (`umzi_native.dll`
SHA-256 `b7b4c132499a9d50415af4eb4cd65736bfb7f49c8166f4bea3d8d82d6cbd4fb4`,
package SHA-256 `36e829d2f1e4e334a48bc4295ee14a26283b929d46c841c9446761fc5a74bae1`).
All three smoke validations (`fp32`, `fp16`, `mixed`) passed before timing.

#### Matched 128×128 full-page baseline (`lb-page-003.png`, batch 1, 32 px overlap, 150 tiles)

Four paired forward/reverse rounds in fresh isolated workers with CUDA Graph
replay, NVML 5 ms VRAM sampling in separate workers, and 150 individually timed
synchronized batch-1 tiles. Fresh untouched eager CUDA FP32 reference at the
same tiling (`output/gfisrv2-session/matched-128-ref/`). Reports:
`output/gfisrv2-session/baseline-matched-128/summary.json`.

| Runner | Median s/page | Range (s/page) | Input MPx/s | Mean batch-1 dev / host (ms/tile) | NVML steady peak above idle (MiB) | Buffers / warmed delta (MiB) | Raw PSNR vs FP32 (dB) | Gray / RGB16 SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical ORT TRT 10 FP16 (target) | **4.8052** | 4.7665–4.9099 | 0.2529 | — | 292.49 | — | 60.4422 | 87.2950 / — |
| Native FP16 (baseline) | 6.3540 | 5.6458–6.4009 | 0.1912 | 40.655 / 43.239 | **128.49** | 39.7 / 66.0 | 64.4828 | 87.3532 / 87.1839 |
| Native Mixed (baseline) | 6.7572 | 5.9912–6.7869 | 0.1798 | 36.953 / 38.030 | 148.49 | 62.3 / 86.0 | 67.4726 | 87.4351 / 87.3302 |
| Native FP32 (baseline) | 8.6323 | 8.6229–9.1936 | 0.1408 | 48.172 / 49.346 | 144.49 | 59.2 / 82.0 | ∞ | 100.000 / 100.000 |

All three native modes matched CUDA Graph vs direct bit-for-bit on the full
page. Native FP16 uses 2.28× less steady VRAM than the historical TRT 10 runner
(128.49 vs 292.49 MiB) and +4.04 dB higher PSNR, but trails by 32.2% in median
page time (6.3540 vs 4.8052 s/page).

#### 256×256 batch-2 native tuning baseline (64 px overlap, all 3 pages, 105 real tiles)

Fresh eager FP32 reference generated with `--overlap 64` at
`output/architecture-baselines-ov64/gfisrv2/batch-2/fp32/report.json`. Reports:
`output/gfisrv2-session/baseline-ov64-{fp32,fp16,mixed}/report.json`.

| Native precision | Batch-2 time, all 3 pages (s) | Input MPx/s | Mean batch-1 tile (ms) | Batch-1 / Batch-2 buffers (MiB) | Batch-1 / Batch-2 warmed delta (MiB) | Raw PSNR (dB) | Mean / Min SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 22.734 | 0.1603 | 212.987 | 184.7 / 352.1 | 186 / 362 | ∞ | 100.000 / 100.000 |
| FP16 | 16.388 | 0.2224 | 153.243 | 133.8 / 259.2 | 138 / 270 | 60.716 | 86.425 / 85.534 |
| Mixed | 17.221 | 0.2117 | 161.076 | 175.8 / 327.2 | 182 / 342 | 63.993 | 86.581 / 85.658 |

#### Baseline kernel profile (`scripts/profile_gfisrv2.py`)

Measured separately with `enable_cuda_graph=False` via CUPTI (`output/gfisrv2-session/baseline-profile-{128,256}.json`):

- **128×128 batch-1 FP16 (50.84 ms GPU kernels; host page 6.99 s = 6.53 s runner + 0.45 s CPU stitch/tile):**
  - cuDNN/GEMM: **29.05 ms (57.2%)** — `expand 48→256` (24×, FP32 acc) 12.80 ms, `project 128→48` (25×, FP32 acc) 7.35 ms, NCHW↔NHWC layout reformat kernels (153×) **6.18 ms**, Fourier 60-ch grouped 3×3 1.91 ms, Fourier 60→60 1×1 0.49 ms.
  - Custom kernels: **17.53 ms (34.5%)** — `add_bias_half` (100×) **7.92 ms**, `gate_half` (24×) 3.59 ms, `rms_norm_half` (72×) 2.12 ms, `residual_half` (24×) 1.63 ms, `depthwise_half` (72×) 1.24 ms, `copy_channels_to_float` 0.49 ms, `add_half_residual` 0.42 ms.
  - cuFFT + pack/unpack/scale: **4.23 ms (8.3%)** — actual FFT kernels ~2.97 ms.
- **128×128 batch-1 Mixed (45.41 ms GPU kernels; host page 6.02 s = 5.71 s runner + 0.31 s CPU):**
  - Same bottleneck hierarchy: `expand` 10.23 ms, `add_bias_half` 5.98 ms, `project` 5.79 ms, NCHW↔NHWC reformat 5.14 ms, `gate_mixed` 2.77 ms, `rms_norm*` 2.99 ms, `residual_mixed` 1.83 ms, Fourier FP32 1×1 1.32 ms, Fourier depthwise 1.23 ms, cuFFT ~3.53 ms.
- **256×256 batch-2 FP16 (284.23 ms GPU kernels):**
  - `expand` 68.23 ms (24.0%), `add_bias_half` 41.44 ms (14.6%), `project` 40.56 ms (14.3%), NCHW↔NHWC reformat **36.23 ms (12.7%)**, `gate_half` 20.11 ms (7.1%), `rms_norm_half` 11.08 ms (3.9%), Fourier grouped 3×3 8.94 ms, `residual_half` 8.86 ms, `depthwise_half` 6.92 ms, cuFFT ~16 ms.

### Optimization pass 1 (2026-11-22)

Build: `output/gfisrv2-session/p1-win` (`--arch sm_86` iteration build).
Switched reduced-precision (`fp16` and `mixed`) to NHWC with canonical block
channel permutation at session open, cuDNN backend timed plans
(`native/kernels/gfisrv2/conv_plan.h`, `CudaPlanCache`, `workspace_mib`, FP16
accumulation in FP16 mode), fused bias/RMSNorm/residual/gate/pack/unpack
kernels, and shared-memory tiled spatial and spectral depthwise kernels. Kept
`precision="fp32"` on its untouched bit-exact NCHW path.

- Same-process alternating A/B probe (`scripts/ab_moesr_builds.py`, random tiles):
  - **128×128 batch 1**: FP16 `42.3 → 22.5 ms/tile` (**1.88×**), Mixed `42.9 → 28.0 ms/tile` (**1.53×**).
  - **256×256 batch 2**: FP16 `273.4 → 150.2 ms` (`75.1 ms/tile`, **1.82×**), Mixed `323.1 → 204.1 ms` (`102.1 ms/tile`, **1.58×**).
- Matched 128×128 full-page (`output/gfisrv2-session/p1-matched-128/summary.json`, 4 rounds):
  - **Pass 1 FP16**: **3.1374 s/page median** (range `3.1302–3.1850 s`, **0.3873 MPx/s**, **19.242 ms/tile** dev / `19.042 ms/tile` host, **116.49 MiB** NVML steady peak VRAM above idle, **59.3372 dB** raw PSNR, **87.1869** gray / `86.9820` RGB16 SSIMULACRA2, graph == direct bit-exact). Already **2.03× faster** than baseline and **1.53× faster** than the 4.8052 s/page TRT 10 target at 2.51× lower VRAM.
- Note caught in Pass 1 smoke check: cuDNN backend selected a TF32 engine (`s1688gemm`) for Mixed's FP32 1×1 spectral projection when Python imported Torch before setting `NVIDIA_TF32_OVERRIDE=0`, causing a Python-vs-CLI mismatch on `mixed`. Pass 2 replaces the spectral 1×1 conv with custom deterministic kernels in both modes.
- Pass 1 CUPTI profile (`output/gfisrv2-session/p1-profile-128.json`, 20.04 ms FP16 / 27.44 ms Mixed):
  - Remaining FP16 bottlenecks: `expand 48→256` 5.43 ms + `gate_plain` 1.13 ms, `project 128→48` 2.88 ms + `residual_rms` 2.82 ms, stride-256 `branch_dw_gate` 1.65 ms, unvectorized `spectral_dw_res` 1.15 ms, cuFFT ~2.21 ms.
  - Remaining Mixed bottlenecks: `expand` (`indexed_wo_smem`) 9.34 ms, `project` 4.30 ms + `residual_rms` 2.86 ms, `spectral_dw_res` 2.05 ms, `gate_plain` 1.68 ms, `branch_dw_gate` 1.41 ms.

### Optimization pass 2 (2026-11-22)

Build: `output/gfisrv2-session/p2-win` (`--arch sm_86` iteration build).
Added custom tensor-core `expand_gate_f16` / `expand_gate_f32` (`48→256` 3×3
fused with 80-ch plain SiLU gate, direct FP32 planar Fourier extraction, and
compact branch/Fourier gate buffers aliased inside the 96-ch `projected` buffer,
eliminating the 256-ch `projected` tensor), `project_res_norm_*` (`128→48` 3×3
fused with bias, SiLU, gamma residual, and quad-shuffle next-block RMSNorm),
compact `branch_dw_gate_compact` & `scale_post_rms_gate_compact`, shared-memory
weight caching in `spectral_dw_res_nhwc_*`, and custom fused
`spectral_proj_gelu_unpack_f16` (FP32-acc tensor-core 64→64 1×1 + bias + GELU +
unpack) and `spectral_proj_gelu_unpack_f32` (pure deterministic FP32 FMA,
fixing the Pass 1 Mixed TF32/CLI mismatch). Only the four bookend convolutions
(`first`, `tail_expand`, `tail_project`, `last`) remain on cuDNN, controlled via
`fused_conv=1` (default).

- All three smoke validations (`fp32`, `fp16`, `mixed`) passed all 18 stress
  cases plus graph/direct, host/device/alternate-stream, and CLI bitwise parity.
- Same-process alternating A/B probe (`base` → `p1` → `p2`):
  - **128×128 batch 1**: FP16 `38.7 → 21.0 → 14.1 ms/tile` (**2.74× vs base**), Mixed `43.0 → 27.7 → 24.0 ms/tile` (**1.79× vs base**).
  - **256×256 batch 2**: FP16 `273.8 → 143.3 → 94.9 ms` (`47.5 ms/tile`, **2.89× vs base**), Mixed `315.3 → 214.2 → 170.5 ms` (`85.2 ms/tile`, **1.85× vs base**).
- Profiled single-page host timing (`output/gfisrv2-session/p2-profile-128.json`):
  - **FP16**: **2.6048 s/page** (`2.261 s` runner + `0.344 s` CPU stitch/tile).
  - **Mixed**: **3.9576 s/page** (`3.593 s` runner + `0.365 s` CPU stitch/tile).
- Remaining Pass 2 bottlenecks (`p2-profile-{128,256}.json`): `expand_gate_f16`
  at `8×16` pixel tile (44.3 ms at 256×256 b2 due to 2× higher weight traffic),
  `spectral_proj_gelu_unpack_f32` at `1×16` micro-tile (27.2 ms at 256×256 b2 /
  3.77 ms at 128×128 b1), and repeated `half→float` conversion in
  `spectral_dw_res_nhwc_f16` (5.95 ms at 256×256 b2). Addressed in Pass 3.

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
