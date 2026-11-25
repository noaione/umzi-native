# Fixed GPU tiling for Umzi Native

Add a page-level CUDA path that extracts fixed tiles, runs the existing model
session, and stitches the output on the GPU. A decoded page crosses to the GPU
once and the finished page crosses back once. Use the current `nmanga` fixed
tiler as the compatibility reference and validate seam quality on real pages.

## Runner and API

- Add `InferenceSession.run_tiled(image, *, tile_size, batch_size, overlap=0,
  background="white", output_dtype=np.uint8)`. Accept a contiguous HWC `uint8`
  page with the model's channel count and return HWC at the model's scale.
  `np.float32` returns the stitched values before clipping and quantization for
  fidelity checks. Keep the Python call synchronous so callers can hand the
  result directly to Pillow. The wheel continues to work with NumPy and its
  existing CUDA dependencies.
- Use an additive `umzi_session_run_tiled` entry point with a size-tagged options
  struct, explicit image dimensions and buffer sizes. Bind the new symbol
  optionally so the existing NumPy `run` API still loads older compatible DLLs.
  The tiled call should report a missing symbol clearly. Validate dimensions,
  overlap, model I/O shape and dtype, buffer sizes, and scaled size arithmetic
  before GPU work. Obtain concrete batch shapes, dtypes and bytes from the
  existing session `describe` API and read scale and context from model metadata.
- Build the tiling kernels once as shared runtime CUDA code through
  [`hatch_build.py`](../hatch_build.py), using the existing cubin and PTX loader.
  Use `KernelSession::enqueue_device`, the implementation behind
  [`umzi_session_enqueue`](../native/include/umzi_native.h), for model work.
  Keep tiling outside the model's CUDA Graph so changing page dimensions
  does not require recapturing a fixed model tile shape.
- Reuse one input batch and one output batch throughout the page. Prepare one
  `[batch_size, C, tile_size, tile_size]` shape. Use the session's selected CUDA
  device and primary context. Order upload, extraction, inference, blending,
  and download on one stream. Synchronize at the page boundary and before
  releasing in-flight buffers on error. The Python session lock covers the
  full page call. Reuse the runtime's out-of-memory status for page allocation
  failures.

```python
restored = session.run_tiled(rgb_page, tile_size=128, batch_size=16, overlap=16)
```

## Grid and blending

Match `nao-manga-rls/nmanga/ml/tiler.py` and the fixed PIL path in
`nao-manga-rls/nmanga/ml/onnx_backend.py` at revision `fbfa061`.

1. Work in source-image pixels. For input tile size `T`, model context `C` per
   side, and total shared overlap `O`, the valid core is `K = T - 2*C` and the
   stride is `D = K - O`. Require `0 <= O < K`. For each axis of length `L`,
   use `max(1, ceil((L - O) / D))` tiles. Tile `i` begins at `i*D` in the
   destination and `i*D - C` in the input. Clip the valid output to the page.
   Match the existing context-trimmed 1× contract. Multiply destination
   positions, valid lengths and overlap by the model scale for stitching.
2. Launch extraction from a compact grid description. The GPU derives tile
   positions from row-major indices and packs HWC bytes into normalized NCHW
   batches in the model's FP16 or FP32 input dtype. Match edge replication when
   overlap or context is active, and use the requested white or black fill
   otherwise. Repeat the last real tile to fill a short batch and exclude those
   repeated outputs from stitching.
3. With overlap, use the current separable linear feather. Taper only toward
   another tile on that axis and derive ramp lengths from the cropped output
   extent. Use the same `ramp_length + 1` denominator and multiply the two
   axis weights at corners. Gather contributors to each touched output pixel
   from the current batch in row-major order. Give each pixel one writer and
   accumulate weighted color and weight sums in FP32 across batches. Normalize
   once after the last batch. Normalization also covers overlap larger than
   half the core, where more than two tiles can contribute along one axis.
   Every in-page pixel should have positive total weight.
4. With zero overlap, write each valid output pixel directly. Clip and
   round the normalized result once for `uint8`, matching Pillow pipeline
   quantization in `nmanga`. Check for non-finite model output in validation.

Blending must preserve flat brightness and avoid dark borders or corner
crosses. A model can still predict different details at opposite edges of the
same overlap. If seams remain visible with the linear feather, compare a
raised-cosine ramp or a larger overlap on the affected model. Keep any change
that improves seam crops without degrading detail.

Keep the first implementation to one page accumulator and one pair of batch
buffers. RGB overlap accumulation costs 16 bytes per output pixel, including
the weight sum. The largest SMoSR input produces a 3624×5200 output with about
288 MiB of accumulation before the model, page, and batch buffers. If measured
footprints become a problem, finalize completed output rows as the grid advances.

## Future `nmanga` integration

- A later integration can add an opt-in `--gpu-tiling` switch to
  `nmanga umzi` in `nmanga/cli/umzi.py`. The fixed `.ummodel` path already
  opens one session through `nmanga/ml/umzi.py`. Have the page wrapper call
  `run_tiled` once per decoded page with the existing tile, batch, overlap,
  and background settings. Reuse Pillow mode conversion, alpha resizing,
  palette restoration, and the existing async image writer.
- Keep the existing CPU tiler as the default and as the route for variable
  shapes, TTA and contrast stretching. Report an actionable error if GPU
  tiling is explicitly selected with those options or with an older runtime.
  The first integration targets native `.ummodel` sessions. ONNX Runtime
  device I/O binding can be evaluated once the CUDA tiler has passed parity
  and page-level benchmarks.

## Validation and implementation order

1. Implement the shared CUDA kernels and check them against the sibling CPU
   tiler using identical synthetic tile outputs. Use identity and nearest
   upscale through the grid, plus tile-specific output offsets to expose
   incorrect blend weights. Cover gradients, patterns, tiny and odd pages,
   right and bottom slivers, batches crossing tile rows, partial final batches,
   1× context trim, and 2× output. Exercise zero overlap, ordinary overlap,
   corner intersections, and overlap above half the core. Assert full
   coverage, preserved constants, and FP32 agreement within rounding tolerance.
   Verify the no-overlap byte path against the CPU conversion on the same
   model outputs.
2. Wire the page entry point into the native and Python sessions. Test
   successive page sizes on one prepared shape, graph enabled and disabled,
   and invalid input and allocation failure handling. Extend
   the manga tiling benchmark with paired CPU and GPU tiling on the same model
   build and precision.
3. Use GateRV3 Original on `samples/input/`, 128×128 tiles, batch 16, and 0 and
   16 source pixels of overlap from the existing
   [manga tiling benchmark](gaterv3-manga-tiling.md). Use SMoSR on
   `samples/2x-test/`, 256×256 tiles, batch 1, and 0 and 32 source pixels of
   overlap. Compare each result against
   the matching CPU tiler with the same model and native precision to isolate
   tiling. Compare raw-float PSNR (peak 1) and rendered SSIMULACRA2 against
   the untouched Torch CUDA FP32 model on the same grid, per image and overall.
   Inspect seam-centered crops and error maps of flat tones, screentones, text,
   and diagonal ink across horizontal and vertical joins, corners and page
   borders. Use a whole-crop reference where the model and VRAM permit it to
   check whether residual seams come from tile context. Require quality at
   least on par with the CPU blend before enabling the integration.
4. On NVIDIA CUDA, compare warmed decoded-page-to-result time, including
   uploads and downloads, and log GPU extraction, inference, blending and
   finalization times plus peak VRAM. Record per-image and aggregate batch
   inference time and useful-tile throughput, including submitted tile counts.
   Measure synchronized batch-1 latency in a separate run and keep every sample
   with its image, tile index and coordinates, plus the mean. Record the device,
   workloads and results in `docs/`. The paired benchmark determines whether
   GPU tiling improves page time and how much it costs in memory.

## Reproduce the local checks

Run these commands from the repository root. The CPU reference uses the
sibling `nao-manga-rls` checkout for comparison. The runtime itself only
needs the Umzi wheel and its usual CUDA libraries.

```powershell
$env:PYTHONPATH = 'src'
$tilingPython = 'C:/Python314/python.exe'
$tilingOutput = 'output/gpu-tiling'
$tilingLibrary = "$tilingOutput/native/umzi_native.dll"
& $tilingPython scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output "$tilingOutput/native"
& $tilingPython -m umzi_native export models/1x_umzi_digital_decompress_gaterv3_1.pth -o "$tilingOutput/gaterv3-original.ummodel" --kernel gaterv3
& $tilingPython -m umzi_native export models/2x_enhancr_da_smosr_v1.safetensors -o "$tilingOutput/smosr.ummodel" --kernel smosr
& $tilingPython scripts/benchmark_gpu_tiling.py gaterv3 --library $tilingLibrary --model "$tilingOutput/gaterv3-original.ummodel" --output "$tilingOutput/gaterv3"
& $tilingPython scripts/benchmark_gpu_tiling.py smosr --library $tilingLibrary --model "$tilingOutput/smosr.ummodel" --output "$tilingOutput/smosr"
```

The benchmark uses FP16 native sessions with CUDA Graphs enabled and writes
raw-float metrics, per-image timings, individual batch-1 tile latencies, and
rendered pages under the output directory. In the Windows sandbox the
exported packages had private temporary-file permissions. The measured run
used byte-identical copies named `gaterv3-original-benchmark.ummodel` and
`smosr-benchmark.ummodel`.

## Measured results

Windows NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), driver 610.62. Page time
includes tiling, model calls, transfers, stitching, and Pillow output. The GPU
path stages the decoded page's NumPy view before timing. PNG encoding, the
quality checks, session startup, and shape tuning are outside the page timing.
Each reported page time is the median of two alternating CPU and GPU runs.
The table sums these medians over all three input pages. Device-wide VRAM peaks
are sampled by NVML and include other GPU activity.

| Model | Overlap | CPU page time | GPU page time | GPU path VRAM peak | Raw PSNR vs Torch FP32 | Mean SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 1.945 s | 1.751 s | 2664 MiB | 71.573 dB | 91.985 |
| GateRV3 Original | 16 px | 2.796 s | 2.382 s | 2707 MiB | 71.479 dB | 91.984 |
| SMoSR | 0 px | 2.821 s | 2.034 s | 1846 MiB | 61.801 dB | 88.724 |
| SMoSR | 32 px | 4.758 s | 2.486 s | 2258 MiB | 61.830 dB | 88.717 |

| Model | Overlap | Batch | Model inference | Useful tiles/s | Mean synchronized batch-1 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 16 | 1.695 s | 311.5 | 4.612 ms |
| GateRV3 Original | 16 px | 16 | 2.315 s | 303.3 | 4.572 ms |
| SMoSR | 0 px | 1 | 1.828 s | 132.4 | 7.556 ms |
| SMoSR | 32 px | 1 | 2.199 s | 131.0 | 7.635 ms |

Model inference is timed separately with device-resident tile buffers.
GateRV3 has a separate batch-1 session. For SMoSR the throughput batch is one,
so the individually synchronized calls also provide the batch-1 latency
samples. The PSNR and SSIMULACRA2 columns use the same fixed tile grid and
padding for the eager CUDA FP32 reference.

Both models matched the fixed CPU tiler exactly before quantization and in the
rendered bytes on all three pages at both overlaps. With overlap, the GPU path
reduced page time by 15% for GateRV3 and 48% for SMoSR. The sampled peak on each
overlapping page was about 70 MiB higher for GateRV3 and 309–345 MiB higher for
SMoSR. Separate extraction and blend kernel timings were outside this run.

Both models also passed the synthetic edge, large overlap, and partial-batch
checks with CUDA Graphs disabled. An additional SMoSR mixed-precision case
matched the CPU tiler within the smoke-test tolerance using FP32 tensor I/O.
The previous GateRV3 DLL still ran NumPy tile inference and returned the
expected feature error when called through `run_tiled`. Both tested models
report zero external tile context.

The reports contain the per-image timings, fidelity, and individual tile
samples. Rendered pages and overlap crops are saved alongside them.
[GateRV3 report](../output/gpu-tiling/gaterv3/report.json)
[SMoSR report](../output/gpu-tiling/smosr/report.json)

## CUDA tiler optimization

Extraction assigns one GPU grid slice to each tile and packs all channels
per pixel. Zero overlap uses a direct tile-to-page copy. Each page uses one
aligned GPU allocation, released when the call returns. The session reuses
one pinned host buffer sized for its largest `uint8` page. Raw FP32 readback
uses temporary pinned storage.

The paired DLL comparison used the same FP16 sessions, images, tile grids,
and CUDA Graph settings as above. Page times are the sum of the three
per-page medians over seven alternating runs. GPU tiling time sums the
extract, stitch, and finalization kernels in one separate trace per build
and setting, on the largest page. Transfers and model kernels are outside
that device-kernel subtotal.

| Model | Overlap | GPU tiling kernels, before → after | Full-page time, before → after |
| --- | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 1.584 → 0.623 ms | 1.727 → 1.727 s |
| GateRV3 Original | 16 px | 2.471 → 1.981 ms | 2.421 → 2.417 s |
| SMoSR | 0 px | 5.304 → 2.923 ms | 2.049 → 2.014 s |
| SMoSR | 32 px | 10.406 → 9.492 ms | 2.322 → 2.318 s |

The zero-overlap kernels took 61% less GPU time for GateRV3 and 45% less
for SMoSR in these traces. Model kernels alone took about 570–655 ms per
zero-overlap page, leaving comparatively little page time to recover in
the tiler. The optimized DLL matched the original DLL bit for bit in raw
float output and rendered bytes on all 12 page cases.
[Paired page timings](../output/gpu-tiling/ab-tiler.json)
[GPU profile summary](../output/gpu-tiling/ab-tiler-profile-summary.json)
