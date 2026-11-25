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

## Variable edge tiling

Expose `tiling="variable"` as an opt-in on `run_tiled`. Keep the interior grid
and overlap widths. Shrink right and bottom tiles to aligned rectangles.
Evaluate a second layout for each axis when its last tile adds at most
`min(64, tile_size // 8)` pixels beyond the previous tile. Select the valid
layout with the fewest submitted input pixels after bucketing, batch padding,
and model padding. Break ties by model calls and then shape groups. Each
small edge group gets a batch no larger than its useful tile count. Include
the regular grid in the comparison so variable shapes can be skipped when
they would increase submitted work.

Group at most four rectangles by shape. Upload the source page once and keep
extraction, blending, and clipping on the GPU. The primary runner handles the
first group, and up to three least-recently-used edge runners retain prepared
CUDA Graphs for subsequent pages. This reuses plans at the cost of separate
edge runner weights and workspaces. Variable edge padding replicates image
pixels, matching the `nmanga` variable layout convention.

For GateRV3 Original, variable mode preserves the fixed geometry and padding
through nominal size 128 and submits an exact final batch. A page with a full
final batch uses one prepared shape. Larger nominal sizes can use rectangular
edge shapes with the larger-tile attention path.

Validate against a CPU stitch using the chosen geometry, exercise a tiny
sliver that can be absorbed, and compare paired fixed and variable page time
on GateRV3 Original (`samples/input`, 128×128, batch 16) and SMoSR
(`samples/2x-test`, 256×256, batch 1), with and without overlap. Use separate
fidelity references for different layouts because edge context can change
model output.

### Variable tiling results

The fixed and variable CUDA paths were paired on the same RTX 3050 Laptop GPU
and native FP16 build. Page timings sum the three per-image medians of five
alternating, warmed runs. Submitted input pixels include batch padding and the
model's own alignment or reflection border.

| Model | Overlap | Fixed page time | Variable page time | Submitted input pixels saved |
| --- | ---: | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 1.700 s | 1.714 s | 0% |
| GateRV3 Original | 16 px | 2.278 s | 2.227 s | 2.50% |
| SMoSR | 0 px | 1.834 s | 1.610 s | 12.53% |
| SMoSR | 32 px | 2.220 s | 2.106 s | 6.06% |

The GPU and CPU stitches matched exactly in raw values on all 12 page cases
using the selected layouts. At zero overlap, SMoSR absorbed a 20-pixel right
tail on the 1812-pixel page and removed one tile column. A synthetic test also
verified that the planner kept a one-pixel tail separate when merging it
would add model work through batch padding.

GateRV3 used batch 10 for the 16-pixel-overlap tail, saving six repeated tiles
per page. The first overlapping variable page took 4.30 s with preparation.
The next two first passes reused the edge runner and took 736 ms each. Without
overlap every GateRV3 batch was already full.

[GateRV3 variable page results](../output/gpu-tiling/variable-gaterv3.json)
[SMoSR variable page results](../output/gpu-tiling/variable-smosr.json)

| Model | Overlap | Raw PSNR vs. matched Torch FP32 | Mean SSIMULACRA2 |
| --- | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 71.573 dB | 91.985 |
| GateRV3 Original | 16 px | 71.478 dB | 91.982 |
| SMoSR | 0 px | 61.798 dB | 88.711 |
| SMoSR | 32 px | 61.831 dB | 88.717 |

[GateRV3 variable fidelity](../output/gpu-tiling/variable-gaterv3-quality.json)
[SMoSR variable fidelity](../output/gpu-tiling/variable-smosr-quality.json)

The independent model-only run synchronizes each device-resident call.
GateRV3 batch-1 calls use a separate prepared session. SMoSR already runs
one tile per call. The reports retain each image, tile index, origin, and
elapsed time.

| Model | Overlap | Batched model time | Useful tiles/s | Mean batch-1 tile latency |
| --- | ---: | ---: | ---: | ---: |
| GateRV3 Original | 0 px | 1.625 s | 324.9 | 4.361 ms |
| GateRV3 Original | 16 px | 2.161 s | 324.9 | 4.378 ms |
| SMoSR | 0 px | 1.539 s | 150.1 | 6.663 ms |
| SMoSR | 32 px | 1.930 s | 149.3 | 6.700 ms |

[GateRV3 model call timings](../output/gpu-tiling/variable-gaterv3-calls.json)
[SMoSR model call timings](../output/gpu-tiling/variable-smosr-calls.json)

## GateRV3 Original tile and batch sweep

Use the three established `samples/input/` pages, native FP16, and CUDA Graphs.
Keep GateRV3 Original shape sweeps at zero overlap. Screen both tiling modes
on `page-001.jpg`. Try batches 8, 11, 12, 16, 18, 22, 24, 26, 32, 39, 44,
48, 64, and 88 with a 128-pixel tile. Screen 120-, 124-, and 127-pixel tiles
with batches 16, 22, 26, and 32. Confirm the baseline and the two fastest
screen settings for each policy on all three pages over three warmed repetitions. Time
from a decoded HWC array through the returned `uint8` page. Keep first-page
setup and model-only inference timing separate.

The native build used for this sweep specialized its channel-attention kernel
to 64 spatial positions. The runner aligns each side to 16 pixels, so the supported
square input range for this checkpoint is 113–128. Smaller candidates in
that range still execute as padded 128×128 model tiles. At a nominal size of
128, the zero-overlap page grid contains 176 tiles.

### Zero-overlap sweep results

All 26 screened configurations ran successfully in both tiling modes on the
RTX 3050 Laptop GPU. The fastest settings on the screen page for each tile
size were as follows. Screening used one warmed measurement per setting.

| Tile | Tiles per page | Fastest fixed setting | Fastest variable setting |
| ---: | ---: | ---: | ---: |
| 120 | 204 | Batch 26, 649.4 ms | Batch 32, 634.5 ms |
| 124 | 204 | Batch 26, 647.7 ms | Batch 32, 639.2 ms |
| 127 | 192 | Batch 32, 593.6 ms | Batch 32, 596.3 ms |
| 128 | 176 | Batch 22, 547.6 ms | Batch 32, 544.5 ms |

Batch 44 was fastest in the three-page confirmation. It beat batch 16 on
every page in a subsequent paired comparison with five interleaved warmed
runs per configuration. The times sum the three page medians.

| Tiling | 128×128, batch 16 | 128×128, batch 44 | Page time reduction |
| --- | ---: | ---: | ---: |
| Fixed | 1.669 s | 1.636 s | 1.9% |
| Variable | 1.673 s | 1.637 s | 2.2% |

Batch 44 covers each page in four full batches. Batch 16 needs eleven full
batches. Both tiling modes submit exactly 176 tiles at either batch size.

[Screening data](../output/gpu-tiling/gaterv3-zero-overlap-sweep/screen.json)
[Confirmation data](../output/gpu-tiling/gaterv3-zero-overlap-sweep/confirm.json)
[Paired batch 16 and batch 44 results](../output/gpu-tiling/gaterv3-zero-overlap-sweep/paired.json)

The matching Torch CUDA FP32 reference gave 71.573 dB raw PSNR at both batch
sizes. Mean SSIMULACRA2 changed from 91.985 to 91.915 with batch 44. A separate
batch-1 session measured all 528 tiles individually, averaging 4.387 ms.
Device-wide NVML peaks and first-page timings below were measured separately
from the paired runs. The first-page timing starts after opening the session
with the disk plan cache available and includes in-memory preparation.

| Tiling | Batch | First page | Device-wide peak | Model inference, 3 pages | Useful tiles/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 16 | 625 ms | 2704 MiB | 1.661 s | 317.9 |
| Fixed | 44 | 701 ms | 2947 MiB | 1.612 s | 327.6 |
| Variable | 16 | 618 ms | 2703 MiB | 1.669 s | 316.3 |
| Variable | 44 | 706 ms | 2958 MiB | 1.661 s | 317.9 |

[Fidelity, VRAM, and individual tile timings](../output/gpu-tiling/gaterv3-zero-overlap-sweep/validation.json)

### Larger square tile compatibility before dynamic attention

The requested sweep was 256, 512, and 1024 pixels at batches 1, 4, 8, 16, and
32. A zero-overlap batch-1 probe through `run_tiled` on `page-001.jpg`
failed during shape preparation for each tile size in both fixed and variable
tiling. The error was `Channel attention prototype needs 64 latent positions`.

| Tile | Latent grid | Spatial positions |
| ---: | ---: | ---: |
| 256 | 16×16 | 256 |
| 512 | 32×32 | 1,024 |
| 1024 | 64×64 | 4,096 |

The check counts positions per tile regardless of batch count. That blocks
batches 4, 8, 16, and 32 for the same three shapes. No page timing was
produced for this matrix.

[Larger tile compatibility probes](../output/gpu-tiling/gaterv3-zero-overlap-sweep/large-tile-compatibility.json)

## VRAM added by GPU tiling

The comparison holds native FP16 inference and CUDA Graphs constant. It uses
the median NVML peak across three warmed runs per path on the RTX 3050 Laptop
GPU. CPU tiling still runs the model on CUDA. GateRV3 uses the 1404×2000
`page-001.jpg` at zero overlap. The SMoSR cases use the 1733×2600 and
1812×2600 pages in `samples/2x-test/`.

| Model and setting | Fixed GPU peak above CPU tiling | Variable GPU peak above CPU tiling |
| --- | ---: | ---: |
| GateRV3 Original, 128 px, batch 16, zero overlap | +23 MiB | +20 MiB |
| GateRV3 Original, 128 px, batch 44, zero overlap | +26 MiB | +26 MiB |
| SMoSR, 256 px, batch 1, zero overlap | +68–70 MiB | +124–132 MiB |
| SMoSR, 256 px, batch 1, 32 px overlap | +342–358 MiB | +382–408 MiB |

The calculated per-page arena for GateRV3 is identical in both modes at each
batch size, at 19.1 MiB for batch 16 and 24.3 MiB for batch 44. Both batches
fill exactly, so neither needs extra edge runners. The 3 MiB difference in
the batch-16 NVML reading reflects a change in the device-wide baseline.

SMoSR needs about 66–70 MiB for page and tile buffers at zero overlap. Its
overlap buffers add about 275–288 MiB for FP32 RGB sums and per-pixel weights.
Variable tiling retains an additional 24–66 MiB for the edge runners on the
tested layouts. Pinned host staging uses system RAM.

[VRAM samples and calculated page arenas](../output/gpu-tiling/tiler-vram-comparison.json)

## Larger GateRV3 Original tiles

Extend the existing channel-attention operation to arbitrary supported latent
heights and widths. Keep the original kernel for 64 latent positions. For other
shapes, accumulate Q/K dot products and norms in FP32, softmax over each
head's 32 channels, and apply the resulting probabilities to V in parallel
spatial blocks. Use the runner's existing scratch allocation for probabilities.

Preserve the existing variable tiling policy through nominal size 128. Enable
rectangular edge planning for larger nominal tiles. Validate the attention
operation directly and check whole-page output against the original Resselt
channel-attention forward in CUDA FP32. GateRV3 testing uses zero overlap and
`samples/input/`. Exercise 256, 512, and 1024 with batches 1, 4, 8, 16, and 32
where the selected GPU has enough memory.

The CUDA cubin can be checked separately with
`scripts/verify_gaterv3_channel_attention.py <gaterv3-sm_86.cubin>`. It tests
multiple batches, a partial spatial block, zero Q/K vectors, and 256, 1024,
and 4096 latent positions against Torch CUDA FP32. The largest absolute
output difference in the NVIDIA run was 0.000488.

### Larger tile page validation

The screen ran the requested 30 tile, batch, and tiling-mode combinations on
`page-001.jpg` with native FP16, CUDA Graphs, and zero overlap. The RTX 3050
Laptop GPU completed these configurations.

| Tile | Fixed batches | Variable batch limits |
| ---: | --- | --- |
| 256 | 1, 4, 8, 16, 32 | 1, 4, 8, 16, 32 |
| 512 | 1, 4, 8, 16 | 1, 4, 8, 16, 32 |
| 1024 | 1, 4 | 1, 4, 8, 16, 32 |

Fixed 512×512 batch 32 and 1024×1024 batch 8 each needed 4,395,630,592 bytes
for the model buffers and failed the available-memory check. Fixed 1024×1024
at batches 16 and 32 exceeded the runner's tensor indexing limits. Variable
batch limits can submit smaller batches. On the test page, the 1024 grid has
four distinct rectangular shapes, each submitted individually. The fixed
1024 batch-4 first page took about 100 seconds including cold preparation.

[Larger tile screening matrix](../output/gpu-tiling/gaterv3-large-tiles/screen.json)

The fastest screened profiles were run on all three pages with two warmed
calls per page. Timings sum the per-image medians and include the page tiler,
model inference, transfers, and stitched `uint8` output.

| Tile | Fixed batch and page time | Variable batch limit and page time |
| ---: | ---: | ---: |
| 128 control | 44, 1.646 s | 44, 1.656 s |
| 256 | 16, 1.874 s | 4, 1.938 s |
| 512 | 1, 1.924 s | 16, 1.666 s |
| 1024 | 4, 2.509 s | 1, 1.708 s |

[Three-page measurements and sampled VRAM](../output/gpu-tiling/gaterv3-large-tiles/confirm.json)

Raw float output at 128 pixels matched the previous DLL bit for bit on all
three pages in both modes. The new tile sizes were compared against the
unchanged Resselt channel-attention model with matching fixed or rectangular
tiles at batch 1. Torch and native ran in separate GPU passes for the largest
tiles so PyTorch's cached activations would not consume native VRAM.

| Tile | Fixed PSNR / mean SSIMULACRA2 | Variable PSNR / mean SSIMULACRA2 |
| ---: | ---: | ---: |
| 256 | 72.056 dB / 92.021 | 72.044 dB / 92.020 |
| 512 | 72.146 dB / 92.161 | 72.123 dB / 92.169 |
| 1024 | 71.089 dB / 91.933 | 71.597 dB / 91.996 |

A 256×384 mixed-precision tile also ran with CUDA Graphs disabled and
measured 87.949 dB against the matching eager FP32 output. The independent
batch-1 calls averaged 13.740, 50.519, and 199.992 ms for fixed 256, 512,
and 1024. Variable layout means were 12.517, 45.442, and 135.145 ms over
their actual rectangular tiles. For fixed batch 16, model-only totals on
the three pages were 1.795 s at 256 and 2.385 s at 512, equivalent to 80.23
and 15.10 useful tiles/s. The reports record the individual latencies,
tile indices, coordinates, and per-image fidelity.

[Matched FP32 fidelity and 128-pixel regression](../output/gpu-tiling/gaterv3-large-tiles/fidelity.json)
[Mixed-precision direct check](../output/gpu-tiling/gaterv3-large-tiles/mixed-direct.json)
[Independent batch and per-tile timings](../output/gpu-tiling/gaterv3-large-tiles/model-calls.json)
