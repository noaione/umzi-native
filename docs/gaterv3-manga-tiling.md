# GateRV3 manga page tiling benchmark

Run the sibling `nao-manga-rls` checkout's actual PIL tiler on the sample
pages with 128-pixel tiles. Compare Windows native GateRV3 and ONNX Runtime
TensorRT-RTX with TTA and contrast stretching off. Use batch 64 and zero
overlap for the application's defaults, batch 16 for the project baseline,
and batch 1 for actual individual tile latency. Also test batch 16 with
16 pixels of overlap to exercise partial final batches.

Record real and submitted tile counts, synchronized host inference time,
page time, non-inference time, and native setup when the batch shape changes.
Compare native exact tail batches with repeat-padded tails. Use the
unchanged Torch CUDA FP32 checkpoint for raw-float PSNR and compare rendered
pages using SSIMULACRA2. Record actual per-tile batch-1 measurements and the
active TRT-RTX provider.

## Reproduce

The script imports `denoise_single_image_with_overlap` directly from
`../nao-manga-rls` (revision `80cdcba` for this run). Pass
`--manga-repo` to use another checkout. It reuses the fixed-shape ONNX
exports under `output/baseline/` and exports any missing batch shape from
the same checkpoint. The native DLL and packed model are the inputs from
the Windows GateRV3 build and export.

```powershell
$py = 'C:\Python314\python.exe'
$dll = 'output/native/umzi_native.dll'
$model = 'output/native/gaterv3.ummodel'
$root = 'output/manga-tiling'
& $py scripts/benchmark_manga.py --library $dll --model $model --batch 64 16 1 --overlap 0 --quality `
    --native-cache "$root/native-cache" --rtx-cache "$root/rtx-cache" --output "$root/default"
& $py scripts/benchmark_manga.py --library $dll --model $model --batch 16 --overlap 16 --quality `
    --native-cache "$root/native-cache" --rtx-cache "$root/rtx-cache" --output "$root/overlap"
```

Page time runs from an already decoded PIL image to the final PIL image.
It includes tile extraction, normalization, synchronous host inference,
tail handling, stitching, blending and conversion to 8-bit pixels.
It excludes initialization, warmup, file decode and PNG encoding.
Each page runs twice per backend. The report sums the per-page medians and
keeps each synchronized call with its submitted and useful batch sizes.
Page and inference ms per useful tile express throughput. Individual tile
latency comes from the separate batch-1 sessions.

## Batch 16 with no overlap

Windows RTX 3050 Laptop, driver 610.62. Each 1404×2000 page has 176 useful
tiles, covered by eleven full batches. Native FP16 and mixed use the
default session flags with CUDA Graphs enabled. TRT-RTX uses fixed FP16
ONNX exports with CUDA Graphs enabled and CPU fallback disabled.

| Backend | Inference, 3 pages (s) | Page pipeline, 3 pages (s) | Inference (ms/useful tile) | Page pipeline (ms/useful tile) | Actual batch-1 latency (ms/tile) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native FP16 | **1.831** | **2.182** | **3.468** | **4.133** | **4.810** |
| Native mixed | 2.125 | 2.456 | 4.025 | 4.651 | 5.185 |
| TRT-RTX FP16 | 2.333 | 2.705 | 4.419 | 5.123 | 6.021 |

Each batch-1 average uses 1,056 individually timed tiles from a separate
session. The per-tile trace includes the page name, tile index and
coordinates.

| Page | Native FP16 inference / pipeline (s) | Native mixed inference / pipeline (s) | TRT-RTX inference / pipeline (s) |
| --- | ---: | ---: | ---: |
| page-001 | 0.612 / 0.727 | 0.717 / 0.832 | 0.775 / 0.891 |
| page-005 | 0.610 / 0.726 | 0.709 / 0.818 | 0.779 / 0.898 |
| page-009 | 0.609 / 0.729 | 0.699 / 0.806 | 0.780 / 0.917 |

FP32-reference fidelity is measured on the stitched floating-point canvas
before clipping and quantization. SSIMULACRA2 uses the rendered PNGs.

| Backend | Aggregate PSNR (dB) | Mean SSIMULACRA2 |
| --- | ---: | ---: |
| Native FP16 | 72.071 | 91.850 |
| Native mixed | **85.563** | **92.362** |
| TRT-RTX FP16 | 72.107 | 91.898 |

[Timing, individual tiles and per-page fidelity](../output/manga-tiling/default/report.json)

## Batch 64 with no overlap

This is the application default. Each 176-tile page produces batches
of 64, 64 and 48 real tiles. The fixed-batch sessions submit 192 tiles
per page and discard 16 repeats. The exact-tail sessions submit 176.
The run below prewarmed both native batch shapes and ended warmup on
the full batch before timing pages.

| Backend | Inference, 3 pages (s) | Page pipeline, 3 pages (s) | Mean page pipeline (s/page) | Page pipeline (ms/useful tile) |
| --- | ---: | ---: | ---: | ---: |
| Native FP16, exact tail | 2.802 | 3.099 | 1.033 | 5.869 |
| Native FP16, padded tail | **1.902** | **2.241** | **0.747** | **4.245** |
| Native mixed, exact tail | 3.331 | 3.643 | 1.214 | 6.900 |
| TRT-RTX FP16, padded tail | 2.497 | 2.810 | 0.937 | 5.322 |

Native FP16 spent 1.052 s on shape setup across the three pages even
with disk cache hits. The median change cost 166 ms going to batch 48
and 220 ms returning to batch 64. Repeat padding reduced full page time
by 28% in this run. During the first pass, an uncached FP16 batch-48 tail
spent 17.2 s in shape setup on the first page. Batch 16 was slightly
faster than padded batch 64 on these pages because 176 divides evenly
by 16.
[Warmed batch-64 run](../output/manga-tiling/batch64-warm/report.json)

## Batch 16 with 16-pixel overlap

The step drops to 112 pixels and each page has 234 useful tiles. A dynamic
session receives fourteen full batches and a 10-tile tail. Fixed native and
TRT-RTX repeat the last tile six times per page and discard the duplicates.

| Backend | Inference, 3 pages (s) | Page pipeline, 3 pages (s) | Page pipeline (ms/useful tile) |
| --- | ---: | ---: | ---: |
| Native FP16, exact tail | 2.701 | 3.379 | 4.813 |
| Native FP16, padded tail | **2.469** | **3.172** | **4.518** |
| Native mixed, exact tail | 3.122 | 3.855 | 5.491 |
| TRT-RTX FP16, padded tail | 3.181 | 3.837 | 5.465 |

After prewarming both native shapes, exact-tail FP16 still spent 287 ms
across the three pages on shape setup. A shape change consumes a warmup
forward pass inside the native runtime and rebuilds the captured CUDA graph.
The median FP16 shape transition spent 37.2 ms on workspace allocation and
warmup, 3.6 ms on graph capture, 2.6 ms on plans, and 1.0 ms on shape
allocation. Repeat padding keeps those setup stages outside page processing.
[Overlap timing, shape switches and fidelity](../output/manga-tiling/overlap/report.json)

## Host transfer and packing check

The no-overlap path often gives the session a strided NumPy batch with
the RGB channels adjacent in memory. The overlap path supplied contiguous
batches on these inputs. In a warmed TRT-RTX probe using a real batch
from page-001, making the batch contiguous took 0.56 ms. Median TRT-RTX
time was 70.56 ms with the original host layout, 69.92 ms with contiguous
host arrays, and 69.23 ms with prebound CUDA input and output buffers.
The bound output matched the host output bit for bit and the profile
recorded only the TensorRT-RTX node provider. Packing and host transfers
accounted for a small share of the batch execution time on this GPU.
[Batch-16 transfer probe](../output/manga-tiling/batch16-transfer-probe.json)
