# umzi-native

Native CUDA inference runtime for [Umzi](https://github.com/umzi2) image
restoration and upscaling models. It packs supported checkpoints into a
single portable `.ummodel` file and runs them on NVIDIA GPUs through one
Python API (`InferenceSession`) with tuned cuBLASLt/cuDNN plans, custom CUDA
kernels, and CUDA Graph replay.

## Supported models

| Family | Precision modes | Notes |
| --- | --- | --- |
| **GateRV3** | `mixed`, `fp16` | Original and IDC variants; opt-in kernel fusions |
| **GateRV2** | `mixed`, `fp16` | Tuned cuBLASLt/cuDNN plans with optional fast cold start |
| **MoESR** | `mixed`, `fp16` | Custom fused tensor-core gate and depthwise kernels |
| **GFISRv2** | `fp32`, `mixed`, `fp16` | Also requires cuFFT 12 at runtime |

See [Supported checkpoints and export options](docs/export.md) for the exact
checkpoint topologies accepted by the exporter, and the
[final timing & fidelity summary](docs/umzi-native-architecture-expansion.md#final-timing-and-fidelity-summary-rtx-3050-laptop)
for RTX 3050 Laptop benchmarks across GateRV2, MoESR, and GFISRv2.

## Requirements

- **Python:** 3.12 or newer
- **GPU runtime:** CUDA 13-compatible NVIDIA driver, cuBLASLt 13, and cuDNN 9
  (plus cuFFT 12 for GFISRv2)

The easiest way to make the CUDA runtime libraries available is to `import torch`
from a CUDA-enabled PyTorch install before opening a session. You can also load
standalone NVIDIA library directories without PyTorch via
`SessionOptions(dll_dirs=...)` — see
[CUDA library loading](docs/gaterv3-python.md#library-loading).

## Install

### Prebuilt wheels (Windows & Linux x64)

Install a tagged release from the private package index (Windows x64 and Linux
`manylinux_2_34_x86_64`, glibc 2.34+ / OpenSSL 3):

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ umzi-native
```

If you also want to convert `.safetensors` / `.pth` checkpoints into `.ummodel`
files, include the `exporter` extra (installs Resselt and PyTorch for CPU-side
export; not needed if you only run existing `.ummodel` files):

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ "umzi-native[exporter]"
```

### Development builds and building from source

Install a wheel downloaded from [GitHub Actions](.github/workflows/build.yml)
with `python -m pip install PATH_TO_WHEEL`, or build from a checkout (requires
MSVC on Windows or a C++ compiler on Linux):

```powershell
python -m pip install .
# Or with the checkpoint exporter included:
python -m pip install ".[exporter]"
```

See [source build instructions](docs/gaterv3-python.md#install) for details.

## Quick start

### 1. Export or inspect a `.ummodel`

Skip the export step if you already have a `.ummodel` file.

```powershell
# Convert a supported checkpoint to .ummodel (runs on CPU)
python -m umzi_native export checkpoint.safetensors -o model.ummodel

# Inspect model metadata and validate the package without initializing CUDA
python -m umzi_native read model.ummodel
```

### 2. Run inference in Python

Pass a normalized RGB NumPy batch in **NCHW** layout `(N, 3, H, W)` with values
in `[0, 1]`. Point `cache_dir` at a writable folder so tuned GPU plans are saved
on the first run and reused on every run after that.

```python
from pathlib import Path

import numpy as np
import torch  # Makes PyTorch's bundled CUDA libraries available.

import umzi_native as umzi

tiles = np.zeros((1, 3, 128, 128), dtype=np.float32)
options = umzi.SessionOptions(
    precision="mixed",  # Use "fp16" with np.float16 input for fastest throughput.
    cache_dir=Path("cache"),
)

with umzi.InferenceSession("model.ummodel", sess_options=options) as session:
    (restored,) = session.run(None, {"input": tiles})
```

- **Full image-to-PNG walkthrough:** [Python inference quickstart](docs/quickstart.md#1-run-your-first-inference)
- **Tuning & options (`precision`, `cache_dir`, `enable_cuda_graph`, `kernel_options`):**
  [Session options](docs/quickstart.md#3-session-options-sessionoptions) ·
  [Model-specific `kernel_options`](docs/quickstart.md#4-model-specific-kernel_options)

## Documentation

### Getting started
- [Python inference quickstart & session options](docs/quickstart.md)
- [Exporting checkpoints to `.ummodel`](docs/export.md)
- [Installation, CUDA library loading, and model inspection](docs/gaterv3-python.md)
- [Changelog](CHANGELOG.md)

### Runtime & architecture design
- [Multi-kernel runtime and `.ummodel` format spec](docs/umzi-native-runtime-plan.md)
- [GateRV2, MoESR, and GFISRv2 architecture expansion](docs/umzi-native-architecture-expansion.md)
- [Built-in model exporter design](docs/umzi-model-exporter-plan.md)
- [Native runtime refactor notes](docs/gaterv3-native-refactor.md)

### Benchmarks & optimization notes
- [GateRV3 native build, verification, and results](docs/gaterv3-native.md)
- [GateRV3 kernel tuning, FP16 experiments, and VRAM measurements](docs/gaterv3-native-optimization.md)
- [GateRV3 PyTorch & TensorRT-RTX baseline](docs/gaterv3-baseline.md)
- [Manga page tiling and batch-size benchmark](docs/gaterv3-manga-tiling.md)
- [GateRV2](docs/handoff-gaterv2.md) · [MoESR](docs/handoff-moesr.md) · [GFISRv2](docs/handoff-gfisrv2.md) engineering handoffs

## Links

- [umzi](https://github.com/umzi2)
- [GateRV3](https://github.com/umzi2/GaterV3)
- [Resselt](https://github.com/rewaifu/resselt)
