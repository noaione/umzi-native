# Export and inspect `.ummodel` packages

Every supported model family uses the same exporter CLI, Python function, and `.ummodel` container format. Once exported, open the file with `InferenceSession` as shown in the [quickstart](quickstart.md).

## Install the exporter

The `exporter` extra installs Resselt and PyTorch for CPU-side conversion from `.safetensors`, `.pth`, `.pt`, and `.ckpt` state dictionaries. You do not need this extra if you only run inference on existing `.ummodel` files.

```powershell
# From a local checkout
python -m pip install ".[exporter]"

# From a published wheel
python -m pip install "umzi-native[exporter]"
```

## Export a checkpoint

```powershell
python -m umzi_native export --help
python -m umzi_native export CHECKPOINT -o MODEL.ummodel [--kernel FAMILY]
```

The exporter detects the architecture automatically; passing `--kernel` makes export fail if the detected family does not match. The output path must end in `.ummodel`. Packages are validated before replacing the destination file, embed the source checkpoint's SHA-256 digest, and produce byte-identical output on repeated runs.

From Python, `export_model()` runs the same flow and returns a `ModelInfo`:

```python
from umzi_native.exporter import export_model

info = export_model("models/4x_dwtp_ds_moesr_v2.pth", "output/moesr.ummodel", kernel="moesr")
```

## Supported checkpoints and topologies

`gaterv3` and `gaterv2` accept compatible architecture configurations (within their supported channel and depth limits). `moesr`, `gfisrv2`, `smosr`, and `figsr` currently target the specific checkpoint topologies below and reject checkpoints with different layer counts or channel sizes.

| Family (`--kernel`) | Reference checkpoint | Scale | Topology / constraints |
| --- | --- | ---: | --- |
| `gaterv3` | `1x_umzi_digital_decompress_gaterv3_1.pth` | 1× | 32 base channels, 16 px alignment (4 encoder/decoder levels) |
| `gaterv3` | `1x_umzi_idc_gaterv3.safetensors` | 1× | IDC variant, 32 base channels, 8 px alignment (3 encoder/decoder levels) |
| `gaterv2` | `1x_umzi_dehalftone_gater2_v1_smooth.safetensors` | 1× | RGB 1×, 16–64 base channels (divisible by 16), 1–4 encoder/decoder levels |
| `moesr` | `4x_dwtp_ds_moesr_v2.pth` | 4× | DWTP DS direct-shuffle topology (64 channels, 9 groups) |
| `gfisrv2` | `4x_wtp_MangaScale_GfisrV2.pth` | 4× | MangaScale topology (48 channels, 24 blocks, direct shuffle) |
| `smosr` | `2x_enhancr_da_smosr_v1.safetensors` | 2× | 48 channels, 6 blocks (2+3+1), direct 2× shuffle |
| `figsr` | `2x_enhancr_da_figsr.safetensors` | 2× | 32 channels, 24 blocks (2×12), direct 2× shuffle |

Examples:

```powershell
python -m umzi_native export models/1x_umzi_digital_decompress_gaterv3_1.pth -o output/native/gaterv3.ummodel
python -m umzi_native export models/1x_umzi_dehalftone_gater2_v1_smooth.safetensors -o output/native/gaterv2.ummodel --kernel gaterv2
python -m umzi_native export models/4x_dwtp_ds_moesr_v2.pth -o output/native/moesr.ummodel --kernel moesr
python -m umzi_native export models/4x_wtp_MangaScale_GfisrV2.pth -o output/native/gfisrv2.ummodel --kernel gfisrv2
python -m umzi_native export models/2x_enhancr_da_smosr_v1.safetensors -o output/native/smosr.ummodel --kernel smosr
python -m umzi_native export models/2x_enhancr_da_figsr.safetensors -o output/native/figsr.ummodel --kernel figsr
```

> [!NOTE]
> Earlier schema 1 SMoSR packages only supported `"mixed"` and `"fp16"`. Re-export the checkpoint to get a schema 2 package if you want `"fp32"` mode.

## Inspect a package

You can validate a `.ummodel` file and inspect its metadata without initializing CUDA or loading the native library:

```powershell
python -m umzi_native read output/native/moesr.ummodel
python -m umzi_native read --json output/native/moesr.ummodel
```

In Python, call `umzi_native.peek_model()`:

```python
import umzi_native as umzi

info = umzi.peek_model("output/native/moesr.ummodel")
print(info.family_name, info.supported, info.problems)
```

`peek_model()` verifies the container header, SHA-256 checksum, kernel family, payload schema, and declared architecture:
- If the file is truncated, malformed, or fails its checksum, it raises `ValueError`.
- If the file is well-formed but unsupported by this build, `info.supported` is `False` and `info.problems` explains why (the CLI `read` command exits non-zero in both cases).
- Shape-dependent validation still runs when `InferenceSession` opens or prepares the model.

For the binary `.ummodel` layout and C ABI details, see the [runtime and format spec](umzi-native-runtime-plan.md).
