# Export a checkpoint to `.ummodel`

Every native kernel family uses the same exporter, command, and package
envelope. After export, open the file with `InferenceSession`, as described in
the [quickstart](quickstart.md).

## Install the exporter

The `exporter` extra installs Resselt and its dependencies, including Torch.
Export runs on the CPU and accepts `.safetensors`, `.pth`, `.pt`, and `.ckpt`
tensor state dictionaries. The base install can read and run packages
without it. Resselt is imported only when you export.

```powershell
# From the checkout
python -m pip install ".[exporter]"
# From a published wheel
python -m pip install "umzi-native[exporter]"
```

## Export

```powershell
python -m umzi_native export --help
python -m umzi_native export CHECKPOINT -o MODEL.ummodel [--kernel FAMILY]
```

The exporter detects the family from the checkpoint. `--kernel` makes it
fail unless the detected family matches. The output name must end in
`.ummodel`. The exporter validates the package before replacing the
destination, and repeated exports of the same checkpoint are byte-identical.
The package records the source checkpoint's SHA-256.

The Python equivalent returns the package's `ModelInfo`:

```python
from umzi_native.exporter import export_model

info = export_model("models/4x_dwtp_ds_moesr_v2.pth", "output/moesr.ummodel", kernel="moesr")
```

## Supported checkpoints

| Family (`--kernel`) | Checkpoint | Scale | Notes |
| --- | --- | ---: | --- |
| `gaterv3` | `1x_umzi_digital_decompress_gaterv3_1.pth` | 1× | 16-pixel alignment |
| `gaterv3` | `1x_umzi_idc_gaterv3.safetensors` | 1× | IDC fine-tune, 8-pixel alignment |
| `gaterv2` | `1x_umzi_dehalftone_gater2_v1_smooth.safetensors` | 1× | Dehalftone, 16-pixel alignment |
| `moesr` | `4x_dwtp_ds_moesr_v2.pth` | 4× | DWTP DS direct-shuffle topology only |
| `gfisrv2` | `4x_wtp_MangaScale_GfisrV2.pth` | 4× | FP32 weights, MangaScale 24-block topology |

For example:

```powershell
python -m umzi_native export models/1x_umzi_digital_decompress_gaterv3_1.pth -o output/native/gaterv3.ummodel
python -m umzi_native export models/1x_umzi_dehalftone_gater2_v1_smooth.safetensors -o output/native/gaterv2.ummodel --kernel gaterv2
python -m umzi_native export models/4x_dwtp_ds_moesr_v2.pth -o output/native/moesr.ummodel --kernel moesr
python -m umzi_native export models/4x_wtp_MangaScale_GfisrV2.pth -o output/native/gfisrv2.ummodel --kernel gfisrv2
```

A checkpoint with a different architecture or configuration is rejected
with the reason.

## Inspect a package

```powershell
python -m umzi_native read output/native/moesr.ummodel
python -m umzi_native read --json output/native/moesr.ummodel
```

`read` and `umzi_native.peek_model()` check the container, checksum, family,
payload schema, and architecture without loading CUDA.
[Package inspection details](gaterv3-python.md#inspect-a-model-package) ·
[Runtime and format spec](umzi-native-runtime-plan.md)
