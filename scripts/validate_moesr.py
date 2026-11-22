"""Check private MoESR CUDA kernels and the FP32 execution path."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from benchmark import image_tensor
from benchmark_architectures import (
    FAMILIES,
    ROOT,
    Runner,
    comparison,
    digest,
    read_image,
    score_stress_render,
    ssimulacra2,
    stress_inputs,
    summarize,
    timed_page,
)
from reference.moesr_native import NativeMoESR, NativeStages
from reference.native_stages import Trace
from resselt.archs.moesr.arch import MSG, GatedCNNBlock, LayerNorm, MoESR
from torch import Tensor
from torch.nn import functional

from umzi_native.exporter._loader import load_checkpoint

FAMILY = FAMILIES["moesr"]
BASELINE = ROOT / "output/architecture-baselines/moesr/batch-2/fp32"
PRIMITIVE_TOLERANCE = {"atol": 3e-6, "rtol": 2e-5}
NETWORK_TOLERANCE = {"atol": 2e-5, "rtol": 5e-5}


def parity(label: str, actual: Tensor, expected: Tensor, tolerance: dict[str, float]) -> dict[str, Any]:
    torch.testing.assert_close(
        actual, expected, atol=tolerance["atol"], rtol=tolerance["rtol"], msg=lambda detail: f"{label}\n{detail}"
    )
    return comparison(actual.cpu().numpy(), expected.cpu().numpy())


def check_primitives(native: NativeStages, reference: MoESR) -> dict[str, Any]:
    def ramp(shape: tuple[int, ...]) -> Tensor:
        size = int(np.prod(shape))
        return (torch.arange(size, device="cuda", dtype=torch.float32) - size / 2).reshape(shape) / size

    rgb = ramp((2, 3, 9, 13))
    down = ramp((2, 16, 8, 12))
    up = ramp((2, 256, 4, 6))
    shortcut = ramp((2, 64, 8, 12))
    logits = ramp((2, 48, 8, 12))
    layout = {
        "reflect": (native.pad(rgb, 2), functional.pad(rgb, (0, 1, 0, 1), mode="reflect")),
        "unshuffle2": (native.down(down), functional.leaky_relu(functional.pixel_unshuffle(down, 2), 0.1)),
        "shuffle2_residual": (
            native.up(up, shortcut),
            functional.leaky_relu(functional.pixel_shuffle(up, 2), 0.1) + shortcut,
        ),
        "shuffle4_crop": (native.output(logits, 7, 11), functional.pixel_shuffle(logits, 4)[:, :, :28, :44]),
    }
    rows = {name: parity(name, actual, expected, {"atol": 0, "rtol": 0}) for name, (actual, expected) in layout.items()}
    random = torch.Generator(device="cuda").manual_seed(42)
    for path in ("blocks.0.blocks.0", "blocks.0.msg.gated.0"):
        block = cast(GatedCNNBlock, reference.get_submodule(path))
        sample = torch.randn((2, block.fc1.in_channels, 7, 11), generator=random, device="cuda")
        norm = block.norm
        for name, source in (("noise", sample), ("constant", torch.full_like(sample, 0.5))):
            key = f"{path}.norm.{name}"
            rows[key] = parity(
                key, native.layer_norm(source, norm.weight, norm.bias, norm.eps), norm(source), PRIMITIVE_TOLERANCE
            )
        projected = block.fc1(norm(sample))
        gate, identity, features = projected.split(block.split_indices, dim=1)
        mix = block.conv
        direct, square, row, column = features.split(mix.split_indexes, dim=1)
        convolved = (mix.dwconv_hw(square), mix.dwconv_w(row), mix.dwconv_h(column))
        expected = block.act(gate) * torch.cat((identity, direct, *convolved), dim=1)
        key = f"{path}.gate"
        rows[key] = parity(key, native.gate(projected, *convolved), expected, PRIMITIVE_TOLERANCE)
        projected = block.fc2(expected)
        key = f"{path}.residual"
        rows[key] = parity(
            key,
            native.residual(projected, sample, block.gamma),
            block.act(projected) * block.gamma + sample,
            PRIMITIVE_TOLERANCE,
        )
    return rows


def reference_trace(model: MoESR, sample: Tensor, record: Trace) -> None:
    def hook(name: str):
        def capture(_module, _args, value):
            record(name, value)

        return capture

    handles = [
        model.in_to_dim.register_forward_pre_hook(lambda _module, args: record("padded", args[0])),
        model.upscale[0].register_forward_pre_hook(lambda _module, args: record("outer_residual", args[0])),
    ]
    for name, module in model.named_modules():
        if (
            name in ("in_to_dim", "upscale.0")
            or name.endswith(".msg.down")
            or isinstance(module, (GatedCNNBlock, LayerNorm, MSG))
        ):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        record("output", model(sample))
    finally:
        for handle in handles:
            handle.remove()


def check_trace(model: NativeMoESR, reference: MoESR, sample: Tensor) -> dict[str, Any]:
    expected: dict[str, Tensor] = {}
    stages: dict[str, Any] = {}

    def record(name: str, value: Tensor) -> None:
        expected[name] = value.cpu()

    def compare_stage(name: str, value: Tensor) -> None:
        stages[name] = parity(name, value.cpu(), expected.pop(name), NETWORK_TOLERANCE)

    reference_trace(reference, sample, record)
    model(sample, compare_stage)
    if expected:
        raise RuntimeError(f"Missing native stages: {sorted(expected)}")
    print(f"{tuple(sample.shape)}: {len(stages)} stages match", flush=True)
    return stages


def check_stress(model: NativeMoESR, reference: MoESR, sources: list[Path], output: Path) -> dict[str, Any]:
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    for height, width in ((2, 2), (3, 5), (8, 9)):
        cases[f"tiny-{height}x{width}"] = torch.zeros((1, 3, height, width))
    rows = {}
    for name, sample in cases.items():
        sample = sample.cuda()
        expected, actual = reference(sample), model(sample)
        row = parity(name, actual, expected, NETWORK_TOLERANCE)
        row["input_shape"] = list(sample.shape)
        row["output_shape"] = list(actual.shape)
        if name in ("checkerboard", "gray", "gray-128", "noise"):
            row["ssimulacra2"] = score_stress_render(output / "stress-renders", name, expected.cpu(), actual.cpu())
        if sample.size(0) > 1:
            row["vs_individual"] = [
                parity(f"{name} tile {i}", part, model(tile), NETWORK_TOLERANCE)
                for i, (part, tile) in enumerate(zip(actual.split(1), sample.split(1), strict=True))
            ]
        rows[name] = row
        print(
            f"{name}: max error {row['max_abs']:.6g}, PSNR {row['psnr_db']}, SSIMULACRA2 {row.get('ssimulacra2')}",
            flush=True,
        )
    for height, width in ((1, 1), (1, 2), (2, 1)):
        sample = torch.zeros((1, 3, height, width), device="cuda")
        for run, message in ((reference, "Padding size should be less"), (model, "reflection padding is invalid")):
            try:
                run(sample)
            except (RuntimeError, ValueError) as error:
                if message not in str(error):
                    raise
            else:
                raise AssertionError(f"Expected reflection padding to reject {height}x{width}")
        rows[f"invalid-{height}x{width}"] = {"padding_rejected": True}
    return rows


def check_pages(model: NativeMoESR, sources: list[Path], baseline: Path, output: Path) -> list[dict[str, Any]]:
    runner = Runner(model, torch.float32)
    pages = []
    runner.measure(torch.zeros((FAMILY.batch, 3, FAMILY.tile, FAMILY.tile)))
    for path in sources:
        row, candidate = timed_page(
            read_image(path), path.name, {1: runner, FAMILY.batch: runner}, FAMILY, FAMILY.batch
        )
        expected = np.load(baseline / f"{path.stem}.npy", mmap_mode="r")
        actual = candidate.numpy()
        np.testing.assert_allclose(actual, expected, err_msg=path.name, **NETWORK_TOLERANCE)
        row.update(comparison(actual, expected))
        rendered = output / f"{path.stem}-native-fp32.png"
        image_tensor(candidate).save(rendered)
        row["ssimulacra2"] = ssimulacra2(baseline / f"{path.stem}.png", rendered)
        pages.append(row)
        print(f"{path.name}: {row['psnr_db']} dB, SSIMULACRA2 {row['ssimulacra2']}", flush=True)
    return pages


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/moesr-native-validation")
    parser.add_argument("--reference-report", type=Path, default=BASELINE / "report.json")
    parser.add_argument("--pages", action="store_true")
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "models" / FAMILY.checkpoint
    sources = sorted((ROOT / "samples" / FAMILY.samples).glob("*.png"))
    config = {
        "checkpoint_sha256": digest(checkpoint),
        "images": {p.name: digest(p) for p in sources},
        "tile": FAMILY.tile,
        "batch": FAMILY.batch,
        "overlap": FAMILY.overlap,
        "scale": FAMILY.scale,
    }
    if args.pages:
        baseline = json.loads(args.reference_report.read_text(encoding="utf-8"))
        if (
            baseline["backend"] != "fp32"
            or baseline["status"] != "complete"
            or any(baseline["config"][key] != value for key, value in config.items())
        ):
            raise ValueError("The saved FP32 report does not match the checkpoint, images, or tiling")
    reference, _ = load_checkpoint(checkpoint)
    if not isinstance(reference, MoESR):
        raise TypeError("The validation checkpoint must load as MoESR")
    reference.eval().cuda()
    report: dict[str, Any] = {
        "config": config,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "resselt": version("resselt"),
        "primitive_tolerance": PRIMITIVE_TOLERANCE,
        "network_tolerance": NETWORK_TOLERANCE,
    }
    report_path = args.output / "report.json"
    with NativeStages(args.nvrtc_root) as native:
        model = NativeMoESR(reference, native)
        report["primitives"] = check_primitives(native, reference)
        generator = torch.Generator(device="cuda").manual_seed(1009)
        report["traces"] = {
            "x".join(map(str, shape)): check_trace(
                model, reference, torch.rand(shape, generator=generator, device="cuda")
            )
            for shape in ((1, 3, 32, 48), (2, 3, 33, 47))
        }
        report["stress"] = check_stress(model, reference, sources, args.output)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if args.pages:
            report["pages"] = check_pages(model, sources, args.reference_report.parent, args.output)
            report["summary"] = summarize(report["pages"])
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
