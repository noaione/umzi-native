"""Check the private GateRV2 CUDA stages and network against eager FP32."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from benchmark import image_tensor
from benchmark_architectures import (
    FAMILIES,
    METRIC,
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
from reference.gaterv2 import linear_attention
from reference.gaterv2_native import NativeGateRV2, NativeStages
from resselt.archs.gaterv2.arch import Attention, GateRV2
from torch import Tensor
from torch.nn import functional

from umzi_native.exporter._loader import load_checkpoint

FAMILY = FAMILIES["gaterv2"]
PRIMITIVE_TOLERANCE = {"atol": 2e-6, "rtol": 2e-5}
NETWORK_TOLERANCE = {"atol": 1e-5, "rtol": 5e-5}


def compare_outputs(actual: Tensor, expected: Tensor) -> dict[str, Any]:
    return comparison(actual.cpu().numpy(), expected.cpu().numpy())


def reference_trace(model: GateRV2, sample: Tensor) -> dict[str, Tensor]:
    records: dict[str, Tensor] = {}

    def record(name: str, value: Tensor) -> None:
        records[name] = value.cpu()

    def hook(name: str):
        def capture(_module, _args, value):
            record(name, value)

        return capture

    names = [
        "in_to_dim",
        *(f"encode.{index}.{op}" for index in range(len(model.encode)) for op in ("gated", "scale")),
        *(f"latent.{index}{suffix}" for index in range(len(model.latent)) for suffix in (".token_mix", "")),
        *(f"decode.{index}.{op}" for index in range(len(model.decode)) for op in ("shor", "gated")),
        "dim_to_in",
    ]
    handles = [model.in_to_dim.register_forward_pre_hook(lambda _module, args: record("padded", args[0]))]
    handles.extend(model.get_submodule(name).register_forward_hook(hook(name)) for name in names)
    try:
        record("output", model(sample))
    finally:
        for handle in handles:
            handle.remove()
    return records


def check_attention(native: NativeStages, model: GateRV2) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(23127)
    attention = model.get_submodule("latent.0.token_mix")
    if not isinstance(attention, Attention):
        raise TypeError("GateRV2 linear attention is required")
    channels = attention.query_conv.out_channels
    results = {}
    for batch, tokens in ((1, 64), (2, 25), (3, 80), (16, 64)):
        query = torch.randn((batch, channels, tokens), generator=generator, device="cuda")
        key = torch.randn(query.shape, generator=generator, device="cuda")
        value = torch.randn((batch, attention.value_conv.out_channels, tokens), generator=generator, device="cuda")
        expected = linear_attention(query, key, value, attention.eps)
        actual = native.attention(query, key, value, attention.eps)
        row = compare_outputs(actual, expected)
        results[f"{batch}x{tokens}"] = row
        torch.testing.assert_close(actual, expected, **PRIMITIVE_TOLERANCE)
        print(f"Attention {batch=}, {tokens=}: max abs {row['max_abs']:.6g}", flush=True)
    return results


def check_layout(native: NativeStages, align: int) -> list[list[int]]:
    shapes = [(1, 3, 128, 128), (2, 3, 65, 79), (1, 3, 9, 9), (1, 3, 128, 160)]
    for batch, channels, height, width in shapes:
        source = torch.arange(batch * channels * height * width, dtype=torch.float32, device="cuda")
        source = source.reshape(batch, channels, height, width)
        padded = native.pad(source, align)
        expected = functional.pad(source, (0, (-width) % align, 0, (-height) % align), mode="reflect")
        torch.testing.assert_close(padded, expected, atol=0, rtol=0)
        torch.testing.assert_close(native.crop(padded, height, width), source, atol=0, rtol=0)
    return [list(shape) for shape in shapes]


def traced_cases(reference: GateRV2, native: NativeGateRV2, cases: dict[str, Tensor]) -> dict[str, Any]:
    result = {}
    for label in ("gray", "checkerboard", "noise", "rect-65x79", "rect-128x160"):
        x = cases[label].cuda()
        before = reference_trace(reference, x)
        actual: dict[str, Tensor] = {}
        native(x, lambda name, value, actual=actual: actual.__setitem__(name, value.cpu()))
        if actual.keys() != before.keys():
            raise ValueError(f"{label} stage coverage differs: {actual.keys() ^ before.keys()}")
        stages = {name: compare_outputs(value, before[name]) for name, value in actual.items()}
        if any(not row["finite"] for row in stages.values()):
            raise ValueError(f"{label} contains non-finite stage output")
        result[label] = stages
        first = next((name for name, errors in stages.items() if not errors["finite"] or errors["mse"]), None)
        print(f"{label}: first difference {first}, output {stages['output']}", flush=True)
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/gaterv2-native-validation")
    parser.add_argument("--nvrtc-root", type=Path)
    parser.add_argument("--pages", action="store_true", help="Also measure the two benchmark pages")
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "models" / FAMILY.checkpoint
    sources = sorted((ROOT / "samples" / FAMILY.samples).glob("*.jpg"))
    reference, _ = load_checkpoint(checkpoint)
    if not isinstance(reference, GateRV2):
        raise TypeError("The validation checkpoint is not GateRV2")
    reference.eval().cuda()
    cases = dict(stress_inputs(FAMILY.tile, sources, FAMILY.batch))
    report: dict[str, Any] = {
        "checkpoint_sha256": digest(checkpoint),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "resselt": version("resselt"),
        "primitive_tolerance": PRIMITIVE_TOLERANCE,
        "output_tolerance": NETWORK_TOLERANCE,
        "inputs": {path.name: digest(path) for path in sources},
    }
    with NativeStages(args.nvrtc_root) as native:
        model = NativeGateRV2(reference, native)
        report["layout_shapes"] = check_layout(native, reference.pad)
        report["attention"] = check_attention(native, reference)
        report["traces"] = traced_cases(reference, model, cases)
        stress = {}
        for label, sample in cases.items():
            x = sample.cuda()
            expected, actual = reference(x), model(x)
            torch.testing.assert_close(actual, expected, msg=f"GateRV2 network parity for {label}", **NETWORK_TOLERANCE)
            stress[label] = compare_outputs(actual, expected)
            if label == "checkerboard":
                stress[label]["ssimulacra2"] = score_stress_render(
                    args.output / "stress-renders", label, expected, actual
                )
            if len(x) > 1:
                stress[label]["vs_individual"] = [
                    compare_outputs(item, model(tile)[0]) for item, tile in zip(actual, x.split(1), strict=True)
                ]
            print(f"{label}: {stress[label]['psnr_db']} dB", flush=True)
        report["stress"] = stress
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if not args.pages:
            return
        if not METRIC.is_file():
            raise FileNotFoundError(METRIC)
        native_runner, eager_runner = Runner(model, torch.float32), Runner(reference, torch.float32)
        pages = []
        for path in sources:
            image = read_image(path)
            row, candidate = timed_page(
                image, path.name, {1: native_runner, FAMILY.batch: native_runner}, FAMILY, FAMILY.batch
            )
            reference_row, expected = timed_page(
                image, path.name, {1: eager_runner, FAMILY.batch: eager_runner}, FAMILY, FAMILY.batch
            )
            torch.testing.assert_close(
                candidate, expected, msg=f"GateRV2 page parity for {path.name}", **NETWORK_TOLERANCE
            )
            row.update(compare_outputs(candidate, expected))
            if not row["finite"]:
                raise ValueError(f"Non-finite output for {path.name}")
            ref, rendered = args.output / f"{path.stem}-fp32.png", args.output / f"{path.stem}-native.png"
            image_tensor(expected).save(ref)
            image_tensor(candidate).save(rendered)
            row["ssimulacra2"] = ssimulacra2(ref, rendered)
            row["reference_timing"] = reference_row
            pages.append(row)
            print(f"{path.name}: {row['psnr_db']} dB, SSIMULACRA2 {row['ssimulacra2']}", flush=True)
        report["pages"] = pages
        report["summary"] = summarize(pages)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
